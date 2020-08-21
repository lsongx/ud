import torch
from torch import nn

class SelectiveFCKDAtt(nn.Module):
    def __init__(self, s_net, t_net, cls_criterion=None, cls_balance=0, alpha=1, mimic_alpha=0.1, temperature=3):
        super().__init__()
        self.s_net = s_net
        self.t_net = t_net
        self.cls_criterion = cls_criterion
        self.temperature = temperature
        self.cls_balance = cls_balance
        self.alpha = alpha
        self.mimic_alpha = mimic_alpha
        self.image_normalization_mean = self.s_net.image_normalization_mean
        self.image_normalization_std = self.s_net.image_normalization_std
        self.get_ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self.get_mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, x, target):
        batch_size = x.shape[0]
        loss = torch.zeros(batch_size).to(x.device)

        s_out = self.s_net.features(x)
        s_pooling_out = self.s_net.max_pooling(s_out).view(batch_size, -1)
        s_bce_out = self.s_net.pooling_bce(s_pooling_out)

        if self.training:
            # out = nn.functional.dropout(out, p=0.2)
            s_softmax_fc_out = self.s_net.pooling_softmax_fc(s_pooling_out)

            if self.cls_balance < 1:
                with torch.no_grad():
                    self.t_net.eval()
                    t_out = self.t_net.features(x)
                    t_pooling_out = self.t_net.max_pooling(t_out).view(batch_size, -1)
                    t_bce_out = self.t_net.pooling_bce(t_pooling_out)
                    t_softmax_fc_out = self.t_net.pooling_softmax_fc(t_pooling_out)

                t_out.requires_grad = True
                t_pooling_out_tmp = self.t_net.max_pooling(t_out).view(batch_size, -1)
                t_bce_out_tmp = self.t_net.pooling_bce(t_pooling_out_tmp)
                t_softmax_fc_out_tmp = self.t_net.pooling_softmax_fc(t_pooling_out_tmp)
                diff = nn.functional.kl_div(
                    t_bce_out_tmp.softmax(dim=1).log(), 
                    t_softmax_fc_out_tmp.softmax(dim=1), reduction='sum')
                diff.backward()
                self.t_net.pooling_bce.zero_grad()
                self.t_net.pooling_softmax_fc.zero_grad()
                t_att = (t_out.grad.sum(dim=[2,3], keepdim=True).softmax(dim=1) *
                         t_out).sum(dim=1).detach()

                # mimic loss
                s_out_tmp = s_out.clone().detach()
                s_out_tmp.requires_grad = True
                s_pooling_out_tmp = self.s_net.max_pooling(s_out_tmp).view(batch_size, -1)
                s_bce_out_tmp = self.s_net.pooling_bce(s_pooling_out_tmp)
                s_softmax_fc_out_tmp = self.s_net.pooling_softmax_fc(s_pooling_out_tmp)
                diff = nn.functional.kl_div(
                    s_bce_out_tmp.softmax(dim=1).log(), 
                    s_softmax_fc_out_tmp.softmax(dim=1), reduction='sum')
                diff.backward()
                self.s_net.pooling_bce.zero_grad()
                self.s_net.pooling_softmax_fc.zero_grad()
                s_att = (s_out_tmp.grad.sum(
                         dim=[2,3], keepdim=True).softmax(dim=1).detach() * s_out).sum(dim=1)

                mimic_loss = self.get_mse_loss(s_att, t_att) * self.mimic_alpha

                # selective kd loss
                select_kd_loss = nn.functional.kl_div(
                    (s_softmax_fc_out / self.temperature).softmax(dim=1).log(),
                    (t_softmax_fc_out / self.temperature).softmax(dim=1),
                    reduction="sum",
                ) * (self.temperature ** 2)

                # multi kd loss
                kd_loss_1 = nn.functional.kl_div(
                    (s_bce_out / self.temperature).sigmoid().log(),
                    (t_bce_out / self.temperature).sigmoid(),
                    reduction="sum",
                )
                kd_loss_2 = nn.functional.kl_div(
                    (1-(s_bce_out / self.temperature).sigmoid()).log(),
                    (1-(t_bce_out / self.temperature).sigmoid()),
                    reduction="sum",
                )
                kd_loss = (kd_loss_1 + kd_loss_2) / 2 *\
                        (self.temperature ** 2) / self.s_net.num_classes
                loss[0] = (select_kd_loss * self.alpha + kd_loss)/2 + mimic_loss
                # loss[0] = select_kd_loss

        return s_bce_out, loss


