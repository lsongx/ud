import torch
from torch import nn

class SelectiveFCKD(nn.Module):
    def __init__(self, s_net, t_net, cls_criterion=None, cls_balance=0, alpha=1, temperature=3, cls_alpha=0.1):
        super().__init__()
        self.s_net = s_net
        self.t_net = t_net
        self.cls_criterion = cls_criterion
        self.temperature = temperature
        self.cls_balance = cls_balance
        self.alpha = alpha
        self.cls_alpha = cls_alpha
        self.image_normalization_mean = self.s_net.image_normalization_mean
        self.image_normalization_std = self.s_net.image_normalization_std
        self.get_ce_loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, x, target):
        batch_size = x.shape[0]
        loss = torch.zeros(batch_size).to(x.device)

        s_out = self.s_net.features(x)
        s_pooling_out = self.s_net.max_pooling(s_out).view(batch_size, -1)
        s_bce_out = self.s_net.pooling_bce(s_pooling_out)

        if self.training:
            s_softmax_fc_out = self.s_net.pooling_softmax_fc(s_pooling_out)

            if self.cls_balance < 1:
                with torch.no_grad():
                    self.t_net.eval()
                    t_out = self.t_net.features(x)
                    t_pooling_out = self.t_net.max_pooling(t_out).view(batch_size, -1)
                    if hasattr(self.t_net, 'fc_after_pooling'):
                        t_pooling_out = self.t_net.fc_after_pooling(t_pooling_out)
                    t_bce_out = self.t_net.pooling_bce(t_pooling_out)
                    t_softmax_fc_out = self.t_net.pooling_softmax_fc(t_pooling_out)

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
                loss[0] = (select_kd_loss * self.alpha + kd_loss)/2
                # loss[0] = select_kd_loss

            if self.cls_balance > 0:
                selective_fc_loss = 0
                for softmax_fc_out_i, target_i in zip(s_softmax_fc_out, target):
                    tmp_out = []
                    tmp_target = []
                    labels = target_i.nonzero()
                    for label in labels:
                        mask = target_i.clone().byte()
                        mask[label] = 0
                        mask = ~mask
                        tmp_target.append(target_i[mask].nonzero()[0])
                        tmp_out.append(softmax_fc_out_i[mask].view((1, -1)))
                    tmp_out = torch.cat(tmp_out, dim=0)
                    tmp_target = torch.cat(tmp_target, dim=0)
                    selective_fc_loss += self.get_ce_loss(tmp_out, tmp_target)
                selective_fc_loss /= len(labels)
                bce_cls_loss = self.cls_criterion(s_bce_out, target) / self.s_net.num_classes
                cls_loss = (bce_cls_loss + selective_fc_loss*self.cls_alpha) / 2
                loss[0] = cls_loss * self.cls_balance + loss[0] * (1-self.cls_balance)
        return s_bce_out, loss


