import torch
from torch import nn

class MLCosKD(nn.Module):
    def __init__(self, s_net, t_net, cls_criterion=None, alpha=0.5, temperature=3):
        super().__init__()
        self.s_net = s_net
        self.t_net = t_net
        self.cls_criterion = cls_criterion
        self.temperature = temperature
        self.cls_balance = cls_balance
        self.alpha = alpha
        self.image_normalization_mean = self.s_net.image_normalization_mean
        self.image_normalization_std = self.s_net.image_normalization_std

        self.get_cos = nn.CosineSimilarity(dim=0)

    def forward(self, x, target):
        batch_size = x.shape[0]
        loss = torch.zeros(batch_size).to(x.device)

        s_feat_out = self.s_net.features(x)
        s_p1 = self.s_net.max_pooling(s_feat_out).view(batch_size, -1)
        s_p2 = self.s_net.avg_pooling(s_feat_out).view(batch_size, -1)
        s_feat = torch.cat([s_p1, s_p2], dim=1)
        s_out = self.s_net.pooling_cat_fc(s_feat)

        if self.training:
            with torch.no_grad():
                t_feat_out = self.t_net.features(x)
                t_p1 = self.t_net.max_pooling(t_feat_out).view(batch_size, -1)
                t_p2 = self.t_net.avg_pooling(t_feat_out).view(batch_size, -1)
                t_feat = torch.cat([t_p1, t_p2], dim=1)
                t_out = self.t_net.pooling_cat_fc(t_feat)

            # cos loss
            cos_loss = 0
            for s_feat_i, t_feat_i in zip(s_feat, t_feat):
                with torch.no_grad():
                    # weight shape: [num_classes, feat_length]
                    t_fc_weight = self.t_net.pooling_cat_fc.weight
                    t_sim = self.get_cos(t_feat_i, t_fc_weight)
                s_fc_weight = self.s_net.pooling_cat_fc.weight
                s_sim = self.get_cos(s_feat_i, s_fc_weight)

                select_kd_loss += tmp_loss / len(labels)

            # multi kd loss
            kd_loss_1 = nn.functional.kl_div(
                (s_out / self.temperature).sigmoid().log(),
                (t_out / self.temperature).sigmoid(),
                reduction="sum",
            )
            kd_loss_2 = nn.functional.kl_div(
                (1-(s_out / self.temperature).sigmoid()).log(),
                (1-(t_out / self.temperature).sigmoid()),
                reduction="sum",
            )
            kd_loss = (kd_loss_1 + kd_loss_2) *\
                      (self.temperature ** 2) / self.s_net.num_classes
            loss[0] = select_kd_loss * self.alpha + kd_loss * (1-self.alpha)

            if self.cls_balance < 1:
                cls_loss = self.cls_criterion(s_out, target)
                loss[0] = cls_loss * (1-self.cls_balance) + loss[0] * self.cls_balance
        return s_out, loss


