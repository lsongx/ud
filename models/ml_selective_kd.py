import torch
from torch import nn

class MLSelectiveKD(nn.Module):
    def __init__(self, s_net, t_net, cls_criterion=None, cls_balance=1, alpha=0.5, temperature=3):
        super().__init__()
        self.s_net = s_net
        self.t_net = t_net
        self.cls_criterion = cls_criterion
        self.temperature = temperature
        self.cls_balance = cls_balance
        self.alpha = alpha
        self.image_normalization_mean = self.s_net.image_normalization_mean
        self.image_normalization_std = self.s_net.image_normalization_std

    def forward(self, x, target):
        s_out = self.s_net(x)
        batch_size = x.shape[0]
        loss = torch.zeros(batch_size).to(x.device)
        if self.training:
            with torch.no_grad():
                t_out = self.t_net(x)

            # selective kd loss
            select_kd_loss = 0
            for s_out_i, t_out_i, target_i in zip(s_out, t_out, target):
                tmp_loss = 0
                labels = target_i.nonzero()
                for label in labels:
                    mask = target_i.clone().byte()
                    mask[label] = 0
                    mask = ~mask
                    s_prob = (s_out_i[mask] / self.temperature).softmax(dim=0)
                    t_prob = (t_out_i[mask] / self.temperature).softmax(dim=0)
                    tmp_loss += nn.functional.kl_div(
                        s_prob.log(), t_prob, reduction="sum",
                    ) * (self.temperature ** 2)
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


