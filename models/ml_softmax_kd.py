import torch
from torch import nn

class MLSoftmaxKD(nn.Module):
    def __init__(self, s_net, t_net, cls_criterion=None, alpha=1, temperature=3):
        super().__init__()
        self.s_net = s_net
        self.t_net = t_net
        self.cls_criterion = cls_criterion
        self.temperature = temperature
        self.alpha = alpha
        self.image_normalization_mean = self.s_net.image_normalization_mean
        self.image_normalization_std = self.s_net.image_normalization_std

    def forward(self, x, target):
        s_out = self.s_net(x)
        loss = torch.zeros(x.shape[0]).to(x.device)
        if self.training:
            with torch.no_grad():
                t_out = self.t_net(x)
            kd_loss = nn.functional.kl_div(
                (s_out / self.temperature).log_softmax(dim=1),
                (t_out / self.temperature).softmax(dim=1),
                reduction="sum",
            ) * (self.temperature ** 2) / self.s_net.num_classes
            if self.cls_criterion is not None:
                cls_loss = self.cls_criterion(s_out, target)
            loss[0] = cls_loss * (1-self.alpha) + kd_loss * self.alpha
        return s_out, loss


