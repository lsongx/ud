from torch import nn

class KDLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, x, target):
        loss = nn.functional.kl_div(
            (x / self.temperature).log_softmax(dim=1),
            (target / self.temperature).softmax(dim=1),
            reduction="batchmean",
        ) * (self.temperature ** 2)
        return loss
