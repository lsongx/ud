import os.path as osp

import torch
from torch import nn

from .vgg_official import VGG, make_layers, cfgs

model_urls = {
    'vgg16': 'vgg16-397923af.pth',
    'vgg16bn': 'vgg16_bn-6c64b313.pth',
    'vgg19bn': 'vgg19_bn-c79401a0.pth',
}


class VGGSelectiveFC(nn.Module):
    def __init__(self, model, num_classes=20, alpha=1, dropout=0):
        super().__init__()
        self.alpha = alpha
        self.features = model.features
        self.num_classes = num_classes

        self.max_pooling = nn.AdaptiveMaxPool2d((7, 7))
        self.fc_after_pooling = nn.Sequential(*model.classifier[:-2])
        self.pooling_bce = nn.Linear(model.out_feat_dim, num_classes, bias=False)
        # self.pooling_softmax = nn.Linear(model.out_feat_dim, num_classes)
        self.pooling_softmax_fc = nn.Sequential(
            nn.Linear(model.out_feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            # nn.Linear(model.out_feat_dim, num_classes),
        )

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        self.get_ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self.get_bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, x, target):
        batch_size = x.size(0)
        loss = torch.zeros(batch_size).to(x.device)

        x = self.features(x)
        x = self.max_pooling(x).view(batch_size, -1)
        x = self.fc_after_pooling(x)
        bce_out = self.pooling_bce(x)

        if self.training:
            bce_loss = self.get_bce_loss(bce_out, target) / self.num_classes

            # selective fc loss
            softmax_fc_out = self.pooling_softmax_fc(x)
            selective_fc_loss = 0
            for softmax_fc_out_i, target_i in zip(softmax_fc_out, target):
                tmp_out = []
                tmp_target = []
                labels = target_i.nonzero()
                for label in labels:
                    mask = target_i.clone().byte()
                    mask[label] = 0
                    mask = ~mask
                    tmp_target.append(target_i[mask].nonzero()[0])
                    tmp_out.append(softmax_fc_out_i[mask].view((1,-1)))
                tmp_out = torch.cat(tmp_out, dim=0)
                tmp_target = torch.cat(tmp_target, dim=0)
                selective_fc_loss += self.get_ce_loss(tmp_out, tmp_target)
            selective_fc_loss /= len(labels)

            loss[0] = (bce_loss + selective_fc_loss * self.alpha)/2

        return bce_out, loss


def vgg16(model_path_prefix=None, **kwargs):
    model = VGG(make_layers(cfgs['D'], batch_norm=False), init_weights=False)
    model.out_feat_dim = 4096
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['vgg16'])
        ))
    return VGGSelectiveFC(model, **kwargs)


def vgg16bn(model_path_prefix=None, **kwargs):
    model = VGG(make_layers(cfgs['D'], batch_norm=True), init_weights=False)
    model.out_feat_dim = 4096
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['vgg16bn'])
        ))
    return VGGSelectiveFC(model, **kwargs)


def vgg19bn(model_path_prefix=None, **kwargs):
    model = VGG(make_layers(cfgs['E'], batch_norm=True), init_weights=False)
    model.out_feat_dim = 4096
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['vgg19bn'])
        ))
    return VGGSelectiveFC(model, **kwargs)
