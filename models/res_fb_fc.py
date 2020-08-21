import os.path as osp

import torch
from torch import nn

from .resnet_official import ResNet, BasicBlock, Bottleneck
from .mobilenet_official import MobileNetV2
from .vgg_official import VGG, make_layers, cfgs

model_urls = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'ig_resnext101_32x8-c38310e5.pth',
    'resnext101_32x16d': 'ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'ig_resnext101_32x32-e4b90b00.pth',
    'mobilenet_v2': 'mobilenet_v2-b0353104.pth',
    'vgg16': 'vgg16-397923af.pth',
}


class ResFBFC(nn.Module):
    def __init__(self, model, num_classes=20, alpha=1, dropout=0, use_feat_layer=False):
        super().__init__()
        self.alpha = alpha
        self.out_feat_dim = model.out_feat_dim
        if use_feat_layer:
            self.features = model.features
        else:
            self.features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
                # nn.Dropout2d(dropout),
            )
        self.num_classes = num_classes

        self.max_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.pooling_bce = nn.Linear(self.out_feat_dim, num_classes, bias=False)
        # self.pooling_softmax = nn.Linear(self.out_feat_dim, num_classes)
        self.pooling_softmax_fc = nn.Sequential(
            nn.Linear(self.out_feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            # nn.Linear(self.out_feat_dim, num_classes),
        )

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        # self.get_ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self.get_ce_loss = nn.KLDivLoss(reduction='sum')
        self.get_bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, x, target):
        batch_size = x.size(0)
        loss = torch.zeros(batch_size).to(x.device)

        x = self.features(x)
        x = self.max_pooling(x).view(batch_size, -1)
        bce_out = self.pooling_bce(x)

        if self.training:
            bce_loss = self.get_bce_loss(bce_out, target) / self.num_classes

            # selective fc loss
            softmax_fc_out = self.pooling_softmax_fc(x)
            target = target.float() / target.sum(dim=1, keepdim=True)
            softmax_loss = self.get_ce_loss(softmax_fc_out.log(), target)

            loss[0] = (bce_loss + softmax_loss * self.alpha)/2

        return bce_out, loss


def resnet18(model_path_prefix=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    model.out_feat_dim = 512
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['resnet18'])
        ))
    return ResFBFC(model, **kwargs)


def resnet34(model_path_prefix=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    model.out_feat_dim = 512
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['resnet34'])
        ))
    return ResFBFC(model, **kwargs)


def resnet50(model_path_prefix=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    model.out_feat_dim = 2048
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['resnet50'])
        ))
    return ResFBFC(model, **kwargs)


def resnet101(model_path_prefix=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    model.out_feat_dim = 2048
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['resnet101'])
        ))
    return ResFBFC(model, **kwargs)


def resnet152(model_path_prefix=None, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    model.out_feat_dim = 2048
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['resnet152'])
        ))
    return ResFBFC(model, **kwargs)


def resnext50_32x4d(model_path_prefix=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4)
    model.out_feat_dim = 2048
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['resnext50_32x4d'])
        ))
    return ResFBFC(model, **kwargs)


def resnext101_32x8d(model_path_prefix=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8)
    model.out_feat_dim = 2048
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['resnext101_32x8d'])
        ))
    return ResFBFC(model, **kwargs)


def resnext101_32x16d(model_path_prefix=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=16)
    model.out_feat_dim = 2048
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['resnext101_32x16d'])
        ))
    return ResFBFC(model, **kwargs)


def resnext101_32x32d(model_path_prefix=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=32)
    model.out_feat_dim = 2048
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['resnext101_32x32d'])
        ))
    return ResFBFC(model, **kwargs)


def mobilenet_v2(model_path_prefix=None, **kwargs):
    model = MobileNetV2()
    model.out_feat_dim = 1280
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['mobilenet_v2'])
        ))
    return ResFBFC(model, use_feat_layer=True, **kwargs)


def vgg16(model_path_prefix=None, **kwargs):
    model = VGG(make_layers(cfgs['D'], batch_norm=False), init_weights=False)
    model.out_feat_dim = 512
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['vgg16'])
        ))
    return ResFBFC(model, use_feat_layer=True, **kwargs)
