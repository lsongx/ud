import os.path as osp

import torch
from torch import nn

from .resnet_official import ResNet, BasicBlock, Bottleneck

model_urls = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
}


class ResNetMaxPooling(nn.Module):
    def __init__(self, model, num_classes=20):
        super().__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes

        self.max_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.pooling = nn.Linear(512 * model.expansion, num_classes, bias=False)
        # self.pooling_cat_fc = nn.Linear(512 * model.expansion * 2, num_classes, bias=False)
        # self.pooling_for_softmax = nn.Linear(512*model.expansion*2, num_classes*2, bias=False)
        # self.pooling_cat_fc1 = nn.Linear(512 * model.expansion * 2, 1024)
        # self.bn = nn.BatchNorm1d(1024)
        # self.final_fc2 = nn.Linear(1024, num_classes)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        x = self.features(x)
        
        x = self.max_pooling(x).view(x.size(0), -1)
        x = self.pooling(x)

        # x1 = self.max_pooling(x).view(x.size(0), -1)
        # x2 = self.avg_pooling(x).view(x.size(0), -1)
        # x = self.pooling_cat_fc(torch.cat([x1, x2], dim=1))
        
        # x = self.pooling_for_softmax(torch.cat([x1, x2], dim=1))
        # x = x.view(x.shape[0], 2, self.num_classes)

        # x = self.pooling_cat_fc1(torch.cat([x1, x2], dim=1))
        # x = self.bn(x)
        # x = self.final_fc2(x)

        return x




def resnet18(model_path_prefix=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    model.expansion = BasicBlock.expansion
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['resnet18'])
        ))
    return ResNetMaxPooling(model, **kwargs)


def resnet34(model_path_prefix=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    model.expansion = BasicBlock.expansion
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['resnet34'])
        ))
    return ResNetMaxPooling(model, **kwargs)


def resnet50(model_path_prefix=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    model.expansion = Bottleneck.expansion
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['resnet50'])
        ))
    return ResNetMaxPooling(model, **kwargs)


def resnet101(model_path_prefix=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    model.expansion = Bottleneck.expansion
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['resnet101'])
        ))
    return ResNetMaxPooling(model, **kwargs)


def resnet152(model_path_prefix=None, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    model.expansion = Bottleneck.expansion
    if model_path_prefix is not None:
        model.load_state_dict(torch.load(
            osp.join(model_path_prefix, model_urls['resnet152'])
        ))
    return ResNetMaxPooling(model, **kwargs)
