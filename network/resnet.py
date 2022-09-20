import random

import numpy as np
import torch
from torch import nn
from torchvision import models


def batch_norm_to_group_norm(layer):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = torch.nn.GroupNorm(num_channels, num_channels)
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = batch_norm_to_group_norm(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer

def batch_norm_to_instance_norm(layer):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = torch.nn.InstanceNorm2d(num_channels, affine=False, momentum=0.)
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = batch_norm_to_instance_norm(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer



def batch_norm_eval(layer):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    sub_layer.eval()
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                batch_norm_eval(sub_layer)


class Resnet34Local(nn.Module):

    def __init__(self, K, freeze_0=True):
        super(Resnet34Local, self).__init__()
        self.net = models.resnet34(pretrained=True)

        self.net = batch_norm_to_group_norm(self.net)
        self.MAX_K = 8

        self.K = K
        self.freeze_0 = freeze_0

        self.net.conv1.stride = 3

        layer0 = nn.Sequential(
                          self.net.conv1, self.net.bn1, self.net.relu, self.net.maxpool
                      )

        # Resblocks (#blocks x memory consumption)
        # layer1: 3 X 8 into 3 groups of 1x8
        layer1_1 = self.net.layer1[0]
        layer1_2 = self.net.layer1[1]
        layer1_3 = self.net.layer1[2]

        # layer2: 4 X 4 into 2 groups of 2x4
        layer2_1 = nn.Sequential(self.net.layer2[0], self.net.layer2[1])
        layer2_2 = nn.Sequential(self.net.layer2[2], self.net.layer2[3])

        # layer3: 6 X 2 into 2 groups of 3x2
        layer3_1 = nn.Sequential(self.net.layer3[0], self.net.layer3[1], self.net.layer3[2])
        layer3_2 = nn.Sequential(self.net.layer3[3], self.net.layer3[4], self.net.layer3[5])

        # layer4: 3 X 1 unchanged
        layer4 = self.net.layer4  # 3 X 1
        self.net_K_list = nn.Sequential(
            layer0,
            layer1_1,
            layer1_2,
            layer1_3,
            layer2_1,
            layer2_2,
            layer3_1,
            layer3_2,
            layer4,
        )

    def forward(self, x, ki=-1):
        if ki == -1:
            for i, net in enumerate(self.net_K_list):
                x = net(x)
                # print("stage: %d, "%i, x.shape)
                # self.print_sparsity(x, i)
        elif ki == 0:
            if self.freeze_0:
                self.net_K_list[0].eval()
                with torch.no_grad():
                    x = self.net_K_list[0](x)
            else:
                x = self.net_K_list[0](x)
        else:
            sep_gap = self.MAX_K // self.K
            for i in range((ki - 1) * sep_gap + 1, ki * sep_gap + 1):
                x = self.net_K_list[i](x)
        return x


class Resnet34LocalBatchNorm(nn.Module):

    def __init__(self, K, freeze_0=True):
        super(Resnet34LocalBatchNorm, self).__init__()
        self.net = models.resnet34(pretrained=True)

        self.MAX_K = 8

        self.K = K
        self.freeze_0 = freeze_0

        self.net.conv1.stride = 3

        layer0 = nn.Sequential(
                          self.net.conv1, self.net.bn1, self.net.relu, self.net.maxpool
                      )

        # Resblocks (#blocks x memory consumption)
        # layer1: 3 X 8 into 3 groups of 1x8
        layer1_1 = self.net.layer1[0]
        layer1_2 = self.net.layer1[1]
        layer1_3 = self.net.layer1[2]

        # layer2: 4 X 4 into 2 groups of 2x4
        layer2_1 = nn.Sequential(self.net.layer2[0], self.net.layer2[1])
        layer2_2 = nn.Sequential(self.net.layer2[2], self.net.layer2[3])

        # layer3: 6 X 2 into 2 groups of 3x2
        layer3_1 = nn.Sequential(self.net.layer3[0], self.net.layer3[1], self.net.layer3[2])
        layer3_2 = nn.Sequential(self.net.layer3[3], self.net.layer3[4], self.net.layer3[5])

        # layer4: 3 X 1 unchanged
        layer4 = self.net.layer4  # 3 X 1
        self.net_K_list = nn.Sequential(
            layer0,
            layer1_1,
            layer1_2,
            layer1_3,
            layer2_1,
            layer2_2,
            layer3_1,
            layer3_2,
            layer4,
        )

    def train(self, mode: bool = True):
        super(Resnet34LocalBatchNorm, self).train(mode)
        batch_norm_eval(self)

    def forward(self, x, ki=-1):
        if ki == -1:
            for i, net in enumerate(self.net_K_list):
                x = net(x)
                # print("stage: %d, "%i, x.shape)
                # self.print_sparsity(x, i)
        elif ki == 0:
            if self.freeze_0:
                self.net_K_list[0].eval()
                with torch.no_grad():
                    x = self.net_K_list[0](x)
            else:
                x = self.net_K_list[0](x)
        else:
            sep_gap = self.MAX_K // self.K
            for i in range((ki - 1) * sep_gap + 1, ki * sep_gap + 1):
                x = self.net_K_list[i](x)
        return x

# ResNet(
#     0: ini conv
#         stride 3, stride 2
#     1: layer 1_0
#         64
#     2: layer 1_1
#         64
#     3: layer 1_2
#         64
#     4: layer 2_0 2_1
#         in 64 out 128
#         down sampling 2
#     5: layer 2_2 2_3
#         in 128 out 128
#     6: layer 3_0 3_1 3_2
#         in 128 out 256
#         dp 2
#     7: layer 3_3 3_4 3_5
#         in 256 out 256
#     8: layer 4
#         in 256 out 512
#         dp 2
# )