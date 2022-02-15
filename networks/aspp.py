#!conda env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


def _AsppConv(in_channels,
              out_channels,
              kernel_size,
              stride=1,
              padding=0,
              dilation=1,
              bn_momentum=0.1):
    asppconv = nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=False),
        nn.BatchNorm2d(out_channels, momentum=bn_momentum), nn.ReLU())
    return asppconv



class AsppModule(nn.Module):
    def __init__(self, bn_momentum=0.1, output_stride=16):
        super(AsppModule, self).__init__()

        # output_stride choice
        if output_stride == 16:
            atrous_rates = [0, 6, 12, 18]
        elif output_stride == 8:
            atrous_rates = 2 * [0, 3, 5, 7]
        else:
            raise Warning("output_stride must be 8 or 16!")
        # atrous_spatial_pyramid_pooling part
        self._atrous_convolution1 = _AsppConv(
            2048, 256, 1, 1, bn_momentum=bn_momentum)
        self._atrous_convolution2 = _AsppConv(
            2048, 256, 3, 1,
            padding=atrous_rates[1],
            dilation=atrous_rates[1],
            bn_momentum=bn_momentum)
        self._atrous_convolution3 = _AsppConv(
            2048, 256, 3, 1,
            padding=atrous_rates[2],
            dilation=atrous_rates[2],
            bn_momentum=bn_momentum)
        self._atrous_convolution4 = _AsppConv(
            2048, 256, 3, 1,
            padding=atrous_rates[3],
            dilation=atrous_rates[3],
            bn_momentum=bn_momentum)

        # image_pooling part
        self._image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, momentum=bn_momentum), nn.ReLU())

        self._init_weight()

    def forward(self, input):
        input1 = self._atrous_convolution1(input)
        input2 = self._atrous_convolution2(input)
        input3 = self._atrous_convolution3(input)
        input4 = self._atrous_convolution4(input)
        input5 = self._image_pool(input)
        input5 = F.interpolate(
            input=input5,
            size=input4.size()[2:4],
            mode='bilinear',
            align_corners=True)

        return torch.cat((input1, input2, input3, input4, input5), dim=1)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()