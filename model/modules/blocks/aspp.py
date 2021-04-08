import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basics import Conv2dBnAct, DepthWiseSeparableConv2dBnAct


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, atrous_rates=(1, 6, 12, 18), separable=False):
        super(ASPP, self).__init__()
        conv = DepthWiseSeparableConv2dBnAct if separable else Conv2dBnAct

        self.branch1 = conv(dim_in, dim_out, 3, padding=atrous_rates[0], dilation=atrous_rates[0],
                            act=nn.ReLU(inplace=True))
        self.branch2 = conv(dim_in, dim_out, 3, padding=atrous_rates[1], dilation=atrous_rates[1],
                            act=nn.ReLU(inplace=True))
        self.branch3 = conv(dim_in, dim_out, 3, padding=atrous_rates[2], dilation=atrous_rates[2],
                            act=nn.ReLU(inplace=True))
        self.branch4 = conv(dim_in, dim_out, 3, padding=atrous_rates[3], dilation=atrous_rates[3],
                            act=nn.ReLU(inplace=True))
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBnAct(dim_in, dim_out, 3, padding=1, act=nn.ReLU(inplace=True)),
        )
        self.conv_cat = Conv2dBnAct(dim_out * 5, dim_out, 1, act=nn.ReLU(inplace=True))

    def forward(self, x):
        row, col = x.size(2), x.size(3)
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = self.branch5(x)
        global_feature = F.interpolate(global_feature, (row, col), None, mode='bilinear', align_corners=True)
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result
