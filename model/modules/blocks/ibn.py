import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basics import Conv2dBnAct
from ..attention import SEModule


class IBN(nn.Module):
    def __init__(self, planes):
        super().__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.Sequential(nn.InstanceNorm2d(half1, affine=True),
                                nn.ReLU(inplace=True))
        self.BN = nn.Sequential(nn.BatchNorm2d(half2),
                                nn.ReLU(inplace=True))

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class ImprovedIBNaDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            IBN(in_channels // 4),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class IBNaDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, enhance_bn=False, use_attention=False):
        super(IBNaDecoderBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, 1, 1, bias=False),
            IBN(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        self.conv2 = Conv2dBnAct(
            in_channels // 4,
            out_channels,
            kernel_size=3,
            padding=1,
            enhance_bn=enhance_bn,
            act=nn.ReLU(inplace=True)
        )

        # attention
        self.se = SEModule(in_channels, reduction=16) if use_attention else nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.se(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
