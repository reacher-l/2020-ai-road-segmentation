import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SCSEModule


class BNET2d(nn.BatchNorm2d):
    def __init__(self, channels, *args, k=3, **kwargs):
        super(BNET2d, self).__init__(channels, *args, affine=False, **kwargs)
        self.bnconv = nn.Conv2d(channels, channels, kernel_size=k, padding=(k - 1) // 2, groups=channels, bias=True)

    def forward(self, x):
        return self.bnconv(super(BNET2d, self).forward(x))


class DepthWiseSeparableConv2dBnAct(nn.Sequential):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 enhance_bn=False,
                 act=nn.ReLU(inplace=True)):
        depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, in_channels, bias=False)
        bn1 = BNET2d(in_channels) if enhance_bn else nn.BatchNorm2d(out_channels)
        act1 = act
        pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        bn2 = BNET2d(out_channels) if enhance_bn else nn.BatchNorm2d(out_channels)
        act2 = act
        super(DepthWiseSeparableConv2dBnAct, self).__init__(depthwise, bn1, act1, pointwise, bn2, act2)


class Conv2dBnAct(nn.Sequential):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 enhance_bn=False,
                 act=nn.ReLU(inplace=True)):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)
        batch_normalization = BNET2d(out_channels) if enhance_bn else nn.BatchNorm2d(out_channels)
        super(Conv2dBnAct, self).__init__(conv, batch_normalization, act)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            enhance_bn=False,
            use_attention=False
    ):
        super().__init__()
        self.conv1 = Conv2dBnAct(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            enhance_bn=enhance_bn,
            act=nn.ReLU(inplace=True))

        self.conv2 = Conv2dBnAct(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            enhance_bn=enhance_bn,
            act=nn.ReLU(inplace=True)
        )

        # attention
        self.attention1 = SCSEModule(in_channels=in_channels + skip_channels) if use_attention else nn.Identity()
        self.attention2 = SCSEModule(in_channels=out_channels) if use_attention else nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x
