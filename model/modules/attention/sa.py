import torch
import torch.nn as nn

from torch.nn import Parameter


class SAModule(nn.Module):
    """
    Constructs a Channel Spatial Group module.
    """

    def __init__(self, channels, groups=64):
        super(SAModule, self).__init__()

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c_weight = Parameter(torch.zeros(1, channels // (2 * groups), 1, 1))
        self.c_bias = Parameter(torch.ones(1, channels // (2 * groups), 1, 1))
        self.s_weight = Parameter(torch.zeros(1, channels // (2 * groups), 1, 1))
        self.s_bias = Parameter(torch.ones(1, channels // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channels // (2 * groups), channels // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.c_weight * xn + self.c_bias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.s_weight * xs + self.s_bias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out
