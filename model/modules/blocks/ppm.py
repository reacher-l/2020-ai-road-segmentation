import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basics import Conv2dBnAct


class PPMModule(nn.Module):
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        super(PPMModule, self).__init__()

        self.stages = nn.ModuleList([self._make_stage(in_channels, size) for size in sizes])
        self.bottleneck = Conv2dBnAct(in_channels * (len(sizes) + 1),
                                      out_channels,
                                      kernel_size=1,
                                      act=nn.ReLU(inplace=True))

    @staticmethod
    def _make_stage(in_channels, size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(size, size)),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        x = [F.upsample(stage(x), size=(h, w), mode='bilinear') for stage in self.stages] + [x]
        x = self.bottleneck(torch.cat(x, dim=1))
        return x
