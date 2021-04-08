import torch.nn as nn
import torch.nn.functional as F

from model.backbone import resnet50_ibn_a
from model.modules.basics import Conv2dBnAct
from model.modules.blocks.ibn import IBNaDecoderBlock


class CenterBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CenterBlock, self).__init__()
        self.conv = Conv2dBnAct(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.conv(x)


class UnetIBN(nn.Module):
    def __init__(self,
                 encoder_pretrained=True,
                 head_channels=512,
                 decoder_channels=[256, 128, 64, 32],
                 dropout=0.,
                 classes=10):
        super(UnetIBN, self).__init__()

        # ENCODER
        self.encoder = resnet50_ibn_a(pretrained=encoder_pretrained)
        encoder_channels = self.encoder.out_channels[1:]

        # CENTER BLOCK
        self.center_block = CenterBlock(encoder_channels[-1], head_channels)

        # DECODER
        skip_channels = encoder_channels[:-1][::-1]
        input_channels = [head_channels] + decoder_channels[:-1]

        self.decoder_modules = nn.ModuleList()
        for in_ch, sk_ch, de_ch in zip(input_channels, skip_channels, decoder_channels):
            self.decoder_modules.append(IBNaDecoderBlock(in_ch + sk_ch, de_ch, use_attention=True))

        # PREDICT
        self.pred_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(32, classes, 1),
        )

    def forward(self, x):
        encoder_feats = self.encoder(x)[1:]
        decoder_feat = self.center_block(encoder_feats[-1])
        skip_feats = encoder_feats[:-1][::-1]
        for idx, decoder_module in enumerate(self.decoder_modules):
            decoder_feat = decoder_module(decoder_feat, skip_feats[idx])
        out = self.pred_head(decoder_feat)
        out = F.interpolate(out, size=x.shape[-2:])
        return out
