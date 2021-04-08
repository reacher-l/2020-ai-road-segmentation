import torch.nn as nn
import segmentation_models_pytorch as smp


class Unet(nn.Module):
    def __init__(self, num_classes):
        super(Unet, self).__init__()

        self.model = smp.Unet(
            encoder_name="se_resnext50_32x4d",
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=[256, 128, 64, 32, 16],
            decoder_attention_type='scse',
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, x):
        logits = self.model(x)
        return [logits]
