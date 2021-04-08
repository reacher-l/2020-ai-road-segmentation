import torch.nn as nn
import segmentation_models_pytorch as smp


class FPN(nn.Module):
    def __init__(self, num_classes):
        super(FPN, self).__init__()

        self.model = smp.FPN(
            encoder_name='resnet50',
            encoder_depth=5,
            encoder_weights=None,
            decoder_pyramid_channels=256,
            decoder_segmentation_channels=128,
            decoder_merge_policy='add',
            decoder_dropout=0.,
            in_channels=3,
            classes=num_classes
        )

    def forward(self, x):
        logits = self.model(x)
        return [logits]
