"""Small CNN for images with any number of channels (1 to 64), such as
multispectral or multiplexed microscopy stacks stored as TIFF.

The first convolution takes ``input_channels``. Picked for a dataset on
tracebloc, the template is filled in with that dataset's channel count, as it is
with its class count; uploaded by hand, set ``input_channels`` to the number of
channels in your images. Three conv blocks with GroupNorm, then global average
pooling, so any ``image_size`` works and it trains in seconds on small images.
"""
import torch.nn as nn

framework = "pytorch"
main_class = "MyModel"
image_size = 64
batch_size = 16
input_channels = 4
output_classes = 2
category = "image_classification"


def _block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.ReLU(inplace=True),
    )


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            _block(input_channels, 16),
            nn.MaxPool2d(2),
            _block(16, 32),
            nn.MaxPool2d(2),
            _block(32, 64),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, output_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
