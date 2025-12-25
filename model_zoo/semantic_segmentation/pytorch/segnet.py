import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
framework = "pytorch"
main_class = "SegNet"
image_size = 256
batch_size = 8
output_classes = 2
category = "semantic_segmentation"


class SegNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=output_classes):
        super(SegNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder (VGG-style)
        self.enc1 = self._make_encoder_block(n_channels, 64, 2)
        self.enc2 = self._make_encoder_block(64, 128, 2)
        self.enc3 = self._make_encoder_block(128, 256, 3)
        self.enc4 = self._make_encoder_block(256, 512, 3)
        self.enc5 = self._make_encoder_block(512, 512, 3)

        # Decoder
        self.dec5 = self._make_decoder_block(512, 512, 3)
        self.dec4 = self._make_decoder_block(512, 256, 3)
        self.dec3 = self._make_decoder_block(256, 128, 3)
        self.dec2 = self._make_decoder_block(128, 64, 2)
        self.dec1 = self._make_decoder_block(64, n_classes, 2)

    def _make_encoder_block(self, in_channels, out_channels, num_layers):
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_decoder_block(self, in_channels, out_channels, num_layers):
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x1_pool, id1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)

        x2 = self.enc2(x1_pool)
        x2_pool, id2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)

        x3 = self.enc3(x2_pool)
        x3_pool, id3 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)

        x4 = self.enc4(x3_pool)
        x4_pool, id4 = F.max_pool2d(x4, kernel_size=2, stride=2, return_indices=True)

        x5 = self.enc5(x4_pool)
        x5_pool, id5 = F.max_pool2d(x5, kernel_size=2, stride=2, return_indices=True)

        # Decoder
        x5_unpool = F.max_unpool2d(x5_pool, id5, kernel_size=2, stride=2)
        x5_dec = self.dec5(x5_unpool)

        x4_unpool = F.max_unpool2d(x5_dec, id4, kernel_size=2, stride=2)
        x4_dec = self.dec4(x4_unpool)

        x3_unpool = F.max_unpool2d(x4_dec, id3, kernel_size=2, stride=2)
        x3_dec = self.dec3(x3_unpool)

        x2_unpool = F.max_unpool2d(x3_dec, id2, kernel_size=2, stride=2)
        x2_dec = self.dec2(x2_unpool)

        x1_unpool = F.max_unpool2d(x2_dec, id1, kernel_size=2, stride=2)
        x1_dec = self.dec1(x1_unpool)

        return x1_dec
