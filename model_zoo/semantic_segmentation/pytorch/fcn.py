import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
framework = "pytorch"
main_class = "FCN"
image_size = 256
batch_size = 8
output_classes = 2
category = "semantic_segmentation"


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(VGGBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        return x, self.pool(x)


class FCN(nn.Module):
    def __init__(self, n_channels=3, n_classes=output_classes):
        super(FCN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # VGG-style encoder
        self.block1 = VGGBlock(n_channels, 64, 2)
        self.block2 = VGGBlock(64, 128, 2)
        self.block3 = VGGBlock(128, 256, 3)
        self.block4 = VGGBlock(256, 512, 3)
        self.block5 = VGGBlock(512, 512, 3)

        # FCN-specific layers
        self.fc6 = nn.Conv2d(512, 4096, 1)
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.score_fc = nn.Conv2d(4096, n_classes, 1)

        # Skip connections for FCN-8s
        self.score_pool3 = nn.Conv2d(256, n_classes, 1)
        self.score_pool4 = nn.Conv2d(512, n_classes, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        pool1, x = self.block1(x)
        pool2, x = self.block2(x)
        pool3, x = self.block3(x)
        pool4, x = self.block4(x)
        pool5, x = self.block5(x)

        # FCN head
        x = self.fc6(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=0.5, training=self.training)

        x = self.fc7(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=0.5, training=self.training)

        x = self.score_fc(x)

        # FCN-8s: Add skip connections
        score_pool3 = self.score_pool3(pool3)
        score_pool4 = self.score_pool4(pool4)

        # Upsample and add skip connections
        x = F.interpolate(
            x, score_pool4.size()[2:], mode="bilinear", align_corners=True
        )
        x = x + score_pool4

        x = F.interpolate(
            x, score_pool3.size()[2:], mode="bilinear", align_corners=True
        )
        x = x + score_pool3

        # Final upsampling to input size
        x = F.interpolate(
            x, size=(x.size(2) * 8, x.size(3) * 8), mode="bilinear", align_corners=True
        )

        return x


class FCNResNet(nn.Module):
    """FCN with ResNet backbone - alternative implementation"""

    def __init__(self, n_channels=input_channels, n_classes=output_classes):
        super(FCNResNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Simple ResNet-style backbone
        self.conv1 = nn.Conv2d(n_channels, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # FCN head
        self.fc = nn.Conv2d(512, n_classes, 1)

        # Skip connections
        self.skip1 = nn.Conv2d(64, n_classes, 1)
        self.skip2 = nn.Conv2d(128, n_classes, 1)
        self.skip3 = nn.Conv2d(256, n_classes, 1)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(self._make_block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(self._make_block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _make_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        skip1 = self.layer1(x)
        skip2 = self.layer2(skip1)
        skip3 = self.layer3(skip2)
        x = self.layer4(skip3)

        # FCN head
        x = self.fc(x)

        # Skip connections
        skip3 = self.skip3(skip3)
        skip2 = self.skip2(skip2)
        skip1 = self.skip1(skip1)

        # Upsample and add skip connections
        x = F.interpolate(x, skip3.size()[2:], mode="bilinear", align_corners=True)
        x = x + skip3

        x = F.interpolate(x, skip2.size()[2:], mode="bilinear", align_corners=True)
        x = x + skip2

        x = F.interpolate(x, skip1.size()[2:], mode="bilinear", align_corners=True)
        x = x + skip1

        # Final upsampling to input size
        x = F.interpolate(
            x, size=(x.size(2) * 4, x.size(3) * 4), mode="bilinear", align_corners=True
        )

        return x
