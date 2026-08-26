"""Single-Person Pose Estimator on a ResNet-50 backbone. Simpler than FasterRCNNSPPE; good default when you don't need detection.

Offline variant: built with ``weights=None`` — no checkpoint download at
construction, so the template constructs anywhere, network or not. A plain
classification ResNet builds the identical architecture (same state_dict
keys and shapes) with or without weights. The pretrained ImageNet tensors
are delivered from the tracebloc model store as the training seed: upload
the matched ``resnet_sppe_weights.pkl`` sitting next to this file via
``upload_model(..., weights=True)``; the platform loads it with
``load_state_dict(strict=True)`` after the model builds. See
``tools/prep_offline_weights.py`` for producing and verifying that file.
"""
import torch.nn as nn
import torchvision.models as models


# Configuration
framework = "pytorch"
model_type = ""
main_class = "ResNetSPPE"
image_size = 64
batch_size = 128
output_classes = 1
category = "keypoint_detection"
num_feature_points = 16

class ResNetSPPE(nn.Module):
    def __init__(self, num_feature_points=num_feature_points, input_channels=3):
        super(ResNetSPPE, self).__init__()
        self.num_feature_points = num_feature_points

        # Build the ResNet-50 architecture with no download; the pretrained
        # tensors arrive from the tracebloc model store as the training seed.
        resnet = models.resnet50(weights=None)

        # Modify the first convolution layer to accommodate different input channels
        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Extract all layers except the fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Define a global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Add a fully connected layer to predict keypoints' x, y coordinates and visibility
        num_features = resnet.fc.in_features
        self.fc = nn.Linear(num_features, num_feature_points * 3)

    def forward(self, x):
        # Pass the input through the ResNet backbone
        x = self.backbone(x)

        # Apply global average pooling
        x = self.global_avg_pool(x)

        # Flatten the pooled output
        x = x.view(x.size(0), -1)

        # Pass through the fully connected layer
        x = self.fc(x)

        # Reshape to (batch_size, num_feature_points, 3)
        x = x.view(-1, self.num_feature_points, 3)
        return x
