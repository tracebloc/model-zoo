import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

# Configuration
framework = "pytorch"
main_class = "DeepLabV3"
image_size = 256
batch_size = 8
output_classes = 2
category = "semantic_segmentation"


class DeepLabV3(nn.Module):
    def __init__(self, backbone="resnet50"):
        super(DeepLabV3, self).__init__()
        
        if backbone == "resnet50":
            self.model = deeplabv3_resnet50(
                pretrained=False, 
                progress=True, 
                num_classes=output_classes
            )
        elif backbone == "resnet101":
            self.model = deeplabv3_resnet101(
                pretrained=False, 
                progress=True, 
                num_classes=output_classes
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.model(x)["out"]


# class DeepLabV3ResNet50(DeepLabV3):
#     """DeepLabV3 with ResNet-50 backbone"""
#     def __init__(self):
#         super(DeepLabV3ResNet50, self).__init__(backbone="resnet50")


# class DeepLabV3ResNet101(DeepLabV3):
#     """DeepLabV3 with ResNet-101 backbone"""
#     def __init__(self):
#         super(DeepLabV3ResNet101, self).__init__(backbone="resnet101") 