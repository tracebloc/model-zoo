"""ConvNeXt V2 via timm. Modern CNN default — FCMAE-pretrained, ~88.9% ImageNet top-1 at Huge scale; Tiny variant here is the practical sweet spot."""
import timm
import torch.nn as nn

framework = "pytorch"
main_class = "MyModel"
license = "Apache-2.0"
image_size = 224
batch_size = 32
output_classes = 2
category = "image_classification"


class MyModel(nn.Module):
    def __init__(self, num_classes=output_classes):
        super().__init__()
        self.model = timm.create_model(
            "convnextv2_tiny", pretrained=False, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)
