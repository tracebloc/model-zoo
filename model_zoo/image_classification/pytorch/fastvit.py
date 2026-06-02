"""FastViT (Apple, ICCV 2023) via timm. Hybrid conv + transformer designed for mobile latency — 1.9–4.9× faster than ConvNeXt at comparable accuracy. Fills the modern-efficient gap (squeezenet is the only other lightweight option in the zoo)."""
import timm
import torch.nn as nn

framework = "pytorch"
main_class = "MyModel"
license = "Apple-Sample-Code-License"
image_size = 256
batch_size = 64
output_classes = 2
category = "image_classification"


class MyModel(nn.Module):
    def __init__(self, num_classes=output_classes):
        super().__init__()
        self.model = timm.create_model(
            "fastvit_s12.apple_in1k", pretrained=False, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)
