"""ResNet-18 via torchvision, ~11M params. Fastest ResNet; good baseline for quick iteration or CPU-constrained runs."""
import torch
import torchvision
import torch.nn as nn

framework = "pytorch"
main_class = "MyModel"
image_size = 224
batch_size = 256
output_classes = 2
category = "image_classification"


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = torchvision.models.resnet18(
            pretrained=False, progress=True, num_classes=output_classes
        )

    def forward(self, x):
        return self.model(x)
