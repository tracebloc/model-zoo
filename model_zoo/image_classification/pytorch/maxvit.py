"""MaxViT via torchvision. Hybrid CNN + transformer. Strong accuracy but memory-hungry; needs a good GPU."""
import torchvision
import torch.nn as nn

framework = "pytorch"
main_class = "MyModel"
image_size = 224
# Sized to train within the smallest client's memory: see tools/template_memory.py.
batch_size = 32
output_classes = 2
category = "image_classification"


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = torchvision.models.maxvit_t(
            pretrained=False, progress=True, num_classes=output_classes
        )

    def forward(self, x):
        return self.model(x)
