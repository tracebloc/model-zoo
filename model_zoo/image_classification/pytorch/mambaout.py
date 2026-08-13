"""MambaOut (NUS) via timm. Gated-CNN backbone from "MambaOut: Do We Really Need Mamba for Vision?" — it removes the SSM from a Mamba-style block and keeps the gated token mixer, arguing SSM is unnecessary for ImageNet classification. Not a MambaVision substitute: different paper, opposite thesis. Trains from scratch, LayerNorm-only with no BatchNorm running stats and no buffers, so every parameter averages cleanly across federated rounds."""
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
            "mambaout_tiny.in1k", pretrained=False, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)
