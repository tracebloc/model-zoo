"""MambaVision (NVIDIA, CVPR 2024). Hybrid Mamba-SSM + self-attention backbone; ImageNet SOTA at its FLOP class. Trains from scratch (no foundation freeze), so all params participate in federated averaging — SSM state buffers should be re-initialized per round, not averaged."""
import timm
import torch.nn as nn

framework = "pytorch"
main_class = "MyModel"
license = "NVIDIA-Source"
image_size = 224
batch_size = 32
output_classes = 2
category = "image_classification"


class MyModel(nn.Module):
    def __init__(self, num_classes=output_classes):
        super().__init__()
        self.model = timm.create_model(
            "mambavision_tiny.fb_in1k", pretrained=False, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)
