"""EVA-02 via timm. First open vision backbone past 90% ImageNet top-1 — ViT + SwiGLU + RoPE + masked-image-modeling pretraining. Base variant is the practical default."""
import timm
import torch.nn as nn

framework = "pytorch"
main_class = "MyModel"
license = "MIT"
image_size = 448
batch_size = 16
output_classes = 2
category = "image_classification"


class MyModel(nn.Module):
    def __init__(self, num_classes=output_classes):
        super().__init__()
        self.model = timm.create_model(
            "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
            pretrained=False,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)
