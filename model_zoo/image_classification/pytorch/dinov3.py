"""DINOv3 backbone (Meta, Aug 2025) with a trainable linear head. Self-supervised ViT trained on 1.7B images; backbone is frozen so federated averaging only sees the small head — BN-free by construction."""
import torch.nn as nn
from transformers import AutoModel

framework = "pytorch"
main_class = "MyModel"
license = "Apache-2.0"
image_size = 224
batch_size = 16
output_classes = 2
category = "image_classification"

_BACKBONE_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


class MyModel(nn.Module):
    def __init__(self, num_classes=output_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(_BACKBONE_ID)
        for p in self.backbone.parameters():
            p.requires_grad = False
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)
