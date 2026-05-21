"""SAM 2 (Meta, Aug 2024). Segment-anything foundation model — image + video, promptable. Used here as a frozen encoder with a trainable linear segmentation head so federated averaging only sees the head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Sam2Model

framework = "pytorch"
main_class = "MyModel"
license = "Apache-2.0"
image_size = 1024
batch_size = 2
output_classes = 2
category = "semantic_segmentation"

_BACKBONE_ID = "facebook/sam2-hiera-tiny"


class MyModel(nn.Module):
    def __init__(self, num_classes=output_classes):
        super().__init__()
        self.backbone = Sam2Model.from_pretrained(_BACKBONE_ID).vision_encoder
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Hiera image encoder outputs C=256 feature map at stride 16
        self.head = nn.Conv2d(256, num_classes, kernel_size=1)
        self._out_size = image_size

    def forward(self, x):
        feats = self.backbone(x).last_hidden_state  # (B, C, H', W')
        logits = self.head(feats)
        return F.interpolate(logits, size=self._out_size, mode="bilinear", align_corners=False)
