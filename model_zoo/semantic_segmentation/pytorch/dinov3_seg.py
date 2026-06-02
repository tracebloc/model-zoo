"""DINOv3 backbone (Meta, Aug 2025) with a trainable linear segmentation head. Currently a top recipe for label-efficient segmentation — frozen self-supervised features + tiny dense head, BN-free, federated-friendly."""
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

framework = "pytorch"
main_class = "MyModel"
license = "Apache-2.0"
image_size = 224
batch_size = 8
output_classes = 2
category = "semantic_segmentation"

_BACKBONE_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


class MyModel(nn.Module):
    def __init__(self, num_classes=output_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(_BACKBONE_ID)
        for p in self.backbone.parameters():
            p.requires_grad = False
        hidden = self.backbone.config.hidden_size
        self.head = nn.Conv2d(hidden, num_classes, kernel_size=1)
        self._patch = 16
        self._out_size = image_size

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        # DINOv3 prepends 1 CLS + N register tokens before the patch tokens;
        # skip both so the remaining sequence is exactly the H'×W' patch grid.
        n_special = 1 + getattr(self.backbone.config, "num_register_tokens", 0)
        tokens = out.last_hidden_state[:, n_special:, :]
        b, n, c = tokens.shape
        h = w = int(n**0.5)
        feat = tokens.transpose(1, 2).reshape(b, c, h, w)
        logits = self.head(feat)
        return F.interpolate(logits, size=self._out_size, mode="bilinear", align_corners=False)
