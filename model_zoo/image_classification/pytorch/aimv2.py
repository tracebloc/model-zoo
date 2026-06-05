"""AIMv2 (Apple, Nov 2024). Autoregressive image pretraining with multimodal targets; strong DINOv3 alternative on classification + retrieval. Backbone frozen → federated averaging only syncs the head."""
import torch.nn as nn
from transformers import AutoModel

framework = "pytorch"
main_class = "MyModel"
license = "apple-amlr"
image_size = 224
batch_size = 16
output_classes = 2
category = "image_classification"

_BACKBONE_ID = "apple/aimv2-large-patch14-224"


class MyModel(nn.Module):
    def __init__(self, num_classes=output_classes):
        super().__init__()
        # Native Aimv2VisionModel (config model_type="aimv2_vision_model") —
        # no trust_remote_code, so it passes the platform security check
        # and avoids the custom-code path's all_tied_weights_keys bug.
        self.backbone = AutoModel.from_pretrained(_BACKBONE_ID)
        for p in self.backbone.parameters():
            p.requires_grad = False
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        feat = out.last_hidden_state.mean(dim=1)
        return self.head(feat)
