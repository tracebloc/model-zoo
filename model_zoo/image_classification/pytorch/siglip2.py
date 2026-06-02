"""SigLIP 2 (Google, Feb 2025). Sigmoid-loss vision-language pretraining successor to SigLIP; now the default frozen backbone for many 2025 multimodal stacks. Backbone frozen → federated averaging only syncs the linear head."""
import torch.nn as nn
from transformers import AutoModel

framework = "pytorch"
main_class = "MyModel"
license = "Apache-2.0"
image_size = 224
batch_size = 16
output_classes = 2
category = "image_classification"

_BACKBONE_ID = "google/siglip2-base-patch16-224"


class MyModel(nn.Module):
    def __init__(self, num_classes=output_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(_BACKBONE_ID).vision_model
        for p in self.backbone.parameters():
            p.requires_grad = False
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state.mean(dim=1)
        return self.head(pooled)
