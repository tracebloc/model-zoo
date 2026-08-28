"""SigLIP 2 (Google, Feb 2025). Sigmoid-loss vision-language pretraining successor to SigLIP; now the default frozen backbone for many 2025 multimodal stacks. Backbone frozen → federated averaging only syncs the linear head.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. Only the vision tower is built (the
text tower is unused for image classification). The pretrained backbone
tensors are delivered from the tracebloc model store as the training seed:
upload the matched ``siglip2_weights.pkl`` sitting next to this file via
``weights=True``::

    user.upload_model("siglip2", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file (vision-tower weights of
``google/siglip2-base-patch16-224``).
"""
import torch.nn as nn
from transformers import AutoConfig, AutoModel

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("head.",)

framework = "pytorch"
main_class = "MyModel"
license = "Apache-2.0"
image_size = 224
batch_size = 16
output_classes = 2
category = "image_classification"

# Architecture config for the vision tower of google/siglip2-base-patch16-224
# (SiglipVisionModel, model_type "siglip_vision_model" — the checkpoint uses
# the original SigLIP architecture with SigLIP-2 training), inlined so the
# model builds with no config fetch. The SDK uploads the .py plus its named
# weight sibling — there is no config.json path — so the config lives here
# in the template.
CONFIG = {
    "model_type": "siglip_vision_model",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_channels": 3,
    "image_size": 224,
    "patch_size": 16,
    "hidden_act": "gelu_pytorch_tanh",
    "layer_norm_eps": 1e-06,
    "attention_dropout": 0.0,
}


class MyModel(nn.Module):
    def __init__(self, num_classes=output_classes):
        super().__init__()
        config = AutoConfig.for_model(**CONFIG)
        self.backbone = AutoModel.from_config(config)
        for p in self.backbone.parameters():
            p.requires_grad = False
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state.mean(dim=1)
        return self.head(pooled)
