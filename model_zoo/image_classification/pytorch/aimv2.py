"""AIMv2 (Apple, Nov 2024). Autoregressive image pretraining with multimodal targets; strong DINOv3 alternative on classification + retrieval. Backbone frozen → federated averaging only syncs the head.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained backbone tensors are
delivered from the tracebloc model store as the training seed: upload the
matched ``aimv2_weights.pkl`` sitting next to this file via ``weights=True``::

    user.upload_model("aimv2", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file (weights of ``apple/aimv2-large-patch14-224``).
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
license = "apple-amlr"
image_size = 224
batch_size = 16
output_classes = 2
category = "image_classification"

# Architecture config for apple/aimv2-large-patch14-224 (native
# Aimv2VisionModel, model_type "aimv2_vision_model" — no custom code path),
# inlined so the model builds with no config fetch. The SDK uploads the .py
# plus its named weight sibling — there is no config.json path — so the
# config lives here in the template.
CONFIG = {
    "model_type": "aimv2_vision_model",
    "hidden_size": 1024,
    "intermediate_size": 2816,
    "num_hidden_layers": 24,
    "num_attention_heads": 8,
    "num_channels": 3,
    "image_size": 224,
    "patch_size": 14,
    "rms_norm_eps": 1e-05,
    "attention_dropout": 0.0,
    "projection_dropout": 0.0,
    "qkv_bias": False,
    "mlp_bias": False,
    "use_bias": False,
    "hidden_act": "silu",
    "initializer_range": 0.02,
    "use_head": False,
    "is_native": False,
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
        feat = out.last_hidden_state.mean(dim=1)
        return self.head(feat)
