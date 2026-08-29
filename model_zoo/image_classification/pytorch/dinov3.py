"""DINOv3 backbone (Meta, Aug 2025) with a trainable linear head. Self-supervised ViT trained on 1.7B images; backbone is frozen so federated averaging only sees the small head — BN-free by construction.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not (the hub checkpoint is license-gated;
this template needs no token). The pretrained backbone tensors are delivered
from the tracebloc model store as the training seed: upload the matched
``dinov3_weights.pkl`` sitting next to this file via ``weights=True``::

    user.upload_model("dinov3", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file (weights of ``facebook/dinov3-vitb16-pretrain-lvd1689m``).
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

# Architecture config for facebook/dinov3-vitb16-pretrain-lvd1689m
# (DINOv3ViTModel, model_type "dinov3_vit"), inlined so the model builds
# with no config fetch. The SDK uploads the .py plus its named weight
# sibling — there is no config.json path — so the config lives here in the
# template.
CONFIG = {
    "model_type": "dinov3_vit",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_channels": 3,
    "image_size": 224,
    "patch_size": 16,
    "num_register_tokens": 4,
    "hidden_act": "gelu",
    "attention_dropout": 0.0,
    "drop_path_rate": 0.0,
    "layer_norm_eps": 1e-05,
    "layerscale_value": 1.0,
    "rope_theta": 100.0,
    "pos_embed_rescale": 2.0,
    "query_bias": True,
    "key_bias": False,
    "value_bias": True,
    "proj_bias": True,
    "mlp_bias": True,
    "use_gated_mlp": False,
    "initializer_range": 0.02,
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
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)
