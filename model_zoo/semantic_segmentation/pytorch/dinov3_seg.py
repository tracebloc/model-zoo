"""DINOv3 backbone (Meta, Aug 2025) with a trainable linear segmentation head. Currently a top recipe for label-efficient segmentation — frozen self-supervised features + tiny dense head, BN-free, federated-friendly.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not (the hub checkpoint is license-gated;
this template needs no token). The pretrained backbone tensors are delivered
from the tracebloc model store as the training seed: upload the matched
``dinov3_seg_weights.pkl`` sitting next to this file via ``weights=True``::

    user.upload_model("dinov3_seg", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file (weights of ``facebook/dinov3-vitb16-pretrain-lvd1689m``).
"""
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

framework = "pytorch"
main_class = "MyModel"
license = "Apache-2.0"
image_size = 224
batch_size = 8
output_classes = 2
category = "semantic_segmentation"

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
    "pos_embed_shift": None,
    "pos_embed_jitter": None,
    "query_bias": True,
    "key_bias": False,
    "value_bias": True,
    "proj_bias": True,
    "mlp_bias": True,
    "use_gated_mlp": False,
    "initializer_range": 0.02,
    "apply_layernorm": True,
    "reshape_hidden_states": True,
    "out_features": ["stage12"],
    "out_indices": [12],
    "stage_names": [
        "stem",
        "stage1",
        "stage2",
        "stage3",
        "stage4",
        "stage5",
        "stage6",
        "stage7",
        "stage8",
        "stage9",
        "stage10",
        "stage11",
        "stage12",
    ],
}


class MyModel(nn.Module):
    def __init__(self, num_classes=output_classes):
        super().__init__()
        config = AutoConfig.for_model(**CONFIG)
        self.backbone = AutoModel.from_config(config)
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
