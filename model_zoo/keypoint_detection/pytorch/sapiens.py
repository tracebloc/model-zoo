"""Sapiens (Meta, ECCV 2024). Human-centric foundation model — pose, depth,
normals, segmentation in one family.

Backbone note
-------------
The official ``facebook/sapiens-pose-*`` Hub repos ship as TorchScript
artifacts (not loadable as HF transformers models), and the MAE-pretrained
``facebook/sapiens-pretrain-0.3b`` repo ships a ``config.json`` that lacks a
``model_type`` key, so it cannot be resolved to an HF architecture either.
Until Meta publishes an HF-loadable sapiens checkpoint, this template builds
the upstream ViT-MAE base backbone sapiens is architecturally built on
(``facebook/vit-mae-base``). The ViT geometry matches; only the
human-centric pretraining is lost.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained backbone tensors are
delivered from the tracebloc model store as the training seed: upload the
matched ``sapiens_weights.pkl`` sitting next to this file via
``weights=True``::

    user.upload_model("sapiens", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file (weights of ``facebook/vit-mae-base``; the position
embeddings are sized to the declared 256px input, freshly initialized —
same as before via the checkpoint's size-mismatch reinit).

Select LoRA-only fine-tuning in the training plan so federated averaging
only syncs the adapter + the final regressor.
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
model_type = "transformer"
main_method = "MyModel"
license = "CC-BY-NC-4.0"
image_size = 256
batch_size = 16
output_classes = 1
category = "keypoint_detection"
num_feature_points = 17

# Architecture config for facebook/vit-mae-base (ViTMAEModel, model_type
# "vit_mae"), inlined so the model builds with no config fetch. The SDK
# uploads the .py plus its named weight sibling — there is no config.json
# path — so the config lives here in the template. ``image_size`` is passed
# at build time (the checkpoint ships at 224; this family declares 256).
CONFIG = {
    "model_type": "vit_mae",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_channels": 3,
    "patch_size": 16,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "qkv_bias": True,
    "decoder_hidden_size": 512,
    "decoder_intermediate_size": 2048,
    "decoder_num_attention_heads": 16,
    "decoder_num_hidden_layers": 8,
    "mask_ratio": 0.75,
    "norm_pix_loss": False,
}


class _SapiensWrapper(nn.Module):
    def __init__(self, backbone, num_feature_points):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.head = nn.Linear(hidden, num_feature_points * 3)
        self.num_feature_points = num_feature_points

    def forward(self, pixel_values, *args, **kwargs):
        out = self.backbone(pixel_values=pixel_values)
        feat = out.last_hidden_state.mean(dim=1)
        coords = self.head(feat).view(-1, self.num_feature_points, 3)
        return coords


def MyModel(num_feature_points=num_feature_points):
    # ViT-MAE base ships at 224x224 by default; we declare 256 above to
    # match the rest of the keypoint model family. ``image_size`` is a
    # config-level override — the position-embedding table is sized to the
    # 256 grid at construction (freshly initialized, exactly as the
    # pre-migration checkpoint load reinitialized it on size mismatch).
    # The patch projection and attention weights carry the pretrained seed.
    config = AutoConfig.for_model(**CONFIG, image_size=image_size)
    base = AutoModel.from_config(config)
    return _SapiensWrapper(base, num_feature_points)
