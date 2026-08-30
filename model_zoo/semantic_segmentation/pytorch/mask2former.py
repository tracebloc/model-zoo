"""Mask2Former (Meta, CVPR 2022). Dominant universal segmentation architecture —
same model handles semantic / instance / panoptic via masked attention; ~57 mIoU
ADE20K at Swin-L.

This wrapper exposes the HF `Mask2FormerForUniversalSegmentation` as a plain
semantic-segmentation module: `forward(x)` returns `[B, num_classes, H, W]`
logits, matching the contract used by the other models in this folder
(deeplab.py, fcn.py, ...). The platform trainer then applies its standard
per-pixel CrossEntropyLoss against the `[B, H, W]` mask targets.

We do NOT use Mask2Former's internal MaskFormerLoss here because the platform
trainer does not pass `mask_labels` / `class_labels`; instead we derive
semantic logits from the (mask_queries, class_queries) outputs via the standard
collapse:

    seg_logits = einsum("bqc,bqhw->bchw",
                        softmax(class_logits)[..., :-1],
                        sigmoid(mask_logits))

then upsample to the input resolution. This is the same formula HF's
`image_processor.post_process_semantic_segmentation` uses.

Offline variant: the architecture is built from the inlined config below
(Swin-tiny backbone sub-config included) — no hub model id, no config fetch,
no download at build time, so the template constructs anywhere, network or
not. The pretrained tensors are delivered from the tracebloc model store as
the training seed: upload the matched ``mask2former_weights.pkl`` sitting
next to this file via ``weights=True``::

    user.upload_model("mask2former", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file (weights of ``facebook/mask2former-swin-tiny-ade-semantic``;
the class-prediction head is sized to ``output_classes`` and freshly
initialized).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, Mask2FormerForUniversalSegmentation

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("model.class_predictor.", "model.criterion.")

# Module-level metadata contract
framework = "pytorch"
main_class = "Mask2Former"
license = "Apache-2.0"
image_size = 384
batch_size = 4
output_classes = 2
category = "semantic_segmentation"

# Architecture config for facebook/mask2former-swin-tiny-ade-semantic
# (Mask2FormerForUniversalSegmentation, model_type "mask2former") with its
# nested Swin backbone sub-config, inlined so the model builds with no
# config fetch. The SDK uploads the .py plus its named weight sibling —
# there is no config.json path — so the config lives here in the template.
CONFIG = {
    "model_type": "mask2former",
    "backbone_config": {
        "model_type": "swin",
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "num_layers": 4,
        "window_size": 7,
        "image_size": 224,
        "patch_size": 4,
        "num_channels": 3,
        "hidden_size": 768,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "drop_path_rate": 0.3,
        "encoder_stride": 32,
        "use_absolute_embeddings": False,
        # The hub config.json carries a stray "path_norm": true — an
        # upstream typo artifact of Swin's original patch_norm option.
        # HF's SwinConfig defines neither spelling and no modeling code
        # reads it (the patch-embedding LayerNorm is unconditional), so
        # the inert key is dropped here rather than carried verbatim.
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-05,
        "out_features": ["stage1", "stage2", "stage3", "stage4"],
        "out_indices": [1, 2, 3, 4],
        "stage_names": ["stem", "stage1", "stage2", "stage3", "stage4"],
    },
    "feature_size": 256,
    "mask_feature_size": 256,
    "hidden_dim": 256,
    "encoder_feedforward_dim": 1024,
    "activation_function": "relu",
    "encoder_layers": 6,
    "decoder_layers": 10,
    "num_attention_heads": 8,
    "dropout": 0.0,
    "dim_feedforward": 2048,
    "pre_norm": False,
    "enforce_input_projection": False,
    "common_stride": 4,
    "ignore_value": 255,
    "num_queries": 100,
    "no_object_weight": 0.1,
    "class_weight": 2.0,
    "mask_weight": 5.0,
    "dice_weight": 5.0,
    "train_num_points": 12544,
    "oversample_ratio": 3.0,
    "importance_sample_ratio": 0.75,
    "init_std": 0.02,
    "init_xavier_std": 1.0,
    "use_auxiliary_loss": True,
    "output_auxiliary_logits": None,
    "feature_strides": [4, 8, 16, 32],
}


class Mask2Former(nn.Module):
    def __init__(self, num_classes: int = output_classes, img_size: int = image_size):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
        self.model = Mask2FormerForUniversalSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We bypass the HF loss path by only passing pixel_values; this returns
        # masks_queries_logits [B, Q, H', W'] and class_queries_logits
        # [B, Q, num_classes + 1] (last channel is the "no object" class).
        outputs = self.model(pixel_values=x)

        mask_logits = outputs.masks_queries_logits          # [B, Q, H', W']
        class_logits = outputs.class_queries_logits          # [B, Q, C+1]

        # Drop the "no object" class and collapse queries into per-class
        # pixel logits.
        class_probs = class_logits.softmax(dim=-1)[..., :-1]  # [B, Q, C]
        mask_probs = mask_logits.sigmoid()                    # [B, Q, H', W']

        seg_logits = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

        # Upsample to the input spatial size so the per-pixel CE loss against
        # [B, H, W] targets lines up.
        seg_logits = F.interpolate(
            seg_logits,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return seg_logits
