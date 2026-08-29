"""OneFormer (CVPR 2023). Single set of weights for semantic / instance / panoptic segmentation; consistently matches or beats Mask2Former with one checkpoint instead of three.

This wrapper exposes the HF `OneFormerForUniversalSegmentation` as a plain
semantic-segmentation module, matching the contract used by the other models
in this folder (deeplab.py, mask2former.py, ...): `forward(x)` returns
`[B, num_classes, H, W]` logits, and the platform trainer applies its
standard per-pixel CrossEntropyLoss against the `[B, H, W]` mask targets.

Two things the bare HF model would otherwise demand from the caller are
supplied internally:

* **task inputs** — OneFormer conditions on a tokenized task string
  ("the task is semantic"); the fixed CLIP-token id sequence for the
  semantic task is inlined below and registered as a (non-persistent)
  buffer, so `forward(x)` needs nothing but pixels and no tokenizer file
  ships with this template.
* **dense logits** — the raw (mask_queries, class_queries) outputs are
  collapsed into per-class pixel logits with the same formula HF's
  `post_process_semantic_segmentation` uses (see mask2former.py):

      seg_logits = einsum("bqc,bqhw->bchw",
                          softmax(class_logits)[..., :-1],
                          sigmoid(mask_logits))

  then upsampled to the input resolution.

Offline variant: the architecture is built from the inlined config below
(Swin-tiny backbone sub-config and task-text-encoder geometry included) —
no hub model id, no config fetch, no download at build time, so the
template constructs anywhere, network or not. The pretrained tensors are
delivered from the tracebloc model store as the training seed: upload the
matched ``oneformer_weights.pkl`` sitting next to this file via
``weights=True``::

    user.upload_model("oneformer", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file (weights of ``shi-labs/oneformer_ade20k_swin_tiny``;
the class-prediction head is sized to ``output_classes`` and freshly
initialized).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, OneFormerForUniversalSegmentation

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("model.criterion.", "model.model.transformer_module.decoder.class_embed.")

framework = "pytorch"
main_method = "MyModel"
license = "MIT"
image_size = 512
batch_size = 4
output_classes = 2
category = "semantic_segmentation"

# Task-conditioning ids for "the task is semantic", exactly as
# OneFormerProcessor._preprocess_text builds them: CLIP-tokenize padded to
# task_seq_len (77), then multiply input_ids by the attention mask — which
# ZEROS every pad position. So the checkpoint-faithful sequence is the 7
# real tokens (49406 BOS, 5 word tokens, 49407 EOS) followed by 70 zeros,
# inlined so forward() needs no tokenizer at runtime.
SEMANTIC_TASK_TOKEN_IDS = [49406, 518, 10549, 533, 29119, 1550, 49407] + [0] * 70

# Architecture config for shi-labs/oneformer_ade20k_swin_tiny
# (OneFormerForUniversalSegmentation, model_type "oneformer") with its
# nested Swin backbone sub-config, inlined so the model builds with no
# config fetch. The SDK uploads the .py plus its named weight sibling —
# there is no config.json path — so the config lives here in the template.
# Note: OneFormer's own ``num_classes`` (loss/matcher sizing, 150 on the
# ADE20K checkpoint) is separate from ``num_labels`` (the class-prediction
# head), matching the pre-migration behavior where only ``num_labels`` was
# overridden.
CONFIG = {
    "model_type": "oneformer",
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
    "ignore_value": 255,
    "num_queries": 150,
    "no_object_weight": 0.1,
    "class_weight": 2.0,
    "mask_weight": 5.0,
    "dice_weight": 5.0,
    "contrastive_weight": 0.5,
    "contrastive_temperature": 0.07,
    "train_num_points": 12544,
    "oversample_ratio": 3.0,
    "importance_sample_ratio": 0.75,
    "init_std": 0.02,
    "init_xavier_std": 1.0,
    "layer_norm_eps": 1e-05,
    "is_training": False,
    "use_auxiliary_loss": True,
    "output_auxiliary_logits": True,
    # The hub config ships output_attentions/output_hidden_states as True;
    # both are runtime-only output flags with no weight impact, and the
    # wrapper only reads mask/class logits — keep them off rather than
    # materializing every Swin hidden state + attention map each forward.
    "output_attentions": False,
    "output_hidden_states": False,
    "strides": [4, 8, 16, 32],
    "task_seq_len": 77,
    "max_seq_len": 77,
    "text_encoder_width": 256,
    "text_encoder_context_length": 77,
    "text_encoder_num_layers": 6,
    "text_encoder_vocab_size": 49408,
    "text_encoder_proj_layers": 2,
    "text_encoder_n_ctx": 16,
    "conv_dim": 256,
    "mask_dim": 256,
    "hidden_dim": 256,
    "encoder_feedforward_dim": 1024,
    "norm": "GN",
    "encoder_layers": 6,
    "decoder_layers": 10,
    "use_task_norm": True,
    "num_attention_heads": 8,
    "dropout": 0.1,
    "dim_feedforward": 2048,
    "pre_norm": False,
    "enforce_input_proj": False,
    "query_dec_layers": 2,
    "common_stride": 4,
    "num_classes": 150,
}


class _OneFormerSemantic(nn.Module):
    def __init__(self, num_classes: int = output_classes, img_size: int = image_size):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
        self.model = OneFormerForUniversalSegmentation(config)

        # Fixed semantic-task conditioning tokens; non-persistent so the
        # constant stays out of the state_dict (the weight dump carries
        # parameters only) while still following .to(device)/.cuda().
        self.register_buffer(
            "task_inputs",
            torch.tensor(SEMANTIC_TASK_TOKEN_IDS, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        task_inputs = self.task_inputs.expand(x.shape[0], -1)
        outputs = self.model(pixel_values=x, task_inputs=task_inputs)

        mask_logits = outputs.masks_queries_logits            # [B, Q, H', W']
        class_logits = outputs.class_queries_logits           # [B, Q, C+1]

        # Drop the "no object" class and collapse queries into per-class
        # pixel logits (same math as mask2former.py).
        class_probs = class_logits.softmax(dim=-1)[..., :-1]  # [B, Q, C]
        mask_probs = mask_logits.sigmoid()                    # [B, Q, H', W']

        seg_logits = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

        # Upsample to the input spatial size so the per-pixel CE loss against
        # [B, H, W] targets lines up.
        return F.interpolate(
            seg_logits,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )


def MyModel(num_classes=output_classes):
    return _OneFormerSemantic(num_classes=num_classes, img_size=image_size)
