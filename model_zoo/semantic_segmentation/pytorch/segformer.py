"""SegFormer (NVIDIA, NeurIPS 2021). Efficient transformer segmenter — hierarchical MiT encoder + lightweight MLP decoder; ~5x fewer params than prior SOTA at similar mIoU. Pairs with Mask2Former: SegFormer for speed, Mask2Former for accuracy.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained tensors are delivered
from the tracebloc model store as the training seed: upload the matched
``segformer_weights.pkl`` sitting next to this file via ``weights=True``::

    user.upload_model("segformer", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file (weights of ``nvidia/segformer-b0-finetuned-ade-512-512``;
the decode head's classifier is sized to ``output_classes`` and freshly
initialized).
"""
from transformers import AutoConfig, AutoModelForSemanticSegmentation

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("decode_head.classifier.",)

framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 512
batch_size = 8
output_classes = 2
category = "semantic_segmentation"

# Architecture config for nvidia/segformer-b0-finetuned-ade-512-512
# (SegformerForSemanticSegmentation, model_type "segformer"), inlined so
# the model builds with no config fetch. The SDK uploads the .py plus its
# named weight sibling — there is no config.json path — so the config lives
# here in the template.
CONFIG = {
    "model_type": "segformer",
    "num_channels": 3,
    "num_encoder_blocks": 4,
    "depths": [2, 2, 2, 2],
    "sr_ratios": [8, 4, 2, 1],
    "hidden_sizes": [32, 64, 160, 256],
    "patch_sizes": [7, 3, 3, 3],
    "strides": [4, 2, 2, 2],
    "num_attention_heads": [1, 2, 5, 8],
    "mlp_ratios": [4, 4, 4, 4],
    "downsampling_rates": [1, 4, 8, 16],
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "classifier_dropout_prob": 0.1,
    "drop_path_rate": 0.1,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-06,
    "decoder_hidden_size": 256,
    "reshape_last_stage": True,
    "semantic_loss_ignore_index": 255,
    "image_size": 224,
}


def MyModel(num_classes=output_classes):
    config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
    return AutoModelForSemanticSegmentation.from_config(config)
