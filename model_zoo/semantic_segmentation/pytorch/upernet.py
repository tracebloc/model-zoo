"""UPerNet (ECCV 2018). PPM + FPN decoder paired here with a ConvNeXt backbone. The canonical decoder used in modern segmentation papers when reporting ConvNeXt / Swin results — strong, well-understood baseline.

Offline variant: the architecture is built from the inlined config below
(ConvNeXt-small backbone sub-config included) — no hub model id, no config
fetch, no download at build time, so the template constructs anywhere,
network or not. The pretrained tensors are delivered from the tracebloc
model store as the training seed: upload the matched ``upernet_weights.pkl``
sitting next to this file via ``weights=True``::

    user.upload_model("upernet", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file (weights of ``openmmlab/upernet-convnext-small``; the
decode head's classifier is sized to ``output_classes`` and freshly
initialized).
"""
from transformers import AutoConfig, AutoModelForSemanticSegmentation

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("auxiliary_head.classifier.", "decode_head.classifier.")

framework = "pytorch"
main_method = "MyModel"
license = "MIT"
image_size = 512
batch_size = 4
output_classes = 2
category = "semantic_segmentation"

# Architecture config for openmmlab/upernet-convnext-small
# (UperNetForSemanticSegmentation, model_type "upernet") with its nested
# ConvNeXt backbone sub-config, inlined so the model builds with no config
# fetch. The SDK uploads the .py plus its named weight sibling — there is
# no config.json path — so the config lives here in the template.
CONFIG = {
    "model_type": "upernet",
    "backbone_config": {
        "model_type": "convnext",
        "num_channels": 3,
        "patch_size": 4,
        "num_stages": 4,
        "hidden_sizes": [96, 192, 384, 768],
        "depths": [3, 3, 27, 3],
        "hidden_act": "gelu",
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "layer_scale_init_value": 1e-06,
        "drop_path_rate": 0.0,
        "image_size": 224,
        "out_features": ["stage1", "stage2", "stage3", "stage4"],
        "out_indices": [1, 2, 3, 4],
        "stage_names": ["stem", "stage1", "stage2", "stage3", "stage4"],
    },
    "hidden_size": 512,
    "initializer_range": 0.02,
    "pool_scales": [1, 2, 3, 6],
    "use_auxiliary_head": True,
    "auxiliary_loss_weight": 0.4,
    "auxiliary_in_channels": 384,
    "auxiliary_channels": 256,
    "auxiliary_num_convs": 1,
    "auxiliary_concat_input": False,
    "loss_ignore_index": 255,
}


def MyModel(num_classes=output_classes):
    config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
    return AutoModelForSemanticSegmentation.from_config(config)
