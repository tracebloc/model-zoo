"""DPT — Dense Prediction Transformer (Intel ISL, ICCV 2021). ViT backbone with a dense prediction head; primarily known for depth but the semantic-segmentation variant is a strong reference baseline on ADE20K.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained tensors are delivered
from the tracebloc model store as the training seed: upload the matched
``dpt_weights.pkl`` sitting next to this file via ``weights=True``::

    user.upload_model("dpt", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file (weights of ``Intel/dpt-large-ade``; the segmentation
head is sized to ``output_classes`` and freshly initialized).
"""
from transformers import AutoConfig, AutoModelForSemanticSegmentation

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("auxiliary_head.head.4.", "head.head.4.")

framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 480
batch_size = 4
output_classes = 2
category = "semantic_segmentation"

# Architecture config for Intel/dpt-large-ade (DPTForSemanticSegmentation,
# model_type "dpt"), inlined so the model builds with no config fetch. The
# SDK uploads the .py plus its named weight sibling — there is no
# config.json path — so the config lives here in the template.
CONFIG = {
    "model_type": "dpt",
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_channels": 3,
    "image_size": 384,
    "patch_size": 16,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "qkv_bias": True,
    "is_hybrid": False,
    "backbone_config": None,
    "backbone_featmap_shape": None,
    "backbone_out_indices": [5, 11, 17, 23],
    "readout_type": "project",
    "reassemble_factors": [4, 2, 1, 0.5],
    "neck_hidden_sizes": [256, 512, 1024, 1024],
    "neck_ignore_stages": [],
    "fusion_hidden_size": 256,
    "use_batch_norm_in_fusion_residual": True,
    "use_bias_in_fusion_residual": None,
    "add_projection": False,
    "head_in_index": -1,
    "use_auxiliary_head": True,
    "auxiliary_loss_weight": 0.4,
    "semantic_loss_ignore_index": 255,
    "semantic_classifier_dropout": 0.1,
    "pooler_output_size": 1024,
    "pooler_act": "tanh",
}


def MyModel(num_classes=output_classes):
    config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
    return AutoModelForSemanticSegmentation.from_config(config)
