"""OneFormer (CVPR 2023). Single set of weights for semantic / instance / panoptic segmentation; consistently matches or beats Mask2Former with one checkpoint instead of three.

Offline variant: the architecture is built from the inlined config below
(Swin-tiny backbone sub-config and task-text-encoder geometry included) —
no hub model id, no config fetch, no download at build time, so the
template constructs anywhere, network or not. The pretrained tensors are
delivered from the tracebloc model store as the training seed: upload the
matched ``oneformer_weights.pkl`` sitting next to this file via
``weights=True``::

    user.upload_model("oneformer", weights=True)

No tokenizer file ships with this template: the task-token text encoder is
only exercised when the caller passes pre-tokenized ``task_inputs`` /
``text_inputs`` ids to ``forward`` — this template's runtime path (the
platform trainer calls ``model(x)``) never tokenizes anything itself.

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file (weights of ``shi-labs/oneformer_ade20k_swin_tiny``;
the class-prediction head is sized to ``output_classes`` and freshly
initialized).
"""
from transformers import AutoConfig, OneFormerForUniversalSegmentation

framework = "pytorch"
main_method = "MyModel"
license = "MIT"
image_size = 512
batch_size = 4
output_classes = 2
category = "semantic_segmentation"

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
        "path_norm": True,
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
    "output_attentions": True,
    "output_hidden_states": True,
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


def MyModel(num_classes=output_classes):
    config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
    return OneFormerForUniversalSegmentation(config)
