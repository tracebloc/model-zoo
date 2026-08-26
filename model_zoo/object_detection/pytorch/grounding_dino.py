"""Grounding DINO (IDEA, ECCV 2024). Open-vocabulary object detection — detects classes given as text rather than fixed integer labels. 52.5 AP zero-shot COCO; fine-tunes well on private class names.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The Swin backbone and the BERT text
encoder are both built from the nested configs, randomly initialized. The
full pretrained tensors (``IDEA-Research/grounding-dino-tiny``, backbone and
text encoder included) are delivered from the tracebloc model store as the
training seed: upload the matched ``grounding_dino_weights.pkl`` sitting
next to this file via ``weights=True``::

    user.upload_model("grounding_dino", weights=True)

No tokenizer sibling ships with this template: the model file itself never
tokenizes — text prompts are tokenized by the data pipeline, and the
engine's HF-transformer detection contract (which will own that pipeline)
is not wired yet. When that contract lands, its increment decides how the
prompt tokenizer is distributed.

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file.
"""
from transformers import GroundingDinoForObjectDetection, GroundingDinoConfig

framework = "pytorch"
model_type = "detr"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 800
batch_size = 4
output_classes = 12
category = "object_detection"

# Architecture config for IDEA-Research/grounding-dino-tiny
# (GroundingDinoConfig, model_type "grounding-dino"), inlined so the model
# builds with no config fetch. The nested Swin backbone config and BERT
# text-encoder config are inlined in full so neither sub-architecture can
# drift with library defaults. The SDK uploads the .py plus its named weight
# sibling — there is no config.json path — so the config lives here in the
# template.
CONFIG = {
    "backbone": None,
    "backbone_config": {
        "model_type": "swin",
        "attention_probs_dropout_prob": 0.0,
        "depths": [2, 2, 6, 2],
        "drop_path_rate": 0.1,
        "embed_dim": 96,
        "encoder_stride": 32,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "hidden_size": 768,
        "image_size": 224,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-05,
        "mlp_ratio": 4.0,
        "num_channels": 3,
        "num_heads": [3, 6, 12, 24],
        "num_layers": 4,
        "out_features": ["stage2", "stage3", "stage4"],
        "out_indices": [2, 3, 4],
        "patch_size": 4,
        "qkv_bias": True,
        "stage_names": ["stem", "stage1", "stage2", "stage3", "stage4"],
        "use_absolute_embeddings": False,
        "window_size": 7,
    },
    "text_config": {
        "model_type": "bert",
        "add_cross_attention": False,
        "attention_probs_dropout_prob": 0.1,
        "bos_token_id": None,
        "classifier_dropout": None,
        "eos_token_id": None,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "is_decoder": False,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "tie_word_embeddings": True,
        "type_vocab_size": 2,
        "use_cache": True,
        "vocab_size": 30522,
    },
}


def MyModel(num_classes=output_classes):
    # Grounding DINO is open-vocabulary: classes are passed as text queries at
    # inference time, not as a fixed integer head size. num_classes is accepted
    # for SDK signature uniformity but intentionally not wired into the model.
    del num_classes
    config = GroundingDinoConfig(**CONFIG)
    return GroundingDinoForObjectDetection(config)
