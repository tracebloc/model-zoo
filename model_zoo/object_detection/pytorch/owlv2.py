"""OWLv2 (Google, NeurIPS 2023). Open-vocabulary detector — like Grounding DINO but built on OWL-ViT lineage with self-training on web-scale pseudo-labels. Complements grounding_dino.py; useful when class vocabulary is fluid or zero-shot.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The CLIP-style text and vision towers
are both built from the nested configs, randomly initialized. The full
pretrained tensors (``google/owlv2-base-patch16-ensemble``, both towers
included) are delivered from the tracebloc model store as the training
seed: upload the matched ``owlv2_weights.pkl`` sitting next to this file
via ``weights=True``::

    user.upload_model("owlv2", weights=True)

No tokenizer sibling ships with this template: the model file itself never
tokenizes — text prompts are tokenized by the data pipeline, and the
engine's HF-transformer detection contract (which will own that pipeline)
is not wired yet. When that contract lands, its increment decides how the
prompt tokenizer is distributed.

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file.
"""
from transformers import Owlv2ForObjectDetection, Owlv2Config

framework = "pytorch"
model_type = "detr"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 960
batch_size = 4
output_classes = 12
category = "object_detection"

# Architecture config for google/owlv2-base-patch16-ensemble (Owlv2Config,
# model_type "owlv2"), inlined so the model builds with no config fetch. The
# nested text-tower and vision-tower configs are inlined in full so neither
# sub-architecture can drift with library defaults. The SDK uploads the .py
# plus its named weight sibling — there is no config.json path — so the
# config lives here in the template.
CONFIG = {
    "text_config": {
        "model_type": "owlv2_text_model",
        "attention_dropout": 0.0,
        "bos_token_id": 49406,
        "eos_token_id": 49407,
        "hidden_act": "quick_gelu",
        "hidden_size": 512,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 2048,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 16,
        "num_attention_heads": 8,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "vocab_size": 49408,
    },
    "vision_config": {
        "model_type": "owlv2_vision_model",
        "attention_dropout": 0.0,
        "hidden_act": "quick_gelu",
        "hidden_size": 768,
        "image_size": 960,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-05,
        "num_attention_heads": 12,
        "num_channels": 3,
        "num_hidden_layers": 12,
        "patch_size": 16,
    },
}


def MyModel(num_classes=output_classes):
    # OWLv2 is open-vocabulary: classes are passed as text queries at inference
    # time, not as a fixed integer head size. num_classes is accepted for SDK
    # signature uniformity but intentionally not wired into the model.
    del num_classes
    config = Owlv2Config(**CONFIG)
    return Owlv2ForObjectDetection(config)
