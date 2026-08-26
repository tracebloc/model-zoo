"""Qwen2.5-0.5B (Alibaba, 2024) as a small decoder-LLM classifier. Decoder-LLM-as-classifier is the dominant 2024-2025 pattern for hard classification tasks; at ~0.5B params this is the lightweight entry point — it fine-tunes on a laptop-class edge and still clearly beats encoder baselines on instruction-heavy label sets. Select LoRA-only fine-tuning in the training plan so federated averaging only syncs the adapter + classification head.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained tensors are delivered
from the tracebloc model store as the training seed: upload the matched
``qwen2_5_0_5b_weights.pkl`` sitting next to this file via ``weights=True``,
with the matching tokenizer as an explicitly named sibling::

    user.upload_model("qwen2_5_0_5b", weights=True,
                      tokenizer="qwen2_5_0_5b_tokenizer.json")

(A bare ``tokenizer.json`` cannot live in this directory: sibling
text_classification templates use different tokenizers, and the SDK
auto-attaches a bare ``tokenizer.json`` to every model in the folder.)
See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file.

Pad-token note: Qwen2.5 ships no dedicated pad token — by convention
``<|endoftext|>`` (id 151643, also the eos token) pads. The inlined config
pins ``pad_token_id`` to it (sequence classification pools the last
non-pad position, so an unset pad id breaks batched inference), and the
committed ``qwen2_5_0_5b_tokenizer.json`` carries that token, so template
and tokenizer agree by construction.

Resources: ~2GB RAM for the fp32 build (~0.5B params, full base) — small
enough for the local SDK self-check and CI construction. Choose LoRA in the
training plan to train only a ~4M adapter + score head.
"""

from transformers import AutoModelForSequenceClassification, AutoConfig

framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
category = "text_classification"
model_type = ""
batch_size = 8
sequence_length = 256
output_classes = 5

# Architecture config for Qwen/Qwen2.5-0.5B, inlined so the model builds
# with no config fetch. The SDK uploads the .py plus its named weight and
# tokenizer siblings — there is no config.json path — so the config lives
# here in the template. pad_token_id is pinned to <|endoftext|> (151643,
# = eos) because Qwen2.5 declares no pad token of its own.
CONFIG = {
    "model_type": "qwen2",
    "vocab_size": 151936,
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "hidden_act": "silu",
    "max_position_embeddings": 32768,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-06,
    "use_cache": True,
    "tie_word_embeddings": True,
    "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
    "use_sliding_window": False,
    "max_window_layers": 24,
    "attention_dropout": 0.0,
    "pad_token_id": 151643,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
}


def MyModel(num_classes=output_classes):
    config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
    return AutoModelForSequenceClassification.from_config(config)
