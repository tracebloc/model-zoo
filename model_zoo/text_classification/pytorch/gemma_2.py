"""Gemma-2 2B (Google, 2024) as a decoder-LLM classifier. Decoder-LLM-as-classifier is the dominant 2024-2025 pattern for hard classification tasks. Select LoRA-only fine-tuning in the training plan so federated averaging only syncs the adapter + classification head.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download, no HuggingFace token at build
time (the Gemma license gate applies to the one-time weight prep, not to
this template), so the template constructs anywhere, network or not. The
pretrained tensors are delivered from the tracebloc model store as the
training seed: upload the matched ``gemma_2_weights.pkl`` sitting next to
this file via ``weights=True``, with the matching tokenizer as an explicitly
named sibling::

    user.upload_model("gemma_2", weights=True,
                      tokenizer="gemma_2_tokenizer.json")

(A bare ``tokenizer.json`` cannot live in this directory: sibling
text_classification templates use different tokenizers, and the SDK
auto-attaches a bare ``tokenizer.json`` to every model in the folder.)
See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file.

Resources: ~10GB RAM for the fp32 build (~2.6B params, full base). ~32GB
system RAM recommended for the local SDK self-check (CPU forward+backward on
synthetic data, expect 2-5 minutes on a laptop). Choose LoRA in the training
plan to train only a ~10M adapter + score head.
"""

from transformers import AutoModelForSequenceClassification, AutoConfig

framework = "pytorch"
main_method = "MyModel"
license = "Gemma"
category = "text_classification"
model_type = ""
batch_size = 4
sequence_length = 256
output_classes = 5

# Architecture config for google/gemma-2-2b, inlined so the model builds
# with no config fetch. The SDK uploads the .py plus its named weight and
# tokenizer siblings — there is no config.json path — so the config lives
# here in the template. Attention alternates sliding-window / full every
# other layer.
CONFIG = {
    "model_type": "gemma2",
    "vocab_size": 256000,
    "hidden_size": 2304,
    "intermediate_size": 9216,
    "num_hidden_layers": 26,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "max_position_embeddings": 8192,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-06,
    "use_cache": True,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "bos_token_id": 2,
    "tie_word_embeddings": True,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    "attention_bias": False,
    "attention_dropout": 0.0,
    "query_pre_attn_scalar": 256,
    "sliding_window": 4096,
    "final_logit_softcapping": 30.0,
    "attn_logit_softcapping": 50.0,
    "cache_implementation": "hybrid",
}


def MyModel(num_classes=output_classes):
    config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
    return AutoModelForSequenceClassification.from_config(config)
