"""ModernBERT-base with a per-token classification head. 2024 long-context encoder (8k tokens, RoPE + GeGLU); state-of-the-art accuracy/throughput for token tagging.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained encoder tensors are
delivered from the tracebloc model store as the training seed: upload the
matched ``modernbert_token_weights.pkl`` sitting next to this file via
``weights=True``, with the matching tokenizer as an explicitly named sibling::

    user.upload_model("modernbert_token", weights=True,
                      tokenizer="modernbert_token_tokenizer.json")

(A bare ``tokenizer.json`` cannot live in this directory: sibling
token_classification templates use different tokenizers, and the SDK
auto-attaches a bare ``tokenizer.json`` to every model in the folder.)
See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file.
"""

from transformers import AutoModelForTokenClassification, AutoConfig

framework = "pytorch"
main_class = "MyModel"
category = "token_classification"
model_type = ""
batch_size = 16
sequence_length = 128
# BIO/IOB2 tag count. Default 9 matches the CoNLL-2003 scheme:
# O + B/I x {PER, ORG, LOC, MISC}. Set to your dataset's tag count.
output_classes = 9
license = "Apache-2.0"

# Architecture config for answerdotai/ModernBERT-base, inlined so the model
# builds with no config fetch. The SDK uploads the .py plus its named weight
# and tokenizer siblings — there is no config.json path — so the config
# lives here in the template. Attention alternates: every 3rd layer global
# (rope theta 160k), the rest sliding-window local (rope theta 10k).
CONFIG = {
    "model_type": "modernbert",
    "vocab_size": 50368,
    "hidden_size": 768,
    "intermediate_size": 1152,
    "num_hidden_layers": 22,
    "num_attention_heads": 12,
    "hidden_activation": "gelu",
    "max_position_embeddings": 8192,
    "initializer_range": 0.02,
    "initializer_cutoff_factor": 2.0,
    "norm_eps": 1e-05,
    "norm_bias": False,
    "pad_token_id": 50283,
    "eos_token_id": 50282,
    "bos_token_id": 50281,
    "cls_token_id": 50281,
    "sep_token_id": 50282,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "global_attn_every_n_layers": 3,
    "rope_parameters": {
        "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    },
    "local_attention": 128,
    "embedding_dropout": 0.0,
    "mlp_bias": False,
    "mlp_dropout": 0.0,
    "decoder_bias": True,
    "classifier_pooling": "mean",
    "classifier_dropout": 0.0,
    "classifier_bias": False,
    "classifier_activation": "gelu",
    "deterministic_flash_attn": False,
    "sparse_prediction": False,
    "sparse_pred_ignore_index": -100,
    "tie_word_embeddings": True,
    "layer_norm_eps": 1e-05,
    "position_embedding_type": "absolute",
}


def MyModel(num_classes=output_classes):
    config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
    return AutoModelForTokenClassification.from_config(config)
