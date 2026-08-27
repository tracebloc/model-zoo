"""GTE-ModernBERT (Alibaba-NLP, 2025). ModernBERT fine-tuned for general text embeddings — top MTEB scores at its size; doubles as a strong classifier via mean-pool + linear head. Select LoRA-only fine-tuning in the training plan for federated averaging.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not (this also clears the #1495 upload gate
that rejected the previous hub-fetching version). The pretrained
encoder tensors are delivered from the tracebloc model store as the training
seed: upload the matched ``gte_modernbert_weights.pkl`` sitting next to this
file via ``weights=True``, with the matching tokenizer as an explicitly
named sibling::

    user.upload_model("gte_modernbert", weights=True,
                      tokenizer="gte_modernbert_tokenizer.json")

(A bare ``tokenizer.json`` cannot live in this directory: sibling
text_classification templates use different tokenizers, and the SDK
auto-attaches a bare ``tokenizer.json`` to every model in the folder.)
See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file.

Resources: ~2GB RAM to build (~150M params, full base). At
``sequence_length=512, batch_size=4`` expect ~1 minute per step on CPU; if
it looks stuck on a bigger seq/batch, halve both. Choose LoRA in the
training plan to train only a ~1M adapter + classifier head.
"""

from transformers import AutoModelForSequenceClassification, AutoConfig

framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
category = "text_classification"
model_type = ""
batch_size = 4
sequence_length = 512
output_classes = 5

# Architecture config for Alibaba-NLP/gte-modernbert-base, inlined so the
# model builds with no config fetch. The SDK uploads the .py plus its named
# weight and tokenizer siblings — there is no config.json path — so the
# config lives here in the template. Same architecture as ModernBERT-base;
# the pretrained tensors (the GTE fine-tune) are what differ.
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
}


def MyModel(num_classes=output_classes):
    config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
    return AutoModelForSequenceClassification.from_config(config)
