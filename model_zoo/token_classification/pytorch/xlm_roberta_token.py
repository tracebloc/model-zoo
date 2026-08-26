"""XLM-RoBERTa-base with a per-token classification head. SOTA multilingual NER/POS across 100 languages; the default for non-English token tagging.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained encoder tensors are
delivered from the tracebloc model store as the training seed: upload the
matched ``xlm_roberta_token_weights.pkl`` sitting next to this file via
``weights=True``, with the matching tokenizer as an explicitly named sibling::

    user.upload_model("xlm_roberta_token", weights=True,
                      tokenizer="xlm_roberta_token_tokenizer.json")

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
license = "MIT"

# Architecture config for xlm-roberta-base, inlined so the model builds with
# no config fetch. The SDK uploads the .py plus its named weight and
# tokenizer siblings — there is no config.json path — so the config lives
# here in the template.
CONFIG = {
    "model_type": "xlm-roberta",
    "vocab_size": 250002,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 514,
    "type_vocab_size": 1,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-05,
    "pad_token_id": 1,
    "bos_token_id": 0,
    "eos_token_id": 2,
    "output_past": True,
    "position_embedding_type": "absolute",
}


def MyModel(num_classes=output_classes):
    config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
    return AutoModelForTokenClassification.from_config(config)
