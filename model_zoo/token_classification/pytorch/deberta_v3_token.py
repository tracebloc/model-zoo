"""DeBERTa-v3-base with a per-token classification head. Disentangled attention + ELECTRA-style pretraining; state-of-the-art on CoNLL/OntoNotes NER.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained encoder tensors are
delivered from the tracebloc model store as the training seed: upload the
matched ``deberta_v3_token_weights.pkl`` sitting next to this file via
``weights=True``, with the matching tokenizer as an explicitly named sibling::

    user.upload_model("deberta_v3_token", weights=True,
                      tokenizer="deberta_v3_token_tokenizer.json")

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

# Architecture config for microsoft/deberta-v3-base (a v2-architecture
# checkpoint, hence model_type "deberta-v2"), inlined so the model builds
# with no config fetch. The SDK uploads the .py plus its named weight and
# tokenizer siblings — there is no config.json path — so the config lives
# here in the template.
CONFIG = {
    "model_type": "deberta-v2",
    "vocab_size": 128100,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "type_vocab_size": 0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-07,
    "relative_attention": True,
    "max_relative_positions": -1,
    "pad_token_id": 0,
    "position_biased_input": False,
    "pos_att_type": ["p2c", "c2p"],
    "position_buckets": 256,
    "norm_rel_ebd": "layer_norm",
    "share_att_key": True,
    "pooler_dropout": 0.0,
    "pooler_hidden_act": "gelu",
    "pooler_hidden_size": 768,
    "legacy": True,
}


def MyModel(num_classes=output_classes):
    config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
    return AutoModelForTokenClassification.from_config(config)
