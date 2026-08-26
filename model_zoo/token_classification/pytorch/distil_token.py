"""DistilBERT via HuggingFace with a per-token classification head. ~60% the size of BERT-base, ~97% of its accuracy; pick when training speed or edge resources matter.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained encoder tensors are
delivered from the tracebloc model store as the training seed: upload the
matched ``distil_token_weights.pkl`` sitting next to this file via
``weights=True``, with the matching tokenizer as an explicitly named sibling::

    user.upload_model("distil_token", weights=True,
                      tokenizer="distil_token_tokenizer.json")

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
batch_size = 64
sequence_length = 128
# BIO/IOB2 tag count. Default 9 matches the CoNLL-2003 scheme:
# O + B/I x {PER, ORG, LOC, MISC}. Set to your dataset's tag count.
output_classes = 9
license = "Apache-2.0"

# Architecture config for distilbert-base-uncased, inlined so the model
# builds with no config fetch. The SDK uploads the .py plus its named weight
# and tokenizer siblings — there is no config.json path — so the config
# lives here in the template.
CONFIG = {
    "model_type": "distilbert",
    "vocab_size": 30522,
    "max_position_embeddings": 512,
    "sinusoidal_pos_embds": False,
    "n_layers": 6,
    "n_heads": 12,
    "dim": 768,
    "hidden_dim": 3072,
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "activation": "gelu",
    "initializer_range": 0.02,
    "qa_dropout": 0.1,
    "seq_classif_dropout": 0.2,
    "pad_token_id": 0,
    "tie_weights_": True,
}


def MyModel(num_classes=output_classes):
    config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
    return AutoModelForTokenClassification.from_config(config)
