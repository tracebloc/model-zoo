"""DistilBERT trained from scratch. Rarely the right choice; usually prefer the pretrained distilbert.py.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. A scratch template random-initializes
by design, so there is no weight file: upload with ``weights=False`` and the
matching tokenizer as an explicitly named sibling::

    user.upload_model("distilbert_scratch",
                      tokenizer="distilbert_scratch_tokenizer.json")

(A bare ``tokenizer.json`` cannot live in this directory: sibling
text_classification templates use different tokenizers, and the SDK
auto-attaches a bare ``tokenizer.json`` to every model in the folder.)
"""

from transformers import AutoModelForSequenceClassification, AutoConfig

framework = "pytorch"
main_class = "MyModel"
category = "text_classification"
model_type = ""
batch_size = 512
sequence_length = 5
output_classes = 5

# Architecture config for distilbert-base-uncased-finetuned-sst-2-english,
# inlined so the model builds with no config fetch. The SDK uploads the .py
# plus its named tokenizer sibling — there is no config.json path — so the
# config lives here in the template.
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
    "tie_word_embeddings": True,
    "output_past": True,
}


def MyModel(num_classes=output_classes):
    config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
    return AutoModelForSequenceClassification.from_config(config)
