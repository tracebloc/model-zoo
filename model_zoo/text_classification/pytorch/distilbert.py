"""DistilBERT via HuggingFace, pretrained. ~60% the size of BERT-base, ~97% of its accuracy; strong default for inference-speed-sensitive setups.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained encoder tensors are
delivered from the tracebloc model store as the training seed: upload the
matched ``distilbert_weights.pkl`` sitting next to this file via
``weights=True``, with the matching tokenizer as an explicitly named sibling::

    user.upload_model("distilbert", weights=True,
                      tokenizer="distilbert_tokenizer.json")

(A bare ``tokenizer.json`` cannot live in this directory: sibling
text_classification templates use different tokenizers, and the SDK
auto-attaches a bare ``tokenizer.json`` to every model in the folder.)
See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file.
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
# plus its named weight and tokenizer siblings — there is no config.json
# path — so the config lives here in the template.
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
