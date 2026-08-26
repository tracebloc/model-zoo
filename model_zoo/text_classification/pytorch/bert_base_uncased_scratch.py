"""BERT-base-uncased trained from scratch. Pick only if you have a massive domain-specific corpus and want to avoid English-pretrained biases.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. A scratch template random-initializes
by design, so there is no weight file: upload with ``weights=False`` and the
matching tokenizer as an explicitly named sibling::

    user.upload_model("bert_base_uncased_scratch",
                      tokenizer="bert_base_uncased_scratch_tokenizer.json")

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

# Architecture config for bert-base-uncased, inlined so the model builds
# with no config fetch. The SDK uploads the .py plus its named tokenizer
# sibling — there is no config.json path — so the config lives here in the
# template.
CONFIG = {
    "model_type": "bert",
    "vocab_size": 30522,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
}


def MyModel(num_classes=output_classes):
    config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
    return AutoModelForSequenceClassification.from_config(config)
