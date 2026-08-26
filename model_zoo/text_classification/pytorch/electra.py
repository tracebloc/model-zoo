"""ELECTRA (Stanford / Google, ICLR 2020). Replaced-token-detection pretraining — same compute, consistently better downstream accuracy than BERT/RoBERTa at small scale. Strong baseline when training-data budget is tight. Select LoRA-only fine-tuning in the training plan if you want federated averaging to sync only the adapter tensors.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained encoder tensors are
delivered from the tracebloc model store as the training seed: upload the
matched ``electra_weights.pkl`` sitting next to this file via
``weights=True``, with the matching tokenizer as an explicitly named sibling::

    user.upload_model("electra", weights=True,
                      tokenizer="electra_tokenizer.json")

(A bare ``tokenizer.json`` cannot live in this directory: sibling
text_classification templates use different tokenizers, and the SDK
auto-attaches a bare ``tokenizer.json`` to every model in the folder.)
See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file.
"""

from transformers import AutoModelForSequenceClassification, AutoConfig

framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
category = "text_classification"
model_type = ""
batch_size = 16
sequence_length = 512
output_classes = 5

# Architecture config for google/electra-base-discriminator, inlined so the
# model builds with no config fetch. The SDK uploads the .py plus its named
# weight and tokenizer siblings — there is no config.json path — so the
# config lives here in the template.
CONFIG = {
    "model_type": "electra",
    "vocab_size": 30522,
    "embedding_size": 768,
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
    "summary_type": "first",
    "summary_use_proj": True,
    "summary_activation": "gelu",
    "summary_last_dropout": 0.1,
    "pad_token_id": 0,
    "classifier_dropout": None,
    "position_embedding_type": "absolute",
}


def MyModel(num_classes=output_classes):
    config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
    return AutoModelForSequenceClassification.from_config(config)
