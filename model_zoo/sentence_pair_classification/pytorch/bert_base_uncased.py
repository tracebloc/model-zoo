"""BERT-base-uncased via HuggingFace, pretrained. Canonical sentence-pair baseline.

BERT is the natural choice for sentence-pair tasks (NLI, paraphrase, similarity):
the client tokenizes the pair as ``tokenizer(text_a, text_b)``, so the two
segments are joined by ``[SEP]`` and distinguished by ``token_type_ids``
(segment embeddings) — which BERT was pretrained with. Fine-tune the head for
your labels.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained encoder tensors are
delivered from the tracebloc model store as the training seed: upload the
matched ``bert_base_uncased_weights.pkl`` sitting next to this file via
``weights=True``, with the matching tokenizer as an explicitly named sibling::

    user.upload_model("bert_base_uncased", weights=True,
                      tokenizer="bert_base_uncased_tokenizer.json")

(A bare ``tokenizer.json`` cannot live in this directory: sibling
sentence_pair_classification templates use different tokenizers, and the SDK
auto-attaches a bare ``tokenizer.json`` to every model in the folder.)
See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file.
"""

from transformers import AutoModelForSequenceClassification, AutoConfig

framework = "pytorch"
main_class = "MyModel"
category = "sentence_pair_classification"
model_type = ""
batch_size = 512
sequence_length = 5
# Sentence-pair defaults are typically binary (paraphrase / duplicate) or
# ternary (NLI: entailment / neutral / contradiction). This is only a default;
# the client passes the dataset's real label count as ``num_classes``.
output_classes = 2

# Architecture config for bert-base-uncased, inlined so the model builds
# with no config fetch. The SDK uploads the .py plus its named weight and
# tokenizer siblings — there is no config.json path — so the config lives
# here in the template.
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
