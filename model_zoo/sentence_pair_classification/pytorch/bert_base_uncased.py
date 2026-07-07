"""BERT-base-uncased via HuggingFace, pretrained. Canonical sentence-pair baseline.

BERT is the natural choice for sentence-pair tasks (NLI, paraphrase, similarity):
the client tokenizes the pair as ``tokenizer(text_a, text_b)``, so the two
segments are joined by ``[SEP]`` and distinguished by ``token_type_ids``
(segment embeddings) — which BERT was pretrained with. Fine-tune the head for
your labels."""
from transformers import AutoModelForSequenceClassification

model_id = 'bert-base-uncased'
# HF tokenizer distributed to every client as the federation's single
# source of truth (#805). Mandatory for HuggingFace NLP models.
tokenizer_id = 'bert-base-uncased'
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


def MyModel(num_classes=output_classes):

    return AutoModelForSequenceClassification.from_pretrained(model_id,
        num_labels=num_classes,
        ignore_mismatched_sizes=True)
