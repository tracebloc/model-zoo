"""DeBERTa-v3-base with a per-token classification head. Disentangled attention + ELECTRA-style pretraining; state-of-the-art on CoNLL/OntoNotes NER."""
from transformers import AutoModelForTokenClassification

model_id = "microsoft/deberta-v3-base"
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


def MyModel(num_classes=output_classes):
    return AutoModelForTokenClassification.from_pretrained(
        model_id, num_labels=num_classes, ignore_mismatched_sizes=True
    )
