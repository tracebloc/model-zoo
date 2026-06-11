"""ELECTRA-base (discriminator) with a per-token classification head. Replaced-token-detection pretraining; strong NER at lower compute, and shares BERT's WordPiece vocab."""
from transformers import AutoModelForTokenClassification

model_id = "google/electra-base-discriminator"
framework = "pytorch"
main_class = "MyModel"
category = "token_classification"
model_type = ""
batch_size = 32
sequence_length = 128
# BIO/IOB2 tag count. Default 9 matches the CoNLL-2003 scheme:
# O + B/I x {PER, ORG, LOC, MISC}. Set to your dataset's tag count.
output_classes = 9
license = "Apache-2.0"


def MyModel(num_classes=output_classes):
    return AutoModelForTokenClassification.from_pretrained(
        model_id, num_labels=num_classes, ignore_mismatched_sizes=True
    )
