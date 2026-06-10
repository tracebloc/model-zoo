"""DistilBERT via HuggingFace with a per-token classification head. ~60% the size of BERT-base, ~97% of its accuracy; pick when training speed or edge resources matter."""
from transformers import AutoModelForTokenClassification

model_id = "distilbert-base-uncased"
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


def MyModel(num_classes=output_classes):

    return AutoModelForTokenClassification.from_pretrained(
        model_id, num_labels=num_classes, ignore_mismatched_sizes=True
    )
