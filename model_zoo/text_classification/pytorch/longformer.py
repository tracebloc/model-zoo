"""Longformer (AllenAI, 2020). Sparse local + global attention — supports 4096-token context with linear-in-length compute. The reference choice for long-document classification (legal, medical, financial filings)."""
from transformers import AutoModelForSequenceClassification

model_id = "allenai/longformer-base-4096"
framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
category = "text_classification"
model_type = ""
batch_size = 8
sequence_length = 4096
output_classes = 5


def MyModel(num_classes=output_classes):
    return AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_classes, ignore_mismatched_sizes=True
    )
