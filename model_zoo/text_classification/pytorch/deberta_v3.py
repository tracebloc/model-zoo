"""DeBERTa-v3 (Microsoft, ICLR 2023). Disentangled attention + ELECTRA-style pretraining. Strongest pure-classification encoder pound-for-pound — controlled 2025 studies show 30-40% better sample efficiency than ModernBERT on equal data."""
from transformers import AutoModelForSequenceClassification

model_id = "microsoft/deberta-v3-base"
framework = "pytorch"
main_method = "MyModel"
license = "MIT"
category = "text_classification"
model_type = ""
batch_size = 32
sequence_length = 512
output_classes = 5


def MyModel(num_classes=output_classes):
    return AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_classes, ignore_mismatched_sizes=True
    )
