"""ModernBERT (Answer.AI / LightOn, Dec 2024). Drop-in BERT replacement — 8192-token context, ~3x training speed, strong on classification + retrieval. Select LoRA-only fine-tuning in the training plan if you want federated averaging to sync only the adapter tensors."""
from transformers import AutoModelForSequenceClassification

model_id = "answerdotai/ModernBERT-base"
framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
category = "text_classification"
model_type = ""
batch_size = 16
sequence_length = 512
output_classes = 5


def MyModel(num_classes=output_classes):
    return AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_classes, ignore_mismatched_sizes=True
    )
