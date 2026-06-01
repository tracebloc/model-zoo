"""ELECTRA (Stanford / Google, ICLR 2020). Replaced-token-detection pretraining — same compute, consistently better downstream accuracy than BERT/RoBERTa at small scale. Strong baseline when training-data budget is tight. Select LoRA-only fine-tuning in the training plan if you want federated averaging to sync only the adapter tensors."""
from transformers import AutoModelForSequenceClassification

model_id = "google/electra-base-discriminator"
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
