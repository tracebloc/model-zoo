"""ELECTRA (Stanford / Google, ICLR 2020). Replaced-token-detection pretraining — same compute, consistently better downstream accuracy than BERT/RoBERTa at small scale. Strong baseline when training-data budget is tight. Fine-tuned LoRA-only so federated averaging only syncs the small adapter tensors."""
from peft import LoraConfig, TaskType, get_peft_model
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
    base = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_classes, ignore_mismatched_sizes=True
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "key", "value"],
        modules_to_save=["classifier"],
    )
    return get_peft_model(base, lora_config)
