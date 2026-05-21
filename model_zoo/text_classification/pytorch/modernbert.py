"""ModernBERT (Answer.AI / LightOn, Dec 2024). Drop-in BERT replacement — 8192-token context, ~3x training speed, strong on classification + retrieval. Fine-tuned LoRA-only so federated averaging only syncs the small adapter tensors."""
from peft import LoraConfig, TaskType, get_peft_model
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
    base = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_classes, ignore_mismatched_sizes=True
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["Wqkv"],
        modules_to_save=["classifier", "head"],
    )
    return get_peft_model(base, lora_config)
