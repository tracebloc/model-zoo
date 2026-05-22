"""DeBERTa-v3 (Microsoft, ICLR 2023). Disentangled attention + ELECTRA-style pretraining. Strongest pure-classification encoder pound-for-pound — controlled 2025 studies show 30-40% better sample efficiency than ModernBERT on equal data. Fine-tuned LoRA-only so federated averaging only syncs the small adapter tensors."""
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

model_id = "microsoft/deberta-v3-base"
framework = "pytorch"
main_method = "MyModel"
license = "MIT"
category = "text_classification"
model_type = ""
batch_size = 8
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
        target_modules=["query_proj", "key_proj", "value_proj"],
        modules_to_save=["classifier", "pooler"],
    )
    return get_peft_model(base, lora_config)
