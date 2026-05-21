"""Longformer (AllenAI, 2020). Sparse local + global attention — supports 4096-token context with linear-in-length compute. The reference choice for long-document classification (legal, medical, financial filings). Fine-tuned LoRA-only so federated averaging only syncs the small adapter tensors."""
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

model_id = "allenai/longformer-base-4096"
framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
category = "text_classification"
model_type = ""
batch_size = 2
sequence_length = 4096
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
        target_modules=[
            "query",
            "key",
            "value",
            "query_global",
            "key_global",
            "value_global",
        ],
        modules_to_save=["classifier"],
    )
    return get_peft_model(base, lora_config)
