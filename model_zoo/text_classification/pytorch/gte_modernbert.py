"""GTE-ModernBERT (Alibaba-NLP, 2025). ModernBERT fine-tuned for general text embeddings — top MTEB scores at its size; doubles as a strong classifier via mean-pool + linear head. LoRA-only fine-tune for federated averaging."""
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

model_id = "Alibaba-NLP/gte-modernbert-base"
framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
category = "text_classification"
model_type = ""
batch_size = 16
sequence_length = 1024
output_classes = 5


def MyModel(num_classes=output_classes):
    base = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_classes, trust_remote_code=True, ignore_mismatched_sizes=True
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
        target_modules=["Wqkv"],
        modules_to_save=["classifier", "head"],
    )
    return get_peft_model(base, lora_config)
