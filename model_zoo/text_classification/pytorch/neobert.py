"""NeoBERT (Chandar Lab, Feb 2025). Next-gen encoder competing with ModernBERT — RoPE, GLU, 4096 context, MTEB-strong. Fine-tuned LoRA-only so federated averaging only syncs the adapter."""
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

model_id = "chandar-lab/NeoBERT"
framework = "pytorch"
main_method = "MyModel"
license = "MIT"
category = "text_classification"
model_type = ""
batch_size = 16
sequence_length = 512
output_classes = 5


def MyModel(num_classes=output_classes):
    base = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_classes, trust_remote_code=True, ignore_mismatched_sizes=True
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["classifier", "score"],
    )
    return get_peft_model(base, lora_config)
