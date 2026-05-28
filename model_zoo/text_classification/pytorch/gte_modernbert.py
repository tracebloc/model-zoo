"""GTE-ModernBERT (Alibaba-NLP, 2025). ModernBERT fine-tuned for general text embeddings — top MTEB scores at its size; doubles as a strong classifier via mean-pool + linear head. LoRA-only fine-tune for federated averaging.

Requirements:
- HuggingFace token NOT required (ungated), but recommended to avoid
  Hub rate-limits. If you set one, read it from env — never commit a
  token to this file.
- ``trust_remote_code=True`` is required — gte-modernbert ships custom
  modeling code. First load downloads the remote code; on slow networks
  this step alone can take a minute.
- Resources: ~600MB download, ~2GB RAM for load. ~8GB system RAM is
  enough for the local SDK self-check. At ``sequence_length=512,
  batch_size=4`` expect ~1 minute on CPU; if it looks stuck on a
  bigger seq/batch, halve both.
- Trainable params (LoRA r=8 + classifier head): ~1M. Base ~150M frozen.
"""
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

model_id = "Alibaba-NLP/gte-modernbert-base"
framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
category = "text_classification"
model_type = ""
batch_size = 4
sequence_length = 512
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
