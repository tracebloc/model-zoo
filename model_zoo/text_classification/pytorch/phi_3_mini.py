"""Phi-3-mini (Microsoft, 2024). 3.8B-param decoder LLM repurposed as a classifier — competitive with Llama-3-8B on reasoning benchmarks at half the size, and ungated on HuggingFace. LoRA-only fine-tune so federated averaging only syncs the adapter + classification head.

Requirements:
- HuggingFace token NOT strictly required (model is ungated), but a
  token avoids HF Hub rate-limits on cold pulls. If you set one, read
  it from env — DO NOT commit a token to this file; uploaded files
  leak any secret embedded in them.
- ``trust_remote_code=True`` is required — Phi-3 ships custom modeling
  code that HF downloads on first load.
- Resources: ~8GB download (fp32), ~16GB RAM for load. ~32GB system
  RAM strongly recommended for the local SDK self-check; the CPU
  forward+backward on synthetic data takes 5-15 minutes on a laptop
  even at ``batch_size=2, sequence_length=1024``. Bump
  ``sequence_length`` down to 256 if the self-check looks stuck.
- Trainable params (LoRA r=16 + score head): ~15M. Base ~3.8B frozen.
"""
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

model_id = "microsoft/Phi-3-mini-4k-instruct"
framework = "pytorch"
main_method = "MyModel"
license = "MIT"
category = "text_classification"
model_type = ""
batch_size = 2
sequence_length = 1024
output_classes = 5



def MyModel(num_classes=output_classes):
    base = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_classes,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["qkv_proj", "o_proj"],
        modules_to_save=["score"],
    )
    return get_peft_model(base, lora_config)
