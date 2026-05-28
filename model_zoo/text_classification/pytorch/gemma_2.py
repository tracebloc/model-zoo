"""Gemma-2 2B (Google, 2024) as a decoder-LLM classifier. Decoder-LLM-as-classifier is the dominant 2024-2025 pattern for hard classification tasks. LoRA-only fine-tune so federated averaging only syncs the adapter + classification head.

Requirements:
- HuggingFace token required. ``google/gemma-2-2b`` is a gated repo —
  request access at https://huggingface.co/google/gemma-2-2b, then read
  the token from an env var. DO NOT commit a real token to this file;
  the file is uploaded to the platform and any token in it is leaked.
- Resources: ~5GB download, ~10GB RAM for fp32 load. ~32GB system RAM
  recommended for the local SDK self-check (CPU forward+backward on
  synthetic data, expect 2-5 minutes on a laptop).
- Trainable params (LoRA + score head): ~10M. Base ~2.5B stays frozen.
"""
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

model_id = "google/gemma-2-2b"
framework = "pytorch"
main_method = "MyModel"
license = "Gemma"
category = "text_classification"
model_type = ""
batch_size = 4
sequence_length = 256
output_classes = 5


def MyModel(num_classes=output_classes):
    base = AutoModelForSequenceClassification.from_pretrained(
        base=AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_classes,
                                                                ignore_mismatched_sizes=True)    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
    )
    return get_peft_model(base, lora_config)
