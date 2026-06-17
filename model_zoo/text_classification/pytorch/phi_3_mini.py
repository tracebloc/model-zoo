"""Phi-3-mini (Microsoft, 2024). 3.8B-param decoder LLM repurposed as a classifier — competitive with Llama-3-8B on reasoning benchmarks at half the size, and ungated on HuggingFace. Select LoRA-only fine-tuning in the training plan so federated averaging only syncs the adapter + classification head.

Requirements:
- HuggingFace token NOT strictly required (model is ungated), but a
  token avoids HF Hub rate-limits on cold pulls. If you set one, read
  it from env — DO NOT commit a token to this file; uploaded files
  leak any secret embedded in them.
- No ``trust_remote_code`` needed — Phi-3's config declares
  ``model_type="phi3"``, which transformers loads with the native
  ``Phi3ForSequenceClassification`` class. Avoiding remote code is
  required to pass the platform's model-security check.
- Resources: ~8GB download (fp32), ~16GB RAM for load. ~32GB system
  RAM strongly recommended for the local SDK self-check; the CPU
  forward+backward on synthetic data takes 5-15 minutes on a laptop
  even at ``batch_size=2, sequence_length=1024``. Bump
  ``sequence_length`` down to 256 if the self-check looks stuck.
- Params: ~3.8B (full base). Choose LoRA in the training plan to train
  only a ~15M adapter + score head.
"""
from transformers import AutoModelForSequenceClassification

model_id = "microsoft/Phi-3-mini-4k-instruct"
# HF tokenizer distributed to every client as the federation's single
# source of truth (#805). Mandatory for HuggingFace NLP models.
tokenizer_id = "microsoft/Phi-3-mini-4k-instruct"
framework = "pytorch"
main_method = "MyModel"
license = "MIT"
category = "text_classification"
model_type = ""
batch_size = 2
sequence_length = 1024
output_classes = 5



def MyModel(num_classes=output_classes):
    return AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
