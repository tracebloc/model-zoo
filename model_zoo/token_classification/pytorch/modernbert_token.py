"""ModernBERT-base with a per-token classification head. 2024 long-context encoder (8k tokens, RoPE + GeGLU); state-of-the-art accuracy/throughput for token tagging."""
from transformers import AutoModelForTokenClassification

model_id = "answerdotai/ModernBERT-base"
# HF tokenizer distributed to every client as the federation's single
# source of truth (#805). Mandatory for HuggingFace NLP models.
tokenizer_id = "answerdotai/ModernBERT-base"
framework = "pytorch"
main_class = "MyModel"
category = "token_classification"
model_type = ""
batch_size = 16
sequence_length = 128
# BIO/IOB2 tag count. Default 9 matches the CoNLL-2003 scheme:
# O + B/I x {PER, ORG, LOC, MISC}. Set to your dataset's tag count.
output_classes = 9
license = "Apache-2.0"


def MyModel(num_classes=output_classes):
    return AutoModelForTokenClassification.from_pretrained(
        model_id, num_labels=num_classes, ignore_mismatched_sizes=True
    )
