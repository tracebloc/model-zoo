"""EuroBERT-210M (2025). Multilingual encoder covering 15 European languages, 8192 context; closes the multilingual gap left by the BERT/RoBERTa lineup. Loads via trust_remote_code — the repo id is hard-coded as a string literal so the model-upload security gate (TBT001) recognises it against its vetted-repos allowlist."""
from transformers import AutoModelForSequenceClassification

framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
category = "text_classification"
model_type = ""
batch_size = 32
sequence_length = 512
output_classes = 5


def MyModel(num_classes=output_classes):
    return AutoModelForSequenceClassification.from_pretrained(
        "EuroBERT/EuroBERT-210m",
        num_labels=num_classes,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
