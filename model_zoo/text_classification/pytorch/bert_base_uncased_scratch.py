"""BERT-base-uncased trained from scratch. Pick only if you have a massive domain-specific corpus and want to avoid English-pretrained biases."""
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig

model_id = "bert-base-uncased"
framework = "pytorch"
main_class = "MyModel"
category = "text_classification"
model_type = ""
batch_size = 512
sequence_length = 5
output_classes = 5

# model version
# https://huggingface.co/bert-base-uncased/commits/main
model_revision = "86b5e0934494bd15c9632b12f734a8a67f723594"


def MyModel(num_classes=output_classes):
    config = AutoConfig.from_pretrained(
        model_id, revision=model_revision, num_labels=num_classes
    )
    return AutoModelForSequenceClassification.from_config(config)
