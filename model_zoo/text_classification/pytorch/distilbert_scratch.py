from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
hf_token = "<PROVIDE HF TOKEN>"
framework = "pytorch"
main_class = "MyModel"
category = "text_classification"
model_type = ""
batch_size = 16
sequence_length = 512
output_classes = 134


def MyModel(num_classes=output_classes):
    config = AutoConfig.from_pretrained(model_id, num_labels=num_classes)
    return AutoModelForSequenceClassification.from_config(config)
