from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
framework = "pytorch"
main_class = "MyModel"
category = "text_classification"
model_type = ""
batch_size = 512
sequence_length = 5
output_classes = 5


def MyModel(num_classes=output_classes):
    config = AutoConfig.from_pretrained(model_id, num_labels=num_classes)
    return AutoModelForSequenceClassification.from_config(config)
