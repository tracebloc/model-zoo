"""BERT-base-uncased via HuggingFace, pretrained. Standard text classification baseline; fine-tune the head for your labels."""
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig

model_id = 'bert-base-uncased'
# Pin a specific commit on HF Hub: the backend security check rejects
# from_pretrained() calls without revision pinning. See
# https://huggingface.co/bert-base-uncased/commits/main
model_revision = "86b5e0934494bd15c9632b12f734a8a67f723594"
framework = "pytorch"
main_class = "MyModel"
category = "text_classification"
model_type = ""
batch_size = 512
sequence_length = 5
output_classes = 5

def MyModel(num_classes=output_classes):

    return AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=model_revision,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
