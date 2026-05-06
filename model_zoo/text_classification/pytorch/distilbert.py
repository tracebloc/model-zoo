"""DistilBERT via HuggingFace, pretrained. ~60% the size of BERT-base, ~97% of its accuracy; strong default for inference-speed-sensitive setups."""
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
# Pin a specific commit on HF Hub: the backend security check rejects
# from_pretrained() calls without revision pinning. Update this when you
# want to consume a newer upstream version — see
# https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/commits/main
model_revision = "714eb0fa89d2f80546fda750413ed43d93601a13"
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
