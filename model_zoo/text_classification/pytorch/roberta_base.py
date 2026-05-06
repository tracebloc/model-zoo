"""RoBERTa-base via HuggingFace, pretrained. Often outperforms BERT-base on downstream tasks; same compute cost."""
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig

model_id = "roberta-base"
# Pin a specific commit on HF Hub: the backend security check rejects
# from_pretrained() calls without revision pinning. See
# https://huggingface.co/roberta-base/commits/main
model_revision = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
tokenizer_id = "roberta-base"
hf_token = ""
framework = "pytorch"
main_class = "MyModel"
category = "text_classification"
model_type = ""
batch_size = 512
sequence_length = 128
output_classes = 5


def MyModel(num_classes=output_classes):
    config = AutoConfig.from_pretrained(
        model_id,
        revision=model_revision,
        token=hf_token,
        num_labels=num_classes,
        problem_type="single_label_classification",
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        classifier_dropout=0.3,
    )
    return AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=model_revision,
        token=hf_token,
        config=config,
        ignore_mismatched_sizes=True,
    )