"""BERT-base warm-started from HuggingFace pretrained weights, adapted for MLM on custom vocab. Fine-tune on domain corpora."""
from transformers import AutoConfig, AutoModelForMaskedLM

framework = "pytorch"
main_method = "NetMedGPTWarmStart"
category = "masked_language_modeling"
model_type = ""
batch_size = 16
sequence_length = 128
vocab_size = 30000

# HuggingFace model to warm-start from
pretrained_model_id = "bert-base-uncased"


def NetMedGPTWarmStart(vocab_size=vocab_size):
    """Load BERT-base-uncased and resize embeddings for a custom vocabulary.

    Warm-starting from general-domain pretrained weights accelerates
    convergence on biomedical corpora compared to training from scratch.
    The embedding layer and LM head are resized to match the custom
    tokenizer's vocabulary — new token embeddings are randomly initialized
    while existing ones retain their pretrained values.

    Returns an ``AutoModelForMaskedLM`` instance whose ``.forward()``
    accepts ``input_ids``, ``attention_mask``, and ``labels``, returning
    a ``MaskedLMOutput`` with ``.loss`` and ``.logits``.
    """
    config = AutoConfig.from_pretrained(pretrained_model_id)
    model = AutoModelForMaskedLM.from_pretrained(
        pretrained_model_id,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.resize_token_embeddings(vocab_size)
    return model
