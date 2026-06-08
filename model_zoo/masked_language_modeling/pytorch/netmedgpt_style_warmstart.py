"""BERT-base warm-started from HuggingFace pretrained weights, adapted for MLM. Fine-tune on domain corpora."""
import torch.nn as nn
from transformers import AutoConfig, AutoModelForMaskedLM

framework = "pytorch"
main_class = "NetMedGPTWarmStart"
category = "masked_language_modeling"
model_type = ""
batch_size = 16
sequence_length = 128
# Must match the bert-base-uncased tokenizer's vocab (30522). The previous
# value (30000) shrank the embedding table below the tokenizer, so token ids
# 30000-30521 caused CUDA index-out-of-bounds at training time.
vocab_size = 30522

# HuggingFace model to warm-start from
pretrained_model_id = "bert-base-uncased"


class NetMedGPTWarmStart(nn.Module):
    """BERT-base-uncased warm-started from pretrained weights for MLM.

    Warm-starting from general-domain pretrained weights accelerates
    convergence on biomedical corpora compared to training from scratch.
    The embedding layer and LM head are resized to the configured vocab —
    new token embeddings are randomly initialized while existing ones
    retain their pretrained values.

    ``forward`` accepts ``input_ids``, ``attention_mask`` and ``labels``
    and returns a ``MaskedLMOutput`` with ``.loss`` and ``.logits``.

    Authored as an ``nn.Module`` subclass (``main_class``) rather than a
    factory function (``main_method``): the platform's model loader resolves
    the class entrypoint reliably, whereas the factory-function form failed
    to load server-side.
    """

    def __init__(self, vocab_size=vocab_size):
        super(NetMedGPTWarmStart, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_id,
            config=config,
            ignore_mismatched_sizes=True,
        )
        self.model.resize_token_embeddings(vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
