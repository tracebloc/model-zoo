"""BERT-base warm-started from pretrained weights, adapted for MLM. Fine-tune on domain corpora.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained encoder tensors are
delivered from the tracebloc model store as the training seed: upload the
matched ``netmedgpt_style_warmstart_weights.pkl`` sitting next to this file
via ``weights=True``, with the matching tokenizer as an explicitly named
sibling (it carries the ``[MASK]`` and ``[PAD]`` special tokens the MLM
ingestor validates)::

    user.upload_model("netmedgpt_style_warmstart", weights=True,
                      tokenizer="netmedgpt_style_warmstart_tokenizer.json")

(The bare ``tokenizer.json`` in this directory serves the scratch MLM
templates; this template names its sibling explicitly so the pairing is
unambiguous.) See ``tools/prep_offline_weights.py`` for producing and
verifying the matched weight file.
"""
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

# Architecture config for bert-base-uncased, inlined so the model builds
# with no config fetch. The SDK uploads the .py plus its named weight and
# tokenizer siblings — there is no config.json path — so the config lives
# here in the template. ``vocab_size`` is passed separately (module
# constant above) so the embedding table always matches the tokenizer.
CONFIG = {
    "model_type": "bert",
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
}


class NetMedGPTWarmStart(nn.Module):
    """BERT-base-uncased warm-started from pretrained weights for MLM.

    Warm-starting from general-domain pretrained weights (seeded from the
    tracebloc model store) accelerates convergence on biomedical corpora
    compared to training from scratch. The architecture is built offline
    from the inlined config, sized directly to the configured vocab — the
    LM-head decoder is tied to the input embeddings, so both track it.

    ``forward`` accepts ``input_ids``, ``attention_mask`` and ``labels``
    and returns a ``MaskedLMOutput`` with ``.loss`` and ``.logits``.

    Authored as an ``nn.Module`` subclass (``main_class``) rather than a
    factory function (``main_method``): the platform's model loader resolves
    the class entrypoint reliably, whereas the factory-function form failed
    to load server-side.
    """

    def __init__(self, vocab_size=vocab_size):
        super(NetMedGPTWarmStart, self).__init__()
        config = AutoConfig.for_model(**CONFIG, vocab_size=vocab_size)
        self.model = AutoModelForMaskedLM.from_config(config)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
