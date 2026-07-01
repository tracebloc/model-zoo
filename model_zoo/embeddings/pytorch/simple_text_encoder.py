"""Minimal transformer text encoder for contrastive embeddings, ~10M params. From-scratch model for smoke-training the embeddings pipeline."""
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

framework = "pytorch"
main_class = "SimpleTextEncoder"
category = "embeddings"
model_type = ""
batch_size = 16
sequence_length = 128
vocab_size = 30522
license = "Apache-2.0"


class SimpleTextEncoder(nn.Module):
    """Lightweight 4-layer bidirectional transformer encoder for embeddings.

    ~10M parameters with default settings. A plain text encoder: it maps a
    batch of token IDs to contextual token vectors and returns them as
    ``last_hidden_state`` shaped ``(batch, seq_len, hidden_size)``. There is no
    classification / language-model head — the self-supervised contrastive
    training container mean-pools ``last_hidden_state`` (attention-mask aware)
    into one sentence embedding per example, so the model deliberately stops at
    the encoder output. Intended as a light from-scratch baseline for pipeline
    validation, not production-quality embeddings.

    The forward pass returns a HuggingFace ``BaseModelOutput`` (exposing
    ``.last_hidden_state``) so the engine's ``model(input_ids=..., attention_mask=...)``
    call reads the pooled-over tensor the same way it would from an ``AutoModel``.
    It does NOT shift labels or build any targets — embeddings training is
    label-free — and bundles no PEFT/LoRA (LoRA is chosen in the training plan,
    not the model file; the package check rejects in-file PEFT).
    """

    def __init__(
        self,
        vocab_size=vocab_size,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
        dropout=0.1,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            # The nested-tensor fast path can silently drop the padding mask;
            # keep it disabled so padded tokens stay masked.
            enable_nested_tensor=False,
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Tokenizers emit integer token IDs. The upload smoke-test may cast
        # inputs to the model's float dtype; nn.Embedding requires Long indices.
        input_ids = input_ids.long()

        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.clamp(max=self.max_position_embeddings - 1)

        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)

        if attention_mask is not None:
            # TransformerEncoder expects True = ignore for the padding mask.
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        last_hidden_state = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return BaseModelOutput(last_hidden_state=last_hidden_state)
