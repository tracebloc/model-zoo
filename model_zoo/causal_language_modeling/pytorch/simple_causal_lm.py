"""Minimal GPT-style decoder for causal LM, ~19M params. Smoke-test model for quick iteration and pipeline validation."""
import torch
import torch.nn as nn

framework = "pytorch"
main_class = "SimpleCausalLM"
category = "causal_language_modeling"
model_type = ""
batch_size = 32
sequence_length = 128
vocab_size = 30522
license = "Apache-2.0"


class SimpleCausalLM(nn.Module):
    """Lightweight 4-layer decoder-only transformer for causal language modeling.

    ~19M parameters with default settings. A self-attention stack with a
    causal (upper-triangular) attention mask so each position only attends to
    itself and earlier positions — i.e. next-token prediction. Intended for
    smoke testing the causal LM training pipeline, not production quality.

    The model emits raw logits shaped ``(batch, seq_len, vocab_size)``. It does
    NOT shift the labels: the training container applies the single next-token
    shift (logits at position t are scored against the token at position t+1).
    """

    def __init__(
        self,
        vocab_size=vocab_size,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512,
        dropout=0.1,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.max_position_embeddings = max_position_embeddings
        self.embed_dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers,
            # The nested-tensor fast path can bypass the attention mask, which
            # would break causality — keep it disabled.
            enable_nested_tensor=False,
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

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
        input_ids = input_ids.long()
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.clamp(max=self.max_position_embeddings - 1)

        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.embed_dropout(x)

        # Causal mask: position i may not attend to any position j > i.
        # Bool mask (True = ignore) to match the bool padding mask below.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )

        if attention_mask is not None:
            # TransformerEncoder expects True = ignore for the padding mask.
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        x = self.decoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)
        return logits
