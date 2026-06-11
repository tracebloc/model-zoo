"""Minimal transformer encoder with a per-token tag head, ~7M params. Smoke-test model for quick iteration and token-classification pipeline validation."""
import torch
import torch.nn as nn

framework = "pytorch"
main_class = "SimpleTokenTagger"
category = "token_classification"
model_type = ""
batch_size = 32
sequence_length = 64
# BIO/IOB2 tag count. Default 9 matches the CoNLL-2003 scheme:
# O + B/I x {PER, ORG, LOC, MISC}. Set to your dataset's tag count.
output_classes = 9
license = "Apache-2.0"

# Must cover the training tokenizer's vocabulary. The client's default
# tokenizer (bert-base-uncased) has vocab_size=30522 — token IDs beyond
# the embedding table cause index-out-of-bounds errors at training time.
_VOCAB_SIZE = 30522


class SimpleTokenTagger(nn.Module):
    """Lightweight 2-layer transformer encoder for token classification.

    Emits per-token logits of shape ``(batch, seq_len, output_classes)`` —
    one BIO tag distribution per sub-word token. Intended for smoke testing
    the token-classification training pipeline — not for production-quality
    predictions.
    """

    def __init__(
        self,
        vocab_size=_VOCAB_SIZE,
        num_classes=output_classes,
        hidden_size=192,
        num_layers=2,
        num_heads=4,
        intermediate_size=384,
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder.enable_nested_tensor = False
        self.classifier = nn.Linear(hidden_size, num_classes)

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
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.clamp(max=self.max_position_embeddings - 1)

        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)

        if attention_mask is not None:
            # TransformerEncoder expects mask where True = ignore
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        # Per-token logits: (batch, seq_len, num_classes)
        return self.classifier(x)
