"""Ultra-light 2-layer transformer for MLM, ~5M params. Fastest smoke-test model for pipeline validation."""
import torch
import torch.nn as nn

framework = "pytorch"
main_class = "TinyMLM"
category = "masked_language_modeling"
model_type = ""
batch_size = 64
sequence_length = 64
vocab_size = 30000


class TinyMLM(nn.Module):
    """2-layer, 128-dim transformer encoder for masked language modeling.

    ~5M parameters. Trains in seconds on CPU — use for verifying the
    end-to-end MLM pipeline before committing GPU time.
    """

    def __init__(
        self,
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
        intermediate_size=256,
        max_position_embeddings=512,
        dropout=0.1,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder.enable_nested_tensor = False
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
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)

        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.lm_head(x)
        return logits
