"""Deep narrow 16-layer transformer for MLM, ~45M params. More depth with less width for capturing hierarchical patterns."""
import torch
import torch.nn as nn

framework = "pytorch"
main_class = "DeepNarrowMLM"
category = "masked_language_modeling"
model_type = ""
batch_size = 32
sequence_length = 128
vocab_size = 30522


class DeepNarrowMLM(nn.Module):
    """16-layer, 384-dim, 6-head transformer encoder for masked language modeling.

    ~45M parameters. Trades width for depth compared to BERT-base —
    narrower hidden dimension (384 vs 768) but more layers (16 vs 12).
    Captures deeper hierarchical patterns in sequential data while
    keeping parameter count moderate.
    """

    def __init__(
        self,
        vocab_size=vocab_size,
        hidden_size=384,
        num_layers=16,
        num_heads=6,
        intermediate_size=1536,
        max_position_embeddings=512,
        dropout=0.1,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder.enable_nested_tensor = False

        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.lm_decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_decoder.weight = self.word_embeddings.weight
        self.lm_bias = nn.Parameter(torch.zeros(vocab_size))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
                if module.out_proj.weight is not None:
                    module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
                if module.out_proj.bias is not None:
                    module.out_proj.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.clamp(max=self.max_position_embeddings - 1)

        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)

        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.final_layer_norm(x)
        logits = self.lm_decoder(x) + self.lm_bias
        return logits
