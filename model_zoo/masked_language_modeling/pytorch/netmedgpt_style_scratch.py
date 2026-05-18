"""BERT-style encoder for biomedical MLM, ~110M params. Train from scratch on domain corpora (e.g. PrimeKG random walks)."""
import torch
import torch.nn as nn

framework = "pytorch"
main_class = "NetMedGPTScratch"
category = "masked_language_modeling"
model_type = ""
batch_size = 16
sequence_length = 128
vocab_size = 30000


class NetMedGPTScratch(nn.Module):
    """BERT-base-scale transformer encoder for masked language modeling.

    ~110M parameters with default settings (12 layers, 768 hidden, 12 heads).
    Designed for from-scratch pretraining on biomedical graph corpora where
    off-the-shelf language models lack domain coverage.

    Architecture mirrors BERT-base so weights can later be compared or
    swapped with ``netmedgpt_style_warmstart``.
    """

    def __init__(
        self,
        vocab_size=vocab_size,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        dropout=0.1,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
        )
        self.lm_decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        # Weight tying: LM decoder shares weights with word embeddings (BERT-style)
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
                # Q/K/V projections stored as raw Parameters, not nn.Linear
                if module.in_proj_weight is not None:
                    module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
                if module.out_proj.weight is not None:
                    module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
                if module.out_proj.bias is not None:
                    module.out_proj.bias.data.zero_()

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
        x = self.lm_transform(x)
        logits = self.lm_decoder(x) + self.lm_bias
        return logits
