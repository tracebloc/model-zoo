"""Mid-size 6-layer transformer for MLM, ~50M params. Balanced speed/quality for moderate datasets."""
import torch
import torch.nn as nn

framework = "pytorch"
main_class = "MediumMLM"
category = "masked_language_modeling"
model_type = ""
batch_size = 32
sequence_length = 128
vocab_size = 30522


class MediumMLM(nn.Module):
    """6-layer, 512-dim transformer encoder for masked language modeling.

    ~50M parameters. Offers a middle ground between the lightweight
    SimpleMaskedLM and the full BERT-base NetMedGPTScratch — suitable
    for datasets with 100K–1M sequences.
    """

    def __init__(
        self,
        vocab_size=vocab_size,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        intermediate_size=2048,
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
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder.enable_nested_tensor = False
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
        logits = self.lm_decoder(x) + self.lm_bias
        return logits
