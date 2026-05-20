"""Convolutional encoder for MLM, ~25M params. No attention — pure CNN baseline for fast training."""
import torch
import torch.nn as nn

framework = "pytorch"
main_class = "ConvMLM"
category = "masked_language_modeling"
model_type = ""
batch_size = 64
sequence_length = 128
vocab_size = 30000


class ConvMLM(nn.Module):
    """Stacked 1D convolutional encoder for masked language modeling.

    ~25M parameters. Uses dilated causal convolutions instead of
    self-attention — no quadratic memory cost in sequence length.
    Trains significantly faster than transformer models on CPU and
    serves as a non-attention baseline for ablation studies.
    """

    def __init__(
        self,
        vocab_size=vocab_size,
        hidden_size=512,
        num_layers=6,
        kernel_size=5,
        max_position_embeddings=512,
        dropout=0.1,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        layers = []
        for i in range(num_layers):
            dilation = 2 ** (i % 3)
            padding = (kernel_size - 1) * dilation // 2
            layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size * 2, kernel_size,
                              padding=padding, dilation=dilation),
                    nn.GLU(dim=1),
                    nn.Dropout(dropout),
                )
            )
        self.conv_layers = nn.ModuleList(layers)
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
            elif isinstance(module, nn.Conv1d):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.clamp(max=self.max_position_embeddings - 1)

        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)

        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        for conv_block in self.conv_layers:
            residual = x
            x = conv_block(x)
            # Trim to match residual length if padding produced extra
            x = x[:, :, :residual.size(2)]
            x = x + residual
        x = x.transpose(1, 2)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        logits = self.lm_head(x)
        return logits
