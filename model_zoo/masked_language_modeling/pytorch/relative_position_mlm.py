"""BERT-base with ALiBi-style relative position bias, ~115M params. Better length generalization than absolute positions."""
import torch
import torch.nn as nn
import math

framework = "pytorch"
main_class = "RelativePositionMLM"
category = "masked_language_modeling"
model_type = ""
batch_size = 16
sequence_length = 128
vocab_size = 30522


class _ALiBiMultiheadAttention(nn.Module):
    """Multi-head attention with ALiBi (Attention with Linear Biases).

    Replaces learned position embeddings with a linear distance bias
    added to attention scores. Generalizes to longer sequences than
    seen during training without extra parameters.
    """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # ALiBi slopes — standard implementation (Press et al., 2021).
        # When num_heads is a power of two, slopes are a single geometric
        # sequence with ratio 2^(-8/n). When it is not (e.g. 12), use the
        # closest power-of-two sequence and interleave a second sequence
        # for the remaining heads.
        self.register_buffer("slopes", self._get_alibi_slopes(num_heads))

    @staticmethod
    def _get_alibi_slopes(num_heads):
        def _get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(num_heads).is_integer():
            slopes = _get_slopes_power_of_2(num_heads)
        else:
            closest_power = 2 ** math.floor(math.log2(num_heads))
            base = _get_slopes_power_of_2(closest_power)
            extra = _get_slopes_power_of_2(2 * closest_power)
            extra = extra[0::2][: num_heads - closest_power]
            slopes = base + extra
        return torch.tensor(slopes)

    def forward(self, x, key_padding_mask=None):
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # ALiBi bias: slope * |i - j|
        positions = torch.arange(S, device=x.device)
        distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
        alibi = -distance.unsqueeze(0) * self.slopes.unsqueeze(-1).unsqueeze(-1)
        attn = attn + alibi.unsqueeze(0)

        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
        return self.out_proj(out)


class _ALiBiEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.attn = _ALiBiMultiheadAttention(hidden_size, num_heads, dropout)
        self.ln1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ln2 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), key_padding_mask))
        x = x + self.ffn(self.ln2(x))
        return x


class RelativePositionMLM(nn.Module):
    """12-layer, 768-dim encoder with ALiBi position biases for MLM.

    ~115M parameters. Replaces absolute position embeddings with
    ALiBi (Attention with Linear Biases) for better generalization
    to sequence lengths not seen during training.
    """

    def __init__(
        self,
        vocab_size=vocab_size,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
        dropout=0.1,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            _ALiBiEncoderLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

        self.lm_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
        )
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

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.word_embeddings(input_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        for layer in self.layers:
            x = layer(x, key_padding_mask)

        x = self.lm_transform(x)
        logits = self.lm_decoder(x) + self.lm_bias
        return logits
