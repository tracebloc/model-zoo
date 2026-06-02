"""Temporal Fusion Transformer (Google, 2019) — minimal native-PyTorch reimplementation.

The original `pytorch_forecasting` TFT requires a structured batch dict
(`encoder_cont`, `decoder_cont`, static features, ...) that does not match the
plain `past_values` tensor convention used by the rest of this zoo. This file
re-implements the core TFT ideas — variable-selection-style gating, LSTM
encoder, interpretable multi-head attention, and a feed-forward forecast head
— in a tensor-in / tensor-out form that matches PatchTST / iTransformer /
TimeMixer and trains under the federated averaging contract (LayerNorm only).
"""
import torch.nn as nn

framework = "pytorch"
model_type = ""
main_class = "MyModel"
license = "MIT"
category = "time_series_forecasting"
batch_size = 64
num_feature_points = 9
sequence_length = 96
forecast_horizon = 24


class _GatedResidualNetwork(nn.Module):
    """Lightweight GRN: dense → ELU → dense → GLU + skip, with LayerNorm."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ELU()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.drop(self.fc2(h))
        a, b = self.gate(h).chunk(2, dim=-1)
        return self.norm(x + a * b.sigmoid())


class MyModel(nn.Module):
    def __init__(self, num_feature_points=num_feature_points,
                 sequence_length=sequence_length, forecast_horizon=forecast_horizon,
                 hidden=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(num_feature_points, hidden)
        self.var_select = _GatedResidualNetwork(hidden, dropout)
        self.encoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.post_lstm = _GatedResidualNetwork(hidden, dropout)
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)
        self.ff = _GatedResidualNetwork(hidden, dropout)
        self.head = nn.Linear(hidden * sequence_length, forecast_horizon * num_feature_points)
        self.num_feature_points = num_feature_points
        self.forecast_horizon = forecast_horizon

    def forward(self, past_values, *args, **kwargs):
        # past_values: (B, L, N)
        h = self.embed(past_values)
        h = self.var_select(h)
        h, _ = self.encoder(h)
        h = self.post_lstm(h)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = self.attn_norm(h + attn_out)
        h = self.ff(h)
        out = self.head(h.flatten(1))
        return out.view(-1, self.forecast_horizon, self.num_feature_points)
