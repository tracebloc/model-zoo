"""iTransformer (ICLR 2024). Inverts the transformer: treats each variate as a token and attention runs across variates, not time steps. Consistently top-3 on long-horizon multivariate benchmarks; now the standard reference baseline."""
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


class MyModel(nn.Module):
    def __init__(self, num_feature_points=num_feature_points,
                 sequence_length=sequence_length, forecast_horizon=forecast_horizon,
                 d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(sequence_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, forecast_horizon)

    def forward(self, past_values, *args, **kwargs):
        # past_values: (B, L, N) — invert to (B, N, L)
        x = past_values.transpose(1, 2)
        x = self.embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        out = self.head(x)  # (B, N, H)
        return out.transpose(1, 2)  # (B, H, N)
