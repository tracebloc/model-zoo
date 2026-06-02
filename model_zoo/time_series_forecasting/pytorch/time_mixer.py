"""TimeMixer (ICLR 2024). Multi-scale mixing of seasonal + trend decompositions via MLPs; consistently strong on long-horizon multivariate without attention. Trains from scratch — federated-safe (no foundation backbone, LayerNorm only)."""
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


class _MLPBlock(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class MyModel(nn.Module):
    def __init__(self, num_feature_points=num_feature_points,
                 sequence_length=sequence_length, forecast_horizon=forecast_horizon,
                 n_blocks=3):
        super().__init__()
        self.time_mix = nn.ModuleList([_MLPBlock(sequence_length) for _ in range(n_blocks)])
        self.chan_mix = nn.ModuleList([_MLPBlock(num_feature_points) for _ in range(n_blocks)])
        self.head = nn.Linear(sequence_length, forecast_horizon)

    def forward(self, past_values, *args, **kwargs):
        # past_values: (B, L, N)
        x = past_values
        for tm, cm in zip(self.time_mix, self.chan_mix):
            x = tm(x.transpose(1, 2)).transpose(1, 2)  # time mixing
            x = cm(x)  # channel mixing
        out = self.head(x.transpose(1, 2))  # (B, N, H)
        return out.transpose(1, 2)
