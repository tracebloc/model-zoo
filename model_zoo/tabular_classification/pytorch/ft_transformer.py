"""FT-Transformer (Yandex, NeurIPS 2021). Per-feature tokenization + transformer; standard tabular-DL reference baseline. LayerNorm-based (BN-free) → federated-averaging-friendly out of the box."""
import torch
import torch.nn as nn

framework = "pytorch"
model_type = ""
main_class = "MyModel"
license = "MIT"
batch_size = 512
output_classes = 5
num_feature_points = 50
category = "tabular_classification"


class _FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

    def forward(self, x):
        # x: (B, F) numeric features
        tok = x.unsqueeze(-1) * self.weight + self.bias  # (B, F, D)
        cls = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([cls, tok], dim=1)


class MyModel(nn.Module):
    def __init__(self, num_classes=output_classes, num_feature_points=num_feature_points,
                 d_token=64, n_layers=3, n_heads=4):
        super().__init__()
        self.tokenizer = _FeatureTokenizer(num_feature_points, d_token)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
            dropout=0.1, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, num_classes)

    def forward(self, x):
        x = torch.nan_to_num(x)
        tokens = self.tokenizer(x)
        h = self.encoder(tokens)
        return self.head(self.norm(h[:, 0]))
