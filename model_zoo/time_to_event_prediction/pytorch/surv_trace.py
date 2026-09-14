"""SurvTRACE (2022, mainstream by 2025). Transformer-based survival with competing-risks support; standard deep-survival baseline alongside DeepSurv/DeepHit. LayerNorm only → federated-safe."""
import torch
import torch.nn as nn

framework = "pytorch"
model_type = ""
main_class = "MyModel"
license = "MIT"
batch_size = 128
output_classes = 1
category = "time_to_event_prediction"
num_feature_points = 12

_NUM_DURATIONS = 10


class MyModel(nn.Module):
    def __init__(self, num_feature_points=num_feature_points,
                 num_durations=_NUM_DURATIONS, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        # Per-feature token embedding (numerical features only)
        self.feat_embed = nn.Linear(1, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, num_feature_points + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_durations)

    def forward(self, x):
        # x: (B, F)
        tok = self.feat_embed(x.unsqueeze(-1))  # (B, F, D)
        cls = self.cls.expand(x.size(0), -1, -1)
        h = torch.cat([cls, tok], dim=1) + self.pos
        h = self.encoder(h)
        return self.head(self.norm(h[:, 0]))  # discrete-time hazards logits
