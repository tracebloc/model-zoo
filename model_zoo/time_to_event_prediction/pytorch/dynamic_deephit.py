"""Dynamic-DeepHit (TBME 2020, now standard in clinical ML). DeepHit extended with an RNN over longitudinal covariates — handles repeated measures and competing risks. Use when subjects have time-varying features rather than a single baseline vector."""
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
sequence_length = 10

_NUM_DURATIONS = 10


class MyModel(nn.Module):
    def __init__(self, num_feature_points=num_feature_points,
                 num_durations=_NUM_DURATIONS, hidden=64):
        super().__init__()
        self.rnn = nn.GRU(num_feature_points, hidden, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.cause_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_durations),
        )

    def forward(self, x):
        # x: (B, T, F) longitudinal covariates; if a 2D tensor is passed, treat as T=1.
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h, _ = self.rnn(x)
        h_last = self.norm(h[:, -1])
        return self.cause_head(h_last)  # discrete-time logits for single risk
