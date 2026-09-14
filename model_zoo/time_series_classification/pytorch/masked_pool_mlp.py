"""Per-timestep MLP with masked mean+max pooling. Simplest sequence classifier; strong baseline when temporal order matters little."""
import torch
import torch.nn as nn

framework = "pytorch"
model_type = ""
main_class = "MaskedPoolMLP"
license = "Apache-2.0"
category = "time_series_classification"
batch_size = 512
output_classes = 2
num_feature_points = 9
sequence_length = 60


class MaskedPoolMLP(nn.Module):
    def __init__(self, input_size=num_feature_points, hidden_size=128, dropout=0.2):
        super(MaskedPoolMLP, self).__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # mean-pool + max-pool concatenated
        self.fc2 = nn.Linear(64, output_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size), zero post-padded
        # Padding mask: a timestep is valid if any feature is non-zero
        mask = (x.abs().sum(-1) > 0)
        # Guard the all-padding edge case: max over zero valid timesteps is
        # -inf, so treat the first timestep of such rows as valid
        all_pad = ~mask.any(dim=1)
        if all_pad.any():
            mask = mask.clone()
            mask[all_pad, 0] = True
        h = self.feature_net(x)  # (batch_size, sequence_length, hidden_size)
        # Masked mean pooling over valid timesteps
        mask_f = mask.unsqueeze(-1).to(h.dtype)
        mean_pool = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        # Masked max pooling — padded timesteps excluded via -inf fill
        max_pool = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        x = torch.cat([mean_pool, max_pool], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
