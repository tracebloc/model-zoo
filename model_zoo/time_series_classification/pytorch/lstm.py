"""LSTM classifier, 2-layer, hidden=128. Good general-purpose sequence classifier; start here before transformers."""
import torch.nn as nn

framework = "pytorch"
model_type = ""
main_class = "LSTMClassifier"
license = "Apache-2.0"
category = "time_series_classification"
batch_size = 512
output_classes = 2
num_feature_points = 9
sequence_length = 60


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=num_feature_points, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size), zero post-padded
        # Padding mask: a timestep is valid if any feature is non-zero
        mask = (x.abs().sum(-1) > 0)
        lstm_out, _ = self.lstm(x)
        # Take the last VALID timestep per sequence — lstm_out[:, -1, :] would
        # read hidden state computed on zero padding
        lengths = mask.sum(dim=1)
        last_valid = (lengths - 1).clamp(min=0)  # all-padding edge case -> index 0
        idx = last_valid.view(-1, 1, 1).expand(-1, 1, lstm_out.size(-1))
        last_output = lstm_out.gather(1, idx).squeeze(1)
        x = self.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
