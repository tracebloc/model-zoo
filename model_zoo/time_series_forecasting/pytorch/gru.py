import torch.nn as nn

framework = "pytorch"
model_type = ""
main_class = "GRUForecaster"
category = "time_series_forecasting"
batch_size = 512
num_feature_points = 9
sequence_length = 60
forecast_horizon = 1


class GRUForecaster(nn.Module):
    def __init__(self, input_size=num_feature_points, hidden_size=128, num_layers=2, dropout=0.2):
        super(GRUForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, forecast_horizon)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        gru_out, _ = self.gru(x)
        # Take the last output from the sequence
        last_output = gru_out[:, -1, :]
        x = self.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
