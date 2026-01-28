import torch
import torch.nn as nn
import math

framework = "pytorch"
model_type = ""
main_class = "TransformerForecaster"
category = "time_series_forecasting"
batch_size = 32
num_feature_points = 10
sequence_length = 60
forecast_horizon = 1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerForecaster(nn.Module):
    def __init__(self, input_size=num_feature_points, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.2):
        super(TransformerForecaster, self).__init__()
        self.d_model = d_model
        
        # Project input to model dimension
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, forecast_horizon)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        # Project to model dimension
        x = self.input_projection(x)
        # Transpose for transformer: (sequence_length, batch_size, d_model)
        x = x.transpose(0, 1)
        # Add positional encoding
        x = self.pos_encoder(x)
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        # Take the last time step
        x = x[-1, :, :]
        # Apply fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
