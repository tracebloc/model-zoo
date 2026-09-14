"""Simple RNN for tabular classification. Lighter than LSTM; use for short sequences."""
import torch
import torch.nn as nn

framework = "pytorch"
model_type = ""
main_class = "SimpleRNN"
batch_size = 4096
output_classes = 5
num_feature_points = 50
category = "tabular_classification"

class SimpleRNN(nn.Module):
    def __init__(self, input_size=num_feature_points, hidden_size=128, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.nan_to_num(x)
        x = x.unsqueeze(1)  # (B, F) -> (B, 1, F): treat each row as a length-1 sequence
        out, _ = self.rnn(x)  # (B, 1, hidden)
        x = out[:, -1, :]  # (B, hidden): last timestep -> 2D for the (B, output_classes) head
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x