"""Temporal Convolutional Network classifier. Dilated causal convs + masked global average pooling; strong on long sequences."""
import torch.nn as nn

framework = "pytorch"
model_type = ""
main_class = "TCNClassifier"
license = "Apache-2.0"
category = "time_series_classification"
batch_size = 512
output_classes = 2
num_feature_points = 9
sequence_length = 60


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = nn.ConstantPad1d((0, -padding), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNClassifier(nn.Module):
    def __init__(self, num_inputs=num_feature_points, num_channels=[64, 128, 64], kernel_size=3, dropout=0.2):
        super(TCNClassifier, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_classes)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, num_features), zero post-padded
        # Padding mask: a timestep is valid if any feature is non-zero
        mask = (x.abs().sum(-1) > 0)
        # Transpose to (batch_size, num_features, sequence_length) for Conv1d
        y = self.network(x.transpose(1, 2))
        # Masked global average pooling — causal convs preserve length, so
        # padded positions align with the mask and are excluded from the mean
        mask_f = mask.unsqueeze(1).to(y.dtype)
        pooled = (y * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp(min=1.0)
        return self.fc(pooled)
