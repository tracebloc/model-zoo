import torch.nn as nn

framework = "pytorch"
model_type = ""
main_class = "TabularMLP"
batch_size = 512
output_classes = 1
category = "tabular_regression"
num_feature_points = 17


class TabularMLP(nn.Module):
    def __init__(self, input_size=num_feature_points):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)