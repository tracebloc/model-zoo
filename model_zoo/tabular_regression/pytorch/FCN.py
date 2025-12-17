import torch.nn as nn

framework = "pytorch"
model_type = ""
main_class = "SimpleFCN"
batch_size = 4
output_classes = 3
category = "tabular_regression"
num_feature_points = 12

class SimpleFCN(nn.Module):
    def __init__(self):
        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(num_feature_points, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
