import torch
import torch.nn as nn

framework = "pytorch"
model_type = ""
main_method = "MyModel"
batch_size = 128
output_classes = 2
category = "time_to_event_prediction"
num_feature_points = 12


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)  # Output: predicted time
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        # Return shape (batch_size, 1) - don't squeeze to avoid scalar when batch_size=1
        return x