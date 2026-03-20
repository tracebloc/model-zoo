import torch
import torch.nn as nn

framework = "pytorch"
model_type = ""
main_method = "MyModel"
batch_size = 128
output_classes = 1
category = "time_to_event_prediction"
num_feature_points = 12


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(12, 64),  # FIXED: was 17, must match num_feature_points
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)  # Output: hazard/log-hazard
        )
        
    def forward(self, x):
        # Return shape (batch_size, 1) - don't squeeze to avoid scalar when batch_size=1
        return self.network(x)
