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
        self.linear = nn.Linear(12, 1)
        
    def forward(self, x):
        # Return shape (batch_size, 1) - don't squeeze to avoid scalar when batch_size=1
        return self.linear(x)