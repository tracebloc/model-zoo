"""DeepHit (AAAI 2018). Deep survival model with discrete-time output — handles competing risks natively and makes no proportional-hazards assumption. Reference architecture for non-PH deep survival."""
import torch.nn as nn
from pycox.models import DeepHitSingle

framework = "pytorch"
model_type = ""
main_method = "MyModel"
license = "BSD-2-Clause"
batch_size = 256
output_classes = 1
category = "time_to_event_prediction"
num_feature_points = 12

_NUM_DURATIONS = 10


def MyModel(num_feature_points=num_feature_points, num_durations=_NUM_DURATIONS):
    net = nn.Sequential(
        nn.Linear(num_feature_points, 64),
        nn.ReLU(),
        nn.LayerNorm(64),
        nn.Dropout(0.1),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.LayerNorm(64),
        nn.Dropout(0.1),
        nn.Linear(64, num_durations),
    )
    return DeepHitSingle(net, optimizer=None, duration_index=None)
