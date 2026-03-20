import torch.nn as nn
from lightgbm import LGBMClassifier

framework = "sklearn"
model_type = "tree"
main_method = "MyModel"
batch_size = 512
output_classes = 2
category = "tabular_classification"
num_feature_points = 254

def MyModel():
    return LGBMClassifier(n_estimators=100, random_state=42)

