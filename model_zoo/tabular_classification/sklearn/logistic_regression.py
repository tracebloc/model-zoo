import torch.nn as nn
from sklearn.linear_model import LogisticRegression

framework = "sklearn"
model_type = "linear"
main_method = "MyModel"
batch_size = 4
output_classes = 5
category = "tabular_classification"
num_feature_points = 50

def MyModel():
    return LogisticRegression(random_state=42)