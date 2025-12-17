import torch.nn as nn
from xgboost import XGBClassifier

framework = "sklearn"
model_type = "tree"
main_method = "MyModel"
batch_size = 4
output_classes = 5
category = "tabular_classification"
num_feature_points = 50

def MyModel():
    return XGBClassifier(n_estimators=100, random_state=42)

