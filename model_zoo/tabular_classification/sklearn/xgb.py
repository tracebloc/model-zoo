import torch.nn as nn
from xgboost import XGBClassifier

framework = "sklearn"
model_type = "tree"
main_method = "MyModel"
batch_size = 512
output_classes = 2
category = "tabular_classification"
num_feature_points = 282

def MyModel():
    return XGBClassifier(n_estimators=100, random_state=42)

