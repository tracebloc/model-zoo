import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier


framework = "sklearn"
model_type = "tree"
main_method = "MyModel"
batch_size = 4
output_classes = 5
category = "tabular_classification"
num_feature_points = 50

def MyModel():
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    return model