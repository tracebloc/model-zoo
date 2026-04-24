"""XGBoost classifier, 100 trees. Strong tabular baseline; usually beats deep models on small/medium datasets."""
import torch.nn as nn
from xgboost import XGBClassifier

framework = "sklearn"
model_type = "tree"
main_method = "MyModel"
batch_size = 512
output_classes = 5
num_feature_points = 50
category = "tabular_classification"

def MyModel():
    return XGBClassifier(n_estimators=100, random_state=42)

