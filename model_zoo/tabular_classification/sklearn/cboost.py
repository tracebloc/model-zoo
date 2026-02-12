import torch.nn as nn
from catboost import CatBoostClassifier

framework = "sklearn"
model_type = "tree"
main_method = "MyModel"
batch_size = 512
output_classes = 2
category = "tabular_classification"
num_feature_points = 183

def MyModel():
    return CatBoostClassifier(n_estimators=100, random_state=42, verbose=0)

