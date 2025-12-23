import torch.nn as nn
from catboost import CatBoostClassifier

framework = "sklearn"
model_type = "catboost"
main_method = "MyModel"
batch_size = 4
output_classes = 5
category = "tabular_classification"
num_feature_points = 50

def MyModel():
    return CatBoostClassifier(n_estimators=100, random_state=42, verbose=0)

