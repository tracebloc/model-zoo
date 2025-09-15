import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier

framework = "sklearn"
model_type = "tree"
main_method = "MyModel"
batch_size = 4
output_classes = 2
category = "tabular_classification"
num_feature_points = 69

def MyModel():
    return DecisionTreeClassifier(random_state=42)