import torch.nn as nn
from sklearn.neural_network import MLPClassifier

framework = "sklearn"
model_type = "neural_network"
main_method = "MyModel"
batch_size = 4
output_classes = 5
category = "tabular_classification"
num_feature_points = 50

def MyModel():
    return MLPClassifier()