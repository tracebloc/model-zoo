import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier

framework = "sklearn"
model_type = "clustering"
main_method = "MyModel"
batch_size = 4
output_classes = 5
category = "tabular_classification"
num_feature_points = 50


def MyModel():
    return KNeighborsClassifier(n_neighbors=5)
