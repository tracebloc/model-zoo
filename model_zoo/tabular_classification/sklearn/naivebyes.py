import torch.nn as nn
from sklearn.naive_bayes import GaussianNB

framework = "sklearn"
model_type = "naive"
main_method = "MyModel"
batch_size = 4
output_classes = 2
category = "tabular_classification"
num_feature_points = 69

def MyModel():
    return GaussianNB()