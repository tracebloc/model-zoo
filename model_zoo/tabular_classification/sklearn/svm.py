import torch.nn as nn
from sklearn.svm import SVC

framework = "sklearn"
model_type = "svm"
main_method = "MyModel"
batch_size = 4
output_classes = 5
category = "tabular_classification"
num_feature_points = 50

def MyModel():
    return SVC(kernel='linear')