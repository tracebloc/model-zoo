"""Ordinary linear regression. Simplest baseline; always run this first to establish a floor."""
from sklearn.linear_model import LinearRegression

framework = "sklearn"
model_type = "linear"
main_method = "MyModel"
batch_size = 128
output_classes = 1
category = "tabular_regression"
num_feature_points = 10


def MyModel():
    return LinearRegression()
