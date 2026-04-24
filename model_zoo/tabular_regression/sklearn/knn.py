"""K-Nearest Neighbors regressor. Simple, non-parametric; pick for small datasets with meaningful distance metrics."""
from sklearn.neighbors import KNeighborsRegressor

framework = "sklearn"
model_type = "ensemble"
main_method = "MyModel"
batch_size = 512
output_classes = 1
category = "tabular_regression"
num_feature_points = 17


def MyModel():
    return KNeighborsRegressor(n_neighbors=5)
