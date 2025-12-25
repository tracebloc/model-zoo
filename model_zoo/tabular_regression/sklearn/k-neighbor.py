from sklearn.neighbors import KNeighborsRegressor

framework = "sklearn"
model_type = "ensemble"
main_method = "MyModel"
batch_size = 128
output_classes = 1
category = "tabular_regression"
num_feature_points = 17


def MyModel():
    return KNeighborsRegressor(n_neighbors=5)
