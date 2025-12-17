from sklearn.neighbors import KNeighborsRegressor

framework = "sklearn"
model_type = "neighbors"
main_method = "MyModel"
batch_size = 4
output_classes = 3
category = "tabular_regression"
num_feature_points = 12

def MyModel():
    return KNeighborsRegressor(n_neighbors=5)
