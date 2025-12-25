from sklearn.neural_network import MLPRegressor

framework = "sklearn"
model_type = "neural_network"
main_method = "MyModel"
batch_size = 128
output_classes = 1
category = "tabular_regression"
num_feature_points = 17


def MyModel():
    return MLPRegressor()
