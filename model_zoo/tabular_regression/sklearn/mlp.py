from sklearn.neural_network import MLPRegressor

framework = "sklearn"
model_type = "neural_network"
main_method = "MyModel"
batch_size = 4
output_classes = 3
category = "tabular_regression"
num_feature_points = 12

def MyModel():
    return MLPRegressor()
