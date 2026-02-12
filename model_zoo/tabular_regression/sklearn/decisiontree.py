from sklearn.tree import DecisionTreeRegressor

framework = "sklearn"
model_type = "tree"
main_method = "MyModel"
batch_size = 512
output_classes = 1
category = "tabular_regression"
num_feature_points = 293


def MyModel():
    return DecisionTreeRegressor(random_state=42)
