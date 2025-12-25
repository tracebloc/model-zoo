from sklearn.linear_model import Ridge

framework = "sklearn"
model_type = "linear"
main_method = "MyModel"
batch_size = 128
output_classes = 1
category = "tabular_regression"
num_feature_points = 17


def MyModel():
    return Ridge(alpha=1.0, random_state=42)
