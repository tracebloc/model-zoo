"""Random Forest regressor. Robust, low-tuning baseline; often competitive out of the box."""
from sklearn.ensemble import RandomForestRegressor

framework = "sklearn"
model_type = "tree"
main_method = "MyModel"
batch_size = 128
output_classes = 1
category = "tabular_regression"
num_feature_points = 10


def MyModel():
    return RandomForestRegressor(n_estimators=100, random_state=42)
