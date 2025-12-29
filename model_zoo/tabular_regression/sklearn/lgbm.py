from lightgbm import LGBMRegressor

framework = "sklearn"
model_type = "lightgbm"
main_method = "MyModel"
batch_size = 128
output_classes = 1
category = "tabular_regression"
num_feature_points = 17

def MyModel():
    return LGBMRegressor(n_estimators=100, random_state=42)
