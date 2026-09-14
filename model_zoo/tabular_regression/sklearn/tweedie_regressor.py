"""Tweedie GLM regressor (sklearn). Exponential dispersion family — the canonical model for non-negative, right-skewed targets with a mass at zero. Standard for insurance claims, healthcare costs, energy consumption."""
from sklearn.linear_model import TweedieRegressor

framework = "sklearn"
model_type = "linear"
main_method = "MyModel"
license = "BSD-3-Clause"
batch_size = 4096
output_classes = 1
num_feature_points = 50
category = "tabular_regression"


def MyModel():
    return TweedieRegressor(power=1.5, alpha=0.5, link="log", max_iter=1000)
