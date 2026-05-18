"""Explainable Boosting Machine regressor (Microsoft Research). Glass-box GAM — competitive with XGBoost on tabular regression while preserving full feature-level interpretability."""
from interpret.glassbox import ExplainableBoostingRegressor

framework = "sklearn"
model_type = ""
main_method = "MyModel"
license = "MIT"
batch_size = 4096
output_classes = 1
num_feature_points = 50
category = "tabular_regression"


def MyModel():
    return ExplainableBoostingRegressor()
