"""Histogram-based Gradient Boosting regressor (sklearn). Modern GBDT — ~10x faster than classic GradientBoosting, handles missing values natively, supports native categorical encoding. Pure-sklearn alternative to LightGBM."""
from sklearn.ensemble import HistGradientBoostingRegressor

framework = "sklearn"
model_type = "tree"
main_method = "MyModel"
license = "BSD-3-Clause"
batch_size = 4096
output_classes = 1
num_feature_points = 50
category = "tabular_regression"


def MyModel():
    return HistGradientBoostingRegressor()
