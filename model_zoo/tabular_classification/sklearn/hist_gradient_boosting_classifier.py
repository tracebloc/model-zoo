"""Histogram-based Gradient Boosting classifier (sklearn). Modern GBDT — ~10x faster than classic GradientBoosting, handles missing values natively, supports native categorical encoding. The sklearn answer to LightGBM."""
from sklearn.ensemble import HistGradientBoostingClassifier

framework = "sklearn"
model_type = "tree"
main_method = "MyModel"
license = "BSD-3-Clause"
batch_size = 4096
output_classes = 2
num_feature_points = 50
category = "tabular_classification"


def MyModel():
    return HistGradientBoostingClassifier()
