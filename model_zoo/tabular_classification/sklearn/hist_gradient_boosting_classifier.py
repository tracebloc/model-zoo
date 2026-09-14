"""Histogram-based Gradient Boosting classifier (sklearn). Modern GBDT — ~10x faster than classic GradientBoosting, handles missing values natively, supports native categorical encoding. The sklearn answer to LightGBM."""
from sklearn.ensemble import HistGradientBoostingClassifier

framework = "sklearn"
# No histgb-specific value exists in the platform's model_type vocabulary
# (model_type_choices.v1.json), so this GBDT stays "tree" — matching its
# regressor sibling hist_gradient_boosting_regressor.py (model-zoo#272).
model_type = "tree"
main_method = "MyModel"
license = "BSD-3-Clause"
batch_size = 4096
output_classes = 2
num_feature_points = 50
category = "tabular_classification"


def MyModel():
    return HistGradientBoostingClassifier()
