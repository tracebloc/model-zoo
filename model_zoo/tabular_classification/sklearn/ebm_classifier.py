"""Explainable Boosting Machine (Microsoft Research). Glass-box GAM — matches XGBoost accuracy with feature-level interpretability. Suitable for healthcare / finance / any regulated domain that requires per-prediction explanations."""
from interpret.glassbox import ExplainableBoostingClassifier

framework = "sklearn"
model_type = ""
main_method = "MyModel"
license = "MIT"
batch_size = 4096
output_classes = 2
num_feature_points = 50
category = "tabular_classification"


def MyModel():
    return ExplainableBoostingClassifier()
