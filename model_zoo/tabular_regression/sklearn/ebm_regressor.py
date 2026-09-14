"""Explainable Boosting Machine regressor (Microsoft Research). Glass-box GAM — competitive with XGBoost on tabular regression while preserving full feature-level interpretability."""
from interpret.glassbox import ExplainableBoostingRegressor

framework = "sklearn"
# `tree`, matching hist_gradient_boosting_regressor.py: an additive boosting
# model with no exact platform vocabulary value, cyclic boosting over shallow
# trees underneath. NOT "" -- the SDK coerces an empty literal to None and the
# sklearn upload path refuses it, so the template was untrainable end to end
# (the EBM declaration incident). See ebm_classifier.py for the full chain.
model_type = "tree"
main_method = "MyModel"
license = "MIT"
batch_size = 4096
output_classes = 1
num_feature_points = 50
category = "tabular_regression"


def MyModel():
    return ExplainableBoostingRegressor()
