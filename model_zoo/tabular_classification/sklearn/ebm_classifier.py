"""Explainable Boosting Machine (Microsoft Research). Glass-box GAM — matches XGBoost accuracy with feature-level interpretability. Suitable for healthcare / finance / any regulated domain that requires per-prediction explanations."""
from interpret.glassbox import ExplainableBoostingClassifier

framework = "sklearn"
# EBM is a glass-box GAM, not a boosted-tree library, so the platform vocabulary
# has no exact value. `tree` is the same choice its sibling
# hist_gradient_boosting_classifier.py makes for the same reason -- an additive
# boosting model whose library has no vocabulary entry -- and EBM is cyclic
# boosting over shallow trees underneath.
#
# NOT "" (the EBM declaration incident). A blank IS stored by the backend as 'default', which is
# what an earlier revision of this comment relied on, but the SDK never sees
# 'default': `_parse_constant_rhs` coerces an empty literal to None
# (tracebloc/validation/rewriter.py), and None is not in the sklearn upload
# path's accepted set, so `average_estimators` raises "model type None is not
# supported for Sklearn" and `model_func_checks` fails. The template was
# untrainable end to end as declared. 'default' would not have worked either --
# it is not in the SDK's set on any route.
model_type = "tree"
main_method = "MyModel"
license = "MIT"
batch_size = 4096
output_classes = 2
num_feature_points = 50
category = "tabular_classification"


def MyModel():
    return ExplainableBoostingClassifier()
