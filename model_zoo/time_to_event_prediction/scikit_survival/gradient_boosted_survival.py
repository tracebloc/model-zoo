"""Gradient Boosted Survival (scikit-survival). Boosted-trees survival analysis — boosted sibling of Random Survival Forest; often beats deep models on tabular medical data with small-to-medium sample sizes."""
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

framework = "scikit_survival"
model_type = ""
main_method = "MyModel"
license = "GPL-3.0"
batch_size = 128
output_classes = 1
category = "time_to_event_prediction"
num_feature_points = 12


def MyModel():
    return GradientBoostingSurvivalAnalysis(
        n_estimators=100, learning_rate=0.1, max_depth=3
    )
