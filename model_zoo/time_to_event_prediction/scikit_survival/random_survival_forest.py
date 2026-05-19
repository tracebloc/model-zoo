"""Random Survival Forest (Ishwaran et al., 2008) via scikit-survival. Tree-based non-parametric survival baseline — handles non-linear effects and interactions without the proportional-hazards assumption."""
from sksurv.ensemble import RandomSurvivalForest

framework = "scikit_survival"
model_type = ""
main_method = "MyModel"
license = "GPL-3.0"
batch_size = 128
output_classes = 1
category = "time_to_event_prediction"
num_feature_points = 12


def MyModel():
    return RandomSurvivalForest(
        n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=-1
    )
