"""Weibull Accelerated Failure Time model (lifelines). Most common parametric survival baseline."""
from lifelines import WeibullAFTFitter

framework = "lifelines"
model_type = ""
main_method = "MyModel"
batch_size = 128
output_classes = 1
category = "time_to_event_prediction"
# Not an input width: lifelines fits whatever covariates the dataset has, so a
# link ignores it. Declared because the upload check requires the field.
num_feature_points = 12


def MyModel():
    return WeibullAFTFitter()