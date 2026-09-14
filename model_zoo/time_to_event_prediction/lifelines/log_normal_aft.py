"""Log-Normal Accelerated Failure Time model (lifelines). Parametric survival model; pick when log(time) is approximately normal."""
from lifelines import LogNormalAFTFitter

framework = "lifelines"
model_type = ""
main_method = "MyModel"
batch_size = 128
output_classes = 1
category = "time_to_event_prediction"
num_feature_points = 12


def MyModel():
    return LogNormalAFTFitter()