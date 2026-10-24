"""Log-Logistic Accelerated Failure Time model (lifelines). Parametric survival model; non-monotonic hazard."""
from lifelines import LogLogisticAFTFitter

framework = "lifelines"
model_type = ""
main_method = "MyModel"
batch_size = 128
output_classes = 1
category = "time_to_event_prediction"
# Not an input width: see "Dataset expectations" in ../README.md.
num_feature_points = 12

def MyModel():
    return LogLogisticAFTFitter()