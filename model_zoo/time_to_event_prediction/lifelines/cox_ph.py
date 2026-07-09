"""Cox Proportional Hazards model (lifelines). Semi-parametric survival baseline;
federates by inverse-variance (precision) weighting of its coefficients."""
from lifelines import CoxPHFitter

framework = "lifelines"
model_type = ""
main_method = "MyModel"
batch_size = 128
output_classes = 1
category = "time_to_event_prediction"
num_feature_points = 12


def MyModel():
    return CoxPHFitter()
