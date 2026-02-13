from sksurv.linear_model import CoxPHSurvivalAnalysis

framework = "scikit_survival"
model_type = ""
main_method = "MyModel"
batch_size = 128
output_classes = 1
category = "time_to_event_prediction"
num_feature_points = 12


def MyModel():
    return CoxPHSurvivalAnalysis()