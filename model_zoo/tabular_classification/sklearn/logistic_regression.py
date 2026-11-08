from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

framework = "sklearn"
model_type = "linear"
main_method = "MyModel"
batch_size = 4
output_classes = 5
category = "tabular_classification"
# One of the three ready-made models a first run trains (index: onboarding).
onboarding = True
num_feature_points = 50

def MyModel():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", LogisticRegression(random_state=42))])