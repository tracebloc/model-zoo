from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

framework = "sklearn"
model_type = "tree"
main_method = "MyModel"
batch_size = 4
output_classes = 5
category = "tabular_classification"
num_feature_points = 50

def MyModel():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])