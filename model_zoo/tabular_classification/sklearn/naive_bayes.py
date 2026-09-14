from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

framework = "sklearn"
model_type = "naive"
main_method = "MyModel"
batch_size = 4
output_classes = 5
category = "tabular_classification"
num_feature_points = 50

def MyModel():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", GaussianNB())])