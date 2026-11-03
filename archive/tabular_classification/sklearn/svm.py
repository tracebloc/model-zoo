from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

framework = "sklearn"
model_type = "svm"
main_method = "MyModel"
batch_size = 4
output_classes = 5
category = "tabular_classification"
num_feature_points = 50

def MyModel():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", SVC(kernel='linear'))])