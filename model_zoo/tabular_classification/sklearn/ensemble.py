from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

framework = "sklearn"
model_type = "ensemble"
main_method = "MyModel"
batch_size = 4
output_classes = 5
category = "tabular_classification"
num_feature_points = 50

def MyModel():
    # Impute inside each base estimator so the imputer is refit within
    # StackingClassifier's internal CV folds. An outer imputer would fit on all
    # training rows before that CV and leak validation-fold values into the
    # meta-features / meta-learner.
    gbm = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("clf", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))])
    svm = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("clf", SVC(probability=True, kernel="rbf", random_state=42))])
    # final_estimator only sees base-model predictions (no NaN) -> no imputer needed.
    return StackingClassifier(
        estimators=[("gbm", gbm), ("svm", svm)],
        final_estimator=LogisticRegression(random_state=42),
    )