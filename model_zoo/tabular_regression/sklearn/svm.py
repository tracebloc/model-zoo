from sklearn.svm import SVR

framework = "sklearn"
model_type = "svm"
main_method = "MyModel"
batch_size = 128
output_classes = 1
category = "tabular_regression"
num_feature_points = 17


def MyModel():
    return SVR(kernel="rbf")
