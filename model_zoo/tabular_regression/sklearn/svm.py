from sklearn.svm import SVR

framework = "sklearn"
model_type = "svm"
main_method = "MyModel"
batch_size = 4
output_classes = 3
category = "tabular_regression"
num_feature_points = 12

def MyModel():
    return SVR(kernel='rbf')
