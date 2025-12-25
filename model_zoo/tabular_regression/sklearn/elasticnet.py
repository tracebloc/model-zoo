from sklearn.linear_model import ElasticNet

framework = "sklearn"
model_type = "linear"
main_method = "MyModel"
batch_size = 128
output_classes = 1
category = "tabular_regression"
num_feature_points = 17


def MyModel():
    return ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
