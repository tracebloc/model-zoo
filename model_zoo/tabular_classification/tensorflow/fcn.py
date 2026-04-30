"""Fully Connected Network for tabular classification (TensorFlow). Default MLP baseline."""
from tensorflow.keras import layers, models

framework = "tensorflow"
main_method = "MyModel"
model_type = ""
batch_size = 4096
output_classes = 5
num_feature_points = 50
category = "tabular_classification"


def MyModel(input_shape=(num_feature_points,), n_outputs=1):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(n_outputs, activation="sigmoid"),
        ]
    )
    return model
