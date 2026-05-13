from tensorflow.keras import layers, models

framework = "tensorflow"
main_method = "MyModel"
model_type = ""
batch_size = 512
output_classes = 1
num_feature_points = 17
category = "tabular_regression"


def MyModel(input_shape=(num_feature_points,), n_outputs=1):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(n_outputs),
        ]
    )
    return model
