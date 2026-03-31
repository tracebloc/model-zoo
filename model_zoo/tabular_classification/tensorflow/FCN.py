from tensorflow.keras import layers, models

framework = "tensorflow"
main_method = "MyModel"
model_type = ""
batch_size = 4096
output_classes = 3
category = "tabular_classification"
num_feature_points = 202


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
