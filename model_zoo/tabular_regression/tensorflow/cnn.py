from tensorflow.keras import layers, models

framework = "tensorflow"
main_method = "MyModel"
model_type = ""
batch_size = 512
output_classes = 1
num_feature_points = 17
category = "tabular_regression"


def MyModel(input_size=num_feature_points, n_outputs=1):
    model = models.Sequential(
        [
            layers.Input(shape=(input_size, 1)),
            layers.Conv1D(16, 3, padding="same", activation="relu"),
            layers.Conv1D(32, 3, padding="same", activation="relu"),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(n_outputs),
        ]
    )
    return model
