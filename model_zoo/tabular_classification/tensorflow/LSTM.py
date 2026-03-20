from tensorflow.keras import layers, models

framework = "tensorflow"
main_method = "MyModel"
model_type = ""
batch_size = 4096
output_classes = 3
category = "tabular_classification"
num_feature_points = 152


def MyModel(input_size=num_feature_points, hidden_size=128, n_outputs=1):
    inputs = layers.Input(shape=(input_size,))
    x = layers.Reshape((1, input_size))(inputs)
    x = layers.LSTM(hidden_size)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(n_outputs, activation="sigmoid")(x)
    return models.Model(inputs, outputs)
