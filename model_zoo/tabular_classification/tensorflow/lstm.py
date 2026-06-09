"""LSTM for tabular classification (TensorFlow). Pick when rows represent sequences."""
import tensorflow as tf
from tensorflow.keras import layers, models

framework = "tensorflow"
main_method = "MyModel"
model_type = ""
batch_size = 4096
output_classes = 5
num_feature_points = 50
category = "tabular_classification"


def MyModel(input_size=num_feature_points, hidden_size=128, n_outputs=output_classes):
    inputs = layers.Input(shape=(input_size,))
    x = layers.Lambda(lambda t: tf.where(tf.math.is_nan(t), tf.zeros_like(t), t))(inputs)
    x = layers.Reshape((1, input_size))(x)
    x = layers.LSTM(hidden_size)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(n_outputs, activation="softmax")(x)
    return models.Model(inputs, outputs)
