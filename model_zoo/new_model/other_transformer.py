import tensorflow as tf
import keras
from keras import layers, ops

framework = "tensorflow"
main_method = "MyModel"
image_size = 224
batch_size = 16
output_classes = 2
category = "image_classification"


def _patch_embedding(
    inputs,
    patch_size: int = 16,
    projection_dim: int = 256,
    name: str = "patch_embedding",
):
    """
    Splits the image into non-overlapping patches using a Conv2D (stride=patch_size)
    and projects them to `projection_dim`. Returns [B, N, D].
    """
    # Inputs expected in [0, 255]; rescale for training stability
    x = layers.Rescaling(1.0 / 255)(inputs)

    # Conv2D with stride = patch size => non-overlapping patches
    x = layers.Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        use_bias=False,
        name=name,
    )(x)  # [B, H/ps, W/ps, D]

    # Flatten to sequence (N = number of patches)
    h = inputs.shape[1] // patch_size
    w = inputs.shape[2] // patch_size
    n = int(h * w)
    x = layers.Reshape((n, projection_dim))(x)  # [B, N, D]
    return x, n


def _positional_embeddings(num_patches: int, dim: int, name: str = "positional_embedding"):
    """
    Learnable positional embeddings, shape [1, N, D].
    """
    pe = keras.Variable(
        initial_value=ops.random.normal(shape=(1, num_patches, dim), stddev=0.02),
        trainable=True,
        name=name,
        dtype="float32",
    )
    return pe


def _transformer_encoder(
    x,
    projection_dim: int,
    num_heads: int,
    mlp_units=(512, 256),
    dropout_rate: float = 0.1,
    name_prefix: str = "encoder",
):
    """
    One encoder block: LN -> MHSA -> Add -> LN -> MLP -> Add
    """
    # Pre-Norm + Self-Attention
    y = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(x)
    y = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim,
        dropout=dropout_rate,
        name=f"{name_prefix}_mhsa",
    )(y, y)
    x = layers.Add(name=f"{name_prefix}_add1")([x, y])

    # Pre-Norm + MLP
    y = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(x)
    for i, units in enumerate(mlp_units):
        y = layers.Dense(units, activation="gelu", name=f"{name_prefix}_mlp_dense{i+1}")(y)
        y = layers.Dropout(dropout_rate, name=f"{name_prefix}_mlp_do{i+1}")(y)
    x = layers.Add(name=f"{name_prefix}_add2")([x, y])
    return x


def MyModel(
    input_shape=(image_size, image_size, 3),
    classes=output_classes,
    patch_size: int = 16,
    projection_dim: int = 256,
    transformer_layers: int = 8,
    num_heads: int = 8,
    mlp_units=(512, 256),
    dropout_rate: float = 0.1,
    use_cls_token: bool = False,
):
    """
    Vision Transformer for image classification.

    Args:
        input_shape: (H, W, 3)
        classes: number of output classes
        patch_size: patch size (pixels)
        projection_dim: token embedding dimension
        transformer_layers: number of encoder blocks
        num_heads: attention heads
        mlp_units: MLP hidden sizes inside each encoder
        dropout_rate: dropout in MHSA and MLP
        use_cls_token: if True, prepend learnable [CLS] and pool that instead of GAP
    """
    inputs = keras.Input(shape=input_shape, name="image")

    # --- Patchify + project ---
    tokens, num_patches = _patch_embedding(
        inputs, patch_size=patch_size, projection_dim=projection_dim
    )  # [B, N, D]

    # --- (Optional) CLS token ---
    if use_cls_token:
        cls = keras.Variable(
            tf.random.normal((1, 1, projection_dim), stddev=0.02),
            name="cls_token", trainable=True,
        )
        cls_broadcast = ops.repeat(cls, repeats=ops.shape(tokens)[0], axis=0)  # [B, 1, D]
        tokens = ops.concatenate([cls_broadcast, tokens], axis=1)  # [B, N+1, D]
        num_tokens = num_patches + 1
    else:
        num_tokens = num_patches

    # --- Positional embeddings ---

    # AFTER
    pos_emb = keras.Variable(
        tf.random.normal((1, num_tokens, projection_dim), stddev=0.02),
        name="positional_embedding", trainable=True,
    )
    # pos_emb = _positional_embeddings(num_tokens, projection_dim)
    x = tokens + pos_emb

    # --- Transformer encoders ---
    for i in range(transformer_layers):
        x = _transformer_encoder(
            x,
            projection_dim=projection_dim,
            num_heads=num_heads,
            mlp_units=mlp_units,
            dropout_rate=dropout_rate,
            name_prefix=f"encoder{i+1}",
        )

    # --- Final norm + pooling ---
    x = layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)
    if use_cls_token:
        # Take the first token ([CLS])
        x = layers.Lambda(lambda t: t[:, 0], name="cls_pool")(x)
    else:
        # Average over patch tokens
        x = layers.GlobalAveragePooling1D(name="gap")(x)

    # --- (Optional) small head, similar to many ViT heads ---
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    x = layers.Dense(256, activation="gelu", name="head_dense")(x)
    x = layers.BatchNormalization(name="head_bn")(x)

    outputs = layers.Dense(classes, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs, outputs, name="vision_transformer")
    return model
