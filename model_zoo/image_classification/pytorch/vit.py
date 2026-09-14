"""ViT-B/16 backbone from HuggingFace transformers + custom classification head. Pick when you want to swap the head or extract features.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. This template has always
random-initialized the backbone (it only ever fetched the config, never the
weights), so there is no weight file: upload with ``weights=False``::

    user.upload_model("vit")

For the pretrained variant of the same architecture, use ``vit_google.py``.
"""
from torch import nn
from transformers import ViTConfig, ViTModel

framework = "pytorch"
main_class = "VisionTransformer"
image_size = 224
batch_size = 16
category = "image_classification"
output_classes = 2

# Architecture config for google/vit-base-patch16-224, inlined so the model
# builds with no config fetch. The SDK uploads only the .py — there is no
# config.json path — so the config lives here in the template.
CONFIG = {
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "image_size": 224,
    "patch_size": 16,
    "num_channels": 3,
    "qkv_bias": True,
    "encoder_stride": 16,
}


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Configuration for ViT, built from the inlined config
        self.config = ViTConfig(**CONFIG, num_labels=output_classes)

        # Initialize the ViT model (random init — this is a backbone template)
        self.vit = ViTModel(self.config)

        # Here you can add more layers if you want, for example a classification head
        self.classification_head = nn.Linear(self.config.hidden_size, output_classes)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classification_head(outputs.last_hidden_state[:, 0])
        return logits
