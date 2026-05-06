"""ViT-B/16 via HuggingFace ViTForImageClassification, google/vit-base-patch16-224 weights. Pretrained classifier; fine-tune the head for your classes."""
import torch
from torch import nn
from transformers import ViTForImageClassification, ViTConfig


framework = "pytorch"
main_class = "VisionTransformer"
image_size = 224
batch_size = 16
category = "image_classification"
output_classes = 2

# model version
# https://huggingface.co/google/vit-base-patch16-224/commits/main
VIT_REVISION = "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3"


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize the model with a specified number of output labels for classification
        config = ViTConfig.from_pretrained(
            "google/vit-base-patch16-224",
            revision=VIT_REVISION,
            num_labels=output_classes,
        )
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            revision=VIT_REVISION,
            config=config,
            ignore_mismatched_sizes=True,
        )

    def forward(self, pixel_values):
        # The model will output a dictionary with various keys.
        outputs = self.vit(pixel_values=pixel_values)
        # The logits are now directly available from the output's 'logits' key.
        logits = outputs.logits
        return logits
