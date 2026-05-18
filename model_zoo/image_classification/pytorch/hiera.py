"""Hiera (Meta, ICML 2023). Hierarchical ViT that strips per-stage tricks (relative position, conv stem, etc.) while matching Swin/MViTv2 accuracy. Simpler architecture, MAE-pretrained, available via transformers."""
from transformers import HieraForImageClassification, HieraConfig

framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 224
batch_size = 32
output_classes = 2
category = "image_classification"

_PRETRAINED_ID = "facebook/hiera-tiny-224-in1k-hf"


def MyModel(num_classes=output_classes):
    config = HieraConfig.from_pretrained(_PRETRAINED_ID, num_labels=num_classes)
    return HieraForImageClassification.from_pretrained(
        _PRETRAINED_ID, config=config, ignore_mismatched_sizes=True
    )
