"""DETR (Meta, ECCV 2020). The original transformer detector — set-prediction with Hungarian matching. Historical reference; pairs with RT-DETR (efficient) and Grounding DINO (open-vocabulary)."""
from transformers import DetrForObjectDetection, DetrConfig

framework = "pytorch"
model_type = "detr"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 800
batch_size = 4
output_classes = 12
category = "object_detection"

_PRETRAINED_ID = "facebook/detr-resnet-50"


def MyModel(num_classes=output_classes):
    config = DetrConfig.from_pretrained(_PRETRAINED_ID, num_labels=num_classes)
    return DetrForObjectDetection.from_pretrained(
        _PRETRAINED_ID, config=config, ignore_mismatched_sizes=True
    )
