"""D-FINE (USTC, ICLR 2025). DETR variant with fine-grained distribution refinement on bbox regression; ~55 AP COCO at S scale while keeping RT-DETR-class latency."""
from transformers import DFineForObjectDetection, DFineConfig

framework = "pytorch"
model_type = "detr"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 640
batch_size = 4
output_classes = 12
category = "object_detection"

_PRETRAINED_ID = "ustc-community/dfine-small-coco"


def MyModel(num_classes=output_classes):
    config = DFineConfig.from_pretrained(_PRETRAINED_ID, num_labels=num_classes)
    return DFineForObjectDetection.from_pretrained(
        _PRETRAINED_ID, config=config, ignore_mismatched_sizes=True
    )
