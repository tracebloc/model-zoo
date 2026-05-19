"""Deformable DETR (SenseTime, ICLR 2021). Multi-scale deformable attention — 10x faster convergence than vanilla DETR and stronger on small objects. The bridge between DETR (2020) and RT-DETR (2024)."""
from transformers import DeformableDetrForObjectDetection, DeformableDetrConfig

framework = "pytorch"
model_type = "detr"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 800
batch_size = 4
output_classes = 12
category = "object_detection"

_PRETRAINED_ID = "SenseTime/deformable-detr"


def MyModel(num_classes=output_classes):
    config = DeformableDetrConfig.from_pretrained(_PRETRAINED_ID, num_labels=num_classes)
    return DeformableDetrForObjectDetection.from_pretrained(
        _PRETRAINED_ID, config=config, ignore_mismatched_sizes=True
    )
