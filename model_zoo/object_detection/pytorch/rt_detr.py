"""RT-DETR (Baidu, CVPR 2024). First real-time DETR; Apache-2.0; ~53 AP COCO. Fills the entire transformer-detector gap left by the YOLO + Faster R-CNN lineup."""
from transformers import RTDetrForObjectDetection, RTDetrConfig

framework = "pytorch"
model_type = "detr"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 640
batch_size = 4
output_classes = 12
category = "object_detection"

_PRETRAINED_ID = "PekingU/rtdetr_r50vd"


def MyModel(num_classes=output_classes):
    config = RTDetrConfig.from_pretrained(_PRETRAINED_ID, num_labels=num_classes)
    return RTDetrForObjectDetection.from_pretrained(
        _PRETRAINED_ID, config=config, ignore_mismatched_sizes=True
    )
