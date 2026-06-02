"""RT-DETRv2 (Baidu, Jul 2024). Bag-of-freebies successor to RT-DETR — better small-object recall and flexible deployment, same Apache-2.0 license. Drop-in upgrade to rt_detr.py."""
from transformers import RTDetrV2ForObjectDetection, RTDetrV2Config

framework = "pytorch"
model_type = "detr"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 640
batch_size = 4
output_classes = 12
category = "object_detection"

_PRETRAINED_ID = "PekingU/rtdetr_v2_r50vd"


def MyModel(num_classes=output_classes):
    config = RTDetrV2Config.from_pretrained(_PRETRAINED_ID, num_labels=num_classes)
    return RTDetrV2ForObjectDetection.from_pretrained(
        _PRETRAINED_ID, config=config, ignore_mismatched_sizes=True
    )
