"""ViTPose (NeurIPS 2022). First transformer pose model in the zoo — ViT backbone + simple decoder; 81+ AP COCO whole-body at Huge scale, fine-tunes well at Base."""
from transformers import VitPoseForPoseEstimation, VitPoseConfig

framework = "pytorch"
model_type = "transformer"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 256
batch_size = 32
output_classes = 1
category = "keypoint_detection"
num_feature_points = 17

_PRETRAINED_ID = "usyd-community/vitpose-base-simple"


def MyModel(num_feature_points=num_feature_points):
    config = VitPoseConfig.from_pretrained(_PRETRAINED_ID, num_labels=num_feature_points)
    return VitPoseForPoseEstimation.from_pretrained(
        _PRETRAINED_ID, config=config, ignore_mismatched_sizes=True
    )
