"""Keypoint R-CNN (Meta, ICCV 2017). Mask R-CNN architecture with a keypoint head — top-down multi-person pose via a two-stage detector. Reference torchvision-native baseline for multi-person keypoint detection."""
import torchvision
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights

framework = "pytorch"
model_type = "rcnn"
main_method = "MyModel"
license = "BSD-3-Clause"
image_size = 448
batch_size = 4
output_classes = 1
category = "keypoint_detection"
num_feature_points = 17


def MyModel(num_feature_points=num_feature_points):
    return torchvision.models.detection.keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
        num_keypoints=num_feature_points,
        weights_backbone=None,
    )
