"""Keypoint R-CNN (Meta, ICCV 2017). Mask R-CNN architecture with a keypoint head — top-down multi-person pose via a two-stage detector. Reference torchvision-native baseline for multi-person keypoint detection."""
import torchvision
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

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
    # torchvision's detection builders raise ValueError when a non-default
    # num_keypoints is paired with COCO pretrained weights (the weights
    # expect 17 keypoints). Load with the pretrained weights' default head,
    # then swap the keypoint predictor for one sized to the caller's
    # num_feature_points — mirrors the head-swap pattern used by
    # object_detection/pytorch/fcos.py and retinanet.py.
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_channels = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
        in_channels, num_feature_points
    )
    return model
