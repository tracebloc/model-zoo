"""FCOS (ICCV 2019). Anchor-free one-stage detector — predicts boxes per-pixel via center-ness, much simpler than anchor-based two-stage approaches. Strong baseline for production deployments."""
import torchvision
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights

framework = "pytorch"
model_type = "fcos"
main_method = "MyModel"
license = "BSD-3-Clause"
image_size = 448
batch_size = 8
output_classes = 12
category = "object_detection"


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background
    return torchvision.models.detection.fcos_resnet50_fpn(
        weights=FCOS_ResNet50_FPN_Weights.DEFAULT,
        num_classes=num_classes,
        weights_backbone=None,
    )
