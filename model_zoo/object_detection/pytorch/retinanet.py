"""RetinaNet (Meta, ICCV 2017). Focal loss + one-stage anchor design — the canonical one-stage detector still widely deployed for production. Pairs with Faster R-CNN as the standard two-stage / one-stage comparison."""
import torchvision
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights

framework = "pytorch"
model_type = "retinanet"
main_method = "MyModel"
license = "BSD-3-Clause"
image_size = 448
batch_size = 8
output_classes = 12
category = "object_detection"


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background
    return torchvision.models.detection.retinanet_resnet50_fpn(
        weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT,
        num_classes=num_classes,
        weights_backbone=None,
    )
