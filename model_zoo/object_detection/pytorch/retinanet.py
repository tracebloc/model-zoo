"""RetinaNet (Meta, ICCV 2017). Focal loss + one-stage anchor design — the canonical one-stage detector still widely deployed for production. Pairs with Faster R-CNN as the standard two-stage / one-stage comparison."""
import torchvision
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

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

    # torchvision's detection builders raise ValueError when a custom
    # num_classes is paired with COCO pretrained weights (the weights expect
    # 91 classes). Load with the pretrained weights' default head, then swap
    # the classification head for one sized to the caller's num_classes —
    # mirrors the pattern in faster_rcnn_resnet.py.
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT
    )
    in_channels = model.backbone.out_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels, num_anchors, num_classes
    )
    return model
