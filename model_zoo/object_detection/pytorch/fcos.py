"""FCOS (ICCV 2019). Anchor-free one-stage detector — predicts boxes per-pixel via center-ness, much simpler than anchor-based two-stage approaches. Strong baseline for production deployments.

Offline variant: the architecture is built without any checkpoint download,
so the template constructs anywhere, network or not. The pretrained
ResNet50-FPN tensors are delivered from the tracebloc model store as the
training seed: upload the matched ``fcos_weights.pkl`` sitting next to this
file via ``upload_model(..., weights=True)``, and the platform loads it with
``load_state_dict(strict=True)`` after ``MyModel()`` builds this
architecture. See ``tools/prep_offline_weights.py`` for producing and
verifying that matched weight file.

The backbone is assembled explicitly instead of via the high-level
``fcos_resnet50_fpn(weights=None)`` builder, for the reasons documented in
``faster_rcnn_resnet.py``: with no weights requested the builder swaps the
backbone norm layers from ``FrozenBatchNorm2d`` to trainable ``BatchNorm2d``
(which changes the state_dict key set) and unfreezes all five backbone
stages instead of the last three. Building the backbone directly reproduces
the checkpoint-path architecture exactly — same norm layers, same three
trainable stages, same P6/P7 pyramid, same state_dict keys and shapes.
(Unlike the R-CNN family and RetinaNet, FCOS's checkpoint path does not
zero FrozenBatchNorm2d eps, so no eps overwrite here.) Verified against
torchvision 0.27.
"""
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.fcos import FCOS, FCOSClassificationHead
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

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

    # Reproduce the checkpoint-path architecture exactly, with no download:
    # frozen batch-norm backbone, FPN over the last 3 stages with extra
    # P6/P7 levels, and the stock 91-class COCO head (replaced below, as
    # before).
    backbone = resnet50(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(
        backbone,
        trainable_layers=3,
        returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(256, 256),
    )
    model = FCOS(backbone, num_classes=91)

    # Replace the classification head (identical to the pre-migration build,
    # so the hosted seed state_dict keys/shapes match this module exactly).
    in_channels = model.backbone.out_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = FCOSClassificationHead(
        in_channels, num_anchors, num_classes
    )
    return model
