"""Faster R-CNN with a ResNet backbone. Two-stage detector; strong accuracy, slower than YOLO variants.

Offline variant: the architecture is built without any checkpoint download,
so the template constructs anywhere, network or not. The pretrained
ResNet50-FPN tensors are delivered from the tracebloc model store as the
training seed: upload the matched ``faster_rcnn_resnet_weights.pkl`` sitting
next to this file via ``upload_model(..., weights=True)``, and the platform
loads it with ``load_state_dict(strict=True)`` after ``MyModel()`` builds
this architecture. See ``tools/prep_offline_weights.py`` for producing and
verifying that matched weight file.

The backbone is assembled explicitly instead of via the high-level
``fasterrcnn_resnet50_fpn(weights=None)`` builder, because that builder keys
its architecture off whether weights were requested: with no weights it
swaps the backbone norm layers from ``FrozenBatchNorm2d`` to trainable
``BatchNorm2d`` and unfreezes all five backbone stages instead of the last
three. Both are training-behavior changes, and the norm swap also changes
the state_dict key set (``BatchNorm2d`` adds ``num_batches_tracked``
buffers), so the hosted pretrained seed would no longer be a key-exact
match. Building the backbone directly reproduces the checkpoint-path
architecture exactly — same norm layers, same three trainable stages, same
state_dict keys and shapes.

``_resnet_fpn_extractor`` is torchvision-private API (stable across recent
releases; this file is verified against torchvision 0.27). If a torchvision
upgrade ever moves it, this template fails loudly at import and the contract
tests catch it.
"""
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.ops import misc as misc_nn_ops


framework = "pytorch"
model_type = "rcnn"
main_class = "MyModel"
image_size = 448
batch_size = 16
output_classes = 12
category = "object_detection"


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background

    # Reproduce the checkpoint-path architecture exactly, with no download:
    # frozen batch-norm backbone, FPN with the last 3 stages trainable, and
    # the stock 91-class COCO head (replaced below, as before).
    backbone = resnet50(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(backbone, trainable_layers=3)
    model = FasterRCNN(backbone, num_classes=91)

    # The checkpoint-path build zeroes FrozenBatchNorm2d eps for this
    # architecture (torchvision's overwrite_eps); match it so numerics are
    # identical, not just the parameter set.
    for module in model.modules():
        if isinstance(module, misc_nn_ops.FrozenBatchNorm2d):
            module.eps = 0.0

    # Replace the classifier head (identical to the pre-migration build, so
    # the hosted seed state_dict keys/shapes match this module exactly).
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
