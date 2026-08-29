"""Keypoint R-CNN (Meta, ICCV 2017). Mask R-CNN architecture with a keypoint head — top-down multi-person pose via a two-stage detector. Reference torchvision-native baseline for multi-person keypoint detection.

Offline variant: the architecture is built without any checkpoint download,
so the template constructs anywhere, network or not. The pretrained
ResNet50-FPN tensors are delivered from the tracebloc model store as the
training seed: upload the matched ``keypoint_rcnn_weights.pkl`` sitting next
to this file via ``upload_model(..., weights=True)``, and the platform loads
it with ``load_state_dict(strict=True)`` after ``MyModel()`` builds this
architecture. See ``tools/prep_offline_weights.py`` for producing and
verifying that matched weight file.

The backbone is assembled explicitly instead of via the high-level
``keypointrcnn_resnet50_fpn(weights=None)`` builder, for the reasons
documented in ``object_detection/pytorch/faster_rcnn_resnet.py``: with no
weights requested the builder swaps the backbone norm layers from
``FrozenBatchNorm2d`` to trainable ``BatchNorm2d`` (which changes the
state_dict key set) and unfreezes all five backbone stages instead of the
last three. Building the backbone directly reproduces the checkpoint-path
architecture exactly — same norm layers, same three trainable stages, same
2-class person detector with the stock 17-keypoint COCO head (replaced
below, as before). Verified against torchvision 0.27.
"""
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.keypoint_rcnn import (
    KeypointRCNN,
    KeypointRCNNPredictor,
)
from torchvision.ops import misc as misc_nn_ops

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("roi_heads.keypoint_predictor.kps_score_lowres.",)

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
    # Reproduce the checkpoint-path architecture exactly, with no download:
    # frozen batch-norm backbone, FPN with the last 3 stages trainable, and
    # the stock COCO configuration (2 classes: background + person, 17
    # keypoints — the keypoint head is replaced below, as before).
    backbone = resnet50(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(backbone, trainable_layers=3)
    model = KeypointRCNN(backbone, num_classes=2, num_keypoints=17)

    # The checkpoint path (COCO_V1) zeroes FrozenBatchNorm2d eps for this
    # architecture (torchvision's overwrite_eps); match it so numerics are
    # identical, not just the parameter set.
    for module in model.modules():
        if isinstance(module, misc_nn_ops.FrozenBatchNorm2d):
            module.eps = 0.0

    # Replace the keypoint predictor with one sized to the caller's
    # num_feature_points (identical to the pre-migration build, so the
    # hosted seed state_dict keys/shapes match this module exactly).
    in_channels = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
        in_channels, num_feature_points
    )
    return model
