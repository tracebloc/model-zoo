"""Mask R-CNN with ResNet-50 FPN backbone. Companion to Faster R-CNN — adds a mask branch for instance segmentation alongside bounding boxes. Canonical reference architecture.

Offline variant: the architecture is built without any checkpoint download,
so the template constructs anywhere, network or not. The pretrained
ResNet50-FPN tensors are delivered from the tracebloc model store as the
training seed: upload the matched ``mask_rcnn_weights.pkl`` sitting next to
this file via ``upload_model(..., weights=True)``, and the platform loads it
with ``load_state_dict(strict=True)`` after ``MyModel()`` builds this
architecture. See ``tools/prep_offline_weights.py`` for producing and
verifying that matched weight file.

The backbone is assembled explicitly instead of via the high-level
``maskrcnn_resnet50_fpn(weights=None)`` builder, for the reasons documented
in ``faster_rcnn_resnet.py``: with no weights requested the builder swaps
the backbone norm layers from ``FrozenBatchNorm2d`` to trainable
``BatchNorm2d`` (which changes the state_dict key set) and unfreezes all
five backbone stages instead of the last three. Building the backbone
directly reproduces the checkpoint-path architecture exactly — same norm
layers, same three trainable stages, same state_dict keys and shapes.
Verified against torchvision 0.27.
"""
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor
from torchvision.ops import misc as misc_nn_ops

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("roi_heads.box_predictor.bbox_pred.", "roi_heads.box_predictor.cls_score.", "roi_heads.mask_predictor.mask_fcn_logits.")

framework = "pytorch"
model_type = "rcnn"
main_method = "MyModel"
license = "BSD-3-Clause"
image_size = 448
batch_size = 8
output_classes = 12
category = "object_detection"


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background

    # Reproduce the checkpoint-path architecture exactly, with no download:
    # frozen batch-norm backbone, FPN with the last 3 stages trainable, and
    # the stock 91-class COCO heads (replaced below, as before).
    backbone = resnet50(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(backbone, trainable_layers=3)
    model = MaskRCNN(backbone, num_classes=91)

    # The checkpoint path (COCO_V1) zeroes FrozenBatchNorm2d eps for this
    # architecture (torchvision's overwrite_eps); match it so numerics are
    # identical, not just the parameter set.
    for module in model.modules():
        if isinstance(module, misc_nn_ops.FrozenBatchNorm2d):
            module.eps = 0.0

    # Replace the box and mask heads (identical to the pre-migration build,
    # so the hosted seed state_dict keys/shapes match this module exactly).
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model
