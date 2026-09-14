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
``keypointrcnn_resnet50_fpn(weights=None)`` builder, because that builder
keys its architecture off whether weights were requested: with no weights it
unfreezes all five backbone stages instead of the last three, and it picks
the backbone norm off the same flag. Building the backbone directly keeps the
three trainable stages under explicit control, along with the 2-class person
detector and the stock 17-keypoint COCO head (replaced below, as before).
The norm is no longer the checkpoint's — see below. Verified against
torchvision 0.27.

Backbone norm: GroupNorm, and it is NOT the checkpoint's norm
----------------------------------------------------------------------------
This template used to build ``norm_layer=misc_nn_ops.FrozenBatchNorm2d`` to
reproduce torchvision's checkpoint-path architecture key-exactly. That was
wrong in the regime the platform actually runs: frozen BN at construction
holds ``weight=1``, ``bias=0``, ``running_mean=0``, ``running_var=1``, so on a
``weights=None`` build it computes ``(x - 0) / sqrt(1 + eps) * 1 + 0`` -- a
bit-exact identity. Its buffers are meaningful only after a checkpoint loads
real statistics into them, so every from-scratch run of this template to date
trained with no backbone normalisation at all. Measured downstream on the OD
roster, which shared this exact line: activations reach sigma ~= 24 at the ROI
head against ~= 3 with a live norm, and ``centernet_resnet`` diverged
1032.6 -> 3.19e+29 with grad norm going to inf, where GroupNorm on the same
script and seeds gave 165.9 -> 15.87 -> 14.61.

GroupNorm normalises per sample, so it is correct with no checkpoint and adds
no running statistics for the averaging service to ship each federated round
-- both halves of the constraint that produced frozen BN in the first place.
This is the same conversion model-zoo#262 applied to twelve OD templates; the
two keypoint templates carrying the identical line were outside that PR's
directory scan.

⚠️ WHAT THIS COSTS, EXPLICITLY, AND IT COSTS MORE HERE THAN ON OD. A
torchvision COCO checkpoint's BN running statistics have nowhere to go in a
GroupNorm tree, so this template can no longer strict-load a seed prepped from
``download.pytorch.org`` weights. Unlike the OD twelve -- where no seed was
ever staged (an internal ticket is blocked on the store decision in the hosting decision) --
``keypoint_rcnn_weights.pkl`` IS a live entry in the backend dump manifest, so
this change invalidates a dump that exists and must be re-prepped.
``tools/prep_offline_weights.py`` fails loudly on the strict load rather than
producing a mismatched dump, which is the right place for it to fail. The
re-prep itself lives in ``backend``, not here. The trade taken is a real
defect on every from-scratch run today against a seed path whose store is
still undecided.

Parameter arithmetic, for any published count: frozen BN holds weight/bias as
BUFFERS and GroupNorm holds them as PARAMETERS, so this is +53,120 parameters
/ -106,240 buffers for the ResNet-50 trunk. It is NOT parameter-neutral.

``_resnet_fpn_extractor`` is torchvision-private API (stable across recent
releases; this file is verified against torchvision 0.27). If a torchvision
upgrade ever moves it, this template fails loudly at import and the contract
tests catch it.
"""
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.keypoint_rcnn import (
    KeypointRCNN,
    KeypointRCNNPredictor,
)

# (internal ref) — the task head is NOT carried by the hosted seed.
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


def _group_norm(channels):
    """GroupNorm with the largest group count ``<= 32`` that divides ``channels``.

    The backbone norm for a from-scratch build — see the module
    docstring for why frozen BN was wrong here.

    The group count is derived, not hardcoded to 32: ``nn.GroupNorm`` requires
    ``channels % num_groups == 0``, which ResNet-50's 64..2048 all satisfy at
    32 (the canonical Wu & He setting), but hardcoding it would break the
    moment this template is retargeted at a backbone with stages that do not
    divide (MobileNetV3's 16/24/40/72/120/184 do not).

    Duplicated per template on purpose -- a zoo template is uploaded as ONE
    file and cannot import a sibling (no relative imports anywhere in this
    repo). ``faster_rcnn_sppe._group_norm`` and the OD family's
    ``_group_norm`` / ``_norm`` / ``_norm_groups`` are the same helper for the
    same reason.
    """
    groups = max(g for g in range(1, 33) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


def MyModel(num_feature_points=num_feature_points):
    # No download: GroupNorm backbone (an internal ticket -- frozen BN is a bit-exact
    # identity from scratch), FPN with the last 3 stages trainable, and the
    # stock COCO configuration (2 classes: background + person, 17 keypoints —
    # the keypoint head is replaced below, as before).
    #
    # There is no `overwrite_eps` loop any more: it existed to zero
    # FrozenBatchNorm2d's eps so numerics matched the COCO_V1 checkpoint path,
    # and with no FrozenBatchNorm2d in the tree it would iterate every module
    # and match none. Dead code that reads as coverage is worse than none.
    backbone = resnet50(weights=None, norm_layer=_group_norm)
    backbone = _resnet_fpn_extractor(backbone, trainable_layers=3)
    model = KeypointRCNN(backbone, num_classes=2, num_keypoints=17)

    # Replace the keypoint predictor with one sized to the caller's
    # num_feature_points (identical to the pre-migration build, so the
    # hosted seed state_dict keys/shapes match this module exactly).
    in_channels = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
        in_channels, num_feature_points
    )
    return model
