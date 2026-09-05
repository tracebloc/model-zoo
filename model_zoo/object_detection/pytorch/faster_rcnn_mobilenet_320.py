"""Faster R-CNN with a MobileNetV3-Large FPN backbone, tuned for 320px input. The low-resolution sibling of ``faster_rcnn_mobilenet.py``: same architecture, but torchvision's small-input configuration — a 320/640 transform and a far shorter RPN proposal list at inference — for the cheapest two-stage option in the zoo.

Offline variant: the architecture is built with ``weights=None``, so nothing
is fetched from ``download.pytorch.org`` — the #199 egress lockdown blocks it
— and the template constructs anywhere, network or not. No seed is hosted for
this template yet, so it random-initialises and there is no weight file:
upload with ``weights=False``::

    user.upload_model("faster_rcnn_mobilenet_320", weights=False)

Hosting the torchvision COCO tensors as a tracebloc model-store seed (the
#1499 pattern: a matched ``<stem>_weights.pkl`` prepped by
``tools/prep_offline_weights.py`` and strict-loaded after ``MyModel()`` has
built the architecture) is follow-up work, not part of this roster addition.
Until a dump is staged, ``tools/check_dump_coverage.py`` classifies this file
NO_SEED and the statement above is what keeps that classification honest. What
seeding would now require is recorded below — it is no longer the torchvision
COCO checkpoint.

The 320 variant differs from ``faster_rcnn_mobilenet.py`` ONLY in
``GeneralizedRCNNTransform`` and RPN inference knobs — none of which are
parameters. The two share a state_dict key for key, so one prepped dump would
serve both once a seed is staged; they are kept as separate templates because
the declared ``image_size`` is the thing a user picks between, and that is
header metadata, not a runtime argument.

Backbone norm: GroupNorm, and it forfeits COCO seeding (backend#3093)
---------------------------------------------------------------------
This template used to build ``norm_layer=misc_nn_ops.FrozenBatchNorm2d``, to
reproduce the checkpoint path's key set exactly so a future COCO seed could
strict-load. That was wrong in the regime the platform actually runs: frozen
BN at construction holds ``weight=1``, ``bias=0``, ``running_mean=0``,
``running_var=1``, so on a ``weights=None`` build it computes
``(x - 0) / sqrt(1 + eps) * 1 + 0`` -- a bit-exact identity. Its buffers are
meaningful only after a checkpoint loads real statistics into them, and NO OD
seed is hosted (backend#3055 is blocked on the store decision in
backend#2659). So every run of this template to date trained with no backbone
normalisation at all.

GroupNorm normalises per sample, so it is correct with no checkpoint and adds
no running statistics for the averaging service to ship each federated round
-- both halves of the constraint that produced frozen BN in the first place.

⚠️ WHAT THIS COSTS, EXPLICITLY. A torchvision COCO checkpoint's BN running
statistics have nowhere to go in a GroupNorm tree, so the follow-up seeding
described above can no longer be done from ``download.pytorch.org`` weights;
``tools/prep_offline_weights.py`` fails loudly on the strict load rather than
producing a mismatched dump. The build is 192 state_dict tensors under
torchvision 0.26.0, against the 284 of the frozen-BN key set this file used to
reproduce -- GroupNorm drops the two running-statistic buffers per norm layer
and keeps ``weight``/``bias``, now as parameters rather than buffers. Whether
this template is seeded from a different source or stays random-initialised
belongs to backend#2659 / backend#3055, not to this file. The trade taken here
is a real defect on every run today against a hypothetical benefit that is
blocked.

The backbone is still assembled explicitly rather than via the builder, but
for a different reason than before: the builder's norm is not this template's
norm, and it unfreezes all backbone stages rather than the last three.

Verified against torchvision 0.26.0 (the engine pin, ``tools/requirements-engine-pin.txt``).
"""
from torch import nn
from torchvision.models import mobilenet_v3_large
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _mobilenet_extractor
from torchvision.models.detection.faster_rcnn import FasterRCNN

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("roi_heads.box_predictor.bbox_pred.", "roi_heads.box_predictor.cls_score.")

framework = "pytorch"
model_type = "torchvision_detection"
main_method = "MyModel"
license = "BSD-3-Clause"
# min_size=320 below, so 320 is what the model actually sees — declaring more
# would be resized away, declaring less would be upscaled back.
image_size = 320
batch_size = 16
output_classes = 12
category = "object_detection"


def _group_norm(channels):
    """GroupNorm with the largest group count ``<= 32`` that divides ``channels``.

    The backbone norm for a from-scratch build (backend#3093). This template
    used ``FrozenBatchNorm2d``, which at construction holds ``weight=1``,
    ``bias=0``, ``running_mean=0``, ``running_var=1`` and therefore computes
    ``(x - 0) / sqrt(1 + eps) * 1 + 0`` -- its input, unchanged. Those buffers
    only mean anything once a pretrained checkpoint loads real statistics into
    them; with ``weights=None`` there is nothing to freeze and the layer is a
    no-op, so the backbone trained with no normalisation at all. GroupNorm
    normalises per sample, so it is correct with no checkpoint AND holds no
    running statistics for the averaging service to ship every federated round
    -- the reason frozen BN was reached for in the first place.

    The group count is derived, not hardcoded to 32: ``nn.GroupNorm`` requires
    ``channels % num_groups == 0``, which ResNet-50's 64..2048 all satisfy at
    32 (the canonical Wu & He setting) but MobileNetV3's 16/24/40/72/120/184
    stages do not.

    Duplicated per template on purpose -- a zoo template is uploaded as ONE
    file and cannot import a sibling (no relative imports anywhere in this
    repo). ``efficientdet_d0._norm`` and ``rtmdet_s``/``yolox_s._norm_groups``
    are the same helper for the same reason.
    """
    groups = max(g for g in range(1, 33) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background

    # No download: GroupNorm backbone (backend#3093 -- frozen BN is an
    # identity from scratch) and the FPN over the last 3 trainable stages.
    backbone = mobilenet_v3_large(
        weights=None, norm_layer=_group_norm
    )
    backbone = _mobilenet_extractor(backbone, True, 3)

    # The 320-variant defaults, restated because the backbone is assembled here
    # rather than by the builder: the small-input transform plus the shortened
    # RPN proposal list that makes this variant cheap at inference.
    anchor_sizes = ((32, 64, 128, 256, 512),) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return FasterRCNN(
        backbone,
        num_classes,
        rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
        min_size=320,
        max_size=640,
        rpn_pre_nms_top_n_test=150,
        rpn_post_nms_top_n_test=150,
        rpn_score_thresh=0.05,
    )
