"""RetinaNet (Meta, ICCV 2017). Focal loss + one-stage anchor design — the canonical one-stage detector still widely deployed for production. Pairs with Faster R-CNN as the standard two-stage / one-stage comparison.

Offline variant: the architecture is built without any checkpoint download,
so the template constructs anywhere, network or not. The pretrained
ResNet50-FPN tensors are delivered from the tracebloc model store as the
training seed: upload the matched ``retinanet_weights.pkl`` sitting next to
this file via ``upload_model(..., weights=True)``, and the platform loads it
with ``load_state_dict(strict=True)`` after ``MyModel()`` builds this
architecture. See ``tools/prep_offline_weights.py`` for producing and
verifying that matched weight file.

The backbone is assembled explicitly instead of via the high-level
``retinanet_resnet50_fpn(weights=None)`` builder, for the reasons documented
in ``faster_rcnn_resnet.py``: with no weights requested the builder unfreezes
all five backbone stages instead of the last three, and it picks the backbone
norm off the same flag. Building the backbone directly keeps the three
trainable stages and the P6/P7 pyramid under explicit control. The norm is no
longer the checkpoint's — see below. Verified against torchvision 0.27.

Backbone norm: GroupNorm, and it is NOT the checkpoint's norm (backend#3093)
----------------------------------------------------------------------------
This template used to build ``norm_layer=misc_nn_ops.FrozenBatchNorm2d`` to
reproduce torchvision's checkpoint-path architecture key-exactly. That was
wrong in the regime the platform actually runs: frozen BN at construction
holds ``weight=1``, ``bias=0``, ``running_mean=0``, ``running_var=1``, so on a
``weights=None`` build it computes ``(x - 0) / sqrt(1 + eps) * 1 + 0`` -- a
bit-exact identity. Its buffers are meaningful only after a checkpoint loads
real statistics into them, and NO OD seed is hosted (backend#3055 is blocked
on the store decision in backend#2659). So every run of this template to date
trained with no backbone normalisation at all; activations were measured at
sigma ~= 24 where a live BN gives ~= 3.

GroupNorm normalises per sample, so it is correct with no checkpoint and adds
no running statistics for the averaging service to ship each federated round
-- both halves of the constraint that produced frozen BN in the first place.

⚠️ WHAT THIS COSTS, EXPLICITLY. A torchvision COCO checkpoint's BN running
statistics have nowhere to go in a GroupNorm tree, so this template can no
longer strict-load a seed prepped from ``download.pytorch.org`` weights, and
the prepped-but-unhosted dump named in the header above is invalidated by this
change. ``tools/prep_offline_weights.py`` fails loudly on the strict load
rather than producing a mismatched dump, which is the right place for it to
fail. Whether this template keeps a seed declaration, drops it, or re-sources
one belongs to backend#2659 / backend#3055 -- not to this file. The trade taken
here is a real defect on every run today against a hypothetical benefit that
is blocked.
"""
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.retinanet import (
    RetinaNet,
    RetinaNetClassificationHead,
)
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("head.classification_head.cls_logits.",)

framework = "pytorch"
model_type = "torchvision_detection"
main_method = "MyModel"
license = "BSD-3-Clause"
image_size = 800
batch_size = 8
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
    # identity from scratch), FPN over the last 3 stages (P2 skipped, per the
    # paper) with extra P6/P7 levels, and the stock 91-class COCO head
    # (replaced below, as before).
    backbone = resnet50(weights=None, norm_layer=_group_norm)
    backbone = _resnet_fpn_extractor(
        backbone,
        trainable_layers=3,
        returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(256, 256),
    )
    model = RetinaNet(backbone, num_classes=91)

    # Replace the classification head. Its keys/shapes are unchanged by the
    # norm swap; SEED_EXCLUDED_PREFIXES above still names exactly this head.
    in_channels = model.backbone.out_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels, num_anchors, num_classes
    )
    return model
