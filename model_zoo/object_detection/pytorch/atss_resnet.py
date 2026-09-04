"""ATSS — Adaptive Training Sample Selection (Zhang et al., CVPR 2020) on a ResNet-50-FPN backbone. ATSS's finding is that the gap between anchor-based and anchor-free detectors is not the anchors, it is how training samples are assigned: replace RetinaNet's fixed IoU thresholds with a per-object adaptive threshold and one anchor per location, and RetinaNet matches FCOS. Same backbone, same head, same losses as the retinanet template — only the assignment differs, which makes this the cheapest real accuracy gain in the roster and a clean baseline for federated assignment experiments.

Offline variant: the architecture is built with ``weights=None`` throughout, so
nothing is fetched from ``download.pytorch.org`` — the #199 egress lockdown
blocks it — and the template constructs anywhere, network or not. No seed is
hosted for this template yet, so it random-initialises and there is
no weight file — upload with ``weights=False``::

    user.upload_model("atss_resnet", weights=False)

Until a dump is staged, ``tools/check_dump_coverage.py`` classifies this file
NO_SEED and the statement above is what keeps that classification honest.

What ATSS changes, and where the seam is
----------------------------------------
torchvision's ``RetinaNet.compute_loss`` assigns anchors to ground truth with a
fixed-threshold ``Matcher`` — IoU >= 0.5 positive, < 0.4 background, the band
between ignored — and then delegates to ``self.head.compute_loss``. ATSS
replaces only the first half. For each ground-truth box:

1. take the ``topk`` anchors per pyramid level whose centres are closest to the
   object centre (L2), so every level gets a say regardless of object size;
2. compute IoU between the object and just those candidates;
3. the threshold is ``mean + std`` of those IoUs — adaptive per object, which
   is the whole idea: a large clear object gets a high bar, a small ambiguous
   one a low bar, and neither is hand-tuned;
4. keep candidates above that threshold **whose centre also falls inside the
   object**;
5. resolve an anchor claimed by two objects in favour of the higher IoU.

``proposal_matcher`` is a constructor argument on ``RetinaNet``, so swapping the
matcher looks like the natural seam — but it is not sufficient, and that is
worth recording. A ``Matcher`` is called as ``matcher(match_quality_matrix)``
and receives **only** the IoU matrix: no anchor coordinates, so it cannot do
step 4, and no per-level boundaries, so it cannot do step 1. ATSS therefore has
to be applied one level up, in ``compute_loss``, which is what
``_ATSSRetinaNet`` below overrides. The head's own ``compute_loss`` is reused
untouched, so the classification and regression losses are exactly RetinaNet's
and the only variable is the assignment.

Getting the per-level split without copying ``forward``
-------------------------------------------------------
Step 1 needs to know how many of the concatenated anchors belong to each
pyramid level. ``RetinaNet.forward`` computes that as
``num_anchors_per_level`` but does not pass it to ``compute_loss``, and copying
``forward`` to thread it through would fork ~80 lines of library code that
would then silently drift on a torchvision upgrade.

``_LevelAwareAnchorGenerator`` records it instead: the anchor generator already
receives the feature maps, so their spatial sizes times the anchors-per-location
*is* the split. Eight lines, no library logic duplicated, and it is recomputed
on every forward so it cannot go stale against a changed input size.

One anchor per location, on purpose
-----------------------------------
The paper's second claim is that with adaptive assignment the anchor tiling
stops mattering, and it demonstrates this with a **single** square anchor per
location (scale 8x stride) against RetinaNet's nine. This template follows
that: ``aspect_ratios=((1.0,),)`` per level. It is not a simplification — a
9-anchor ATSS is a different (and per the paper no better) model, and the
single anchor is what makes the head a third of RetinaNet's width.

Contract
--------
``model(images, targets)`` returns RetinaNet's loss dict (``classification``,
``bbox_regression``); ``model(images)`` returns ``List[Dict]`` of pixel-xyxy
``boxes``/``scores``/``labels`` via RetinaNet's unmodified postprocessing. NMS
is unchanged — ATSS is a training-time change only, so nothing about inference
differs from the ``retinanet`` template.

Verified against torchvision 0.26.0 (the engine pin, ``tools/requirements-engine-pin.txt``).
"""
import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead
from torchvision.ops import boxes as box_ops
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
# GeneralizedRCNNTransform's default is min_size=800, max_size=1333, and it
# UPSCALES anything smaller straight back to 800, so 800 is the resolution this
# model actually runs at.
image_size = 800
# Matches the retinanet template: same backbone and same head width per anchor,
# and ATSS's single anchor per location makes the head narrower, not wider.
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


#: Candidates drawn per pyramid level, per ground-truth box. 9 is the paper's
#: value and its ablation shows the result is flat between roughly 7 and 17, so
#: this is not a knob worth exposing on the template.
ATSS_TOPK = 9

#: torchvision's ``Matcher`` sentinels, restated rather than imported from
#: ``detection._utils``: the head's ``compute_loss`` treats ``>= 0`` as a
#: ground-truth index, ``-1`` as background, and ``-2`` as ignore, and this
#: file must produce that same encoding. Naming them here keeps the assignment
#: readable and pins the contract this override has to honour.
BACKGROUND = -1


class _LevelAwareAnchorGenerator(AnchorGenerator):
    """``AnchorGenerator`` that records how many anchors each level produced.

    ATSS picks its candidates *per pyramid level*, so the assignment needs the
    level boundaries of the concatenated anchor tensor. See the module docstring
    for why this is recorded here rather than threaded through ``forward``.
    """

    def forward(self, image_list, feature_maps):
        self.num_anchors_per_level = [
            feature_map.shape[-2] * feature_map.shape[-1] * anchors_per_location
            for feature_map, anchors_per_location in zip(
                feature_maps, self.num_anchors_per_location()
            )
        ]
        return super().forward(image_list, feature_maps)


def _centres(boxes):
    """``(N, 2)`` centres of ``(N, 4)`` xyxy boxes."""
    return torch.stack(
        ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2), dim=1
    )


def _atss_assign(anchors, gt_boxes, num_anchors_per_level, topk):
    """ATSS assignment for one image.

    Returns an ``int64`` tensor of shape ``(num_anchors,)`` holding, per anchor,
    the index of the ground-truth box it is assigned to, or ``BACKGROUND``. The
    ignore sentinel is deliberately never produced — ATSS partitions anchors
    into positive and negative with no ambiguous band, which is one of the
    things it removes relative to fixed thresholds.
    """
    num_anchors = anchors.shape[0]
    matched = anchors.new_full((num_anchors,), BACKGROUND, dtype=torch.int64)
    if gt_boxes.numel() == 0:
        return matched

    anchor_centres = _centres(anchors)
    gt_centres = _centres(gt_boxes)

    # (num_gt, num_anchors) centre distances, and the IoU matrix on the same axes.
    distances = (gt_centres[:, None, :] - anchor_centres[None, :, :]).pow(2).sum(-1).sqrt()
    ious = box_ops.box_iou(gt_boxes, anchors)

    # Step 1 — the topk closest candidates WITHIN each level, so a level with
    # few anchors cannot be crowded out by a finer one.
    candidate_mask = torch.zeros_like(distances, dtype=torch.bool)
    start = 0
    for level_size in num_anchors_per_level:
        end = start + level_size
        if level_size > 0:
            k = min(topk, level_size)
            _, local_idx = distances[:, start:end].topk(k, dim=1, largest=False)
            candidate_mask[:, start:end].scatter_(1, local_idx, True)
        start = end

    # Steps 2-3 — the threshold is mean + std of the candidate IoUs, per object.
    # Masked statistics: std over a single candidate is NaN under the unbiased
    # estimator, so it is computed explicitly and zero-filled.
    candidate_ious = ious.masked_fill(~candidate_mask, 0.0)
    counts = candidate_mask.sum(dim=1).clamp(min=1)
    mean = candidate_ious.sum(dim=1) / counts
    variance = (
        (candidate_ious - mean[:, None]).pow(2).masked_fill(~candidate_mask, 0.0).sum(dim=1)
        / counts
    )
    thresholds = mean + variance.sqrt()

    # Step 4 — above threshold AND anchor centre inside the object. The centre
    # test is what stops a high-IoU anchor sitting outside a thin diagonal
    # object from being called positive.
    inside = (
        (anchor_centres[None, :, 0] >= gt_boxes[:, None, 0])
        & (anchor_centres[None, :, 0] <= gt_boxes[:, None, 2])
        & (anchor_centres[None, :, 1] >= gt_boxes[:, None, 1])
        & (anchor_centres[None, :, 1] <= gt_boxes[:, None, 3])
    )
    positive = candidate_mask & (ious >= thresholds[:, None]) & inside

    # Step 5 — an anchor claimed by two objects goes to the higher IoU.
    # -1.0 rather than -inf as the sentinel: the max over an all-negative column
    # stays finite, so the ">= 0" test below is the only thing deciding.
    claim = ious.masked_fill(~positive, -1.0)
    best_iou, best_gt = claim.max(dim=0)
    assigned = best_iou >= 0.0
    matched[assigned] = best_gt[assigned]
    return matched


class _ATSSRetinaNet(RetinaNet):
    """RetinaNet with ATSS assignment. Only ``compute_loss`` differs."""

    def compute_loss(self, targets, head_outputs, anchors):
        num_anchors_per_level = getattr(self.anchor_generator, "num_anchors_per_level", None)
        # Belt and braces: a forward has always run by the time compute_loss is
        # reached, so this is unreachable in practice — but assigning every
        # anchor to one level would silently degrade ATSS to global topk, which
        # is a wrong model that still trains. Fail instead.
        if num_anchors_per_level is None:
            raise RuntimeError(
                "ATSS assignment needs the per-level anchor split, which "
                "_LevelAwareAnchorGenerator records during forward; none was "
                "recorded. Was the model built with a plain AnchorGenerator?"
            )

        matched_idxs = [
            _atss_assign(
                anchors_per_image,
                targets_per_image["boxes"],
                num_anchors_per_level,
                ATSS_TOPK,
            )
            for anchors_per_image, targets_per_image in zip(anchors, targets)
        ]
        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background

    # weights=None: architecture only, no download (the #199 egress lockdown
    # blocks download.pytorch.org). GroupNorm (backend#3093) and
    # trainable_layers=3 match the rest of this family. GroupNorm, not
    # FrozenBatchNorm2d: frozen BN answers the federated-averaging problem --
    # BN running statistics average badly across non-IID clients -- but on a
    # weights=None build its weight=1 / bias=0 / running_mean=0 /
    # running_var=1 buffers make it a bit-exact identity, so this backbone
    # trained with no normalisation at all. GroupNorm normalises per sample:
    # correct with no checkpoint, and no running statistics to average.
    backbone = resnet50(weights=None, norm_layer=_group_norm)
    backbone = _resnet_fpn_extractor(
        backbone,
        trainable_layers=3,
        returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(256, 256),
    )

    # P3..P7 with ONE square anchor per location at 8x the level stride --
    # strides 8/16/32/64/128, so sizes 64/128/256/512/1024. See the module
    # docstring: the single anchor is the paper's design, not a shortcut.
    anchor_generator = _LevelAwareAnchorGenerator(
        sizes=((64,), (128,), (256,), (512,), (1024,)),
        aspect_ratios=((1.0,),) * 5,
    )

    head = RetinaNetHead(
        backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        num_classes,
        norm_layer=None,
    )

    return _ATSSRetinaNet(
        backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        head=head,
    )
