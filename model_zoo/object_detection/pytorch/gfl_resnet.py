"""GFL — Generalized Focal Loss (Li et al., NeurIPS 2020) on a ResNet-50-FPN backbone. GFL makes two changes to the one-stage dense detector and both are about what the network is asked to predict rather than how it is built. Classification predicts localisation *quality* — the IoU the box will achieve — as a continuous target, which removes the separate centre-ness/IoU branch and the train/test misalignment that came with it. Box regression predicts a discrete *distribution* over each edge distance instead of a single number, which lets the network express uncertainty about an ambiguous boundary. Assignment is ATSS, so this is the ``atss_resnet`` template plus a different head and different losses.

Offline variant: the architecture is built with ``weights=None`` throughout, so
nothing is fetched from ``download.pytorch.org`` — the #199 egress lockdown
blocks it — and the template constructs anywhere, network or not. No seed is
hosted for this template yet, so it random-initialises and there is
no weight file — upload with ``weights=False``::

    user.upload_model("gfl_resnet", weights=False)

Until a dump is staged, ``tools/check_dump_coverage.py`` classifies this file
NO_SEED and the statement above is what keeps that classification honest.

The two ideas, concretely
------------------------
**Quality Focal Loss.** A standard dense detector trains classification against
a hard 1/0 label and localisation quality against a separate branch, then
multiplies the two at inference — so the ranking used at test time was never
trained. QFL makes the classification target the **IoU between the predicted
box and its assigned ground truth**, a continuous value in ``[0, 1]``, and
keeps focal loss's modulation around that soft target. One head, one score,
trained the way it is used.

**Distribution Focal Loss.** Instead of regressing ``l, t, r, b`` as four
scalars, the head emits ``reg_max + 1`` logits per side. Softmax over those
bins is a distribution over the edge distance in stride units, and the box is
its expectation — the *integral* in the paper. DFL supervises only the two bins
straddling the continuous target, weighted by how close each is, which is what
keeps the distribution sharp instead of drifting flat.

What this template reuses, and what it replaces
-----------------------------------------------
``RetinaNet.forward`` does a lot of work that has nothing to do with either
idea: the input transform, anchor generation, splitting the flat head outputs
back into pyramid levels, and mapping detections to original image coordinates.
All of that is inherited. Three things are overridden:

- ``head`` — a ``_GFLHead`` emitting ``num_classes`` and ``4 * (reg_max + 1)``
  channels, with GroupNorm towers as the paper specifies (not BatchNorm: the
  batch sizes a federated edge trains at are small enough that BN statistics
  are the dominant noise source)
- ``compute_loss`` — ATSS assignment, then QFL + DFL + GIoU
- ``postprocess_detections`` — decode the distribution before NMS

⚠️ Output ordering is load-bearing and silent if wrong
------------------------------------------------------
``AnchorGenerator`` emits anchors location-major: for each spatial position in
row-major ``(H, W)`` order, all ``A`` base anchors. The head's conv output is
``(N, A * K, H, W)``, so it has to be permuted to match before anything is
indexed by a shared anchor index. ``_GFLHead`` uses exactly torchvision's own
permutation — ``view(N, A, K, H, W) -> permute(0, 3, 4, 1, 2) -> reshape``.

Getting that wrong does not raise: the tensors are the same shape either way,
so the model trains against boxes decoded at the wrong pixels and simply learns
badly. ``tests/test_gfl_head.py`` pins the ordering by decoding a hand-built
distribution at a known anchor and checking the box lands where that anchor is.

ATSS assignment is duplicated here, on purpose
----------------------------------------------
GFL assigns with ATSS, and the code below is a copy of ``atss_resnet.py``'s.
Zoo templates are uploaded to the platform one file at a time and there is not
a single sibling or relative import anywhere in ``model_zoo/``, so a shared
helper would make both files fail on upload. The duplication is the contract,
not an oversight.

``reg_max`` bounds what a box can express
-----------------------------------------
An edge distance is representable only up to ``reg_max`` strides — 16 strides,
so 128px at P3 and 2048px at P7. ATSS sends large objects to coarse levels, so
this is not normally reached, but the target is clamped rather than left to
produce an out-of-range DFL bin. A clamped target is a slightly wrong box; an
unclamped one is an indexing error at ``scatter_``.

Verified against torchvision 0.26.0 (the engine pin, ``tools/requirements-engine-pin.txt``).
"""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.ops import boxes as box_ops
from torchvision.ops import generalized_box_iou_loss
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("head.cls_logits.",)

framework = "pytorch"
model_type = "torchvision_detection"
main_method = "MyModel"
license = "BSD-3-Clause"
# GeneralizedRCNNTransform's default is min_size=800, max_size=1333, and it
# UPSCALES anything smaller straight back to 800, so 800 is the resolution this
# model actually runs at.
image_size = 800
# The distribution head is 4 * (reg_max + 1) = 68 regression channels against
# RetinaNet's 4 per anchor, so activations are heavier than atss_resnet's at
# the same resolution. 4 rather than 8.
batch_size = 4
output_classes = 12
category = "object_detection"

#: Bins per edge distance. The head emits ``reg_max + 1`` logits per side, so
#: an edge can be placed at most ``reg_max`` strides from its anchor centre.
#: 16 is the paper's value.
REG_MAX = 16

#: Candidates drawn per pyramid level, per ground-truth box, by ATSS.
ATSS_TOPK = 9

#: Focal modulation exponent for QFL. 2.0 is the paper's beta.
QFL_BETA = 2.0

#: torchvision's ``Matcher`` background sentinel. Restated rather than imported
#: so the assignment below reads on its own; ``>= 0`` is a ground-truth index.
BACKGROUND = -1

#: FPN strides for the P3..P7 pyramid this template builds.
PYRAMID_STRIDES = (8, 16, 32, 64, 128)


class _LevelAwareAnchorGenerator(AnchorGenerator):
    """``AnchorGenerator`` that records the per-level anchor counts.

    ATSS picks candidates *per pyramid level*, and the distribution decode needs
    a per-anchor stride; both need the level boundaries of the concatenated
    anchor tensor, which ``RetinaNet.forward`` computes but does not pass to
    ``compute_loss``. Recording it here costs a few lines and duplicates no
    library logic — copying ``forward`` to thread it through would fork ~80
    lines that then drift silently on a torchvision upgrade.
    """

    def forward(self, image_list, feature_maps):
        self.num_anchors_per_level = [
            feature_map.shape[-2] * feature_map.shape[-1] * anchors_per_location
            for feature_map, anchors_per_location in zip(
                feature_maps, self.num_anchors_per_location()
            )
        ]
        return super().forward(image_list, feature_maps)


class _Scale(nn.Module):
    """Learnable per-level scalar on the regression branch.

    The towers are shared across pyramid levels, but the distances each level
    should predict differ by a factor of two per step. GFL gives each level one
    trainable multiplier so the shared tower does not have to encode that.
    """

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.scale


def _tower(in_channels, num_convs=4):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False))
        # GroupNorm, not BatchNorm: a federated edge trains at small batch
        # sizes, where BN's running statistics are the dominant noise source.
        layers.append(nn.GroupNorm(32, in_channels))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class _GFLHead(nn.Module):
    """Shared-tower head emitting class logits and an edge-distance distribution."""

    def __init__(self, in_channels, num_anchors, num_classes, num_levels, reg_max=REG_MAX):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.reg_max = reg_max
        self.cls_tower = _tower(in_channels)
        self.reg_tower = _tower(in_channels)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        self.bbox_regression = nn.Conv2d(
            in_channels, num_anchors * 4 * (reg_max + 1), 3, padding=1
        )
        self.scales = nn.ModuleList(_Scale() for _ in range(num_levels))

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Prior probability 0.01, as every focal-loss detector does: without it
        # the first steps are dominated by the ~10^4 easy negatives per image.
        nn.init.constant_(self.cls_logits.bias, -4.59511985013459)

    @staticmethod
    def _flatten(output, channels_per_anchor):
        """``(N, A * K, H, W)`` -> ``(N, H * W * A, K)``.

        Exactly torchvision's own permutation. See the module docstring: this
        has to agree with ``AnchorGenerator``'s location-major ordering, and a
        mismatch is shape-identical and therefore silent.
        """
        n, _, h, w = output.shape
        output = output.view(n, -1, channels_per_anchor, h, w)
        output = output.permute(0, 3, 4, 1, 2)
        return output.reshape(n, -1, channels_per_anchor)

    def forward(self, features):
        all_cls, all_reg = [], []
        for level, feature in enumerate(features):
            cls_logits = self.cls_logits(self.cls_tower(feature))
            bbox_regression = self.scales[level](
                self.bbox_regression(self.reg_tower(feature))
            )
            all_cls.append(self._flatten(cls_logits, self.num_classes))
            all_reg.append(self._flatten(bbox_regression, 4 * (self.reg_max + 1)))
        return {
            "cls_logits": torch.cat(all_cls, dim=1),
            "bbox_regression": torch.cat(all_reg, dim=1),
        }


def _centres(boxes):
    """``(N, 2)`` centres of ``(N, 4)`` xyxy boxes."""
    return torch.stack(
        ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2), dim=1
    )


def _atss_assign(anchors, gt_boxes, num_anchors_per_level, topk):
    """ATSS assignment for one image. Duplicated from ``atss_resnet.py`` — see
    the module docstring for why sharing is not available.

    Returns an ``int64`` tensor of shape ``(num_anchors,)`` holding, per anchor,
    the index of its assigned ground-truth box or ``BACKGROUND``.
    """
    num_anchors = anchors.shape[0]
    matched = anchors.new_full((num_anchors,), BACKGROUND, dtype=torch.int64)
    if gt_boxes.numel() == 0:
        return matched

    anchor_centres = _centres(anchors)
    gt_centres = _centres(gt_boxes)
    distances = (gt_centres[:, None, :] - anchor_centres[None, :, :]).pow(2).sum(-1).sqrt()
    ious = box_ops.box_iou(gt_boxes, anchors)

    candidate_mask = torch.zeros_like(distances, dtype=torch.bool)
    start = 0
    for level_size in num_anchors_per_level:
        end = start + level_size
        if level_size > 0:
            k = min(topk, level_size)
            _, local_idx = distances[:, start:end].topk(k, dim=1, largest=False)
            candidate_mask[:, start:end].scatter_(1, local_idx, True)
        start = end

    candidate_ious = ious.masked_fill(~candidate_mask, 0.0)
    counts = candidate_mask.sum(dim=1).clamp(min=1)
    mean = candidate_ious.sum(dim=1) / counts
    # Population deviation, computed explicitly: the unbiased estimator is NaN
    # over a single candidate, and `iou >= nan` is False everywhere, which
    # would silently leave that object with no anchors at all.
    variance = (
        (candidate_ious - mean[:, None]).pow(2).masked_fill(~candidate_mask, 0.0).sum(dim=1)
        / counts
    )
    thresholds = mean + variance.sqrt()

    inside = (
        (anchor_centres[None, :, 0] >= gt_boxes[:, None, 0])
        & (anchor_centres[None, :, 0] <= gt_boxes[:, None, 2])
        & (anchor_centres[None, :, 1] >= gt_boxes[:, None, 1])
        & (anchor_centres[None, :, 1] <= gt_boxes[:, None, 3])
    )
    positive = candidate_mask & (ious >= thresholds[:, None]) & inside

    claim = ious.masked_fill(~positive, -1.0)
    best_iou, best_gt = claim.max(dim=0)
    assigned = best_iou >= 0.0
    matched[assigned] = best_gt[assigned]
    return matched


def _anchor_strides(num_anchors_per_level, device, dtype):
    """Per-anchor stride for the concatenated anchor tensor."""
    return torch.cat(
        [
            torch.full((count,), float(stride), device=device, dtype=dtype)
            for count, stride in zip(num_anchors_per_level, PYRAMID_STRIDES)
        ]
    )


def _integral(distribution_logits, reg_max=REG_MAX):
    """Expectation of the per-edge distribution — the paper's integral.

    ``(..., 4 * (reg_max + 1))`` logits -> ``(..., 4)`` distances in stride units.
    """
    shape = distribution_logits.shape[:-1]
    logits = distribution_logits.reshape(*shape, 4, reg_max + 1)
    probabilities = logits.softmax(dim=-1)
    bins = torch.arange(reg_max + 1, device=logits.device, dtype=probabilities.dtype)
    return (probabilities * bins).sum(dim=-1)


def _distance_to_box(centres, distances, strides):
    """``l, t, r, b`` in stride units, measured from ``centres``, to xyxy pixels."""
    scaled = distances * strides[:, None]
    return torch.stack(
        (
            centres[:, 0] - scaled[:, 0],
            centres[:, 1] - scaled[:, 1],
            centres[:, 0] + scaled[:, 2],
            centres[:, 1] + scaled[:, 3],
        ),
        dim=-1,
    )


def _box_to_distance(centres, boxes, strides, reg_max=REG_MAX):
    """Inverse of ``_distance_to_box``, clamped into the representable range.

    Clamped rather than trusted: an edge further than ``reg_max`` strides from
    its anchor centre has no bin, and an unclamped target is an out-of-range
    index at ``scatter_`` rather than a slightly wrong box.
    """
    distances = torch.stack(
        (
            centres[:, 0] - boxes[:, 0],
            centres[:, 1] - boxes[:, 1],
            boxes[:, 2] - centres[:, 0],
            boxes[:, 3] - centres[:, 1],
        ),
        dim=-1,
    )
    # 1e-3 below reg_max so the upper bin index stays inside the tensor.
    return (distances / strides[:, None]).clamp(min=0.0, max=reg_max - 1e-3)


def _distribution_focal_loss(distribution_logits, target, reg_max=REG_MAX):
    """DFL — cross-entropy on the two bins straddling a continuous target.

    ``distribution_logits`` is ``(P, 4, reg_max + 1)``, ``target`` is ``(P, 4)``
    in stride units. Each side's loss is the linear interpolation of the
    negative log-likelihood of its two neighbouring bins, so the distribution is
    pushed to concentrate there rather than anywhere that averages correctly.
    """
    lower = target.floor().long()
    upper = lower + 1
    weight_upper = target - lower.to(target.dtype)
    weight_lower = 1.0 - weight_upper

    log_probabilities = distribution_logits.log_softmax(dim=-1)
    loss_lower = -log_probabilities.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
    loss_upper = -log_probabilities.gather(-1, upper.clamp(max=reg_max).unsqueeze(-1)).squeeze(-1)
    return (loss_lower * weight_lower + loss_upper * weight_upper).sum(dim=-1)


def _quality_focal_loss(cls_logits, target_scores, beta=QFL_BETA):
    """QFL — focal loss against a *continuous* quality target.

    ``target_scores`` is dense and mostly zero: for an assigned anchor it holds
    the IoU its predicted box achieved, in the ground-truth class column. The
    modulation ``|target - sigmoid|**beta`` reduces to standard focal loss when
    the target is 0 or 1, which is what makes the soft target a generalisation
    rather than a different loss.
    """
    probabilities = cls_logits.sigmoid()
    modulation = (target_scores - probabilities).abs().pow(beta)
    return (
        F.binary_cross_entropy_with_logits(cls_logits, target_scores, reduction="none")
        * modulation
    ).sum()


class _GFL(RetinaNet):
    """RetinaNet's plumbing with GFL's head, losses and box decode."""

    def __init__(self, *args, reg_max=REG_MAX, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_max = reg_max

    def _level_split(self):
        split = getattr(self.anchor_generator, "num_anchors_per_level", None)
        if split is None:
            raise RuntimeError(
                "GFL needs the per-level anchor split, which "
                "_LevelAwareAnchorGenerator records during forward; none was "
                "recorded. Was the model built with a plain AnchorGenerator?"
            )
        return split

    def compute_loss(self, targets, head_outputs, anchors):
        split = self._level_split()
        cls_logits = head_outputs["cls_logits"]
        bbox_regression = head_outputs["bbox_regression"]

        losses_qfl, losses_dfl, losses_box = [], [], []
        total_positives = 0

        for image_index, (anchors_per_image, targets_per_image) in enumerate(
            zip(anchors, targets)
        ):
            strides = _anchor_strides(
                split, anchors_per_image.device, anchors_per_image.dtype
            )
            centres = _centres(anchors_per_image)
            logits_per_image = cls_logits[image_index]
            regression_per_image = bbox_regression[image_index]

            matched = _atss_assign(
                anchors_per_image, targets_per_image["boxes"], split, ATSS_TOPK
            )
            foreground = matched >= 0
            num_positives = int(foreground.sum())
            total_positives += num_positives

            # QFL's target is dense; every anchor contributes, and a negative
            # anchor's target is an all-zero row rather than an omitted one.
            target_scores = torch.zeros_like(logits_per_image)

            if num_positives:
                matched_gt = matched[foreground]
                gt_boxes = targets_per_image["boxes"][matched_gt]
                gt_labels = targets_per_image["labels"][matched_gt]

                positive_regression = regression_per_image[foreground]
                positive_centres = centres[foreground]
                positive_strides = strides[foreground]

                predicted_distances = _integral(positive_regression, self.reg_max)
                predicted_boxes = _distance_to_box(
                    positive_centres, predicted_distances, positive_strides
                )

                # The quality target is the IoU the box actually achieved, and
                # it is DETACHED: it is a label, so gradient must reach the
                # boxes through the box loss, never through the classifier.
                with torch.no_grad():
                    quality = box_ops.box_iou(predicted_boxes, gt_boxes).diagonal().clamp(min=0)
                target_scores[foreground, gt_labels] = quality

                # Box loss weighted by quality, as the paper specifies: a
                # confident anchor's geometry matters more than a marginal one's.
                weight = quality
                normaliser = weight.sum().clamp(min=1e-6)
                losses_box.append(
                    (
                        generalized_box_iou_loss(predicted_boxes, gt_boxes, reduction="none")
                        * weight
                    ).sum()
                    / normaliser
                )

                target_distances = _box_to_distance(
                    positive_centres, gt_boxes, positive_strides, self.reg_max
                )
                losses_dfl.append(
                    (
                        _distribution_focal_loss(
                            positive_regression.reshape(-1, 4, self.reg_max + 1),
                            target_distances,
                            self.reg_max,
                        )
                        * weight
                    ).sum()
                    / normaliser
                )

            losses_qfl.append(_quality_focal_loss(logits_per_image, target_scores))

        # Normalise QFL by the positive count across the batch, the convention
        # every focal-loss detector in this family uses.
        denominator = max(1, total_positives)
        zero = cls_logits.sum() * 0.0
        return {
            "classification": torch.stack(losses_qfl).sum() / denominator,
            "bbox_regression": (
                torch.stack(losses_box).mean() if losses_box else zero
            ),
            "distribution_focal": (
                torch.stack(losses_dfl).mean() if losses_dfl else zero
            ),
        }

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        """Decode the edge distributions, then RetinaNet's usual trim and NMS.

        ⚠️ Note the nesting, which is easy to get backwards and was: after
        ``RetinaNet.forward`` splits them, ``head_outputs[k]`` is a list over
        **pyramid levels**, each entry shaped ``(N, anchors_in_level, K)``,
        while ``anchors`` is a list over **images** of lists over levels. So the
        image index is an index *into* each level tensor, not a position in the
        outer list. Iterating the outer list as if it were images silently
        processes level 0 of image 0 only — ``zip`` truncates to the shorter
        sequence rather than raising — and that is invisible on a freshly built
        model, because the prior-0.01 classification bias puts every score
        below ``score_thresh`` so no box is produced to be wrong.
        """
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]
        detections = []

        for index, image_shape in enumerate(image_shapes):
            logits_per_image = [level[index] for level in class_logits]
            regression_per_image = [level[index] for level in box_regression]
            anchors_per_image = anchors[index]

            image_boxes, image_scores, image_labels = [], [], []

            for level, (logits, regression, level_anchors) in enumerate(
                zip(logits_per_image, regression_per_image, anchors_per_image)
            ):
                num_classes = logits.shape[-1]
                scores = logits.sigmoid().flatten()

                # Same pre-NMS trimming RetinaNet does: threshold, then keep at
                # most topk_candidates per level.
                keep = scores > self.score_thresh
                scores = scores[keep]
                keep_idxs = torch.where(keep)[0]
                num_topk = min(self.topk_candidates, scores.numel())
                scores, sort_idx = scores.topk(num_topk)
                keep_idxs = keep_idxs[sort_idx]

                anchor_idxs = torch.div(keep_idxs, num_classes, rounding_mode="floor")
                labels = keep_idxs % num_classes

                selected_anchors = level_anchors[anchor_idxs]
                # This level's OWN stride: an edge distance is in stride units,
                # so decoding a level-4 prediction with P3's stride shrinks the
                # box 16-fold while still producing valid xyxy.
                stride = float(PYRAMID_STRIDES[level])
                strides = selected_anchors.new_full((selected_anchors.shape[0],), stride)
                distances = _integral(regression[anchor_idxs], self.reg_max)
                boxes = _distance_to_box(_centres(selected_anchors), distances, strides)
                boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

                image_boxes.append(boxes)
                image_scores.append(scores)
                image_labels.append(labels)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            keep = box_ops.batched_nms(
                image_boxes, image_scores, image_labels, self.nms_thresh
            )[: self.detections_per_img]
            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )
        return detections


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background

    # weights=None: architecture only, no download (the #199 egress lockdown
    # blocks download.pytorch.org). FrozenBatchNorm2d and trainable_layers=3
    # match the retinanet and atss_resnet templates.
    backbone = resnet50(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(
        backbone,
        trainable_layers=3,
        returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(256, 256),
    )

    # P3..P7 with ONE square anchor per location at 8x the level stride, as
    # ATSS establishes: with adaptive assignment the tiling stops mattering,
    # and GFL inherits that. The anchor is a reference point for the edge
    # distances, not a box prior to regress from.
    anchor_generator = _LevelAwareAnchorGenerator(
        sizes=tuple((8 * stride,) for stride in PYRAMID_STRIDES),
        aspect_ratios=((1.0,),) * len(PYRAMID_STRIDES),
    )

    head = _GFLHead(
        backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        num_classes,
        num_levels=len(PYRAMID_STRIDES),
    )

    return _GFL(
        backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        head=head,
    )
