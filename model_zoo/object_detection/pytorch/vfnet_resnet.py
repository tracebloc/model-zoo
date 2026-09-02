"""VarifocalNet (VFNet, Zhang et al., CVPR 2021) on a ResNet-50-FPN backbone. VFNet is the roster's answer to a specific question: given that a dense detector's ranking should reflect localisation quality, how should the classifier be *trained* to produce that ranking, and how can the box it is ranking be made good enough to rank? Two answers, and they are independent contributions — the **Varifocal Loss**, which is asymmetric where focal loss is symmetric, and **star-shaped box refinement**, which spends one deformable convolution sampling the box a location already predicted and correcting it.

Offline variant: the architecture is built with ``weights=None`` throughout, so
nothing is fetched from ``download.pytorch.org`` — the #199 egress lockdown
blocks it — and the template constructs anywhere, network or not. No seed is
hosted for this template yet, so it random-initialises and there is
no weight file — upload with ``weights=False``::

    user.upload_model("vfnet_resnet", weights=False)

Until a dump is staged, ``tools/check_dump_coverage.py`` classifies this file
NO_SEED and the statement above is what keeps that classification honest.

The Varifocal Loss — asymmetric on purpose
------------------------------------------
Focal loss down-weights *every* easy example. VFL down-weights only the
negatives::

    positive (q > 0):  -q * (q * log(p) + (1 - q) * log(1 - p))
    negative (q = 0):  -alpha * p ** gamma * log(1 - p)

The positive term carries **no** ``(1 - p) ** gamma`` modulation, so a
well-localised anchor is never discounted for being easy — it is weighted by
``q``, its IoU, which is precisely the quantity the ranking is supposed to
express. The negative term keeps the focal modulation, because there are ~10^4
of them per image and they do need discounting. That asymmetry is the whole
idea, and it is why VFL is not focal loss with a soft target: ``gfl_resnet``'s
Quality Focal Loss modulates both sides symmetrically, and the two behave
differently on a confident positive.

Star-shaped box refinement — and it is genuinely deformable
-----------------------------------------------------------
The head predicts distances once, cheaply. Those distances define a box, and
nine points on that box — its centre, four edge midpoints and four corners —
are where the features that describe it actually live. A single 3x3 deformable
convolution whose offsets are *derived from the predicted box* samples exactly
those nine points, and its output predicts a per-edge scale factor that
corrects the first estimate.

⚠️ **This uses a real deformable convolution, and that is a deliberate,
verified choice.** ``torchvision.ops.deform_conv2d`` ships compiled in the
engine's pinned wheel and works on CPU — forward and backward, nothing to build.
It is **not** the ``MultiScaleDeformableAttention`` op the RFC warns about for
the Tier 3 DETR family: that one is a separate custom CUDA extension that the
slim ``cv`` image cannot compile. Plain deformable convolution is free here.

Two details of that op were measured rather than assumed, because both are
silent if wrong:

- **Offsets are ``(y, x)`` pairs.** Verified by feeding a one-hot input through
  an identity kernel and shifting one tap: moving the *second* channel of a pair
  by ``+2`` moves the sample in x. An ``(x, y)`` ordering would transpose every
  sampling point — no error, just a head reading the wrong pixels.
- **Offsets are relative to the kernel's own 3x3 grid**, so the base grid
  ``[-1, 0, 1] x [-1, 0, 1]`` has to be subtracted from the absolute point
  positions. Without that subtraction every point is displaced by its grid
  position, which for the centre tap happens to be zero — so the model half
  works and is hard to spot.

``GRADIENT_MUL`` partially detaches the offsets (0.1 of the gradient flows).
The offsets are a *function of* the box prediction, so without this the box
branch is trained mostly to make the sampling convenient rather than to be
correct — the reference implementation's value, kept.

Both boxes are supervised
-------------------------
The initial distances and the refined distances each get a GIoU loss. The
refinement can only correct what the first estimate roughly located, so leaving
the first unsupervised makes the second's job impossible. The classification
target is the IoU of the **refined** box, since that is what inference emits.

What is reused
--------------
``RetinaNet.forward`` is inherited whole — transform, anchor generation,
per-level splitting, mapping detections back. ``head``, ``compute_loss`` and
``postprocess_detections`` are overridden. Assignment is ATSS, duplicated here
because zoo templates cannot import siblings; its guard tests are duplicated
alongside it in ``tests/test_vfnet_head.py`` for the same reason.

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
from torchvision.ops import deform_conv2d, generalized_box_iou_loss
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("head.cls_star.",)

framework = "pytorch"
model_type = "torchvision_detection"
main_method = "MyModel"
license = "BSD-3-Clause"
# GeneralizedRCNNTransform's default is min_size=800, max_size=1333, and it
# UPSCALES anything smaller straight back to 800, so 800 is the resolution this
# model actually runs at (backend#3058).
image_size = 800
# Two deformable convolutions per pyramid level on top of the shared towers.
batch_size = 4
output_classes = 12
category = "object_detection"

#: Varifocal Loss parameters. alpha weights the negative term, gamma is its
#: focal exponent. The positive term is deliberately unmodulated.
VFL_ALPHA = 0.75
VFL_GAMMA = 2.0

#: Fraction of the gradient allowed to flow into the box prediction through the
#: deformable offsets. The reference implementation's value.
GRADIENT_MUL = 0.1

#: Candidates drawn per pyramid level, per ground-truth box, by ATSS.
ATSS_TOPK = 9

#: torchvision's ``Matcher`` background sentinel; ``>= 0`` is a GT index.
BACKGROUND = -1

#: FPN strides for the P3..P7 pyramid this template builds.
PYRAMID_STRIDES = (8, 16, 32, 64, 128)


class _LevelAwareAnchorGenerator(AnchorGenerator):
    """``AnchorGenerator`` that records the per-level anchor counts.

    ATSS picks candidates per pyramid level and the distance decode needs a
    per-anchor stride; both need the level boundaries of the concatenated anchor
    tensor, which ``RetinaNet.forward`` computes but does not pass down.
    Recording it here duplicates no library logic.
    """

    def forward(self, image_list, feature_maps):
        self.num_anchors_per_level = [
            feature_map.shape[-2] * feature_map.shape[-1] * anchors_per_location
            for feature_map, anchors_per_location in zip(
                feature_maps, self.num_anchors_per_location()
            )
        ]
        return super().forward(image_list, feature_maps)


def _tower(channels, num_convs=3):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(channels, channels, 3, padding=1, bias=False))
        # GroupNorm: a federated edge trains at batch sizes where BatchNorm's
        # running statistics are the dominant noise source.
        layers.append(nn.GroupNorm(32, channels))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class _StarDeformable(nn.Module):
    """3x3 deformable convolution sampling the nine points of a predicted box.

    The offsets are not learned; they are *computed* from the box the head has
    already predicted, so the kernel reads the pixels that describe that box.
    """

    #: The kernel's own 3x3 sampling grid as ``(y, x)`` pairs, in the order
    #: ``deform_conv2d`` expects. Offsets are RELATIVE to this, so it must be
    #: subtracted from the absolute point positions.
    BASE_GRID = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )

    def __init__(self, channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.normal_(self.weight, std=0.01)
        base = torch.tensor(
            [value for pair in self.BASE_GRID for value in pair], dtype=torch.float32
        )
        self.register_buffer("base_offset", base.view(1, 18, 1, 1))

    def star_offsets(self, distances):
        """``(N, 4, H, W)`` distances in stride units -> ``(N, 18, H, W)`` offsets.

        The nine sampled points are the box's centre, its four edge midpoints
        and its four corners, expressed relative to the location itself. Written
        out explicitly rather than looped: the mapping from ``(l, t, r, b)`` to
        nine ``(y, x)`` pairs in the kernel's raster order is the part that is
        silent if wrong, so it reads as a table.
        """
        # Partially detached: the offsets are a function of the box prediction,
        # and letting the full gradient through trains the box to make sampling
        # convenient rather than to be correct.
        distances = (
            1 - GRADIENT_MUL
        ) * distances.detach() + GRADIENT_MUL * distances

        left = distances[:, 0]
        top = distances[:, 1]
        right = distances[:, 2]
        bottom = distances[:, 3]

        offsets = distances.new_zeros(
            distances.shape[0], 18, distances.shape[2], distances.shape[3]
        )
        #      kernel tap        y            x
        offsets[:, 0] = -top;    offsets[:, 1] = -left      # top-left corner
        offsets[:, 2] = -top                                # top-centre  (x = 0)
        offsets[:, 4] = -top;    offsets[:, 5] = right      # top-right corner
        offsets[:, 7] = -left                               # left-centre (y = 0)
        #      tap 4 is the location itself: both offsets stay 0
        offsets[:, 11] = right                              # right-centre(y = 0)
        offsets[:, 12] = bottom; offsets[:, 13] = -left     # bottom-left corner
        offsets[:, 14] = bottom                             # bottom-centre
        offsets[:, 16] = bottom; offsets[:, 17] = right     # bottom-right corner

        # Relative to the kernel's own grid, not absolute.
        return offsets - self.base_offset

    def forward(self, feature, distances):
        offsets = self.star_offsets(distances)
        return deform_conv2d(feature, offsets, self.weight, self.bias, padding=1)


class _VFNetHead(nn.Module):
    """Shared towers, an initial distance prediction, then star-shaped refinement."""

    def __init__(self, in_channels, num_anchors, num_classes, num_levels):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.cls_tower = _tower(in_channels)
        self.reg_tower = _tower(in_channels)

        self.reg_initial = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)
        self.reg_scales = nn.ParameterList(
            nn.Parameter(torch.ones(1)) for _ in range(num_levels)
        )
        # Refinement: sample the predicted box, predict a per-edge SCALE.
        self.reg_refine = _StarDeformable(in_channels, num_anchors * 4)
        # Classification also reads the box's own nine points, which is what
        # makes the score IoU-aware rather than merely co-located.
        self.cls_star = _StarDeformable(in_channels, num_anchors * num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Distances start at ~1 stride rather than ~0: a degenerate first box
        # gives the refinement nothing to sample and makes every IoU ~0, which
        # is the cold start that broke tood_resnet before it was fixed.
        nn.init.constant_(self.reg_initial.bias, 1.0)
        # Prior probability 0.01 on the classifier.
        nn.init.constant_(self.cls_star.bias, -4.59511985013459)
        # The refinement predicts a scale FACTOR, so it must start at 1.0.
        nn.init.constant_(self.reg_refine.bias, 1.0)

    @staticmethod
    def _flatten(output, channels_per_anchor):
        """``(N, A * K, H, W)`` -> ``(N, H * W * A, K)``.

        Exactly torchvision's own permutation, so it agrees with
        ``AnchorGenerator``'s location-major ordering. A mismatch is
        shape-identical and therefore silent.
        """
        n, _, h, w = output.shape
        output = output.view(n, -1, channels_per_anchor, h, w)
        output = output.permute(0, 3, 4, 1, 2)
        return output.reshape(n, -1, channels_per_anchor)

    def forward(self, features):
        all_cls, all_initial, all_refined = [], [], []
        for level, feature in enumerate(features):
            cls_feature = self.cls_tower(feature)
            reg_feature = self.reg_tower(feature)

            initial = F.relu(self.reg_initial(reg_feature)) * self.reg_scales[level]
            # A per-edge multiplicative correction, kept positive.
            scale = F.relu(self.reg_refine(reg_feature, initial))
            refined = initial * scale

            cls_logits = self.cls_star(cls_feature, initial)

            all_cls.append(self._flatten(cls_logits, self.num_classes))
            all_initial.append(self._flatten(initial, 4))
            all_refined.append(self._flatten(refined, 4))
        return {
            "cls_logits": torch.cat(all_cls, dim=1),
            "bbox_regression": torch.cat(all_refined, dim=1),
            "bbox_initial": torch.cat(all_initial, dim=1),
        }


def _centres(boxes):
    """``(N, 2)`` centres of ``(N, 4)`` xyxy boxes."""
    return torch.stack(
        ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2), dim=1
    )


def _anchor_strides(num_anchors_per_level, device, dtype):
    return torch.cat(
        [
            torch.full((count,), float(stride), device=device, dtype=dtype)
            for count, stride in zip(num_anchors_per_level, PYRAMID_STRIDES)
        ]
    )


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


def _atss_assign(anchors, gt_boxes, num_anchors_per_level, topk):
    """ATSS assignment for one image. Duplicated — zoo templates cannot import
    siblings, so its guards are duplicated in the test file too."""
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
    # over a single candidate, and `iou >= nan` is False everywhere, which would
    # silently leave that object with no anchors at all.
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


def _varifocal_loss(cls_logits, target_scores, alpha=VFL_ALPHA, gamma=VFL_GAMMA):
    """Varifocal Loss — ASYMMETRIC, which is the whole point.

    The positive term (``target > 0``) carries **no** focal modulation: it is
    weighted by ``target``, the anchor's IoU, so a well-localised anchor is
    never discounted for being easy. The negative term keeps
    ``alpha * p ** gamma``, because there are ~10^4 of them per image.

    Symmetric modulation would make this Quality Focal Loss (see
    ``gfl_resnet``), which is a different loss.
    """
    probabilities = cls_logits.sigmoid()
    positive = target_scores > 0
    # Positives: weight = target. Negatives: weight = alpha * p ** gamma.
    weight = torch.where(
        positive, target_scores, alpha * probabilities.pow(gamma).detach()
    )
    return (
        F.binary_cross_entropy_with_logits(cls_logits, target_scores, reduction="none")
        * weight
    ).sum()


class _VFNet(RetinaNet):
    """RetinaNet's plumbing with VFNet's head, losses and distance decode."""

    def _level_split(self):
        split = getattr(self.anchor_generator, "num_anchors_per_level", None)
        if split is None:
            raise RuntimeError(
                "VFNet needs the per-level anchor split, which "
                "_LevelAwareAnchorGenerator records during forward; none was "
                "recorded. Was the model built with a plain AnchorGenerator?"
            )
        return split

    def compute_loss(self, targets, head_outputs, anchors):
        split = self._level_split()
        cls_logits = head_outputs["cls_logits"]
        refined = head_outputs["bbox_regression"]
        initial = head_outputs["bbox_initial"]

        losses_cls, losses_refined, losses_initial = [], [], []
        total_quality = 0.0

        for index, (anchors_per_image, targets_per_image) in enumerate(zip(anchors, targets)):
            strides = _anchor_strides(split, anchors_per_image.device, anchors_per_image.dtype)
            centres = _centres(anchors_per_image)
            logits = cls_logits[index]

            refined_boxes = _distance_to_box(centres, refined[index], strides)
            initial_boxes = _distance_to_box(centres, initial[index], strides)

            matched = _atss_assign(
                anchors_per_image, targets_per_image["boxes"], split, ATSS_TOPK
            )
            foreground = matched >= 0

            target_scores = torch.zeros_like(logits)
            if bool(foreground.any()):
                matched_gt = matched[foreground]
                gt_boxes = targets_per_image["boxes"][matched_gt]
                gt_labels = targets_per_image["labels"][matched_gt]

                # The classification target is the IoU of the REFINED box,
                # because that is what inference emits. Detached: it is a label,
                # so gradient must reach the boxes through the box losses and
                # never through the classifier.
                with torch.no_grad():
                    quality = (
                        box_ops.box_iou(refined_boxes[foreground], gt_boxes)
                        .diagonal()
                        .clamp(min=0)
                    )
                target_scores[foreground, gt_labels] = quality
                total_quality += float(quality.sum())

                weight = quality
                normaliser = weight.sum().clamp(min=1e-6)
                losses_refined.append(
                    (
                        generalized_box_iou_loss(
                            refined_boxes[foreground], gt_boxes, reduction="none"
                        )
                        * weight
                    ).sum()
                    / normaliser
                )
                # The initial box is supervised too: the refinement can only
                # correct what the first estimate roughly located.
                losses_initial.append(
                    generalized_box_iou_loss(
                        initial_boxes[foreground], gt_boxes, reduction="mean"
                    )
                )

            losses_cls.append(_varifocal_loss(logits, target_scores))

        denominator = max(total_quality, 1.0)
        zero = cls_logits.sum() * 0.0
        return {
            "classification": torch.stack(losses_cls).sum() / denominator,
            "bbox_regression": (
                torch.stack(losses_refined).mean() if losses_refined else zero
            ),
            "bbox_initial": (
                torch.stack(losses_initial).mean() if losses_initial else zero
            ),
        }

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        """Decode the REFINED distances, then RetinaNet's usual trim and NMS.

        Note the nesting: after ``RetinaNet.forward`` splits them,
        ``head_outputs[k]`` is a list over pyramid LEVELS, each
        ``(N, anchors_in_level, K)``, while ``anchors`` is a list over images of
        lists over levels. Iterating the outer list as images silently processes
        one level of one image — ``zip`` truncates rather than raising — and
        that is invisible on a fresh model whose prior-0.01 scores all sit below
        ``score_thresh``. This shape is the fix for that class of bug, found in
        ``gfl_resnet``.
        """
        class_logits = head_outputs["cls_logits"]
        box_distances = head_outputs["bbox_regression"]
        detections = []

        for index, image_shape in enumerate(image_shapes):
            logits_per_image = [level[index] for level in class_logits]
            distances_per_image = [level[index] for level in box_distances]
            anchors_per_image = anchors[index]

            image_boxes, image_scores, image_labels = [], [], []

            for level, (logits, distances, level_anchors) in enumerate(
                zip(logits_per_image, distances_per_image, anchors_per_image)
            ):
                num_classes = logits.shape[-1]
                scores = logits.sigmoid().flatten()

                keep = scores > self.score_thresh
                scores = scores[keep]
                keep_idxs = torch.where(keep)[0]
                num_topk = min(self.topk_candidates, scores.numel())
                scores, sort_idx = scores.topk(num_topk)
                keep_idxs = keep_idxs[sort_idx]

                anchor_idxs = torch.div(keep_idxs, num_classes, rounding_mode="floor")
                labels = keep_idxs % num_classes

                selected = level_anchors[anchor_idxs]
                # This level's OWN stride: the distances are in stride units.
                stride = float(PYRAMID_STRIDES[level])
                strides = selected.new_full((selected.shape[0],), stride)
                boxes = _distance_to_box(
                    _centres(selected), distances[anchor_idxs], strides
                )
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
    # match the rest of the family.
    backbone = resnet50(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(
        backbone,
        trainable_layers=3,
        returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(256, 256),
    )

    # P3..P7 with ONE square anchor per location at 8x the level stride. The
    # anchor is a reference POINT for the edge distances, not a box prior —
    # VFNet is anchor-free in the same sense FCOS is.
    anchor_generator = _LevelAwareAnchorGenerator(
        sizes=tuple((8 * stride,) for stride in PYRAMID_STRIDES),
        aspect_ratios=((1.0,),) * len(PYRAMID_STRIDES),
    )

    head = _VFNetHead(
        backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        num_classes,
        num_levels=len(PYRAMID_STRIDES),
    )

    return _VFNet(
        backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        head=head,
    )
