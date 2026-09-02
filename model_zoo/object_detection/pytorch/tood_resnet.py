"""TOOD — Task-aligned One-stage Object Detection (Feng et al., ICCV 2021) on a ResNet-50-FPN backbone. TOOD attacks a misalignment the rest of this family lives with: classification and localisation are learned by two parallel branches from the same features, so the anchor that scores highest is often not the anchor that localises best, and NMS then keeps the wrong one. Two changes address it — a head whose two task features are *derived from each other* rather than computed independently, and an assignment that scores an anchor by how well it does both at once.

Offline variant: the architecture is built with ``weights=None`` throughout, so
nothing is fetched from ``download.pytorch.org`` — the #199 egress lockdown
blocks it — and the template constructs anywhere, network or not. No seed is
hosted for this template yet, so it random-initialises and there is
no weight file — upload with ``weights=False``::

    user.upload_model("tood_resnet", weights=False)

Until a dump is staged, ``tools/check_dump_coverage.py`` classifies this file
NO_SEED and the statement above is what keeps that classification honest.

The T-Head — interactive, then task-specific
--------------------------------------------
A conventional head runs two independent towers. TOOD runs **one** stack of
inter-level convolutions and then lets each task pick its own weighted
combination of that stack's layers:

1. six 3x3 convolutions produce a list of six feature maps, each a deeper view
   of the same FPN level;
2. for each task, a *layer attention* module — global average pool over the six
   concatenated maps, two fully-connected layers, a sigmoid — emits six scalars;
3. the task's feature is the sigmoid-weighted sum of the six maps, passed
   through a 1x1 convolution.

So the classification and regression features are two different reads of the
same computation rather than two separate computations. Early layers carry
localisation detail and later ones carry semantics; the attention lets each task
weight them differently instead of hard-coding one depth for both.

TAL — one score for both tasks
------------------------------
The alignment metric is ``t = s ** alpha * u ** beta``, where ``s`` is the
predicted classification score for the ground-truth class and ``u`` is the IoU
of the predicted box. An anchor good at only one task scores near zero. The
classification target is then a *normalised* ``t`` rather than 1, so the
best-aligned anchor is trained to be the most confident one — which is exactly
what NMS later assumes.

⚠️ **Cold start, and what this template does about it.** ``t`` depends on the
model's own predictions, so at initialisation both factors are meaningless and
a pure TAL assignment is random for the first steps. The reference
implementation handles this by running **ATSS for the first epoch** and
switching afterwards. A zoo template cannot do that: it is a constructor, it
never sees an epoch counter, and the engine's training loop does not offer one.

Rather than pretend, the assignment here is **staged**: ATSS's static candidate
selection (the ``topk`` anchors per pyramid level nearest the object centre)
builds the pool, and TAL's alignment metric ranks *within* that pool. At
initialisation the pool is geometric and sensible, so the model trains from step
one; as the predictions become meaningful the alignment ranking takes over. It
is a documented adaptation, not the paper's schedule, and the consequence is
recorded honestly: convergence should be close but is not guaranteed identical,
and this template's COCO number is not the published one.

What is reused, and what is not
-------------------------------
``RetinaNet.forward`` is inherited whole — transform, anchor generation,
per-level splitting of the flat head outputs, mapping detections back to
original coordinates. ``head``, ``compute_loss`` and ``postprocess_detections``
are overridden. The boxes are regressed as **distances from the anchor centre**
(``l, t, r, b`` in stride units), the anchor-free parameterisation TOOD uses,
so ``RetinaNet``'s delta-based decode does not apply and the override is
required rather than stylistic.

⚠️ **Task-aligned prediction (the "O" half) is deliberately NOT implemented.**
The paper adds a deformable-convolution layer that spatially aligns each task's
prediction map. ``torchvision.ops.deform_conv2d`` is available and works on CPU
in the engine's pinned wheel (verified — forward and backward, no custom
extension to compile, unlike the ``MultiScaleDeformableAttention`` op the
Tier 3 DETR family needs). It is left out because the paper's own ablation puts
almost all of the gain in T-Head plus TAL, and a deformable alignment layer
whose offsets are learned from scratch with no pretrained backbone is the part
least likely to help a randomly-initialised model. Recorded as available and
declined, not as unavailable.

Output ordering is load-bearing
-------------------------------
``AnchorGenerator`` emits anchors location-major; the head's conv output is
``(N, A * K, H, W)``. Permuting those to disagree is *shape-identical*, so the
model would train against boxes decoded at the wrong pixels with nothing
raising. ``_flatten`` uses torchvision's own permutation and
``tests/test_tood_head.py`` pins it.

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
# model actually runs at (backend#3058).
image_size = 800
# The interactive head keeps six 256-channel feature maps alive per level
# simultaneously so the layer attention can weight them, which is heavier in
# activations than a sequential tower of the same depth.
batch_size = 4
output_classes = 12
category = "object_detection"

#: Inter-level convolutions in the T-Head stack. 6 is the paper's value.
STACKED_CONVS = 6

#: Alignment metric exponents, ``t = s ** ALPHA * u ** BETA``. The paper's
#: values: beta >> alpha, so a confident anchor that localises poorly is
#: punished much harder than an unconfident one that localises well.
TAL_ALPHA = 1.0
TAL_BETA = 6.0

#: Anchors selected per ground-truth box by the alignment metric.
TAL_TOPK = 13

#: Candidates drawn per pyramid level, per ground-truth box, to build the pool
#: the alignment ranks within. See the cold-start note in the module docstring.
POOL_TOPK = 9

#: torchvision's ``Matcher`` background sentinel; ``>= 0`` is a GT index.
BACKGROUND = -1

#: FPN strides for the P3..P7 pyramid this template builds.
PYRAMID_STRIDES = (8, 16, 32, 64, 128)


class _LevelAwareAnchorGenerator(AnchorGenerator):
    """``AnchorGenerator`` that records the per-level anchor counts.

    Both the candidate pool and the distance decode need the level boundaries of
    the concatenated anchor tensor, which ``RetinaNet.forward`` computes but does
    not pass to ``compute_loss``. Recording it here duplicates no library logic;
    copying ``forward`` to thread it through would fork ~80 lines that then
    drift silently on a torchvision upgrade.
    """

    def forward(self, image_list, feature_maps):
        self.num_anchors_per_level = [
            feature_map.shape[-2] * feature_map.shape[-1] * anchors_per_location
            for feature_map, anchors_per_location in zip(
                feature_maps, self.num_anchors_per_location()
            )
        ]
        return super().forward(image_list, feature_maps)


class _LayerAttention(nn.Module):
    """Per-task weights over the T-Head's stacked layers.

    Global average pool over the concatenated stack, two fully-connected
    layers, a sigmoid. One scalar per stacked layer, so the task can choose its
    own depth mixture rather than being handed a fixed one.
    """

    def __init__(self, channels, num_layers, reduction=32):
        super().__init__()
        self.num_layers = num_layers
        self.fc1 = nn.Conv2d(channels * num_layers, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, num_layers, 1)
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, stacked):
        # stacked: list of (N, C, H, W), one per stacked conv.
        concatenated = torch.cat(stacked, dim=1)
        pooled = F.adaptive_avg_pool2d(concatenated, 1)
        weights = torch.sigmoid(self.fc2(F.relu(self.fc1(pooled))))
        # (N, num_layers, 1, 1) -> weight each map and sum.
        return sum(
            stacked[index] * weights[:, index : index + 1] for index in range(self.num_layers)
        )


class _TOODHead(nn.Module):
    """The T-Head: one interactive stack, two task-specific reads of it."""

    def __init__(self, in_channels, num_anchors, num_classes, num_levels,
                 num_stacked=STACKED_CONVS):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.inter_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_stacked)
        )
        self.cls_attention = _LayerAttention(in_channels, num_stacked)
        self.reg_attention = _LayerAttention(in_channels, num_stacked)
        self.cls_reduce = nn.Conv2d(in_channels, in_channels, 1)
        self.reg_reduce = nn.Conv2d(in_channels, in_channels, 1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        self.bbox_regression = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)
        # One learnable scalar per level on the regression branch: the stack is
        # shared across levels but the distances each level predicts differ by a
        # factor of two per step.
        self.scales = nn.ParameterList(nn.Parameter(torch.ones(1)) for _ in range(num_levels))

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.weight.shape[-1] == 3:
                    nn.init.normal_(module.weight, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        # Prior probability 0.01: without it the first steps are dominated by
        # the ~10^4 easy negatives per image.
        nn.init.constant_(self.cls_logits.bias, -4.59511985013459)
        # Distances start at ~1 stride rather than ~0. Measured: with a zero
        # bias the ReLU'd head emits distances with mean 0.08, so the predicted
        # boxes are near-degenerate points and their IoU with any object is
        # ~0.004. The alignment metric is u ** 6, so that becomes ~1e-15 and
        # TAL assigns nothing at all. A positive bias makes the first boxes
        # real enough for an IoU to mean something.
        nn.init.constant_(self.bbox_regression.bias, 1.0)

    @staticmethod
    def _flatten(output, channels_per_anchor):
        """``(N, A * K, H, W)`` -> ``(N, H * W * A, K)``.

        Exactly torchvision's own permutation, so it agrees with
        ``AnchorGenerator``'s location-major ordering. A mismatch here is
        shape-identical and therefore silent.
        """
        n, _, h, w = output.shape
        output = output.view(n, -1, channels_per_anchor, h, w)
        output = output.permute(0, 3, 4, 1, 2)
        return output.reshape(n, -1, channels_per_anchor)

    def forward(self, features):
        all_cls, all_reg = [], []
        for level, feature in enumerate(features):
            stacked, current = [], feature
            for conv in self.inter_convs:
                current = conv(current)
                stacked.append(current)

            cls_feature = self.cls_reduce(self.cls_attention(stacked))
            reg_feature = self.reg_reduce(self.reg_attention(stacked))

            cls_logits = self.cls_logits(F.relu(cls_feature))
            # ⚠️ The per-level scale is applied INSIDE the ReLU, not after it.
            #
            # `scales[level]` is an unconstrained Parameter taking gradient from
            # the GIoU loss, so it can cross zero. With the scale applied last
            # it is the final operation and nothing keeps distances
            # non-negative: measured, a scale of -1 makes EVERY box on that
            # level inverted (428 of 428 with x2 < x1).
            #
            # And NOTHING WOULD NOTICE. Measured across the torchvision box
            # path: neither `generalized_box_iou` nor `generalized_box_iou_loss`
            # nor `box_iou` validates corner ordering at all — the only `raise`
            # in either loss guards an invalid `reduction` argument, and
            # `box_iou`'s guards an unsupported format. On an inverted target
            # `[[10, 10, 5, 5]]` against `[[0, 0, 10, 10]]` the loss returns a
            # finite 0.75, and `generalized_box_iou` returns 0.25, neither
            # raising. So this does not fail loudly anywhere downstream; it
            # trains forever on inverted boxes producing plausible,
            # differentiable numbers.
            #
            # Inside the ReLU the clamp is the last word, whatever the scale
            # does. (gfl_resnet is immune by construction rather than by care —
            # its scale sits in front of the softmax integral, so the decoded
            # distance stays in [0, reg_max]; verified, not assumed.)
            distances = F.relu(self.bbox_regression(F.relu(reg_feature)) * self.scales[level])

            all_cls.append(self._flatten(cls_logits, self.num_classes))
            all_reg.append(self._flatten(distances, 4))
        return {
            "cls_logits": torch.cat(all_cls, dim=1),
            "bbox_regression": torch.cat(all_reg, dim=1),
        }


def _centres(boxes):
    """``(N, 2)`` centres of ``(N, 4)`` xyxy boxes."""
    return torch.stack(
        ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2), dim=1
    )


def _anchor_strides(num_anchors_per_level, device, dtype):
    """Per-anchor stride for the concatenated anchor tensor."""
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


def _inside_mask(anchor_centres, gt_boxes):
    """``(num_gt, num_anchors)`` — is this anchor's centre inside this object?"""
    return (
        (anchor_centres[None, :, 0] >= gt_boxes[:, None, 0])
        & (anchor_centres[None, :, 0] <= gt_boxes[:, None, 2])
        & (anchor_centres[None, :, 1] >= gt_boxes[:, None, 1])
        & (anchor_centres[None, :, 1] <= gt_boxes[:, None, 3])
    )


def _candidate_pool(anchor_centres, gt_centres, num_anchors_per_level, topk):
    """ATSS's static step: the ``topk`` nearest anchors WITHIN each level.

    This is the cold-start fix described in the module docstring — a geometric
    pool that is meaningful before the model can predict anything, which the
    alignment metric then ranks inside.
    """
    distances = (gt_centres[:, None, :] - anchor_centres[None, :, :]).pow(2).sum(-1).sqrt()
    pool = torch.zeros_like(distances, dtype=torch.bool)
    start = 0
    for level_size in num_anchors_per_level:
        end = start + level_size
        if level_size > 0:
            k = min(topk, level_size)
            _, local_idx = distances[:, start:end].topk(k, dim=1, largest=False)
            pool[:, start:end].scatter_(1, local_idx, True)
        start = end
    return pool


def _tal_assign(anchors, gt_boxes, gt_labels, scores, predicted_boxes,
                num_anchors_per_level, topk=TAL_TOPK, pool_topk=POOL_TOPK,
                alpha=TAL_ALPHA, beta=TAL_BETA):
    """Task Alignment Learning assignment for one image.

    Returns ``(matched, alignment)``: the ground-truth index per anchor (or
    ``BACKGROUND``), and the *normalised* alignment metric per anchor, which
    becomes the classification target for the positives.
    """
    num_anchors = anchors.shape[0]
    matched = anchors.new_full((num_anchors,), BACKGROUND, dtype=torch.int64)
    alignment = anchors.new_zeros((num_anchors,))
    if gt_boxes.numel() == 0:
        return matched, alignment

    anchor_centres = _centres(anchors)
    ious = box_ops.box_iou(gt_boxes, predicted_boxes)          # (num_gt, num_anchors)
    # The predicted score for each object's OWN class, per anchor.
    class_scores = scores[:, gt_labels].transpose(0, 1)        # (num_gt, num_anchors)

    metric = class_scores.clamp(min=0).pow(alpha) * ious.clamp(min=0).pow(beta)

    # Candidates must be geometrically plausible AND inside the object.
    distances = (_centres(gt_boxes)[:, None, :] - anchor_centres[None, :, :]).pow(2).sum(-1).sqrt()
    pool = _candidate_pool(anchor_centres, _centres(gt_boxes), num_anchors_per_level, pool_topk)
    eligible = pool & _inside_mask(anchor_centres, gt_boxes)
    metric = metric.masked_fill(~eligible, 0.0)

    # ⚠️ THE COLD START, handled per object rather than per epoch.
    #
    # `t = s ** alpha * u ** beta` is only meaningful once the model can
    # predict. Measured at initialisation: the best IoU between a predicted box
    # and any object is ~0.004, and with beta = 6 that makes the metric ~1e-15
    # for every anchor — so a pure TAL assignment selects nothing, the
    # classification target is all zeros and the box loss has no positives. The
    # model trains, reports finite losses, and learns NOTHING. The reference
    # implementation avoids this by running ATSS for the first epoch; a template
    # is a constructor and never sees an epoch counter.
    #
    # So the staging is driven by the data instead: an object whose candidates
    # are all degenerate is ranked GEOMETRICALLY (nearest centre, which is
    # ATSS's static rule) and supervised with a HARD target of 1, exactly as a
    # fixed-threshold detector would. As soon as its predictions become good
    # enough for the metric to separate candidates, that object switches to
    # alignment ranking with the soft `t_hat` target. Per object and per step,
    # so different objects can be at different stages in the same batch.
    degenerate = metric.max(dim=1, keepdim=True).values <= 1e-12
    geometric = (1.0 / (1.0 + distances)).masked_fill(~eligible, 0.0)
    ranking = torch.where(degenerate, geometric, metric)

    # Top-m by whichever ranking applies, per object.
    selected = torch.zeros_like(ranking, dtype=torch.bool)
    k = min(topk, num_anchors)
    _, top_idx = ranking.topk(k, dim=1)
    selected.scatter_(1, top_idx, True)
    # ONE filter, deliberately, and it is `ranking > 0` rather than `eligible`.
    #
    # `topk` always returns k columns, so for an object with fewer than k
    # candidates the surplus columns are anchors the ranking scored at zero —
    # and without a filter they become positives with an alignment weight of
    # zero: they contribute nothing to the box loss and push their own class
    # towards zero in the classification loss. Positives in name only.
    #
    # `ranking > 0` is strictly stronger than `eligible`, because both `metric`
    # and `geometric` are already masked to zero outside `eligible` — so a
    # non-zero ranking IMPLIES eligibility, while eligibility does not imply a
    # non-zero ranking. An earlier version applied both; a mutation sweep showed
    # that made each individually un-testable (removing either left the other
    # doing the work, so no single-filter mutation could be caught) while adding
    # nothing. One filter that a test can actually defend beats two that
    # cover for each other.
    selected &= ranking > 0

    # An anchor claimed by two objects goes to the higher IoU, as in ATSS.
    claim = ious.masked_fill(~selected, -1.0)
    best_iou, best_gt = claim.max(dim=0)
    assigned = best_iou >= 0.0
    matched[assigned] = best_gt[assigned]

    # Normalise the metric per object so its maximum equals that object's best
    # achieved IoU. This is what makes the classification target a *quality*
    # rather than an arbitrary product of two numbers below 1 — an unnormalised
    # t with beta = 6 is vanishingly small and the classifier would learn to
    # predict ~0 everywhere.
    normalised = metric.masked_fill(~selected, 0.0)
    per_gt_max = normalised.max(dim=1, keepdim=True).values.clamp(min=1e-12)
    per_gt_iou_max = ious.masked_fill(~selected, 0.0).max(dim=1, keepdim=True).values
    normalised = normalised / per_gt_max * per_gt_iou_max
    # A degenerate object gets the hard target instead, per the note above.
    normalised = torch.where(
        degenerate & selected, torch.ones_like(normalised), normalised
    )

    # ⚠️ Gathered from `best_gt`, NOT `max(dim=0)`.
    #
    # `matched` resolves a multi-claimed anchor by IoU, so the soft label has to
    # come from the SAME object whose class channel it is written into. Taking
    # the maximum across every ground truth decouples the two: an anchor
    # selected by both a well-predicted object and a degenerate one would get
    # the degenerate one's hard 1.0 written onto the good object's channel. The
    # cold-start branch above makes that concrete rather than hypothetical.
    # mmdet's TaskAlignedAssigner gathers from the same argmax, for this reason.
    safe_gt = best_gt.clamp(min=0).unsqueeze(0)
    anchor_alignment = normalised.gather(0, safe_gt).squeeze(0)
    alignment[assigned] = anchor_alignment[assigned]
    return matched, alignment


def _quality_focal_loss(cls_logits, target_scores, beta=2.0):
    """Focal loss against a continuous quality target.

    Reduces to standard sigmoid focal loss when the target is 0 or 1, which is
    what makes a soft alignment target a generalisation rather than a different
    loss. Shared in spirit with ``gfl_resnet``; TOOD's target is the alignment
    metric rather than the IoU alone.
    """
    probabilities = cls_logits.sigmoid()
    modulation = (target_scores - probabilities).abs().pow(beta)
    return (
        F.binary_cross_entropy_with_logits(cls_logits, target_scores, reduction="none")
        * modulation
    ).sum()


class _TOOD(RetinaNet):
    """RetinaNet's plumbing with the T-Head, TAL assignment and a distance decode."""

    def _level_split(self):
        split = getattr(self.anchor_generator, "num_anchors_per_level", None)
        if split is None:
            raise RuntimeError(
                "TOOD needs the per-level anchor split, which "
                "_LevelAwareAnchorGenerator records during forward; none was "
                "recorded. Was the model built with a plain AnchorGenerator?"
            )
        return split

    def compute_loss(self, targets, head_outputs, anchors):
        split = self._level_split()
        cls_logits = head_outputs["cls_logits"]
        distances = head_outputs["bbox_regression"]

        losses_cls, losses_box = [], []
        total_alignment = 0.0

        for index, (anchors_per_image, targets_per_image) in enumerate(zip(anchors, targets)):
            strides = _anchor_strides(split, anchors_per_image.device, anchors_per_image.dtype)
            centres = _centres(anchors_per_image)
            logits = cls_logits[index]
            predicted = _distance_to_box(centres, distances[index], strides)

            # The assignment reads the model's own predictions, so it must not
            # contribute gradient: it is a label-producing step.
            with torch.no_grad():
                matched, alignment = _tal_assign(
                    anchors_per_image,
                    targets_per_image["boxes"],
                    targets_per_image["labels"],
                    logits.sigmoid(),
                    predicted,
                    split,
                )

            foreground = matched >= 0
            target_scores = torch.zeros_like(logits)
            if bool(foreground.any()):
                matched_gt = matched[foreground]
                gt_boxes = targets_per_image["boxes"][matched_gt]
                gt_labels = targets_per_image["labels"][matched_gt]
                target_scores[foreground, gt_labels] = alignment[foreground]

                weight = alignment[foreground]
                normaliser = weight.sum().clamp(min=1e-6)
                total_alignment += float(weight.sum())
                losses_box.append(
                    (
                        generalized_box_iou_loss(
                            predicted[foreground], gt_boxes, reduction="none"
                        )
                        * weight
                    ).sum()
                    / normaliser
                )

            losses_cls.append(_quality_focal_loss(logits, target_scores))

        # Normalised by the summed alignment, the paper's denominator: a batch
        # of poorly-aligned positives should not be scaled as if they were
        # confident ones.
        denominator = max(total_alignment, 1.0)
        zero = cls_logits.sum() * 0.0
        return {
            "classification": torch.stack(losses_cls).sum() / denominator,
            "bbox_regression": torch.stack(losses_box).mean() if losses_box else zero,
        }

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        """Decode distances, then RetinaNet's usual trim and NMS.

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
    # anchor is a reference POINT for the edge distances, not a box prior to
    # regress from — TOOD is anchor-free in the same sense FCOS is.
    anchor_generator = _LevelAwareAnchorGenerator(
        sizes=tuple((8 * stride,) for stride in PYRAMID_STRIDES),
        aspect_ratios=((1.0,),) * len(PYRAMID_STRIDES),
    )

    head = _TOODHead(
        backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        num_classes,
        num_levels=len(PYRAMID_STRIDES),
    )

    return _TOOD(
        backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        head=head,
    )
