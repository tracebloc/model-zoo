"""Cascade R-CNN — multi-stage IoU refinement (Cai & Vasconcelos, CVPR 2018) on a ResNet-50-FPN backbone. A single-stage R-CNN head is trained at one IoU threshold and is therefore good at exactly one thing: 0.5 is what makes training work at all (a higher threshold starves the head of positives), and 0.5 is also what makes the detector's high-IoU output mediocre. Cascade R-CNN resolves that by training THREE heads in sequence at 0.5 / 0.6 / 0.7, each one taking the previous head's regressed boxes as its proposals. Each stage sees a proposal distribution that is already better localised than the last, so a threshold that would have starved stage 1 is well populated by the time stage 3 sees it.

Offline variant: the architecture is built with ``weights=None`` throughout, so
nothing is fetched from ``download.pytorch.org`` — the #199 egress lockdown
blocks it — and the template constructs anywhere, network or not. No seed is
hosted for this template yet, so it random-initialises and there is
no weight file — upload with ``weights=False``::

    user.upload_model("cascade_rcnn", weights=False)

Until a dump is staged, ``tools/check_dump_coverage.py`` classifies this file
NO_SEED and the statement above is what keeps that classification honest.

Why three stages and not one head trained harder
------------------------------------------------
The paper's observation is that a box regressor is a *distribution shift*
machine: feed it proposals at IoU ~0.55 and its output lands around IoU ~0.65.
That output is the right input for a head trained at 0.6, which shifts it again.
So the cascade is not an ensemble of three attempts at the same problem — each
stage is trained on the distribution the previous stage actually produces, which
is the only reason a 0.7 head is trainable at all.

Two consequences that show up in the code:

- **Per-stage box-coder weights get tighter** — ``(10, 10, 5, 5)`` then
  ``(20, 20, 10, 10)`` then ``(30, 30, 15, 15)``. A later stage's residuals are
  smaller, so the same weight would leave it regressing near-zero deltas.
- **The refined boxes are detached** before becoming the next stage's
  proposals. Stage 3's loss must not flow back through stage 2's regressor;
  the stages are trained jointly but each on its own objective.

Ground truth is added to the proposals at EVERY stage
-----------------------------------------------------
This is the detail that decides whether the model trains at all, and getting it
wrong is silent. At initialisation the RPN emits essentially random boxes: on a
freshly built model the best RPN proposal against a ground-truth box reaches an
IoU in the low single-digit percents, so a stage matching at 0.7 would select
**zero** positives, contribute a zero box loss, and never learn to localise —
the model would report finite losses forever while stage 3 learned nothing. That
is the same cold-start shape TOOD hit with ``t = s^alpha * u^beta``.

Adding the ground-truth boxes themselves to the proposal set (torchvision's
``RoIHeads`` does this for its one stage; here it happens for all three) makes a
self-IoU of 1.0 available to every stage, so each stage has at least
``num_gt`` positives on step one regardless of the RPN's state. Measured on a
freshly built model: **stage 3 selects exactly the ground-truth proposals and
nothing else** on the first step, which is the intended cold start rather than
an empty one.

What is reused, and what is written here
----------------------------------------
``GeneralizedRCNN.forward`` supplies the transform, the backbone/RPN call and
the eval/train output switch; ``RegionProposalNetwork``, ``AnchorGenerator``,
``MultiScaleRoIAlign``, ``TwoMLPHead``, ``FastRCNNPredictor``,
``fastrcnn_loss``, ``Matcher``, ``BoxCoder`` and
``BalancedPositiveNegativeSampler`` are all torchvision's. What torchvision has
no equivalent of is the ROI head itself: ``RoIHeads`` is single-stage by
construction — one ``box_head``, one ``box_predictor``, one
``proposal_matcher``, one ``box_coder`` — so ``_CascadeRoIHeads`` below replaces
it rather than subclassing it. Subclassing would have meant inheriting four
attributes that describe a stage count of one and then never using them.

⚠️ Two things in here are silent when wrong
-------------------------------------------
**Per-class regression indexing.** ``FastRCNNPredictor`` emits
``num_classes * 4`` deltas per proposal and only one class's four are used. In
*training* the refinement uses the ASSIGNED label (a background proposal has no
meaningful class, so it takes column 0's deltas and is refined arbitrarily —
harmless, it will be re-assigned next stage); at *inference* it uses the
foreground argmax. Indexing with the wrong class produces valid-looking boxes
decoded from another class's regressor, and nothing raises.

**Score ensembling.** Inference averages the three stages' softmax scores, as
the paper specifies — the three classifiers are trained on different proposal
distributions and the ensemble is measurably better than stage 3 alone. Using
only the last stage's scores is a shape-identical change.
``tests/test_cascade_rcnn_stages.py`` pins both against stub predictors emitting
known constants, and drives the eval decode at batch >= 2 with scores forced
above ``score_thresh`` — a fresh model's scores sit near ``1/num_classes`` and a
decode bug would otherwise hide behind a well-formed empty result.

Verified against torch 2.11.0 / torchvision 0.26.0 (the engine pin,
``tools/requirements-engine-pin.txt``).
"""
from typing import List

import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = (
    "roi_heads.stages.0.box_predictor.bbox_pred.",
    "roi_heads.stages.0.box_predictor.cls_score.",
    "roi_heads.stages.1.box_predictor.bbox_pred.",
    "roi_heads.stages.1.box_predictor.cls_score.",
    "roi_heads.stages.2.box_predictor.bbox_pred.",
    "roi_heads.stages.2.box_predictor.cls_score.",
)

framework = "pytorch"
model_type = "torchvision_detection"
main_method = "MyModel"
license = "BSD-3-Clause"
# GeneralizedRCNNTransform's default is min_size=800, max_size=1333, and it
# UPSCALES anything smaller straight back to 800, so 800 is the resolution this
# model actually runs at. Read off the built model's transform rather than
# asserted from here — tests/test_od_declared_resolution.py compares the two
# (backend#3058).
image_size = 800
# Three ROI heads over the same pooled features. The backbone pass is shared, so
# the cost over faster_rcnn_resnet is three 1024-wide MLP heads on 512 sampled
# proposals per image rather than one — enough to warrant 4 rather than 8.
batch_size = 4
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


#: Per-stage foreground IoU thresholds — the paper's 0.5 / 0.6 / 0.7. Three
#: values, therefore three stages: the stage count is not configurable
#: separately, because a stage without its own threshold is not a cascade stage.
CASCADE_IOU_THRESHOLDS = (0.5, 0.6, 0.7)

#: Per-stage box-coder weights, tightening with the stage. A later stage's
#: residuals are smaller, so reusing stage 1's weights would leave stages 2-3
#: regressing deltas near zero.
CASCADE_BBOX_REG_WEIGHTS = (
    (10.0, 10.0, 5.0, 5.0),
    (20.0, 20.0, 10.0, 10.0),
    (30.0, 30.0, 15.0, 15.0),
)

#: Per-stage loss weights, the paper's 1 / 0.5 / 0.25. Later stages see fewer
#: positives, and an unweighted sum lets stage 3's noisier gradient dominate.
CASCADE_STAGE_LOSS_WEIGHTS = (1.0, 0.5, 0.25)

#: RoIAlign output edge, and therefore the ROI head's input resolution.
ROI_OUTPUT_SIZE = 7

#: Width of each stage's shared MLP before the classifier and regressor.
REPRESENTATION_SIZE = 1024


class _CascadeStage(nn.Module):
    """One stage: a two-layer MLP over the pooled ROI, then class + box heads.

    A module rather than two parallel ``ModuleList``s so that a stage is one
    named parameter subtree (``stages.<i>.box_head`` / ``.box_predictor``).
    That is what makes ``SEED_EXCLUDED_PREFIXES`` above expressible, and what
    makes "are there really three stages?" a checkable question rather than a
    matter of reading the forward pass.
    """

    def __init__(self, in_channels, representation_size, num_classes, roi_output_size):
        super().__init__()
        self.box_head = TwoMLPHead(
            in_channels * roi_output_size * roi_output_size, representation_size
        )
        self.box_predictor = FastRCNNPredictor(representation_size, num_classes)

    def forward(self, pooled_features):
        return self.box_predictor(self.box_head(pooled_features))


class _CascadeRoIHeads(nn.Module):
    """Three sequentially-refined R-CNN heads in place of torchvision's one.

    Speaks ``RoIHeads``' interface — ``forward(features, proposals,
    image_shapes, targets) -> (result, losses)`` — because that is what
    ``GeneralizedRCNN.forward`` calls, and returns the same shapes.
    """

    def __init__(
        self,
        box_roi_pool,
        in_channels,
        num_classes,
        iou_thresholds=CASCADE_IOU_THRESHOLDS,
        bbox_reg_weights=CASCADE_BBOX_REG_WEIGHTS,
        stage_loss_weights=CASCADE_STAGE_LOSS_WEIGHTS,
        representation_size=REPRESENTATION_SIZE,
        roi_output_size=ROI_OUTPUT_SIZE,
        batch_size_per_image=512,
        positive_fraction=0.25,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
    ):
        super().__init__()
        if not len(iou_thresholds) == len(bbox_reg_weights) == len(stage_loss_weights):
            raise ValueError(
                "cascade_rcnn: iou_thresholds, bbox_reg_weights and "
                "stage_loss_weights must describe the same number of stages, got "
                f"{len(iou_thresholds)}, {len(bbox_reg_weights)}, "
                f"{len(stage_loss_weights)}"
            )
        self.box_roi_pool = box_roi_pool
        self.num_classes = num_classes
        self.stages = nn.ModuleList(
            _CascadeStage(in_channels, representation_size, num_classes, roi_output_size)
            for _ in iou_thresholds
        )
        # One matcher and one box coder PER STAGE, held as plain lists: neither
        # carries parameters or buffers, so nn.ModuleList would add nothing and
        # would make the state_dict imply state that is not there.
        self.proposal_matchers = [
            det_utils.Matcher(threshold, threshold, allow_low_quality_matches=False)
            for threshold in iou_thresholds
        ]
        self.box_coders = [det_utils.BoxCoder(weights) for weights in bbox_reg_weights]
        self.stage_loss_weights = tuple(stage_loss_weights)
        self.iou_thresholds = tuple(iou_thresholds)
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    # --- training-sample selection -----------------------------------------

    def _assign(self, stage_index, proposals, targets):
        """Per-image ``(matched_gt_index, label)`` for one stage's threshold.

        ``matched_gt_index`` is clamped to ``>= 0`` so it can index the target
        boxes unconditionally; ``label == 0`` is what marks a background
        proposal, exactly as torchvision's ``assign_targets_to_proposals`` does.
        """
        matched_idxs, labels = [], []
        matcher = self.proposal_matchers[stage_index]
        for proposals_in_image, targets_in_image in zip(proposals, targets):
            gt_boxes = targets_in_image["boxes"]
            if gt_boxes.numel() == 0:
                # The engine emits an explicit zero-object target for an
                # unannotated image; every proposal is background.
                zeros = torch.zeros(
                    (proposals_in_image.shape[0],),
                    dtype=torch.int64,
                    device=proposals_in_image.device,
                )
                matched_idxs.append(zeros)
                labels.append(zeros)
                continue
            match_quality_matrix = box_ops.box_iou(gt_boxes, proposals_in_image)
            matched = matcher(match_quality_matrix)
            clamped = matched.clamp(min=0)
            labels_in_image = targets_in_image["labels"][clamped].to(torch.int64)
            labels_in_image[matched < 0] = 0
            matched_idxs.append(clamped)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def _select_training_samples(self, stage_index, proposals, targets):
        """Add ground truth, assign, subsample, and encode regression targets.

        The ground-truth boxes are prepended to the proposals at every stage —
        see the module docstring: without them a 0.7-threshold stage selects
        nothing on a freshly initialised RPN and silently never learns.
        """
        proposals = [
            torch.cat((targets_in_image["boxes"].to(proposals_in_image.dtype), proposals_in_image))
            for proposals_in_image, targets_in_image in zip(proposals, targets)
        ]
        matched_idxs, labels = self._assign(stage_index, proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        sampled_proposals, sampled_labels, regression_targets = [], [], []
        box_coder = self.box_coders[stage_index]
        for image_index, (pos_inds, neg_inds) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            keep = torch.where(pos_inds | neg_inds)[0]
            proposals_in_image = proposals[image_index][keep]
            labels_in_image = labels[image_index][keep]
            gt_boxes = targets[image_index]["boxes"]
            if gt_boxes.numel() == 0:
                matched_gt = torch.zeros_like(proposals_in_image)
            else:
                matched_gt = gt_boxes[matched_idxs[image_index][keep]]
            sampled_proposals.append(proposals_in_image)
            sampled_labels.append(labels_in_image)
            regression_targets.append(
                box_coder.encode([matched_gt], [proposals_in_image])[0]
            )
        return sampled_proposals, sampled_labels, regression_targets

    # --- refinement --------------------------------------------------------

    def _refine(self, stage_index, proposals, class_logits, box_regression, labels, image_shapes):
        """Decode this stage's boxes and hand them to the next stage.

        Detached: the stages are trained jointly but each on its own proposal
        distribution, so stage k+1's loss must not reach stage k's regressor.
        """
        box_coder = self.box_coders[stage_index]
        counts = [p.shape[0] for p in proposals]
        deltas = box_regression.reshape(box_regression.shape[0], -1, 4)
        if labels is None:
            # Inference: the foreground argmax. Column 0 is background and is
            # excluded, then the +1 puts the index back on the full range.
            selected = class_logits[:, 1:].argmax(dim=1) + 1
        else:
            # Training: the assigned label. A background proposal takes column
            # 0's deltas and is refined arbitrarily — it is re-assigned by the
            # next stage's matcher anyway.
            selected = torch.cat(labels)
        deltas = deltas[torch.arange(deltas.shape[0], device=deltas.device), selected]

        refined = []
        offset = 0
        for count, proposals_in_image, image_shape in zip(counts, proposals, image_shapes):
            boxes = box_coder.decode_single(
                deltas[offset : offset + count], proposals_in_image
            )
            refined.append(box_ops.clip_boxes_to_image(boxes, image_shape).detach())
            offset += count
        return refined

    # --- inference ---------------------------------------------------------

    def _postprocess(self, stage_scores, boxes_per_image, image_shapes):
        """Ensemble the stages' scores, then the usual score filter + NMS.

        ``stage_scores`` is a list over stages of ``(P, num_classes)`` softmax
        probabilities, all sharing the proposal ordering; ``boxes_per_image``
        holds the LAST stage's refined boxes, which are the best-localised ones
        available. Averaging the classifiers is the paper's inference rule.
        """
        scores = torch.stack(stage_scores, dim=0).mean(dim=0)
        results = []
        offset = 0
        for boxes, image_shape in zip(boxes_per_image, image_shapes):
            count = boxes.shape[0]
            scores_in_image = scores[offset : offset + count]
            offset += count

            # One box per proposal (the refined box), scored for every class:
            # expand the boxes across the class axis so class-wise NMS below
            # sees a (box, score, label) triple per candidate.
            num_classes = scores_in_image.shape[1]
            boxes_in_image = boxes[:, None, :].expand(-1, num_classes, -1)
            labels = torch.arange(num_classes, device=scores.device)
            labels = labels[None, :].expand(count, num_classes)

            # Drop the background column, then flatten.
            boxes_in_image = boxes_in_image[:, 1:].reshape(-1, 4)
            scores_flat = scores_in_image[:, 1:].reshape(-1)
            labels = labels[:, 1:].reshape(-1)

            keep = torch.where(scores_flat > self.score_thresh)[0]
            boxes_in_image, scores_flat, labels = (
                boxes_in_image[keep],
                scores_flat[keep],
                labels[keep],
            )
            boxes_in_image = box_ops.clip_boxes_to_image(boxes_in_image, image_shape)
            keep = box_ops.remove_small_boxes(boxes_in_image, min_size=1e-2)
            boxes_in_image, scores_flat, labels = (
                boxes_in_image[keep],
                scores_flat[keep],
                labels[keep],
            )
            keep = box_ops.batched_nms(boxes_in_image, scores_flat, labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]
            results.append(
                {
                    "boxes": boxes_in_image[keep],
                    "scores": scores_flat[keep],
                    "labels": labels[keep],
                }
            )
        return results

    # --- the RoIHeads interface -------------------------------------------

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training and targets is None:
            raise ValueError("cascade_rcnn: targets are required in training mode")

        losses = {}
        stage_scores: List[torch.Tensor] = []
        current = proposals

        for stage_index, stage in enumerate(self.stages):
            if self.training:
                current, labels, regression_targets = self._select_training_samples(
                    stage_index, current, targets
                )
            else:
                labels, regression_targets = None, None

            pooled = self.box_roi_pool(features, current, image_shapes)
            class_logits, box_regression = stage(pooled)

            if self.training:
                loss_classifier, loss_box_reg = fastrcnn_loss(
                    class_logits, box_regression, labels, regression_targets
                )
                weight = self.stage_loss_weights[stage_index]
                losses[f"loss_classifier_stage{stage_index}"] = loss_classifier * weight
                losses[f"loss_box_reg_stage{stage_index}"] = loss_box_reg * weight
            else:
                stage_scores.append(class_logits.softmax(dim=-1))

            current = self._refine(
                stage_index, current, class_logits, box_regression, labels, image_shapes
            )

        if self.training:
            return [], losses
        return self._postprocess(stage_scores, current, image_shapes), {}


class _CascadeRCNN(GeneralizedRCNN):
    """``GeneralizedRCNN`` with a three-stage ROI head.

    No behaviour of its own: the subclass exists so the model's repr names the
    architecture, and so ``isinstance`` checks in downstream tooling see
    something more specific than the generic base.
    """


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
    backbone = _resnet_fpn_extractor(backbone, trainable_layers=3)

    # torchvision's Faster R-CNN FPN defaults: one anchor size per level, three
    # aspect ratios. Cascade changes the ROI head, not the proposal generator.
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    rpn = RegionProposalNetwork(
        anchor_generator,
        RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0]),
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.3,
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_top_n={"training": 2000, "testing": 1000},
        post_nms_top_n={"training": 2000, "testing": 1000},
        nms_thresh=0.7,
    )

    roi_heads = _CascadeRoIHeads(
        MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=ROI_OUTPUT_SIZE,
            sampling_ratio=2,
        ),
        backbone.out_channels,
        num_classes,
    )

    transform = GeneralizedRCNNTransform(
        min_size=image_size,
        max_size=1333,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    return _CascadeRCNN(backbone, rpn, roi_heads, transform)
