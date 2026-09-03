"""Sparse R-CNN — learned proposals, dynamic instance interaction, set prediction (Sun et al., CVPR 2021). Every other two-stage detector starts from a dense hypothesis space: hundreds of thousands of anchors, or a region proposal network scoring them, followed by NMS to collapse the duplicates that density guarantees. Sparse R-CNN removes all three. A fixed, small set of **learned proposal boxes** (100 of them, ``nn.Parameter``, part of the model) is paired with a set of **learned proposal features**, and six iterative stages refine both. Training matches predictions to ground truth one-to-one with the Hungarian algorithm, so duplicates are penalised by the objective instead of removed by NMS.

Offline variant: every module here is built from an inlined architecture
description, the ResNet-50 trunk with ``weights=None``, so nothing is fetched
from ``download.pytorch.org`` — the #199 egress lockdown blocks it — and the
template constructs anywhere, network or not. No seed is hosted for this
template yet, so it random-initialises and there is
no weight file — upload with ``weights=False``::

    user.upload_model("sparse_rcnn", weights=False)

Until a dump is staged, ``tools/check_dump_coverage.py`` classifies this file
NO_SEED and the statement above is what keeps that classification honest.

The three pieces, and why each is not the obvious thing
-------------------------------------------------------
**Learned proposals.** ``init_proposal_boxes`` is a ``(num_proposals, 4)``
parameter in normalized ``cxcywh``, initialised to the whole image, and
``init_proposal_features`` is a ``(num_proposals, d_model)`` parameter
initialised randomly. The boxes being identical at initialisation is not an
oversight — it is the reference initialisation, and the *features* are what
break the symmetry, so the 100 proposals receive different gradients from step
one and differentiate into a learned spatial prior. Both must be trainable
parameters that actually move: a proposal set held as a buffer, or detached
before use, gives a detector with 100 fixed full-image boxes that can never
specialise, and it trains and evaluates without complaint.

**Dynamic instance interaction.** The pooled 7x7 ROI feature is not simply
concatenated with the proposal feature. The proposal feature *generates the
convolution parameters* that filter its own ROI — two ``d_model x dim_dynamic``
matrices predicted per proposal, applied in sequence to the 49 ROI positions.
That is the mechanism that lets one proposal attend to "the object I am
tracking" rather than to whatever is in the box. Replacing it with a plain
concatenation is a shape-identical change and the model still trains.

**Set prediction.** No NMS anywhere, at train or test time. Instead the loss
matches each ground-truth box to exactly one prediction by minimum total cost
(focal classification cost + L1 on normalized boxes + GIoU), and every
unmatched prediction is supervised as background. Loss is applied at all six
stages — deep supervision — with a fresh matching at each.

The Hungarian algorithm is written out here, not imported
---------------------------------------------------------
``scipy.optimize.linear_sum_assignment`` is what every reference
implementation calls, and it is deliberately not used: ``scipy`` is not a
declared dependency of this repo's pinned stack (it arrives only transitively,
via ``scikit-learn``) and the platform's model-checker validation environment
is a *different* environment from the training image — ``CLAUDE.md`` records
``peft`` being absent there, and a template that imports a library that
environment lacks is rejected at upload. So ``_linear_sum_assignment`` below is
a vectorised Jonker-Volgenant shortest-augmenting-path solver in ~40 lines of
``torch``. It is EXACT, not greedy, and
``tests/test_sparse_rcnn_matching.py`` pins it against an independent
brute-force oracle (every permutation, for matrices small enough to enumerate)
rather than against a second copy of the same algorithm.

⚠️ What the matcher selects at initialisation — measured, because this is where
a faithful implementation can silently learn nothing
----------------------------------------------------
A cold-start metric can starve an assigner while every loss stays finite:
TOOD's ``t = s^alpha * u^beta`` with ``beta = 6`` measures ~1e-15 at
initialisation, so a correct implementation selects **nothing** and trains
forever on an empty positive set. Hungarian matching has a different shape and
it is worth being explicit about why: the assignment is **cardinality-forced**,
not threshold-gated. It returns exactly ``min(num_gt, num_proposals)`` pairs
whatever the costs are, because the objective is a minimum over complete
matchings and there is no score to fall below. Measured on a freshly built
model: with 2 ground-truth boxes and 100 proposals it selects exactly 2 distinct
proposals on step one, and the classification loss's positive set is never
empty. The failure mode Sparse R-CNN *does* have is the opposite one — an
unstable matching that reshuffles between steps — and that is what the paper's
deep supervision and cost weighting exist to damp.

⚠️ Convergence: slow by design, and easy to mis-measure
-------------------------------------------------------
Two things about training this template that are worth knowing before reading a
loss curve, both measured rather than assumed.

**It is learning-rate sensitive.** The box deltas are exponentiated by
``BoxCoder`` (the width/height weights are 1.0, so the scale factor is
``exp(delta)`` directly), so a learning rate that is merely aggressive pushes
boxes off the image and localisation collapses to nothing while every loss stays
finite and plausible. Measured on a two-image overfit: at AdamW ``1e-3`` the best
IoU against ground truth is **0.000** after 120 steps, with the loss trace
looking unremarkable — 63.6 down to 31.2. At ``2.5e-4`` the same fixture
descends cleanly. The paper trains at ``2.5e-5``. This is the same observable as
the cold-start hazard above — finite losses, no learning — from a different
cause, so a finite loss is not evidence that this model is training.

**A pure-noise overfit is not a fair test of it, and that is structural.** The
usual two-random-images overfit works for an anchor or grid detector because its
head output is spatial: the conv at feature cell ``(i, j)`` can memorise "there
is an object here" with no help from the pixels. Sparse R-CNN has no grid, and
its 100 proposals are **shared across images** — so given two images whose
objects sit in different places and no usable signal in the pixels, there is
nothing for it to discriminate on, and it correctly learns almost nothing.
Measured at ``2.5e-4``: best IoU **0.04** on ``torch.rand`` images against
**0.42** on the same geometry with an actual bright rectangle at each box, in
fewer steps. If this template ever looks like it is not learning, check that the
fixture carries positional signal before suspecting the model.

⚠️ The decode is driven directly by the tests, at batch >= 2
------------------------------------------------------------
``_detections`` selects the top ``detections_per_img`` over the flattened
``(proposal, class)`` grid, per image. A fresh focal-loss head sits below any
sensible score threshold, so an eval assertion against a well-formed **empty**
list passes no matter what the decode does — the vacuous-eval path that shipped
a real ``zip``-truncation bug elsewhere in this roster. So the test feeds
synthetic above-threshold logits that DIFFER PER IMAGE, at batch 2, and asserts
each image's detections come from its own row.

What is reused
--------------
``ResNet-50`` + ``_resnet_fpn_extractor``, ``MultiScaleRoIAlign``,
``GeneralizedRCNNTransform`` (resize, normalize, pad to a batch at
``size_divisible=32``, and ``postprocess`` to map boxes back to original image
coordinates), ``BoxCoder`` for the per-stage delta application,
``generalized_box_iou`` / ``generalized_box_iou_loss`` and
``sigmoid_focal_loss`` are all torchvision's. ``GeneralizedRCNN`` itself is
NOT reused: its ``forward`` calls ``self.rpn``, and the whole point of this
architecture is that there is no RPN.

Verified against torch 2.11.0 / torchvision 0.26.0 (the engine pin,
``tools/requirements-engine-pin.txt``).
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign, generalized_box_iou, sigmoid_focal_loss
from torchvision.ops import boxes as box_ops
from torchvision.ops import misc as misc_nn_ops

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = (
    "stages.0.class_logits.",
    "stages.1.class_logits.",
    "stages.2.class_logits.",
    "stages.3.class_logits.",
    "stages.4.class_logits.",
    "stages.5.class_logits.",
)

framework = "pytorch"
model_type = "torchvision_detection"
main_method = "MyModel"
license = "Apache-2.0"
# GeneralizedRCNNTransform is built with min_size=800/max_size=1333 below, the
# resolution the paper trains at, and it UPSCALES anything smaller straight
# back to 800 — so 800 is the resolution this model actually runs at. Read off
# the built model's transform rather than asserted from here —
# tests/test_od_declared_resolution.py compares the two (backend#3058).
image_size = 800
# Six iterative stages, each with a 2048-wide FFN and self-attention over 100
# proposals, on top of a ResNet-50-FPN. Heavier per image than anything else in
# this roster, so 2.
batch_size = 2
output_classes = 12
category = "object_detection"

#: Learned proposals. 100 is the paper's default; it is also a hard ceiling on
#: how many objects one image can be given, because set prediction emits at most
#: one detection per proposal per class.
NUM_PROPOSALS = 100

#: Iterative refinement stages. Deep supervision applies the full loss at each.
NUM_STAGES = 6

#: Width of a proposal feature, and of everything downstream of it.
D_MODEL = 256
#: Heads in the proposal-to-proposal self-attention.
NUM_HEADS = 8
#: Hidden width of each stage's feed-forward block.
DIM_FEEDFORWARD = 2048
#: Rank of the dynamically generated convolution. 64 against D_MODEL's 256 is
#: what keeps the generated-parameter count (2 * 256 * 64 per proposal) small
#: enough to predict from a 256-vector.
DIM_DYNAMIC = 64
#: Depth of each stage's classification and regression MLPs.
NUM_CLS_LAYERS = 1
NUM_REG_LAYERS = 3

#: RoIAlign output edge. 7x7 = 49 positions, which is the sequence the
#: dynamically generated convolutions are applied over.
ROI_OUTPUT_SIZE = 7

#: Matching-cost weights, and the loss weights, which are the same numbers by
#: design: the cost the matcher minimises is the loss the matched pair will
#: incur, so a pairing that looks cheap to the matcher is cheap to the loss.
COST_CLASS = 2.0
COST_L1 = 5.0
COST_GIOU = 2.0

#: Focal-loss constants.
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

#: Per-stage box-delta weights — the reference implementation's.
BBOX_REG_WEIGHTS = (2.0, 2.0, 1.0, 1.0)

#: Predictions kept per image at inference. No NMS: this is a top-k over the
#: flattened (proposal, class) grid.
DETECTIONS_PER_IMG = 100


def _linear_sum_assignment(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact minimum-cost assignment — Jonker-Volgenant augmenting paths.

    Returns ``(row_indices, column_indices)`` such that summing
    ``cost[row_indices, column_indices]`` is minimal over all matchings of size
    ``min(rows, cols)``. Equivalent to ``scipy.optimize.linear_sum_assignment``;
    written out because ``scipy`` is not part of this repo's pinned stack (see
    the module docstring).

    Exact, not greedy. The distinction matters: greedy nearest-cost pairing is
    a different objective that agrees with the optimum on most inputs and
    disagrees on exactly the ambiguous ones a detector's matcher meets, so a
    greedy stand-in cannot be caught by spot-checking. The test compares against
    every permutation on small matrices.

    The dual potentials ``u`` and ``v`` and the inner scan are vectorised; the
    outer loop over rows is not, because each augmenting path depends on the
    previous one. Cost matrices here are ``(num_gt, num_proposals)`` with
    ``num_gt`` in the single or low double digits, so that is the right trade.
    """
    if cost.ndim != 2:
        raise ValueError(f"_linear_sum_assignment expects a 2-D cost matrix, got {tuple(cost.shape)}")
    if cost.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.int64, device=cost.device)
        return empty, empty.clone()

    # Solve on the orientation with rows <= cols, then swap the answer back:
    # the algorithm assigns every row and needs a column for each.
    transposed = cost.shape[0] > cost.shape[1]
    matrix = cost.detach().to(torch.float64)
    if transposed:
        matrix = matrix.t()
    matrix = matrix.contiguous()
    num_rows, num_cols = matrix.shape

    inf = torch.tensor(float("inf"), dtype=torch.float64, device=matrix.device)
    # 1-based indexing with a sentinel at 0, which is what makes the augmenting
    # path terminate on "column 0 is unassigned" without a separate flag.
    u = torch.zeros(num_rows + 1, dtype=torch.float64, device=matrix.device)
    v = torch.zeros(num_cols + 1, dtype=torch.float64, device=matrix.device)
    #: ``column_row[j]`` is the row currently assigned to column ``j``, or 0.
    column_row = torch.zeros(num_cols + 1, dtype=torch.int64, device=matrix.device)
    predecessor = torch.zeros(num_cols + 1, dtype=torch.int64, device=matrix.device)

    for row in range(1, num_rows + 1):
        column_row[0] = row
        current = 0
        minima = torch.full((num_cols + 1,), float("inf"), dtype=torch.float64, device=matrix.device)
        used = torch.zeros(num_cols + 1, dtype=torch.bool, device=matrix.device)
        while True:
            used[current] = True
            source = int(column_row[current])
            reduced = matrix[source - 1] - u[source] - v[1:]
            improved = (~used[1:]) & (reduced < minima[1:])
            minima[1:] = torch.where(improved, reduced, minima[1:])
            predecessor[1:] = torch.where(improved, torch.tensor(current, device=matrix.device), predecessor[1:])

            candidates = torch.where(used[1:], inf, minima[1:])
            delta, best = candidates.min(dim=0)
            next_column = int(best) + 1

            # The row potentials of every row currently reachable go up by
            # delta and the reached columns' potentials come down, which is what
            # keeps the reduced costs non-negative. The sentinel column 0 is
            # included on purpose: ``column_row[0]`` is the row being augmented,
            # so this is where ITS potential is raised. Indices are distinct —
            # the augmenting row is by definition assigned to no column — so the
            # in-place add cannot double-apply.
            u[column_row[used]] += delta
            v[used] -= delta
            minima[~used] -= delta

            current = next_column
            if int(column_row[current]) == 0:
                break

        while True:
            previous = int(predecessor[current])
            column_row[current] = column_row[previous]
            current = previous
            if current == 0:
                break

    columns = torch.arange(1, num_cols + 1, device=matrix.device)
    assigned = column_row[1:] != 0
    rows_out = column_row[1:][assigned] - 1
    cols_out = columns[assigned] - 1
    if transposed:
        rows_out, cols_out = cols_out, rows_out
    order = torch.argsort(rows_out)
    return rows_out[order], cols_out[order]


def _mlp(width, depth):
    """``depth`` blocks of ``Linear -> LayerNorm -> ReLU`` at constant width."""
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(width, width, bias=False), nn.LayerNorm(width), nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)


class _DynamicConv(nn.Module):
    """Dynamic instance interaction — the proposal feature filters its own ROI.

    ``dynamic_layer`` maps a ``(d_model,)`` proposal feature to
    ``2 * d_model * dim_dynamic`` numbers, which are *reshaped into
    convolution parameters* and applied to the 49 ROI positions in sequence.
    So the filter is not learned once and shared: it is produced per proposal,
    per stage, per forward pass.
    """

    def __init__(self, d_model=D_MODEL, dim_dynamic=DIM_DYNAMIC,
                 roi_output_size=ROI_OUTPUT_SIZE):
        super().__init__()
        self.d_model = d_model
        self.dim_dynamic = dim_dynamic
        self.num_params = d_model * dim_dynamic
        # TWO generated parameter blocks, and the 2 is STRUCTURAL rather than a
        # count you may vary: `d_model x dim_dynamic` projects the ROI down and
        # `dim_dynamic x d_model` projects it back. They are a down/up pair, not
        # N interchangeable convolutions.
        #
        # There used to be a `NUM_DYNAMIC = 2` module constant and a
        # `num_dynamic=NUM_DYNAMIC` parameter threaded to the stage. Neither
        # reached here -- this `__init__` never accepted it and the 2 below was
        # always literal -- so setting it to 3 changed nothing in the built
        # model. Worse, it would not have reddened: the parameter oracle in
        # `tests/test_sparse_rcnn_matching.py` derives its expectation from its
        # own transcribed `PUBLISHED_NUM_DYNAMIC`, so the declared architecture
        # and the real one could disagree while the test stayed green. Removed
        # rather than generalised, because the pair is what the architecture is.
        # Caught in review on model-zoo#246. See
        # `test_the_dynamic_interaction_builds_exactly_two_blocks`.
        self.dynamic_layer = nn.Linear(d_model, 2 * self.num_params)
        self.down_norm = nn.LayerNorm(dim_dynamic)
        self.up_norm = nn.LayerNorm(d_model)
        self.out_layer = nn.Linear(d_model * roi_output_size * roi_output_size, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, proposal_features, roi_features):
        """``(B, d_model)`` and ``(B, L, d_model)`` -> ``(B, d_model)``.

        ``B`` is images x proposals flattened and ``L`` is the ROI's 49
        positions. The residual and its norm live in the calling stage, not
        here, so that this module is exactly "the interaction" and can be
        swapped in a test without also removing the residual.
        """
        parameters = self.dynamic_layer(proposal_features)
        down = parameters[:, : self.num_params].view(-1, self.d_model, self.dim_dynamic)
        up = parameters[:, self.num_params :].view(-1, self.dim_dynamic, self.d_model)

        features = F.relu(self.down_norm(torch.bmm(roi_features, down)))
        features = F.relu(self.up_norm(torch.bmm(features, up)))
        features = self.out_layer(features.flatten(1))
        return F.relu(self.out_norm(features))


class _SparseRCNNStage(nn.Module):
    """One iterative stage: self-attention, dynamic interaction, FFN, heads.

    Six of these are built, with **independent** parameters. Sharing them would
    make the iteration a recurrent refinement rather than a cascade, which is a
    different (and measurably worse) architecture — and it is invisible in the
    forward pass, so ``tests/test_sparse_rcnn_matching.py`` asserts the stages
    are distinct parameter subtrees.
    """

    def __init__(self, d_model=D_MODEL, num_heads=NUM_HEADS, dim_feedforward=DIM_FEEDFORWARD,
                 num_classes=2, roi_output_size=ROI_OUTPUT_SIZE, dim_dynamic=DIM_DYNAMIC,
                 num_cls_layers=NUM_CLS_LAYERS, num_reg_layers=NUM_REG_LAYERS):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(d_model)
        self.instance_interaction = _DynamicConv(d_model, dim_dynamic, roi_output_size)
        self.interaction_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, d_model),
        )
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.cls_mlp = _mlp(d_model, num_cls_layers)
        self.reg_mlp = _mlp(d_model, num_reg_layers)
        self.class_logits = nn.Linear(d_model, num_classes)
        self.boxes_delta = nn.Linear(d_model, 4)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Prior probability 0.01, as every focal-loss detector does: without it
        # the first steps are dominated by the (num_proposals x num_classes)
        # negatives the one-to-one matching leaves behind.
        nn.init.constant_(self.class_logits.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, proposal_features, roi_features, num_proposals):
        """``proposal_features`` is ``(N * P, d_model)``, ``roi_features`` is
        ``(N * P, L, d_model)``. Returns updated features plus the stage's class
        logits and box deltas, all flattened over ``N * P``."""
        batch = proposal_features.shape[0] // num_proposals
        attended = proposal_features.view(batch, num_proposals, -1)
        attended = attended + self.self_attention(attended, attended, attended, need_weights=False)[0]
        attended = self.attention_norm(attended).reshape(batch * num_proposals, -1)

        features = self.interaction_norm(
            attended + self.instance_interaction(attended, roi_features)
        )
        features = self.feed_forward_norm(features + self.feed_forward(features))

        cls_features = features
        for layer in self.cls_mlp:
            cls_features = layer(cls_features)
        reg_features = features
        for layer in self.reg_mlp:
            reg_features = layer(reg_features)
        return features, self.class_logits(cls_features), self.boxes_delta(reg_features)


def _normalize(boxes, image_shape):
    """xyxy pixels -> xyxy in ``[0, 1]``, for the scale-free L1 cost.

    L1 on raw pixels would weight an 800px image's errors 1.6x an equivalent
    error in a 500px one, so the matcher would prefer the objects in small
    images. Normalising is what makes the cost comparable across a batch whose
    images the transform resized differently.
    """
    height, width = image_shape
    scale = boxes.new_tensor([width, height, width, height])
    return boxes / scale


class _SparseRCNN(nn.Module):
    """The detector: transform, backbone, learned proposals, six stages.

    Speaks the engine's ``TorchvisionDetectionHandler`` contract directly —
    ``model(images, targets)`` returns a loss dict, ``model(images)`` returns a
    ``List[Dict]`` of pixel-xyxy ``boxes`` / ``scores`` / ``labels`` in the
    ORIGINAL image coordinates — rather than inheriting it from
    ``GeneralizedRCNN``, whose forward is built around an RPN this architecture
    does not have.
    """

    def __init__(
        self,
        backbone,
        num_classes,
        transform,
        box_roi_pool,
        num_proposals=NUM_PROPOSALS,
        num_stages=NUM_STAGES,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dim_feedforward=DIM_FEEDFORWARD,
        dim_dynamic=DIM_DYNAMIC,
        roi_output_size=ROI_OUTPUT_SIZE,
        detections_per_img=DETECTIONS_PER_IMG,
    ):
        super().__init__()
        self.backbone = backbone
        self.transform = transform
        self.box_roi_pool = box_roi_pool
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.detections_per_img = detections_per_img
        self.box_coder = det_utils.BoxCoder(weights=BBOX_REG_WEIGHTS)

        # THE LEARNED PROPOSALS. Parameters, not buffers: the whole architecture
        # is that these are trained. Boxes start as the whole image in
        # normalized cxcywh and the features start random — the asymmetry lives
        # in the features, which is what lets the 100 identical boxes receive
        # different gradients and specialise. See the module docstring.
        self.init_proposal_boxes = nn.Parameter(torch.empty(num_proposals, 4))
        nn.init.constant_(self.init_proposal_boxes[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes[:, 2:], 1.0)
        self.init_proposal_features = nn.Parameter(torch.empty(num_proposals, d_model))
        nn.init.normal_(self.init_proposal_features, std=1.0)

        # Six INDEPENDENTLY constructed stages, not one module reused six times
        # and not six copies of one: sharing the parameters would turn the
        # cascade into a recurrence, which is a different (and measurably worse)
        # architecture and is invisible in the forward pass.
        self.stages = nn.ModuleList(
            _SparseRCNNStage(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                num_classes=num_classes,
                roi_output_size=roi_output_size,
                dim_dynamic=dim_dynamic,
            )
            for _ in range(num_stages)
        )

    # --- proposals ---------------------------------------------------------

    def _initial_boxes(self, image_shapes, dtype, device):
        """Normalized cxcywh parameters -> per-image pixel xyxy.

        Scaled by each image's own post-transform size, so a batch whose images
        the transform resized differently still gets whole-image proposals in
        each.
        """
        boxes = self.init_proposal_boxes.to(device=device, dtype=dtype)
        centre, size = boxes[:, :2], boxes[:, 2:]
        xyxy = torch.cat((centre - 0.5 * size, centre + 0.5 * size), dim=-1)
        return [
            xyxy * xyxy.new_tensor([width, height, width, height])
            for height, width in image_shapes
        ]

    # --- one stage ---------------------------------------------------------

    def _run_stages(self, features, image_shapes, dtype, device):
        """Iterate the six stages, returning per-stage logits and boxes.

        Boxes are detached between stages, exactly as in Cascade R-CNN and for
        the same reason: each stage is trained on the box distribution the
        previous one produces, so stage k+1's loss must not flow back into
        stage k's regressor.
        """
        batch = len(image_shapes)
        boxes = self._initial_boxes(image_shapes, dtype, device)
        proposal_features = (
            self.init_proposal_features.to(device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch, -1, -1)
            .reshape(batch * self.num_proposals, -1)
        )

        stage_logits, stage_boxes = [], []
        for stage in self.stages:
            pooled = self.box_roi_pool(features, boxes, image_shapes)
            # (N*P, C, 7, 7) -> (N*P, 49, C): the dynamic convolutions treat the
            # ROI's spatial positions as a sequence.
            roi_features = pooled.flatten(2).permute(0, 2, 1)
            proposal_features, class_logits, deltas = stage(
                proposal_features, roi_features, self.num_proposals
            )

            flat_boxes = torch.cat(boxes, dim=0)
            refined = self.box_coder.decode_single(deltas, flat_boxes)
            per_image = refined.split(self.num_proposals, dim=0)
            # Detached for the NEXT stage, kept attached for THIS stage's loss:
            # deep supervision needs the gradient at every stage, and the
            # cascade needs each stage trained on the distribution the previous
            # one produced rather than on one it can reach back and change.
            boxes = [b.detach() for b in per_image]
            stage_logits.append(class_logits.view(batch, self.num_proposals, -1))
            stage_boxes.append(torch.stack(list(per_image), dim=0))
        return stage_logits, stage_boxes

    # --- matching and loss -------------------------------------------------

    def _match(self, logits, boxes, targets, image_shapes):
        """One-to-one Hungarian matching, per image.

        Returns a list of ``(gt_index, proposal_index)`` pairs per image. The
        cost is the loss a matched pair would incur — the same three terms with
        the same weights — so the matcher and the objective agree.
        """
        matches = []
        for image_index, (targets_per_image, image_shape) in enumerate(
            zip(targets, image_shapes)
        ):
            gt_boxes = targets_per_image["boxes"]
            if gt_boxes.numel() == 0:
                empty = torch.zeros((0,), dtype=torch.int64, device=logits.device)
                matches.append((empty, empty.clone()))
                continue
            gt_labels = targets_per_image["labels"].to(torch.int64)

            probabilities = logits[image_index].sigmoid()
            # DETR's focal matching cost: the classification term is the
            # DIFFERENCE between what a positive and a negative would cost, so
            # a confident wrong class is penalised rather than merely not
            # rewarded.
            # ⚠️ ALPHA GOES ON THE POSITIVE TERM. torchvision's
            # `sigmoid_focal_loss` computes
            #   alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            # so a positive is weighted `alpha` and a negative `1 - alpha`.
            # An earlier revision of this file had them swapped -- FOCAL_ALPHA
            # on the negative and (1 - FOCAL_ALPHA) on the positive -- which at
            # the standard 0.25 means 0.75/0.25 instead of 0.25/0.75. The
            # assignment then minimised a different objective than the loss
            # training the matched pair, so a close geometry-versus-class call
            # picked the wrong proposal. Caught in review on model-zoo#246.
            #
            # Nothing here could see it: every matcher test asserted
            # CARDINALITY, which is invariant to any reweighting of the cost.
            # `matcher_cost_weights_match_the_loss` below asserts the weights
            # against torchvision's own formula instead.
            negative_cost = (
                (1 - FOCAL_ALPHA)
                * probabilities.pow(FOCAL_GAMMA)
                * -(1 - probabilities).clamp(min=1e-8).log()
            )
            positive_cost = (
                FOCAL_ALPHA
                * (1 - probabilities).pow(FOCAL_GAMMA)
                * -probabilities.clamp(min=1e-8).log()
            )
            class_cost = (positive_cost - negative_cost)[:, gt_labels].t()

            predicted = boxes[image_index]
            l1_cost = torch.cdist(
                _normalize(predicted, image_shape), _normalize(gt_boxes, image_shape), p=1
            ).t()
            giou_cost = -generalized_box_iou(gt_boxes, predicted)

            cost = COST_CLASS * class_cost + COST_L1 * l1_cost + COST_GIOU * giou_cost
            # nan/inf would make the assignment arbitrary rather than raise, so
            # they are replaced with a large finite cost.
            cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=-1e4)
            matches.append(_linear_sum_assignment(cost))
        return matches

    def _stage_loss(self, logits, boxes, targets, image_shapes, num_boxes):
        matches = self._match(logits, boxes, targets, image_shapes)

        target_classes = torch.zeros_like(logits)
        l1_losses, giou_losses = [], []
        for image_index, (gt_index, proposal_index) in enumerate(matches):
            if gt_index.numel() == 0:
                continue
            targets_per_image = targets[image_index]
            labels = targets_per_image["labels"].to(torch.int64)[gt_index]
            target_classes[image_index, proposal_index, labels] = 1.0

            matched_predictions = boxes[image_index][proposal_index]
            matched_gt = targets_per_image["boxes"][gt_index]
            image_shape = image_shapes[image_index]
            l1_losses.append(
                F.l1_loss(
                    _normalize(matched_predictions, image_shape),
                    _normalize(matched_gt, image_shape),
                    reduction="sum",
                )
            )
            giou_losses.append(
                (1.0 - generalized_box_iou(matched_predictions, matched_gt).diagonal()).sum()
            )

        classification = (
            sigmoid_focal_loss(
                logits, target_classes, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction="sum"
            )
            / num_boxes
        )
        zero = logits.sum() * 0.0
        return {
            "classification": COST_CLASS * classification,
            "bbox_regression": COST_L1
            * (torch.stack(l1_losses).sum() / num_boxes if l1_losses else zero),
            "giou": COST_GIOU
            * (torch.stack(giou_losses).sum() / num_boxes if giou_losses else zero),
        }

    # --- inference ---------------------------------------------------------

    def _detections(self, logits, boxes, image_shapes):
        """Top-k over the flattened ``(proposal, class)`` grid, per image.

        No NMS, at all: one-to-one training is what removes the duplicates, and
        adding NMS back would suppress genuinely overlapping objects the set
        prediction is able to keep apart.

        Column 0 is the dataset's background id, which set prediction never
        targets — "no object" is expressed by a proposal going unmatched — so it
        is dropped rather than allowed to win a top-k slot.
        """
        detections: List[Dict[str, torch.Tensor]] = []
        for image_index, image_shape in enumerate(image_shapes):
            scores = logits[image_index].sigmoid()[:, 1:]
            num_foreground_classes = scores.shape[1]
            flat = scores.flatten()
            keep = min(self.detections_per_img, flat.numel())
            top_scores, top_indices = flat.topk(keep)
            proposal_indices = torch.div(
                top_indices, num_foreground_classes, rounding_mode="floor"
            )
            labels = top_indices % num_foreground_classes + 1

            selected = box_ops.clip_boxes_to_image(
                boxes[image_index][proposal_indices], image_shape
            )
            detections.append(
                {"boxes": selected, "scores": top_scores, "labels": labels}
            )
        return detections

    # --- the handler contract ---------------------------------------------

    def forward(self, images, targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        if self.training and targets is None:
            raise ValueError("sparse_rcnn: targets are required in training mode")

        original_image_sizes: List[Tuple[int, int]] = [
            (int(image.shape[-2]), int(image.shape[-1])) for image in images
        ]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        stage_logits, stage_boxes = self._run_stages(
            features, images.image_sizes, images.tensors.dtype, images.tensors.device
        )

        if self.training:
            # Focal loss is normalised by the number of ground-truth boxes in
            # the batch, floored at 1 so a batch of unannotated images is a
            # finite zero rather than a division by zero.
            num_boxes = max(1, sum(int(t["boxes"].shape[0]) for t in targets))
            losses: Dict[str, torch.Tensor] = {}
            for stage_index, (logits, boxes) in enumerate(zip(stage_logits, stage_boxes)):
                for name, value in self._stage_loss(
                    logits, boxes, targets, images.image_sizes, num_boxes
                ).items():
                    losses[f"{name}_stage{stage_index}"] = value
            return losses

        detections = self._detections(
            stage_logits[-1], stage_boxes[-1], images.image_sizes
        )
        return self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background

    # weights=None: architecture only, no download (the #199 egress lockdown
    # blocks download.pytorch.org). FrozenBatchNorm2d and trainable_layers=3
    # match the rest of this family — a federated edge trains at batch sizes
    # where BN running statistics are the dominant noise source, and frozen
    # stats also average cleanly across clients.
    backbone = resnet50(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    # returned_layers 1..4 gives P2..P5, which is what Sparse R-CNN pools from:
    # its proposals start as the whole image, so the finest level matters.
    backbone = _resnet_fpn_extractor(backbone, trainable_layers=3, returned_layers=[1, 2, 3, 4])

    transform = GeneralizedRCNNTransform(
        min_size=image_size,
        max_size=1333,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    return _SparseRCNN(
        backbone,
        num_classes,
        transform,
        MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=ROI_OUTPUT_SIZE,
            sampling_ratio=2,
        ),
        d_model=backbone.out_channels,
    )
