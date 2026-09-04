"""YOLOv8-S (Ultralytics, 2023) — anchor-free C2f/PAFPN detector with a
decoupled DFL head and task-aligned label assignment, written from scratch in
PyTorch.

Offline variant: nothing is fetched at construction — no hub id, no
``timm``, no ``transformers``, no ``ultralytics``, no torchvision pretrained
enum, no ``download.pytorch.org`` (the #199 egress lockdown blocks it). Every
layer is built from the inlined width/depth multipliers and arch table below,
so the template constructs on a closed edge. No seed is hosted for this
template and there is no weight file: upload with ``weights=False``::

    user.upload_model("yolov8_s", weights=False)

Hosting COCO tensors as a tracebloc model-store seed (the #1499 pattern: a
matched ``<stem>_weights.pkl`` prepped by ``tools/prep_offline_weights.py``
and strict-loaded after the architecture is built) is follow-up work, not part
of this roster addition. Until a dump is staged,
``tools/check_dump_coverage.py`` classifies this file NO_SEED and the sentence
above is what keeps that classification honest.

⚠️ AND THE UPSTREAM CHECKPOINT WOULD NOT FIT ANYWAY, for two reasons worth
recording before anyone tries. First, the norm layers here are GroupNorm rather
than the published BatchNorm (see the federated note below), so every norm
tensor's *identity* differs even where its shape does not — a BN checkpoint
carries ``running_mean``/``running_var`` this tree has no slot for, and a strict
load fails loudly, which is the good case. Second, Ultralytics ships YOLOv8
weights under AGPL-3.0 while this file is an independent re-implementation
declared Apache-2.0; the seed for this template, if one is ever prepped, is
prepped by us against THIS build (backend#3055).

Why it is NOT in the ``yolo`` family — the whole point of this file
-------------------------------------------------------------------
``model_type = "yolo"`` is the legacy YOLOv1 grid contract: a fixed 448px
input, a ``[7, 7, num_classes + 10]`` target tensor, **one object per cell** (so
at most 49 objects per image, silently overwriting co-located ones) and an
external customer ``loss.py``. That family is frozen at its three existing
templates (backend#2982).

The zoo already ships ``object_detection/pytorch/yolo_v8/`` in that family, and
it is **a YOLOv8 backbone bent onto a YOLOv1 head** — the single 7x7 grid at
448px, none of the multi-scale structure the architecture exists for. This file
is the genuine article and replaces nothing: the legacy template stays where it
is, frozen, for the customers whose ``loss.py`` targets it.

What "genuine multi-scale" means here, concretely:

* a real **PAN neck** — one top-down pass and one bottom-up pass over the
  stride-8/16/32 feature maps, so the fine level carries semantic context and
  the coarse level carries localisation detail;
* **per-level heads** at all three strides, each predicting at the scale it is
  responsible for, rather than one head on one grid;
* **8400 anchor points** at 640px (80x80 + 40x40 + 20x20) against the legacy
  contract's 49 cells, with no per-cell object limit anywhere in the pipeline.

So it declares ``torchvision_detection``, which is a **duck-typed contract
rather than a library dependency**:

* ``model(images, targets)`` in train mode -> a dict of scalar losses
* ``model(images)`` in eval mode -> ``List[Dict]`` with ``boxes`` (pixel
  xyxy), ``scores``, ``labels``
* ``images`` is a **list** of differently-sized 3-D tensors, because
  ``_rcnn_collate`` builds tuples rather than stacking (object counts vary per
  image)

Nothing about that contract mentions torchvision, and the engine's
``TorchvisionDetectionHandler`` needs no change to train this file.

Why this is ONE file, and duplicated against its siblings
---------------------------------------------------------
A zoo template is uploaded as a single ``.py`` and there are zero relative
imports anywhere under ``model_zoo/`` — a template that imported a sibling would
arrive at the model checker with the sibling missing. So the backbone, the neck,
the head, the assigner and the losses all live here, and the handful of blocks
``yolox_s.py`` / ``rtmdet_s.py`` also need (a conv-norm-act, a group-count
helper, a PAN neck skeleton) are **deliberately duplicated** rather than shared.
Its tests are duplicated with it, for the same reason: copied code that leaves
its guard behind is how a duplicated assigner silently loses a rule.

The duplication is also thinner than it looks. This backbone's stage is a
**C2f**, not YOLOX's ``CSPLayer`` — the split halves *and every intermediate
block output* are concatenated, so an ``n``-block stage feeds ``(2 + n)``
branches into its fusion conv rather than 2. The head regresses a **discrete
distribution** over 16 bins per box edge (DFL) instead of a log-size, carries no
objectness branch at all, and is assigned by a **task-aligned** metric rather
than by SimOTA's optimal-transport cost.

Batching and resolution — reused, not hand-rolled
-------------------------------------------------
``GeneralizedRCNNTransform`` does the resize, the normalize, the pad-to-batch
at ``size_divisible=32`` and the ``postprocess()`` that maps predicted boxes
back to each image's original coordinates. Reusing it is what makes the
variable-size list contract safe, and the 32-divisibility is a hard requirement
because the head's three levels sit at strides 8/16/32.

``image_size`` IS the resolution this model runs at
---------------------------------------------------
``min_size`` and ``max_size`` are both set to ``image_size``, so a square
``data_shape x data_shape`` image from the engine's dataset scales by exactly
1.0 and the backbone sees ``image_size x image_size``. That is stated because
the previously shipped non-yolo OD templates declared ``image_size = 448``
while their builders resized to ``min_size=800`` (backend#3058): the declared
edge was decorative. Here it is the edge, and it is **measured off the built
model, never read from this docstring** —
``tests/test_od_declared_resolution.py`` compares declared against the
transform's configured resolution family-wide, and ``tests/test_yolov8_s.py``'s
``declared_size_measured`` guard hooks the transform to check the tensor the
backbone is actually handed is square at exactly this edge.

Label space is MODEL space ``[1, C]``
-------------------------------------
Since backend#3062 the family handler owns the translation: it shifts dataset
labels ``[0, C-1]`` up to ``[1, C]`` on the way in
(``_targets_to_model_space``) and shifts predictions back with the background
row dropped on the way out (``_detections_to_dataset_space``, which keeps only
``labels >= BACKGROUND_LABEL_OFFSET``).

So the head allocates ``output_classes + 1`` sigmoid channels and uses the
incoming label **directly** as the channel index. Channel 0 is therefore never a
positive target — it is trained only as a negative — and ``_predictions``
**slices it off BEFORE the score threshold and the top-k**, not after. That
ordering is the whole point: the engine does drop channel-0 rows, but it does so
downstream of this decode's budget, so a background candidate that survives to
there has already spent a detection slot a real object should have had. Same
argument as CenterNet's pre-slice fix in model-zoo#236 and the siblings' in
model-zoo#237.

Consequence, stated plainly: this template **requires** the family handler's
shift. Fed raw 0-based dataset labels it would discard the first class.

Regression parameterisation (DFL) and where the bins live
---------------------------------------------------------
Each box edge is predicted as a ``REG_MAX``-bin discrete distribution over
distances from the anchor point, in **cell units** (multiples of that level's
stride), and decoded by its expectation. Two consequences:

* the maximum representable distance is ``REG_MAX - 1 = 15`` cells, i.e. 120px
  at stride 8 and 480px at stride 32. That is the published design and it is
  why the coarse level exists; a large object is assigned to a coarse level
  where 15 cells reaches far enough.
* every decoded box is a valid xyxy **by construction** — the distances are a
  softmax expectation and so non-negative, giving ``x2 >= x1`` without a clamp
  that would silently kill the gradient on the clamped side.

The bin index vector is built with ``torch.arange`` inside the decode rather
than stored as a parameter or a buffer. Upstream keeps it as a frozen
``Conv2d`` weight with ``requires_grad=False``; here it is neither, which keeps
buffer count at exactly zero (see below) and keeps a dead-parameter sweep from
having to special-case a constant. It also means this build carries the
published **trainable** parameter count exactly — see the arithmetic in
``tests/test_yolov8_s.py``.

Federated note (GroupNorm, not BatchNorm)
-----------------------------------------
The norm layers are GroupNorm. Upstream YOLOv8 uses BatchNorm.

BN's ``running_mean``/``running_var`` are **buffers the averaging service ships
and averages every federated round**, and they average badly across non-IID
clients (see CLAUDE.md). The torchvision family used to avoid that with
``norm_layer=misc_nn_ops.FrozenBatchNorm``, whose statistics never update — and
on a ``weights=None`` backbone ``FrozenBatchNorm2d`` is a **bit-exact identity**
(``weight=1``, ``bias=0``, ``running_mean=0``, ``running_var=1``, verified in
backend#3093), so freezing it on a from-scratch template normalises nothing at
all. That whole family moved to GroupNorm in model-zoo#259; this note is kept
because it is the reasoning, not a description of the other templates.

GroupNorm is preferred here over both for a specific reason: Frozen BN registers
``weight``/``bias`` as BUFFERS, which changes the parameter count and would
silently invalidate ``guard_matches_the_published_architecture`` — the check
that compares this model against the published YOLOv8-S figure. GroupNorm keeps
``weight``/``bias`` as parameters (identical count to BN), carries no running
statistics, and actually normalises on a from-scratch build.

``num_groups`` is **derived** from the channel count rather than fixed at 32.
At the shipped ``WIDTH_MULT = 0.50`` every channel count in this tree happens to
be a multiple of 32, so a hardcoded 32 would work *for this build only*; at
YOLOv8-N's ``0.25`` the stem is 16 channels and ``GroupNorm(32, 16)`` raises.
The width multiplier is a live knob (it is what selects the scale), so the
derivation is what keeps it usable — and ``guard_norm_groups_survive_a_narrower_width``
rebuilds at 0.25 to prove the knob reaches the norm, rather than asserting it
from here.

Verified against torch 2.11.0 / torchvision 0.26.0 (the engine pin,
``tools/requirements-engine-pin.txt``).

Reference: Glenn Jocher, Ayush Chaurasia, Jing Qiu, "Ultralytics YOLOv8" (2023),
https://github.com/ultralytics/ultralytics — ``cfg/models/v8/yolov8.yaml`` for
the arch table and ``nn/modules/{block,head}.py`` for the C2f and Detect
shapes; Feng et al., "Generalized Focal Loss", arXiv:2006.04388 for DFL; and
TOOD (Feng et al., arXiv:2108.07755) for the task-aligned metric the assigner
uses. Architecture re-implemented from those specifications; no upstream code
is vendored.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import batched_nms, box_iou, complete_box_iou_loss

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("head.cls_preds.0.", "head.cls_preds.1.", "head.cls_preds.2.")

framework = "pytorch"
model_type = "torchvision_detection"
main_class = "YOLOv8S"
license = "Apache-2.0"
# The resolution the backbone actually sees: the transform below is built with
# min_size == max_size == this value, so a square input scales by 1.0.
image_size = 640
batch_size = 8
output_classes = 12
category = "object_detection"

#: YOLOv8-S scaling, from ``yolov8.yaml``'s ``scales:`` table. ``DEPTH_MULT``
#: multiplies the C2f block counts, ``WIDTH_MULT`` the channel counts, and
#: ``MAX_CHANNELS`` caps a stage's width BEFORE the multiplier is applied.
#: (n = 0.33/0.25/1024, s = 0.33/0.50/1024, m = 0.67/0.75/768,
#: l = 1.00/1.00/512, x = 1.00/1.25/512 — this triple is the only thing that
#: would change to rescale.)
DEPTH_MULT = 0.33
WIDTH_MULT = 0.50
MAX_CHANNELS = 1024

#: Channel counts are rounded to a multiple of this, as ``make_divisible`` does.
CHANNEL_DIVISOR = 8

#: Feature-map strides of the three head levels, smallest object first.
STRIDES = (8, 16, 32)

#: DFL bins per box edge. Each edge's distance from the anchor point is a
#: distribution over ``[0, REG_MAX - 1]`` **cell units**, decoded by expectation.
REG_MAX = 16

#: Task-aligned assigner knobs, at the published values.
TAL_TOPK = 10  # candidates considered per ground truth
TAL_ALPHA = 0.5  # classification-score exponent in the alignment metric
TAL_BETA = 6.0  # IoU exponent in the alignment metric

#: Loss weights, at the published values (``box`` / ``cls`` / ``dfl``).
BOX_LOSS_WEIGHT = 7.5
CLS_LOSS_WEIGHT = 0.5
DFL_LOSS_WEIGHT = 1.5

#: Inference post-processing, at the published validation values
#: (``conf=0.001``, ``iou=0.7``, ``max_det=300``) — the settings the mAP the
#: engine measures is computed under, not the ``conf=0.25`` predict default.
#:
#: DELIBERATELY NOT CARRIED: upstream also applies a ``max_nms=30000`` cap to
#: the candidate list BEFORE NMS. It is a latency bound, not a correctness one —
#: ``DETECTIONS_PER_IMAGE`` already bounds what leaves this function, and
#: ``torchvision.ops.batched_nms`` handles a large input correctly, just
#: slowly. It is omitted rather than added because a fourth budget constant
#: would need its own guard and its own mutation to be worth declaring, and an
#: unguarded knob is how a constant that reaches nothing gets shipped. If edge
#: eval latency ever becomes the complaint, add the cap AND its guard together.
SCORE_THRESH = 0.001
NMS_THRESH = 0.7
DETECTIONS_PER_IMAGE = 300

#: torchvision's ImageNet normalisation, matching every other CV template here.
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

_EPS = 1e-9


def _norm_groups(channels, maximum=32):
    """Largest group count ``<= maximum`` that divides ``channels``.

    GroupNorm requires ``channels % num_groups == 0``, and a hardcoded 32
    crashes as soon as ``WIDTH_MULT`` is lowered: YOLOv8-N's 0.25 puts 16
    channels in the stem. Deriving the count keeps the norm valid at any width
    the multiplier produces, not only the one built today.

    DUPLICATED from the sibling hand-written detectors on purpose -- zoo
    templates cannot import siblings (zero relative imports repo-wide). Its
    test is duplicated alongside it for the same reason.
    """
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _round_channels(channels: int) -> int:
    """Width-scale a channel count the way ``yolov8.yaml`` is parsed.

    ``make_divisible(min(channels, MAX_CHANNELS) * WIDTH_MULT, 8)``. The cap is
    applied BEFORE the multiplier and is what makes the m/l/x scales narrower
    than a naive product would give — it is transcribed rather than simplified
    away so the arch table stays comparable with the published one.
    """
    scaled = min(channels, MAX_CHANNELS) * WIDTH_MULT
    return max(
        CHANNEL_DIVISOR, int(math.ceil(scaled / CHANNEL_DIVISOR) * CHANNEL_DIVISOR)
    )


def _round_depth(blocks: int) -> int:
    """Depth-scale a block count: ``max(round(n * DEPTH_MULT), 1)``."""
    return max(int(round(blocks * DEPTH_MULT)), 1)


class ConvNormAct(nn.Module):
    """conv -> GroupNorm -> SiLU, the unit every block here is built from."""

    def __init__(self, in_ch, out_ch, ksize=1, stride=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            ksize,
            stride,
            padding=(ksize - 1) // 2,
            groups=groups,
            bias=False,
        )
        # GroupNorm, NOT BatchNorm. BN's running_mean/running_var are buffers
        # the averaging service ships and averages every round, and they
        # average badly across non-IID clients. The torchvision family used
        # to avoid this with FrozenBatchNorm2d -- which on a weights=None
        # backbone is a bit-exact identity (backend#3093), i.e. no
        # normalisation at all, and it moved to GroupNorm in model-zoo#259.
        # GroupNorm is used here instead because Frozen BN also moves
        # weight/bias into buffers and would change the parameter count,
        # silently invalidating the published-architecture guard. GroupNorm
        # keeps weight+bias as parameters (identical count to BN), normalises
        # for real, and has no running statistics.
        self.norm = nn.GroupNorm(_norm_groups(out_ch), out_ch, eps=1e-3)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Bottleneck(nn.Module):
    """3x3 -> 3x3 at full width, with an optional identity branch.

    Both convs are 3x3 and ``expansion`` has NO DEFAULT, for the same reason it
    has none in ``yolox_s.py``: inside a ``C2f`` it must be ``1.0`` because the
    split has already halved the channel count, and squeezing again there
    narrows the whole backbone and neck while leaving every loss finite. A
    default invited exactly that slip once and it shipped past thirty guards
    (see the note on ``_PINNED_TOTALS`` in the test file).

    YOLOv8's bottleneck differs from YOLOX's in the FIRST kernel: 3x3 here,
    1x1 there. That is a ~4x parameter difference per block and is the kind of
    drift only the published-count comparison catches.
    """

    def __init__(self, in_ch, out_ch, expansion, shortcut=True):
        super().__init__()
        hidden = max(1, int(out_ch * expansion))
        self.conv1 = ConvNormAct(in_ch, hidden, 3, stride=1)
        self.conv2 = ConvNormAct(hidden, out_ch, 3, stride=1)
        self.use_add = shortcut and in_ch == out_ch

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.use_add else y


class C2f(nn.Module):
    """YOLOv8's cross-stage stage, and the thing that makes it not a YOLOv5.

    ``cv1`` splits into two half-width branches. One is passed straight to the
    fusion; the other is fed through ``n`` bottlenecks **and every intermediate
    output is kept**, so ``cv2`` fuses ``2 + n`` branches rather than 2. That is
    the whole difference from YOLOv5/YOLOX's ``C3``, which concatenates only the
    final block output and the skip.

    A C3-shaped implementation of this class trains perfectly happily at a
    slightly smaller parameter count, which is why
    ``guard_c2f_fuses_every_intermediate_block`` reconstructs the expected
    branch list and compares it against the tensor ``cv2`` is actually handed,
    rather than reading the constructor.
    """

    def __init__(self, in_ch, out_ch, n=1, shortcut=False, expansion=0.5):
        super().__init__()
        self.hidden = int(out_ch * expansion)
        self.cv1 = ConvNormAct(in_ch, 2 * self.hidden, 1, stride=1)
        self.cv2 = ConvNormAct((2 + n) * self.hidden, out_ch, 1, stride=1)
        # expansion=1.0, NOT the stage's own 0.5: `hidden` is already the halved
        # branch, so the inner bottleneck runs at full branch width.
        self.m = nn.ModuleList(
            Bottleneck(self.hidden, self.hidden, 1.0, shortcut) for _ in range(n)
        )

    def forward(self, x):
        branches = list(self.cv1(x).chunk(2, dim=1))
        for block in self.m:
            branches.append(block(branches[-1]))
        return self.cv2(torch.cat(branches, dim=1))


class SPPF(nn.Module):
    """Spatial pyramid pooling, fast: ONE 5x5 max-pool applied three times in
    series, which has the receptive field of 5/9/13 pooling at a third of the
    cost. Widens the deepest stage's receptive field without extra stride."""

    def __init__(self, in_ch, out_ch, ksize=5, repeats=3):
        super().__init__()
        hidden = in_ch // 2
        self.repeats = repeats
        self.cv1 = ConvNormAct(in_ch, hidden, 1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=ksize, stride=1, padding=ksize // 2)
        self.cv2 = ConvNormAct(hidden * (repeats + 1), out_ch, 1, stride=1)

    def forward(self, x):
        outputs = [self.cv1(x)]
        for _ in range(self.repeats):
            outputs.append(self.pool(outputs[-1]))
        return self.cv2(torch.cat(outputs, dim=1))


class YOLOv8Backbone(nn.Module):
    """The C2f backbone. Returns the stride-8/16/32 feature maps.

    Stem is a plain 3x3 stride-2 conv, NOT YOLOX's ``Focus`` slice — YOLOv8
    dropped that.
    """

    #: ``(out channels at full width, blocks at full depth, shortcut)`` per
    #: stage after the stem, from ``yolov8.yaml``'s backbone list. Strides are
    #: 4 / 8 / 16 / 32.
    STAGES = (
        (128, 3, True),
        (256, 6, True),
        (512, 6, True),
        (1024, 3, True),
    )

    def __init__(self):
        super().__init__()
        stem_ch = _round_channels(64)
        self.stem = ConvNormAct(3, stem_ch, 3, stride=2)

        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        in_ch = stem_ch
        widths = []
        for out_full, blocks_full, shortcut in self.STAGES:
            out_ch = _round_channels(out_full)
            self.downsamples.append(ConvNormAct(in_ch, out_ch, 3, stride=2))
            self.stages.append(
                C2f(out_ch, out_ch, n=_round_depth(blocks_full), shortcut=shortcut)
            )
            widths.append(out_ch)
            in_ch = out_ch

        self.sppf = SPPF(widths[-1], widths[-1])
        #: (stride 8, stride 16, stride 32) channel counts, read by the neck.
        self.out_channels = (widths[1], widths[2], widths[3])

    def forward(self, x):
        x = self.stem(x)
        outputs = []
        for downsample, stage in zip(self.downsamples, self.stages):
            x = stage(downsample(x))
            outputs.append(x)
        # SPPF sits on the deepest map, after its C2f, exactly as in the yaml.
        return outputs[1], outputs[2], self.sppf(outputs[3])


class YOLOv8PAFPN(nn.Module):
    """Path-aggregation neck: one top-down pass then one bottom-up pass, so the
    stride-8 map carries semantic context and the stride-32 map carries
    localisation detail. Fusion stages are ``C2f`` without the identity branch,
    as in the yaml."""

    def __init__(self, in_channels, blocks=None):
        super().__init__()
        c3, c4, c5 = in_channels
        n = _round_depth(3) if blocks is None else blocks
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # top-down
        self.td_p4 = C2f(c5 + c4, c4, n=n, shortcut=False)
        self.td_p3 = C2f(c4 + c3, c3, n=n, shortcut=False)
        # bottom-up
        self.bu_conv3 = ConvNormAct(c3, c3, 3, stride=2)
        self.bu_p4 = C2f(c3 + c4, c4, n=n, shortcut=False)
        self.bu_conv4 = ConvNormAct(c4, c4, 3, stride=2)
        self.bu_p5 = C2f(c4 + c5, c5, n=n, shortcut=False)

        #: YOLOv8 keeps the backbone's widths at the head, unlike RTMDet which
        #: projects all three levels to a common width.
        self.out_channels = (c3, c4, c5)

    def forward(self, features):
        c3, c4, c5 = features

        p4 = self.td_p4(torch.cat((self.upsample(c5), c4), dim=1))
        p3_out = self.td_p3(torch.cat((self.upsample(p4), c3), dim=1))

        p4_out = self.bu_p4(torch.cat((self.bu_conv3(p3_out), p4), dim=1))
        p5_out = self.bu_p5(torch.cat((self.bu_conv4(p4_out), c5), dim=1))
        return p3_out, p4_out, p5_out


class YOLOv8Head(nn.Module):
    """The decoupled DFL head.

    Two things distinguish it from YOLOX's decoupled head:

    * there is **no objectness branch**. Classification confidence is the only
      score, which is why the assigner's alignment metric and the classifier's
      soft target have to carry localisation quality — see ``assign``.
    * the box branch emits ``4 * REG_MAX`` channels, a discrete distribution per
      edge rather than four numbers.

    Classification and regression get **separate** conv towers per level
    (``cls_convs`` and ``box_convs`` share no parameters). A coupled head trains
    perfectly happily and would be silently wrong, so the test file asserts the
    two subtrees are parameter-disjoint rather than reading it off here.
    """

    def __init__(self, num_classes, in_channels, strides=STRIDES, reg_max=REG_MAX):
        super().__init__()
        self.num_classes = num_classes
        self.strides = tuple(strides)
        self.reg_max = reg_max

        # The published widths: max(16, ch[0] // 4, reg_max * 4) for the box
        # tower and max(ch[0], min(num_classes, 100)) for the class tower.
        #
        # THE SECOND ONE MATTERS TO THE SEED CONTRACT. `min(num_classes, 100)`
        # is capped at 100 and `in_channels[0]` is 128 at WIDTH_MULT = 0.50, so
        # `cls_hidden` is 128 for EVERY class count -- which is what keeps
        # SEED_EXCLUDED_PREFIXES down to the three 1x1 predictors instead of
        # sweeping in the towers. At a narrower width (YOLOv8-N's 64) the max
        # can be won by the class term and the tower becomes class-count
        # dependent; `guard_class_tower_width_is_class_count_invariant` asserts
        # the property this build actually relies on rather than the formula.
        self.box_hidden = max(16, in_channels[0] // 4, reg_max * 4)
        self.cls_hidden = max(in_channels[0], min(num_classes, 100))

        self.box_convs = nn.ModuleList()
        self.box_preds = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        for channels in in_channels:
            self.box_convs.append(
                nn.Sequential(
                    ConvNormAct(channels, self.box_hidden, 3, stride=1),
                    ConvNormAct(self.box_hidden, self.box_hidden, 3, stride=1),
                )
            )
            self.box_preds.append(nn.Conv2d(self.box_hidden, 4 * reg_max, 1))
            self.cls_convs.append(
                nn.Sequential(
                    ConvNormAct(channels, self.cls_hidden, 3, stride=1),
                    ConvNormAct(self.cls_hidden, self.cls_hidden, 3, stride=1),
                )
            )
            self.cls_preds.append(nn.Conv2d(self.cls_hidden, num_classes, 1))

        self._init_prediction_biases()

    def _init_prediction_biases(self):
        """Upstream's ``Detect.bias_init``.

        The class bias is ``log(5 / nc / (640 / stride) ** 2)`` — "expect about
        five objects per image, spread over this level's cells" — which puts the
        prior far below ``SCORE_THRESH``. Without it the first batches are
        dominated by the ~8400 negative anchors per image.

        ⚠️ That prior is also why a freshly built model returns **zero
        detections**, so any eval assertion taken from a forward pass on an
        untrained model is vacuous. The decode is therefore driven directly with
        synthetic above-threshold head outputs in the test file, at batch >= 2.

        The box bias is 1.0, not 0: it makes the initial distribution favour the
        low bins, so a fresh anchor proposes a box about one cell across at its
        own level rather than at an arbitrary scale.
        """
        for stride, module in zip(self.strides, self.cls_preds):
            cells = (640.0 / float(stride)) ** 2
            nn.init.constant_(
                module.bias, math.log(5.0 / float(self.num_classes) / cells)
            )
        for module in self.box_preds:
            nn.init.constant_(module.bias, 1.0)

    def forward(self, features):
        """Return ``(cls_logits, dist_logits, anchors)``.

        ``cls_logits`` is ``(B, N, num_classes)``; ``dist_logits`` is
        ``(B, N, 4, reg_max)`` still in bin space; ``anchors`` is ``(N, 3)``
        holding each anchor point's ``x`` and ``y`` **in cell units** (cell
        index + 0.5, the centre of the cell) and its stride. Decoding is a
        separate step because the loss needs both forms: the assigner works in
        pixels, the DFL loss works in bin space.
        """
        cls_outputs = []
        dist_outputs = []
        anchors = []
        for level, (feature, stride) in enumerate(zip(features, self.strides)):
            cls_output = self.cls_preds[level](self.cls_convs[level](feature))
            dist_output = self.box_preds[level](self.box_convs[level](feature))

            batch, _, height, width = cls_output.shape
            cls_outputs.append(
                cls_output.permute(0, 2, 3, 1).reshape(batch, height * width, -1)
            )
            dist_outputs.append(
                dist_output.permute(0, 2, 3, 1).reshape(
                    batch, height * width, 4, self.reg_max
                )
            )

            yv, xv = torch.meshgrid(
                torch.arange(height, device=cls_output.device, dtype=cls_output.dtype),
                torch.arange(width, device=cls_output.device, dtype=cls_output.dtype),
                indexing="ij",
            )
            anchors.append(
                torch.stack(
                    (
                        xv.reshape(-1) + 0.5,
                        yv.reshape(-1) + 0.5,
                        torch.full_like(xv.reshape(-1), float(stride)),
                    ),
                    dim=1,
                )
            )

        return (
            torch.cat(cls_outputs, dim=1),
            torch.cat(dist_outputs, dim=1),
            torch.cat(anchors, dim=0),
        )


def _distribution_to_distance(dist_logits):
    """``(..., 4, reg_max)`` bin logits -> ``(..., 4)`` distances in cell units.

    The DFL decode: softmax over the bins, then the **expectation** under that
    distribution. An ``argmax`` here would train (the loss is computed on the
    logits, not on this) and would quantise every box to whole cells; dropping
    the softmax would let unnormalised logits scale the distance arbitrarily.
    Both leave the train step green, so this is a named function with its own
    guard.

    The bin vector is built here rather than stored, so the model carries no
    constant tensor for it — see the module docstring.
    """
    bins = torch.arange(
        dist_logits.shape[-1], device=dist_logits.device, dtype=dist_logits.dtype
    )
    return (dist_logits.softmax(dim=-1) * bins).sum(dim=-1)


def _decode_boxes(dist_logits, anchors):
    """Head output -> pixel ``xyxy``.

    ``anchors[:, 2]`` is the per-anchor stride, so this is where the head's
    multi-level structure enters the geometry: the anchor point and the four
    distances are both in cell units and both scaled by *that anchor's* stride.
    Using one stride for every level is a silent bug — the boxes stay finite and
    the model still trains, it just cannot represent small objects.
    """
    distance = _distribution_to_distance(dist_logits)
    anchor_x, anchor_y, stride = anchors[:, 0], anchors[:, 1], anchors[:, 2]
    return torch.stack(
        (
            (anchor_x - distance[..., 0]) * stride,
            (anchor_y - distance[..., 1]) * stride,
            (anchor_x + distance[..., 2]) * stride,
            (anchor_y + distance[..., 3]) * stride,
        ),
        dim=-1,
    )


def _boxes_to_distance(boxes_xyxy, anchors, reg_max):
    """Pixel ``xyxy`` -> the four cell-unit distances the DFL loss targets.

    Clamped to just below the top bin: a distance of exactly ``reg_max - 1``
    would make the upper interpolation bin ``reg_max``, i.e. out of range.
    """
    anchor_x = anchors[:, 0]
    anchor_y = anchors[:, 1]
    stride = anchors[:, 2]
    scaled = boxes_xyxy / stride.unsqueeze(-1)
    return torch.stack(
        (
            anchor_x - scaled[:, 0],
            anchor_y - scaled[:, 1],
            scaled[:, 2] - anchor_x,
            scaled[:, 3] - anchor_y,
        ),
        dim=-1,
    ).clamp(min=0.0, max=float(reg_max) - 1.0 - 0.01)


def _distribution_focal_loss(dist_logits, target_distance):
    """DFL: cross-entropy against the two bins a real-valued target falls
    between, weighted by how close it is to each.

    ``(P, 4, reg_max)`` and ``(P, 4)`` in, ``(P,)`` out (averaged over the four
    edges). This is what makes the distribution learn a *sharp* mode at the
    right distance rather than any distribution with the right mean — the
    expectation decode is many-to-one, so an L1 on the decoded distance alone
    leaves the shape unconstrained.
    """
    bins = dist_logits.shape[-1]
    lower = target_distance.long()
    upper = lower + 1
    weight_upper = target_distance - lower.to(target_distance.dtype)
    weight_lower = 1.0 - weight_upper

    flat = dist_logits.reshape(-1, bins)
    loss_lower = F.cross_entropy(flat, lower.reshape(-1), reduction="none").reshape(
        lower.shape
    )
    loss_upper = F.cross_entropy(flat, upper.reshape(-1), reduction="none").reshape(
        upper.shape
    )
    return (loss_lower * weight_lower + loss_upper * weight_upper).mean(dim=-1)


class YOLOv8S(nn.Module):
    """YOLOv8-S speaking the ``torchvision_detection`` contract."""

    def __init__(self, num_classes=output_classes, input_size=image_size):
        super().__init__()
        # +1 sigmoid channel for the background index the family handler
        # reserves. Since backend#3062 targets arrive in model space [1, C], so
        # the incoming label indexes the head DIRECTLY and channel 0 is only
        # ever a negative. See "Label space" in the module docstring.
        self.num_classes = int(num_classes) + 1
        self.input_size = int(input_size)
        self.reg_max = REG_MAX

        self.transform = GeneralizedRCNNTransform(
            min_size=self.input_size,
            max_size=self.input_size,
            image_mean=IMAGE_MEAN,
            image_std=IMAGE_STD,
            size_divisible=32,
        )
        self.backbone = YOLOv8Backbone()
        self.neck = YOLOv8PAFPN(self.backbone.out_channels)
        # reg_max passed EXPLICITLY, not left to the head's default. A default
        # argument is evaluated once at class-definition time, so a head reading
        # REG_MAX from its own signature would keep the value the module had at
        # import while `self.reg_max` above tracked the current one — the two
        # would disagree silently and the head's reshape would be the first
        # thing to notice. `guard_reg_max_reaches_the_head_and_the_decode`
        # rebuilds at a different REG_MAX to prove the knob is live rather than
        # decorative.
        self.head = YOLOv8Head(
            self.num_classes, self.neck.out_channels, reg_max=self.reg_max
        )

        self.score_thresh = SCORE_THRESH
        self.nms_thresh = NMS_THRESH
        self.detections_per_image = DETECTIONS_PER_IMAGE

    # -- contract entry point ------------------------------------------------

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError(
                "yolov8_s: train mode requires targets — the engine calls "
                "model(images, targets) for the loss dict and model(images) "
                "only in eval mode"
            )

        original_image_sizes = [
            (int(img.shape[-2]), int(img.shape[-1])) for img in images
        ]
        image_list, targets = self.transform(list(images), targets)

        cls_logits, dist_logits, anchors = self.head(
            self.neck(self.backbone(image_list.tensors))
        )

        if self.training:
            return self._losses(cls_logits, dist_logits, anchors, targets)

        detections = self._predictions(
            cls_logits, dist_logits, anchors, image_list.image_sizes
        )
        return self.transform.postprocess(
            detections, image_list.image_sizes, original_image_sizes
        )

    # -- training ------------------------------------------------------------

    def _losses(self, cls_logits, dist_logits, anchors, targets):
        """Task-aligned losses, returned as the handler's loss dict."""
        boxes = _decode_boxes(dist_logits, anchors)
        scores = cls_logits.detach().sigmoid()
        # Anchor points in PIXELS, for the geometry the assigner works in.
        anchor_points = anchors[:, :2] * anchors[:, 2:3]

        cls_targets = torch.zeros_like(cls_logits)
        fg_masks = []
        matched_boxes = []
        for index, target in enumerate(targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            if int(gt_boxes.shape[0]) == 0:
                fg_masks.append(
                    torch.zeros(
                        anchors.shape[0], dtype=torch.bool, device=anchors.device
                    )
                )
                continue

            fg_mask, labels, boxes_for_fg, aligned = self.assign(
                gt_boxes, gt_labels, scores[index], boxes[index].detach(), anchor_points
            )
            fg_masks.append(fg_mask)
            matched_boxes.append(boxes_for_fg)
            if int(fg_mask.sum()) > 0:
                cls_targets[index, fg_mask, labels] = aligned

        fg_mask = torch.stack(fg_masks, dim=0)
        # Upstream normalises every term by the summed soft target rather than
        # by the positive count, so a batch of poorly-aligned positives does not
        # get the same gradient scale as a batch of well-aligned ones.
        divisor = cls_targets.sum().clamp(min=1.0)

        loss_cls = (
            F.binary_cross_entropy_with_logits(
                cls_logits, cls_targets, reduction="sum"
            )
            / divisor
        )

        if bool(fg_mask.any()):
            target_boxes = torch.cat(matched_boxes, dim=0)
            weight = cls_targets[fg_mask].sum(dim=-1)
            pred_boxes = boxes[fg_mask]
            loss_box = (
                complete_box_iou_loss(pred_boxes, target_boxes, reduction="none")
                * weight
            ).sum() / divisor

            # Anchors repeated per image so each positive keeps its own stride.
            repeated = anchors.unsqueeze(0).expand(fg_mask.shape[0], -1, -1)[fg_mask]
            target_distance = _boxes_to_distance(target_boxes, repeated, self.reg_max)
            loss_dfl = (
                _distribution_focal_loss(dist_logits[fg_mask], target_distance)
                * weight
            ).sum() / divisor
        else:
            # Keep the dict shape and the graph: ``* 0.0`` on a real prediction
            # tensor keeps both branches connected to the loss, so a batch of
            # empty images does not produce ``None`` grads.
            loss_box = boxes.sum() * 0.0
            loss_dfl = dist_logits.sum() * 0.0

        return {
            "loss_box": BOX_LOSS_WEIGHT * loss_box,
            "loss_cls": CLS_LOSS_WEIGHT * loss_cls,
            "loss_dfl": DFL_LOSS_WEIGHT * loss_dfl,
        }

    @torch.no_grad()
    def assign(self, gt_boxes, gt_labels, pred_scores, pred_boxes, anchor_points):
        """Task-aligned assignment for one image.

        Returns ``(fg_mask, matched_labels, matched_boxes, aligned_scores)``,
        where ``aligned_scores`` is the **normalised alignment metric** each
        positive anchor's classifier is trained towards — not a hard 1. All
        inputs are for one image: ``pred_scores`` ``(N, C)`` probabilities,
        ``pred_boxes`` ``(N, 4)`` pixel xyxy, ``anchor_points`` ``(N, 2)`` pixel
        centres.

        Four things happen here, and every one of them fails **silently** if it
        is wrong — an all-negative image still yields a finite, small loss and a
        clean train step:

        1. the alignment metric is ``score ** TAL_ALPHA * iou ** TAL_BETA``, so
           a candidate is judged on classification and localisation *jointly*.
           The two exponents are wildly asymmetric on purpose (0.5 against 6.0):
           IoU dominates. Swapping them reverses which candidate wins and
           changes no cardinality at all, which is why the guards assert **which
           anchor** is selected and with what score, never how many;
        2. a candidate must have its anchor point **inside** the ground-truth
           box. Dropping that admits well-overlapping boxes anchored outside the
           object;
        3. the top ``TAL_TOPK`` candidates per ground truth are selected, by
           metric;
        4. an anchor claimed by two ground truths is awarded to the one it
           overlaps **best by IoU** — not by alignment metric, which is a
           different tie-break and a plausible-looking wrong one.

        Then the target is normalised: each positive's soft label is its metric
        rescaled so the best-aligned anchor for a ground truth is trained
        towards that ground truth's best IoU. A hard 1.0 target trains happily
        and simply loses the model's ability to say "this is a car, but I have
        it badly boxed" — the score it ranks by at inference *is* this quantity,
        because there is no objectness branch to carry it.

        ``tests/test_yolov8_s.py`` registers a mutation against each and proves
        it goes red.
        """
        num_anchors = int(pred_boxes.shape[0])
        labels = gt_labels.long()

        ious = box_iou(gt_boxes, pred_boxes).clamp(min=0.0)
        # (num_gt, N): each ground truth scored against its OWN class channel.
        scores = pred_scores[:, labels].t().clamp(min=0.0)
        alignment = scores.pow(TAL_ALPHA) * ious.pow(TAL_BETA)

        inside = self._anchors_inside(gt_boxes, anchor_points)
        candidate = alignment * inside.to(alignment.dtype)

        topk = min(TAL_TOPK, num_anchors)
        _, positions = torch.topk(candidate, topk, dim=1)
        selected = torch.zeros_like(candidate, dtype=torch.bool)
        selected.scatter_(1, positions, True)
        # topk pads with zero-metric entries when a ground truth has fewer than
        # `topk` geometrically valid candidates; those are not candidates.
        matching = selected & inside & (candidate > 0.0)

        claimed_by = matching.sum(dim=0)
        contested = claimed_by > 1
        if bool(contested.any()):
            # By IoU, which is the published tie-break — NOT by the alignment
            # metric. The two disagree whenever one ground truth is better
            # classified and the other better localised.
            best = (ious * matching.to(ious.dtype)).argmax(dim=0)
            matching[:, contested] = False
            matching[best[contested], contested] = True

        fg_mask = matching.any(dim=0)
        if not bool(fg_mask.any()):
            return (
                fg_mask,
                torch.zeros(0, dtype=torch.int64, device=pred_boxes.device),
                torch.zeros((0, 4), dtype=pred_boxes.dtype, device=pred_boxes.device),
                torch.zeros(0, dtype=alignment.dtype, device=pred_boxes.device),
            )

        matched_gt = matching.to(alignment.dtype).argmax(dim=0)[fg_mask]

        # Normalisation. Per ground truth: rescale its anchors' metrics so the
        # best-aligned one lands exactly on its best IoU.
        assigned = alignment * matching.to(alignment.dtype)
        best_alignment = assigned.amax(dim=1, keepdim=True)
        best_iou = (ious * matching.to(ious.dtype)).amax(dim=1, keepdim=True)
        normalised = (assigned * best_iou / (best_alignment + _EPS)).amax(dim=0)

        return (
            fg_mask,
            gt_labels.long()[matched_gt],
            gt_boxes[matched_gt],
            normalised[fg_mask],
        )

    @staticmethod
    def _anchors_inside(gt_boxes, anchor_points):
        """``(num_gt, N)`` — whether each anchor POINT lies inside each box.

        Strictly inside on all four sides, so a zero-area box admits nothing.
        This is the geometric prefilter; without it the metric alone would
        happily assign an anchor sitting outside the object to a box that merely
        overlaps well, and every loss stays finite.
        """
        x = anchor_points[:, 0].unsqueeze(0)
        y = anchor_points[:, 1].unsqueeze(0)
        left, top, right, bottom = (
            gt_boxes[:, 0:1],
            gt_boxes[:, 1:2],
            gt_boxes[:, 2:3],
            gt_boxes[:, 3:4],
        )
        return (x > left) & (x < right) & (y > top) & (y < bottom)

    # -- inference -----------------------------------------------------------

    def _predictions(self, cls_logits, dist_logits, anchors, image_sizes):
        decoded = _decode_boxes(dist_logits, anchors)
        scores = cls_logits.sigmoid()

        results = []
        for boxes, class_scores, (height, width) in zip(decoded, scores, image_sizes):
            boxes = boxes.clone()
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=float(width))
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=float(height))

            # ⚠️ CHANNEL 0 IS DROPPED BEFORE THE THRESHOLD, not after.
            #
            # The head is `output_classes + 1` wide and indexes by the incoming
            # label directly. Since backend#3062 the family handler hands this
            # model space `[1, C]` (`_targets_to_model_space`), so channel 0 is
            # never a positive target -- it is trained only as a negative, and
            # emitting it produces detections carrying dataset class `-1` once
            # the handler shifts back.
            #
            # The engine DOES drop them: `_detections_to_dataset_space` keeps
            # only `labels >= BACKGROUND_LABEL_OFFSET`. But that is downstream
            # of THIS function's score threshold and top-k, so a channel-0
            # candidate still consumes a detection slot a real object should
            # have had. Acute here: SCORE_THRESH is 0.001 and DETECTIONS_PER_IMAGE
            # is 300 against 8400 anchors, so the budget genuinely binds.
            #
            # Consequence worth stating plainly: this template REQUIRES the
            # family handler's shift. Fed raw 0-based dataset labels it would
            # discard the first class.
            class_scores = class_scores[:, 1:]
            num_anchors, num_classes = class_scores.shape
            flat_scores = class_scores.reshape(-1)
            labels = (
                torch.arange(1, num_classes + 1, device=boxes.device)
                .unsqueeze(0)
                .expand(num_anchors, num_classes)
                .reshape(-1)
            )
            box_index = (
                torch.arange(num_anchors, device=boxes.device)
                .unsqueeze(1)
                .expand(num_anchors, num_classes)
                .reshape(-1)
            )

            keep = flat_scores > self.score_thresh
            flat_scores, labels, box_index = (
                flat_scores[keep],
                labels[keep],
                box_index[keep],
            )
            candidate_boxes = boxes[box_index]

            keep = batched_nms(candidate_boxes, flat_scores, labels, self.nms_thresh)
            keep = keep[: self.detections_per_image]
            results.append(
                {
                    "boxes": candidate_boxes[keep],
                    "scores": flat_scores[keep],
                    "labels": labels[keep],
                }
            )
        return results
