"""YOLO11-S (Ultralytics, 2024) — C3k2/C2PSA PAFPN detector with a lightweight
decoupled DFL head, NMS post-processing and task-aligned label assignment,
written from scratch in PyTorch.

Offline variant: nothing is fetched at construction — no hub id, no ``timm``,
no ``transformers``, no ``ultralytics``, no torchvision pretrained enum, no
``download.pytorch.org`` (the #199 egress lockdown blocks it). Every layer is
built from the inlined width/depth multipliers and arch table below, so the
template constructs on a closed edge. No seed is hosted for this template and
there is no weight file: upload with ``weights=False``::

    user.upload_model("yolo11_s", weights=False)

Hosting COCO tensors as a tracebloc model-store seed (the #1499 pattern: a
matched ``<stem>_weights.pkl`` prepped by ``tools/prep_offline_weights.py`` and
strict-loaded after the architecture is built) is follow-up work, not part of
this roster addition. Until a dump is staged,
``tools/check_dump_coverage.py`` classifies this file NO_SEED and the sentence
above is what keeps that classification honest.

⚠️ AND THE UPSTREAM CHECKPOINT WOULD NOT FIT ANYWAY, for the two reasons
``yolov8_s.py`` records: the norm layers here are GroupNorm rather than the
published BatchNorm (see the federated note below), so a BN checkpoint carries
``running_mean``/``running_var`` this tree has no slot for and a strict load
fails loudly; and Ultralytics ships YOLO11 weights under AGPL-3.0 while this
file is an independent re-implementation declared Apache-2.0. The seed for this
template, if one is ever prepped, is prepped by us against THIS build
(backend#3055).

The name is ``yolo11``, with no ``v``
--------------------------------------
Upstream dropped the ``v`` at this generation: the config is
``cfg/models/11/yolo11.yaml`` and the weights are ``yolo11s.pt``. So the stem
here is ``yolo11_s`` and not ``yolov11_s``, which is a name upstream has never
used. Worth stating because "yolov11" is the more natural thing to type and
searching the zoo for it finds nothing.

Not to be confused with ``object_detection/pytorch/yolo_v8/``, which is a
different family entirely — see below.

Why it is NOT in the ``yolo`` family
-------------------------------------
``model_type = "yolo"`` is the legacy YOLOv1 grid contract: a fixed 448px
input, a ``[7, 7, num_classes + 10]`` target tensor, **one object per cell** (so
at most 49 objects per image, silently overwriting co-located ones) and an
external customer ``loss.py``. That family is frozen at its three existing
templates (backend#2982), one of which — ``yolo_v8/`` — is a YOLOv8 backbone
bent onto that YOLOv1 head.

This file declares ``torchvision_detection``, which is a **duck-typed contract
rather than a library dependency**:

* ``model(images, targets)`` in train mode -> a dict of scalar losses
* ``model(images)`` in eval mode -> ``List[Dict]`` with ``boxes`` (pixel
  xyxy), ``scores``, ``labels``
* ``images`` is a **list** of differently-sized 3-D tensors, because
  ``_rcnn_collate`` builds tuples rather than stacking (object counts vary per
  image)

Nothing about that contract mentions torchvision, and the engine's
``TorchvisionDetectionHandler`` needs no change to train this file. It gets a
real PAN neck, per-level heads at strides 8/16/32 and **8400 anchor points** at
640px against the legacy contract's 49 cells.

What YOLO11 changes from YOLOv8, and what it does NOT
------------------------------------------------------
This is the sibling this file is closest to, so the differences are the whole
content of the file. Four of them, and the fourth is the one that costs
parameters:

1. **C2f becomes C3k2** (``C3k2`` below). Same cross-stage skeleton — split in
   two, feed one half through ``n`` blocks, fuse ``2 + n`` branches — but the
   inner block differs in two ways that matter, see ``C3k2``'s docstring.
2. **A ``C2PSA`` follows the SPPF** on the deepest stage: stacked partial
   self-attention blocks. YOLOv8's backbone ends at the SPPF.
3. **The class tower is depthwise-separable**, not two dense 3x3 convolutions.
   Same change YOLOv10 made, and most of the head's parameter saving.
4. **The stage widths are not the downsample widths.** In ``yolov8.yaml`` a
   downsample conv and the stage after it share an output width
   (``Conv, [128, 3, 2]`` then ``C2f, [128, True]``). In ``yolo11.yaml`` the
   stage **doubles** it (``Conv, [128, 3, 2]`` then ``C3k2, [256, ...]``), and
   the same holds in the neck, where the stride-8 output is 128 channels while
   the backbone's stride-8 map is 256. So ``STAGES`` below carries a downsample
   width AND a stage width per row, and ``NECK_WIDTHS`` is its own table rather
   than the backbone's widths reused. A neck that reuses the backbone widths —
   the natural thing to copy from ``yolov8_s.py`` — builds, trains, and is
   wrong by about a million parameters.

And what it does NOT change: **YOLO11 keeps the NMS-based head.** It has one
detection branch, assigned once, and its predictions go through
``batched_nms``. That is the sibling relationship worth being careful about,
because ``yolov10_s.py`` sits next to this file and is *NMS-free*: it carries a
duplicated one2one branch, a ``detach`` isolating that branch from the trunk,
and a decode with no suppression at all. **None of that belongs here**, and
carrying it across by reflex would be a different architecture that trains
perfectly happily. YOLO11 is, in head terms, a YOLOv8 with a cheaper class
tower.

Why this is ONE file, and duplicated against its siblings
---------------------------------------------------------
A zoo template is uploaded as a single ``.py`` and there are zero relative
imports anywhere under ``model_zoo/`` — a template that imported a sibling
would arrive at the model checker with the sibling missing. So the backbone,
the neck, the head, the assigner and the losses all live here, and the blocks
the siblings also need (a conv-norm-act, a group-count helper, an SPPF, a PAN
skeleton, the DFL decode, the task-aligned assigner) are **deliberately
duplicated** rather than shared. Its tests are duplicated with it, for the same
reason: copied code that leaves its guard behind is how a duplicated assigner
silently loses a rule.

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
model, never read from this docstring** — ``tests/test_yolo11_s.py``'s
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
incoming label **directly** as the channel index. Channel 0 is therefore never
a positive target — it is trained only as a negative — and ``_predictions``
**slices it off BEFORE the score threshold and the NMS**, not after. That
ordering is the whole point: the engine does drop channel-0 rows, but it does
so downstream of this decode's budget, so a background candidate that survives
to there has already spent a detection slot a real object should have had. Same
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
buffer count at exactly zero and is the entire reason this build carries the
published **gradient** count rather than the published **parameter** count —
the two differ by exactly those 16 numbers. See ``tests/test_yolo11_s.py``.

Federated note (GroupNorm, not BatchNorm)
-----------------------------------------
The norm layers are GroupNorm. Upstream YOLO11 uses BatchNorm.

BN's ``running_mean``/``running_var`` are **buffers the averaging service ships
and averages every federated round**, and they average badly across non-IID
clients (see CLAUDE.md). The rest of this family avoids that with
``norm_layer=misc_nn_ops.FrozenBatchNorm``, whose statistics never update — and
on a ``weights=None`` backbone ``FrozenBatchNorm2d`` is a **bit-exact identity**
(``weight=1``, ``bias=0``, ``running_mean=0``, ``running_var=1``, verified in
backend#3093), so freezing it on a from-scratch template normalises nothing at
all.

GroupNorm is preferred here over both for a specific reason: Frozen BN registers
``weight``/``bias`` as BUFFERS, which changes the parameter count and would
silently invalidate ``guard_matches_the_published_architecture`` — the check
that compares this model against the published YOLO11-S figure. GroupNorm keeps
``weight``/``bias`` as parameters (identical count to BN), carries no running
statistics, and actually normalises on a from-scratch build.

``num_groups`` is **derived** from the channel count rather than fixed at 32.
This matters more here than on ``yolov8_s.py``: the C3k2 stages squeeze their
inner bottleneck to *half* the split branch, so at ``WIDTH_MULT = 0.50`` the
shallowest stage already carries a 16-channel norm (branch 32, inner 16) and a
hardcoded ``GroupNorm(32, ...)`` does not construct **at the shipped scale at
all**, never mind at YOLO11-N's 0.25. The width multiplier is a live knob (it is
what selects the scale), so the derivation is what keeps it usable, and
``guard_norm_groups_survive_a_narrower_width`` rebuilds at 0.25 to prove the
knob reaches the norm rather than asserting it from here.

Verified against torch 2.11.0 / torchvision 0.26.0 (the engine pin,
``tools/requirements-engine-pin.txt``).

Reference: Rahima Khanam and Muhammad Hussain, "YOLOv11: An Overview of the Key
Architectural Enhancements", arXiv:2410.17725 (2024), for the architectural
description; and Glenn Jocher and Jing Qiu, "Ultralytics YOLO11" (2024),
https://github.com/ultralytics/ultralytics — ``cfg/models/11/yolo11.yaml`` for
the arch table and scales and ``nn/modules/{block,conv,head}.py`` for the C3k2,
C3k, C2PSA, PSABlock, Attention and Detect shapes. Feng et al., "Generalized
Focal Loss", arXiv:2006.04388 for DFL; TOOD (Feng et al., arXiv:2108.07755) for
the task-aligned metric the assigner uses. Architecture re-implemented from
those specifications; no upstream code is vendored.
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
#
# THREE prefixes, not six: YOLO11 has ONE detection branch. `yolov10_s.py` next
# door needs six because its NMS-free design duplicates the whole head, and a
# list copied from there would name three prefixes this tree does not have.
SEED_EXCLUDED_PREFIXES = ("head.cls_preds.0.", "head.cls_preds.1.", "head.cls_preds.2.")

framework = "pytorch"
model_type = "torchvision_detection"
main_class = "YOLO11S"
license = "Apache-2.0"
# The resolution the backbone actually sees: the transform below is built with
# min_size == max_size == this value, so a square input scales by 1.0.
image_size = 640
batch_size = 8
output_classes = 12
category = "object_detection"

#: YOLO11-S scaling, from ``yolo11.yaml``'s ``scales:`` table. ``DEPTH_MULT``
#: multiplies the block counts, ``WIDTH_MULT`` the channel counts, and
#: ``MAX_CHANNELS`` caps a stage's width BEFORE the multiplier is applied.
#: (n = 0.50/0.25/1024, s = 0.50/0.50/1024, m = 0.50/1.00/512,
#: l = 1.00/1.00/512, x = 1.00/1.50/512 — this triple is the only thing that
#: would change to rescale, WITH ONE EXCEPTION, see ``C3K_STAGES``.)
#:
#: ⚠️ NOTE THE DEPTH: YOLO11's is 0.50 at BOTH n and s, where YOLOv8's is 0.33.
#: Copying 0.33 across from ``yolov8_s.py`` builds a model whose every block
#: count is right anyway (``max(round(2 * 0.33), 1) == 1``), so it is invisible
#: at this scale and wrong at l/x. ``guard_arch_table_is_live`` rebuilds at the
#: published N triple, where the same is true — so the depth multiplier is
#: additionally pinned against the published l scale, which is the only place
#: it separates.
DEPTH_MULT = 0.50
WIDTH_MULT = 0.50
MAX_CHANNELS = 1024

#: Channel counts are rounded UP to a multiple of this, as ``make_divisible``
#: does (``ceil``, not ``round`` — a nearest-rounding version agrees at every
#: width this table produces and would disagree at some other multiplier).
CHANNEL_DIVISOR = 8

#: Feature-map strides of the three head levels, smallest object first.
STRIDES = (8, 16, 32)

#: DFL bins per box edge. Each edge's distance from the anchor point is a
#: distribution over ``[0, REG_MAX - 1]`` **cell units**, decoded by expectation.
REG_MAX = 16

#: ``C3k2`` expansion — the fraction of the stage's OUTPUT width each of the two
#: split branches carries. The two shallow stages run at 0.25 and everything
#: else at 0.5, straight from the yaml's third positional argument
#: (``C3k2, [256, False, 0.25]`` against ``C3k2, [512, True]``).
C3K2_EXPANSION = 0.50
C3K2_SHALLOW_EXPANSION = 0.25

#: ⚠️ ``C3k2``'s inner bottleneck squeezes to HALF the split branch, and this is
#: the single easiest number in this file to get wrong, because ``C2f`` — the
#: block C3k2 subclasses and the one ``yolov8_s.py`` implements — does NOT.
#:
#: Upstream: ``C2f`` builds ``Bottleneck(self.c, self.c, shortcut, g, e=1.0)``,
#: passing the expansion EXPLICITLY; ``C3k2`` overrides ``self.m`` with
#: ``Bottleneck(self.c, self.c, shortcut, g)`` and lets ``Bottleneck``'s own
#: ``e=0.5`` default apply. So the same class name means a full-width inner conv
#: pair in YOLOv8 and a half-width one in YOLO11.
#:
#: ``yolov8_s.py``'s ``Bottleneck`` gives ``expansion`` no default at all,
#: precisely so a copy cannot inherit the wrong one silently — and that comment
#: reads as though 1.0 were the only correct value. It is the correct value
#: THERE. Here it is 0.5, and the difference is ~700k parameters at this scale.
C3K2_INNER_EXPANSION = 0.50

#: When ``C3k2``'s inner block is a ``C3k`` instead of a plain bottleneck, this
#: is the ``C3k``'s own block count / expansion / kernel — from
#: ``C3k2.__init__``'s ``C3k(self.c, self.c, 2, shortcut, g)`` and ``C3k``'s
#: defaults. TWO blocks, always: the count comes from that literal and is NOT
#: depth-scaled (``parse_model`` has already consumed the depth gain by the time
#: the C3k2 is constructed).
C3K_BLOCKS = 2
C3K_EXPANSION = 0.50
C3K_KERNEL = 3

#: ⚠️ A YOLO11 "scale" is the multiplier triple PLUS a per-stage block kind, and
#: this is the second half.
#:
#: ``C3k2``'s ``c3k`` flag selects between a plain bottleneck and a nested
#: ``C3k``. The yaml sets it per layer — ``True`` on the two deep backbone
#: stages and the neck's stride-32 fusion, ``False`` on the rest — and then
#: ``parse_model`` OVERRIDES it to ``True`` everywhere for the m/l/x scales::
#:
#:     if m is C3k2 and scale in "mlx":  # for M/L/X sizes
#:         args[3] = True
#:
#: So the flag is a property of (layer, scale), not of the layer alone, and it is
#: worth a constant of its own because it is invisible in the yaml for three of
#: the five scales. At n/s — this template — the yaml's own flags apply.
C3K_AT_SCALE_OVERRIDE_SCALES = ("m", "l", "x")

#: ``C2PSA`` — the partial-self-attention block after the SPPF. ``ratio`` is the
#: half it splits into; ``blocks`` is the yaml's ``[-1, 2, C2PSA, [1024]]``
#: repeat count, which IS depth-scaled (1 at n/s/m, 2 at l/x).
C2PSA_RATIO = 0.50
C2PSA_BLOCKS = 2

#: Attention inside ``PSABlock``. ``ATTN_RATIO`` sets the key width as a fraction
#: of the head width; ``ATTENTION_HEAD_DIM`` is what the head COUNT is derived
#: from — upstream writes ``num_heads=max(c // 64, 1)``, i.e. "one head per 64
#: channels". Deriving it rather than hardcoding a count is deliberate: see
#: ``Attention``'s docstring for why a hardcoded count is invisible to every
#: parameter and every shape in this file.
ATTN_RATIO = 0.50
ATTENTION_HEAD_DIM = 64

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
#: slowly. Omitted rather than added because a fourth budget constant would need
#: its own guard and its own mutation to be worth declaring, and an unguarded
#: knob is how a constant that reaches nothing gets shipped.
SCORE_THRESH = 0.001
NMS_THRESH = 0.7
DETECTIONS_PER_IMAGE = 300

#: torchvision's ImageNet normalisation, matching every other CV template here.
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

_EPS = 1e-9


def _norm_groups(channels, maximum=32):
    """Largest group count ``<= maximum`` that divides ``channels``.

    GroupNorm requires ``channels % num_groups == 0``, and a hardcoded 32 does
    not construct **at the shipped scale**: the shallowest C3k2 stage's inner
    bottleneck is 16 channels wide (see ``C3K2_INNER_EXPANSION``), and
    ``GroupNorm(32, 16)`` raises. Deriving the count keeps the norm valid at any
    width the multiplier produces, not only the one built today.

    DUPLICATED from the sibling hand-written detectors on purpose -- zoo
    templates cannot import siblings (zero relative imports repo-wide). Its
    test is duplicated alongside it for the same reason.
    """
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _round_channels(channels: int) -> int:
    """Width-scale a channel count the way ``yolo11.yaml`` is parsed.

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
    """Depth-scale a block count: ``max(round(n * DEPTH_MULT), 1)``.

    Upstream applies the gain only when ``n > 1``; every depth-scaled count in
    ``yolo11.yaml`` is 2, so the guard is unreachable here and is left out
    rather than written and never exercised.
    """
    return max(int(round(blocks * DEPTH_MULT)), 1)


class ConvNormAct(nn.Module):
    """conv -> GroupNorm -> SiLU, the unit every block here is built from.

    ``groups`` is exposed because YOLO11 needs genuinely depthwise convolutions
    in three places — the attention's positional encoding, and both spatial
    mixers of the head's class tower. ``groups=in_ch=out_ch`` is depthwise;
    anything else is a grouped conv and a different operator with a different
    parameter count.
    """

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
        # average badly across non-IID clients. The shipped family avoids this
        # with FrozenBatchNorm2d -- which on a weights=None backbone is a
        # bit-exact identity (backend#3093), i.e. no normalisation at all.
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
    """3x3 -> 3x3, with an optional identity branch.

    ``expansion`` has NO DEFAULT, and the reason is sharper here than in the
    siblings that say the same thing: YOLO11 uses this block at **two different
    expansions in the same stage**. ``C3k2``'s plain path squeezes to
    ``C3K2_INNER_EXPANSION`` (0.5), while the ``C3k`` nested inside ``C3k2``'s
    other path runs its bottlenecks at full branch width (1.0). A default would
    silently pick one of them for both.

    ``kernel`` is likewise explicit: ``C3k``'s whole reason to exist upstream is
    that its bottleneck kernel is configurable (``C3k(..., k=3)``), and the
    ``C3`` it subclasses uses ``1x1 -> 3x3`` where this uses ``kxk -> kxk``.
    """

    def __init__(self, in_ch, out_ch, expansion, shortcut=True, kernel=3):
        super().__init__()
        hidden = max(1, int(out_ch * expansion))
        self.conv1 = ConvNormAct(in_ch, hidden, kernel, stride=1)
        self.conv2 = ConvNormAct(hidden, out_ch, kernel, stride=1)
        self.use_add = shortcut and in_ch == out_ch

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.use_add else y


class C3k(nn.Module):
    """The nested block ``C3k2`` uses on YOLO11's deep stages: a ``C3``.

    Three convs, not two. ``cv1`` and ``cv2`` each take the FULL input; ``cv1``'s
    output is the one that goes through the bottlenecks and ``cv2``'s is an
    untouched skip; ``cv3`` fuses the two. That is the YOLOv5-era ``C3``
    topology, and the ``k`` in the name is the bottleneck kernel, which ``C3``
    fixes at ``1x1 -> 3x3`` and ``C3k`` makes ``kxk -> kxk``.

    Two silent ways to build this wrong, both guarded functionally because both
    leave every channel count, every shape and the parameter total identical:

    * routing the **skip** through the bottlenecks instead of ``cv1``'s branch
      (``cv3(cat(cv1(x), m(cv2(x))))``) — the block still mixes, just not the
      half the design mixes;
    * dropping the bottlenecks' identity branch. ``shortcut`` is ``True`` here
      because ``C3k2`` passes its own ``shortcut`` down and the yaml never sets
      it false at this generation (see ``C3k2``).
    """

    def __init__(self, in_ch, out_ch, blocks=C3K_BLOCKS, shortcut=True):
        super().__init__()
        hidden = int(out_ch * C3K_EXPANSION)
        self.cv1 = ConvNormAct(in_ch, hidden, 1, stride=1)
        self.cv2 = ConvNormAct(in_ch, hidden, 1, stride=1)
        self.cv3 = ConvNormAct(2 * hidden, out_ch, 1, stride=1)
        # expansion 1.0 -- FULL branch width, unlike C3k2's own plain path. See
        # Bottleneck's docstring: the two coexist one level apart.
        self.m = nn.Sequential(
            *(
                Bottleneck(hidden, hidden, 1.0, shortcut, kernel=C3K_KERNEL)
                for _ in range(blocks)
            )
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3k2(nn.Module):
    """YOLO11's cross-stage stage. A ``C2f`` skeleton with a different filling.

    The skeleton is YOLOv8's: ``cv1`` splits into two half-width branches, one
    goes straight to the fusion, the other is fed through ``n`` blocks and
    **every intermediate output is kept**, so ``cv2`` fuses ``2 + n`` branches.
    A ``C3``-shaped implementation that keeps only the last block's output is
    caught by ``cv2``'s channel count at ``n >= 2`` but NOT at ``n == 1``, which
    is what this scale builds — so the fusion input is checked functionally.

    Two things differ from ``C2f``, and both are invisible in a diff of the
    class skeletons:

    1. **the inner block can be a ``C3k``** rather than a bottleneck, selected
       by ``c3k``. See ``C3K_AT_SCALE_OVERRIDE_SCALES``: the flag is per (layer,
       scale), and flipping it on one stage moves ~300k parameters.
    2. **the plain path squeezes.** ``C3K2_INNER_EXPANSION`` is 0.5, where
       ``C2f`` passes 1.0. Read that constant's comment before changing this.

    ``shortcut`` is ``True`` on every C3k2 in ``yolo11.yaml``, INCLUDING the neck
    ones — another difference from ``yolov8.yaml``, whose neck ``C2f``s are built
    ``shortcut=False``. The yaml's ``False`` literals (``C3k2, [512, False]``)
    are the ``c3k`` flag, not the shortcut, and reading them as shortcuts is an
    easy mistake to make: it changes **no parameter and no shape**, only whether
    the neck's blocks are residual. Hence a functional guard rather than a
    comment alone.
    """

    def __init__(self, in_ch, out_ch, blocks=1, c3k=False, expansion=C3K2_EXPANSION):
        super().__init__()
        self.hidden = int(out_ch * expansion)
        self.c3k = bool(c3k)
        self.cv1 = ConvNormAct(in_ch, 2 * self.hidden, 1, stride=1)
        self.cv2 = ConvNormAct((2 + blocks) * self.hidden, out_ch, 1, stride=1)
        self.m = nn.ModuleList(
            C3k(self.hidden, self.hidden)
            if self.c3k
            else Bottleneck(self.hidden, self.hidden, C3K2_INNER_EXPANSION)
            for _ in range(blocks)
        )

    def forward(self, x):
        branches = list(self.cv1(x).chunk(2, dim=1))
        for block in self.m:
            branches.append(block(branches[-1]))
        return self.cv2(torch.cat(branches, dim=1))


class SPPF(nn.Module):
    """Spatial pyramid pooling, fast: ONE 5x5 max-pool applied three times in
    series, which has the receptive field of 5/9/13 pooling at a third of the
    cost. Widens the deepest stage's receptive field without extra stride.

    In series, not in parallel: the second pool sees the FIRST pool's output.
    Three parallel 5x5 pools produce a tensor of exactly the same shape from
    exactly the same parameters, so the difference is only visible in what the
    fusion conv is handed.
    """

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


class Attention(nn.Module):
    """The attention inside ``PSABlock``, at the published shape.

    ⚠️ ``num_heads`` IS PARAMETER-INVARIANT, which makes it the textbook "a
    constant that reaches nothing". ``nh_kd = num_heads * int((dim / num_heads)
    * ATTN_RATIO)`` equals ``dim * ATTN_RATIO`` for any head count that divides
    ``dim``, so ``qkv``'s output width — and therefore every parameter in this
    module — is the same whatever the head count is. A hardcoded ``num_heads =
    8`` would change no parameter, no shape, no loss key and no published
    figure; it would only re-factorise the attention into eight narrower heads.
    So the count is DERIVED from ``ATTENTION_HEAD_DIM`` and pinned on the built
    module, with a mutation that proves the derivation is live.

    ``pe`` is a depthwise positional encoding applied to ``v`` and added to the
    attention output — the reason this attention needs no learned position
    table and works at any feature-map size.
    """

    def __init__(self, dim, head_dim=None, attn_ratio=ATTN_RATIO):
        super().__init__()
        if head_dim is None:
            head_dim = ATTENTION_HEAD_DIM
        self.num_heads = max(1, dim // head_dim)
        self.head_dim = dim // self.num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        self.dim = dim

        qkv_channels = dim + 2 * self.key_dim * self.num_heads
        self.qkv = ConvNormAct(dim, qkv_channels, 1, stride=1, act=False)
        self.proj = ConvNormAct(dim, dim, 1, stride=1, act=False)
        self.pe = ConvNormAct(dim, dim, 3, stride=1, groups=dim, act=False)

    def forward(self, x):
        batch, channels, height, width = x.shape
        tokens = height * width
        qkv = self.qkv(x).view(
            batch, self.num_heads, 2 * self.key_dim + self.head_dim, tokens
        )
        query, key, value = qkv.split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        # Routed through SDPA rather than an explicit matmul-softmax-matmul, so
        # it dispatches to the fused attention kernels on hardware that has them
        # (backend#2090 established this for `relative_position_mlm.py`;
        # `yolov10_s.py` was the second template to adopt it and this is the
        # third).
        #
        # ⚠️ THE LAYOUT TRANSPOSE IS PART OF THE CALL, not tidiness. Upstream
        # holds q/k/v CHANNELS-FIRST — `(batch, heads, width, tokens)` — while
        # SDPA's contract is tokens-second-to-last, `(batch, heads, tokens,
        # width)`. Feeding the channels-first tensors straight in would attend
        # over CHANNELS instead of over spatial positions: same output shape,
        # finite losses, and a completely different operator.
        #
        # `scale` is passed EXPLICITLY even though SDPA's default is already
        # `1 / sqrt(query.size(-1))` and `self.scale` equals it exactly today.
        # Relying on the default would make `self.scale` a constant that reaches
        # nothing, so a future change to ATTN_RATIO — which moves `key_dim` and
        # therefore the correct scale — would be silently ignored by the kernel
        # while still reading as configuration here.
        attended = F.scaled_dot_product_attention(
            query.transpose(-2, -1),
            key.transpose(-2, -1),
            value.transpose(-2, -1),
            scale=self.scale,
        ).transpose(-2, -1)
        attended = attended.reshape(batch, channels, height, width)
        return self.proj(
            attended + self.pe(value.reshape(batch, channels, height, width))
        )


class PSABlock(nn.Module):
    """Attention then feed-forward, each as a **residual**.

    Both residuals are load-bearing and both are invisible to every structural
    check: dropping either leaves the parameter count, every tensor shape, every
    ``state_dict`` key and every loss key identical, and the model trains. So
    each is checked by comparing the block's output against a hand-computed
    ``x + attn(x)`` / ``y + ffn(y)`` rather than by reading the constructor.

    The feed-forward's second conv carries **no activation** — it is the block's
    output projection, and a SiLU there would make every residual contribution
    non-negative.
    """

    def __init__(self, channels, shortcut=True):
        super().__init__()
        self.attn = Attention(channels)
        self.ffn = nn.Sequential(
            ConvNormAct(channels, 2 * channels, 1, stride=1),
            ConvNormAct(2 * channels, channels, 1, stride=1, act=False),
        )
        self.add = bool(shortcut)

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


class C2PSA(nn.Module):
    """Partial self-attention, cross-stage — YOLO11's addition to the backbone.

    ``cv1`` splits the stage in two. **Only one half attends**; the other half
    bypasses every ``PSABlock`` entirely and is concatenated back at ``cv2``.
    That partial application is what the name means and what makes attention
    affordable at the stride-32 width at all — it halves both the token-mixing
    cost and the attention modules' own parameters.

    Attending both halves keeps every parameter, every shape and every key
    identical, so the bypass is checked functionally: the tensor ``cv2`` is
    handed must contain the split's first half UNCHANGED.

    Width-preserving by construction (``cv2`` maps ``2 * hidden`` back to
    ``in_ch``), which upstream asserts as ``assert c1 == c2``.
    """

    def __init__(self, in_ch, out_ch, blocks=1, ratio=C2PSA_RATIO):
        super().__init__()
        if in_ch != out_ch:
            raise ValueError(
                f"yolo11_s: C2PSA preserves its width, got in_ch={in_ch} "
                f"out_ch={out_ch}"
            )
        self.hidden = int(in_ch * ratio)
        self.cv1 = ConvNormAct(in_ch, 2 * self.hidden, 1, stride=1)
        self.cv2 = ConvNormAct(2 * self.hidden, in_ch, 1, stride=1)
        self.m = nn.Sequential(*(PSABlock(self.hidden) for _ in range(blocks)))

    def forward(self, x):
        bypass, attend = self.cv1(x).split((self.hidden, self.hidden), dim=1)
        return self.cv2(torch.cat((bypass, self.m(attend)), dim=1))


class YOLO11Backbone(nn.Module):
    """The C3k2 backbone. Returns the stride-8/16/32 feature maps.

    Stem is a plain 3x3 stride-2 conv (yaml layer 0), then four
    downsample-plus-stage pairs, then SPPF and C2PSA on the deepest map.

    ⚠️ THE DEEPEST OUTPUT IS ``C2PSA``'s, AND THE NECK USES IT TWICE. The yaml's
    bottom-up stride-32 fusion concatenates with layer **10** (the C2PSA), not
    with layer 9 (the SPPF) — so C2PSA's output is the only stride-32 tensor
    that leaves this module. Returning the SPPF output instead would be
    shape-identical and silent, which is why ``guard_deepest_stage_modules_are_applied``
    checks that both modules are actually called rather than merely built.
    """

    #: One row per downsample-plus-stage pair, transcribed from
    #: ``yolo11.yaml``'s backbone list. Strides after each pair are 4/8/16/32.
    #:
    #: ``(downsample out, stage out, blocks, c3k, expansion)`` at FULL width and
    #: FULL depth. The first two columns are separate because YOLO11's stage
    #: DOUBLES its downsample's width — see the module docstring's point 4.
    STAGES = (
        # yaml 1-2   P2/4
        (128, 256, 2, False, C3K2_SHALLOW_EXPANSION),
        # yaml 3-4   P3/8   -> head level 0
        (256, 512, 2, False, C3K2_SHALLOW_EXPANSION),
        # yaml 5-6   P4/16  -> head level 1
        (512, 512, 2, True, C3K2_EXPANSION),
        # yaml 7-8   P5/32  -> SPPF, C2PSA, head level 2
        (1024, 1024, 2, True, C3K2_EXPANSION),
    )

    #: yaml layer 0.
    STEM_CHANNELS = 64

    #: yaml layers 9 and 10 both take ``[1024]``.
    DEEPEST_CHANNELS = 1024

    def __init__(self):
        super().__init__()
        stem_ch = _round_channels(self.STEM_CHANNELS)
        self.stem = ConvNormAct(3, stem_ch, 3, stride=2)

        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        in_ch = stem_ch
        widths = []
        for down_full, out_full, blocks_full, c3k, expansion in self.STAGES:
            down_ch = _round_channels(down_full)
            out_ch = _round_channels(out_full)
            self.downsamples.append(ConvNormAct(in_ch, down_ch, 3, stride=2))
            self.stages.append(
                C3k2(
                    down_ch,
                    out_ch,
                    blocks=_round_depth(blocks_full),
                    c3k=c3k,
                    expansion=expansion,
                )
            )
            widths.append(out_ch)
            in_ch = out_ch

        deepest = _round_channels(self.DEEPEST_CHANNELS)
        self.sppf = SPPF(widths[-1], deepest)
        self.c2psa = C2PSA(deepest, deepest, blocks=_round_depth(C2PSA_BLOCKS))
        #: (stride 8, stride 16, stride 32) channel counts, read by the neck.
        self.out_channels = (widths[1], widths[2], deepest)

    def forward(self, x):
        x = self.stem(x)
        outputs = []
        for downsample, stage in zip(self.downsamples, self.stages):
            x = stage(downsample(x))
            outputs.append(x)
        # SPPF then C2PSA on the deepest map, exactly as in the yaml (9 then 10).
        return outputs[1], outputs[2], self.c2psa(self.sppf(outputs[3]))


class YOLO11PAFPN(nn.Module):
    """Path-aggregation neck: one top-down pass then one bottom-up pass, so the
    stride-8 map carries semantic context and the stride-32 map carries
    localisation detail.

    ⚠️ THE FUSION WIDTHS ARE THE YAML'S OWN, NOT THE BACKBONE'S. ``yolov8.yaml``
    lets each neck fusion return the width of the backbone level it meets, so
    ``yolov8_s.py``'s neck can be written entirely in terms of ``in_channels``.
    ``yolo11.yaml`` does not: its stride-8 fusion returns ``[256]``, which is
    128 channels at this scale against the backbone's stride-8 width of 256. So
    ``NECK_WIDTHS`` is transcribed separately, and the head's ``ch[0]`` is 128 —
    which is also what makes the head's class-tower width 128 rather than 256.

    Writing this neck in terms of ``in_channels`` builds, trains and is about a
    million parameters too heavy. It is the single most likely thing to be
    carried across from the sibling by mistake.
    """

    #: ``(top-down P4, top-down P3, bottom-up P4, bottom-up P5)`` fusion output
    #: widths at FULL width, from yaml layers 13 / 16 / 19 / 22; and the two
    #: bottom-up downsample convs from layers 17 / 20.
    NECK_WIDTHS = (512, 256, 512, 1024)
    DOWNSAMPLE_WIDTHS = (256, 512)

    #: ``c3k`` per fusion, from the same four yaml rows: only the stride-32 one
    #: is ``True`` at n/s.
    NECK_C3K = (False, False, False, True)

    #: yaml ``[-1, 2, C3k2, ...]`` on all four fusions.
    NECK_BLOCKS = 2

    def __init__(self, in_channels):
        super().__init__()
        c3, c4, c5 = in_channels
        td_p4_ch, td_p3_ch, bu_p4_ch, bu_p5_ch = (
            _round_channels(w) for w in self.NECK_WIDTHS
        )
        down3_ch, down4_ch = (_round_channels(w) for w in self.DOWNSAMPLE_WIDTHS)
        blocks = _round_depth(self.NECK_BLOCKS)
        c3k_td_p4, c3k_td_p3, c3k_bu_p4, c3k_bu_p5 = self.NECK_C3K

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # top-down (yaml 11-16)
        self.td_p4 = C3k2(c5 + c4, td_p4_ch, blocks=blocks, c3k=c3k_td_p4)
        self.td_p3 = C3k2(td_p4_ch + c3, td_p3_ch, blocks=blocks, c3k=c3k_td_p3)
        # bottom-up (yaml 17-22)
        self.bu_conv3 = ConvNormAct(td_p3_ch, down3_ch, 3, stride=2)
        self.bu_p4 = C3k2(down3_ch + td_p4_ch, bu_p4_ch, blocks=blocks, c3k=c3k_bu_p4)
        self.bu_conv4 = ConvNormAct(bu_p4_ch, down4_ch, 3, stride=2)
        self.bu_p5 = C3k2(down4_ch + c5, bu_p5_ch, blocks=blocks, c3k=c3k_bu_p5)

        self.out_channels = (td_p3_ch, bu_p4_ch, bu_p5_ch)

    def forward(self, features):
        c3, c4, c5 = features

        p4 = self.td_p4(torch.cat((self.upsample(c5), c4), dim=1))
        p3_out = self.td_p3(torch.cat((self.upsample(p4), c3), dim=1))

        p4_out = self.bu_p4(torch.cat((self.bu_conv3(p3_out), p4), dim=1))
        # `c5` again -- the C2PSA output, which the yaml's layer 21 reuses.
        p5_out = self.bu_p5(torch.cat((self.bu_conv4(p4_out), c5), dim=1))
        return p3_out, p4_out, p5_out


class YOLO11Head(nn.Module):
    """The decoupled DFL head, with YOLO11's lightweight class tower.

    ONE detection branch. YOLO11 is NMS-based: this head is assigned once, its
    predictions are suppressed by ``batched_nms`` in ``_predictions``, and there
    is no one2one branch, no duplicated tower and no ``detach`` anywhere.
    ``yolov10_s.py``'s head has all three and is otherwise nearly identical, so
    the absence is deliberate rather than unfinished.

    Three things distinguish it from ``yolov8_s.py``'s head, of which only the
    first is YOLO11-specific:

    * the class tower is **two depthwise-separable pairs** (depthwise 3x3 then
      pointwise 1x1, twice) where YOLOv8 spends two DENSE 3x3 convolutions.
      Same change YOLOv10 made. This is most of the head's parameter saving and
      is caught by the published count, which is worth noting because the
      structural guard on it is therefore redundant — it is kept for the error
      message, not for the coverage.
    * there is **no objectness branch**. Classification confidence is the only
      score, which is why the assigner's alignment metric and the classifier's
      soft target have to carry localisation quality — see ``assign``.
    * the box branch emits ``4 * REG_MAX`` channels, a discrete distribution per
      edge rather than four numbers.

    Classification and regression get **separate** towers per level
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
        # sweeping in the towers. At YOLO11-N's width ch[0] is 64 and the max IS
        # won by the class term (80 > 64 at the published 80 classes), so the
        # tower becomes class-count dependent there; the seed guard asserts the
        # property this build actually relies on rather than the formula.
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
            self.cls_convs.append(self._class_tower(channels))
            self.cls_preds.append(nn.Conv2d(self.cls_hidden, num_classes, 1))

        self._init_prediction_biases()

    def _class_tower(self, channels):
        """YOLO11's lightweight class tower: two depthwise-separable pairs.

        ``ConvNormAct(c, c, 3, groups=c)`` then ``ConvNormAct(c, hidden, 1)``,
        then the same shape again at ``hidden``. A DENSE 3x3 in either spatial
        position is YOLOv8's tower; it trains identically and is ~1.5M
        parameters too heavy at this scale.
        """
        return nn.Sequential(
            nn.Sequential(
                ConvNormAct(channels, channels, 3, stride=1, groups=channels),
                ConvNormAct(channels, self.cls_hidden, 1, stride=1),
            ),
            nn.Sequential(
                ConvNormAct(
                    self.cls_hidden,
                    self.cls_hidden,
                    3,
                    stride=1,
                    groups=self.cls_hidden,
                ),
                ConvNormAct(self.cls_hidden, self.cls_hidden, 1, stride=1),
            ),
        )

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


class YOLO11S(nn.Module):
    """YOLO11-S speaking the ``torchvision_detection`` contract."""

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
        self.backbone = YOLO11Backbone()
        self.neck = YOLO11PAFPN(self.backbone.out_channels)
        # reg_max passed EXPLICITLY, not left to the head's default. A default
        # argument is evaluated once at class-definition time, so a head reading
        # REG_MAX from its own signature would keep the value the module had at
        # import while `self.reg_max` above tracked the current one — the two
        # would disagree silently and the head's reshape would be the first
        # thing to notice. `guard_reg_max_reaches_the_head_and_the_decode`
        # rebuilds at a different REG_MAX to prove the knob is live rather than
        # decorative.
        self.head = YOLO11Head(
            self.num_classes, self.neck.out_channels, reg_max=self.reg_max
        )

        self.score_thresh = SCORE_THRESH
        self.nms_thresh = NMS_THRESH
        self.detections_per_image = DETECTIONS_PER_IMAGE

    # -- contract entry point ------------------------------------------------

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError(
                "yolo11_s: train mode requires targets — the engine calls "
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

        ``tests/test_yolo11_s.py`` registers a mutation against each and proves
        it goes red.

        Identical in every rule to ``yolov8_s.py``'s and ``yolov10_s.py``'s —
        the task-aligned assigner did not change at this generation. Duplicated
        rather than shared for the reason the module docstring gives, and
        re-guarded here rather than assumed, because a duplicated assigner that
        leaves its guards behind is exactly how one of these four rules goes
        missing.
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
            # of THIS function's score threshold and NMS budget, so a channel-0
            # candidate still consumes a detection slot a real object should
            # have had. Acute here: SCORE_THRESH is 0.001 and
            # DETECTIONS_PER_IMAGE is 300 against 8400 anchors, so the budget
            # genuinely binds.
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

            # YOLO11 IS NMS-BASED. This is the line `yolov10_s.py` does not have
            # and must not acquire, and the line this template must not lose:
            # the head is assigned one-to-MANY, so several anchors are trained
            # to fire on the same object and duplicate boxes are the design's
            # expected output, not an artefact.
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
