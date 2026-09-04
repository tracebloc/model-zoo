"""YOLOv10-S (Wang, Chen, Liu, Lin, Han et al., 2024) — NMS-free end-to-end
detector with consistent dual assignments, written from scratch in PyTorch.

Offline variant: nothing is fetched at construction — no hub id, no ``timm``,
no ``transformers``, no ``ultralytics``, no torchvision pretrained enum, no
``download.pytorch.org`` (the #199 egress lockdown blocks it). Every layer is
built from the inlined multipliers and architecture table below, so the
template constructs on a closed edge. No seed is hosted for this template and
there is no weight file: upload with ``weights=False``::

    user.upload_model("yolov10_s", weights=False)

Hosting COCO tensors as a tracebloc model-store seed (the #1499 pattern: a
matched ``<stem>_weights.pkl`` prepped by ``tools/prep_offline_weights.py`` and
strict-loaded after the architecture is built) is follow-up work, not part of
this roster addition. Until a dump is staged, ``tools/check_dump_coverage.py``
classifies this file NO_SEED and the sentence above is what keeps that
classification honest.

⚠️ AND THE UPSTREAM CHECKPOINT WOULD NOT FIT ANYWAY, for two reasons worth
recording before anyone tries. First, the norm layers here are GroupNorm rather
than the published BatchNorm (see the federated note below), so every norm
tensor's *identity* differs even where its shape does not — a BN checkpoint
carries ``running_mean``/``running_var`` this tree has no slot for, and a strict
load fails loudly, which is the good case. Second, both YOLOv10 distributions
(the authors' reference implementation and Ultralytics' port) ship weights under
AGPL-3.0 while this file is an independent re-implementation declared
Apache-2.0; the seed for this template, if one is ever prepped, is prepped by us
against THIS build (backend#3055).

Why it is NOT in the ``yolo`` family
------------------------------------
``model_type = "yolo"`` is the legacy YOLOv1 grid contract: a fixed 448px
input, a ``[7, 7, num_classes + 10]`` target tensor, **one object per cell** (so
at most 49 objects per image, silently overwriting co-located ones) and an
external customer ``loss.py``. That family is frozen at its three existing
templates (backend#2982), and nothing here touches them.

This file declares ``torchvision_detection``, which is a **duck-typed contract
rather than a library dependency**:

* ``model(images, targets)`` in train mode -> a dict of scalar losses
* ``model(images)`` in eval mode -> ``List[Dict]`` with ``boxes`` (pixel xyxy),
  ``scores``, ``labels``
* ``images`` is a **list** of differently-sized 3-D tensors, because
  ``_rcnn_collate`` builds tuples rather than stacking (object counts vary per
  image)

Nothing about that contract mentions torchvision, and the engine's
``TorchvisionDetectionHandler`` needs no change to train this file. The
platform's eval side is equally permissive: metrics are torchmetrics
``MeanAveragePrecision`` over xyxy, so an NMS-free detector scores exactly like
an NMS one — it simply emits its final predictions, which is what makes this
architecture shippable here at all.

WHAT IS NEW IN YOLOv10, AND WHY IT IS THE POINT OF THIS FILE
=============================================================
YOLOv10's contribution is **end-to-end detection without NMS**, achieved by
training two heads and deploying one.

1. Consistent dual assignments
------------------------------
Every previous YOLO in this zoo (``yolov8_s``, ``yolov9_s``) assigns each
ground truth to ``TAL_TOPK = 10`` anchors. That is what makes them fast to
train — dense supervision — and it is also what forces NMS at inference: ten
anchors were *taught* to fire on one object, so ten boxes come out.

YOLOv10 keeps that head and adds a second one:

* the **one2many** head, assigned with ``TAL_TOPK`` (10) anchors per ground
  truth. Rich supervision, and it is the branch that trains the backbone;
* the **one2one** head, assigned with ``ONE2ONE_TOPK`` (**1**) anchor per
  ground truth. One object, one positive anchor, so its predictions need no
  de-duplication and it is the only head that runs at inference.

The **consistency** in "consistent dual assignments" is the part that is easy
to implement and get subtly wrong: both heads use the **same alignment metric**
(``score ** TAL_ALPHA * iou ** TAL_BETA``, at the same published exponents), so
the one2one head's single positive is the anchor the one2many head *also* ranked
first. That is the whole mechanism — the supervision of the deployed head
agrees with the supervision of the branch that shaped the features, so the rich
one2many gradient is not pulling the features away from what the one2one head
is being asked to do. Give the two heads different metrics and the model trains
perfectly happily to a worse place. ``ONE2ONE_TOPK`` is therefore the ONLY
assigner knob that differs between the two calls, and
``guard_dual_assignment_is_consistent`` asserts exactly that rather than
reading it off here.

2. ⚠️ THE ``detach`` — THE SILENT-FAILURE CENTREPIECE OF THIS TEMPLATE
-----------------------------------------------------------------------
The one2one head is fed **detached** features::

    one2one = self._branch_forward(
        [feature.detach() for feature in features], ...
    )

so its gradient stops at the head and never reaches the neck or the backbone.

This is not an optimisation. The one2many branch exists precisely because
one-to-one supervision alone is too sparse to shape a backbone — that is the
convergence problem DETR spent four papers fixing. If the one2one head's
gradient is allowed into the backbone, it competes with the one2many branch for
exactly the features the one2many branch is there to provide, and the published
result does not reproduce.

**And it fails silently in every way this repo can normally see.** Remove the
``.detach()`` and:

* the model still constructs, and every shape is unchanged;
* the parameter count is identical — a ``detach`` is not a parameter;
* ``model(images, targets)`` returns the same six finite loss keys;
* the losses still go down, and the template still overfits one object;
* ``tests/test_od_torchvision_family_train_step.py`` stays green.

Nothing about the *value* of anything is wrong. What is wrong is the **shape of
the gradient graph**, so the guard that covers it has to measure the gradient
graph: ``guard_one2one_head_is_detached_from_the_backbone`` backpropagates the
one2one losses ALONE and asserts every backbone and neck parameter receives
either no gradient or an exactly-zero one, while the one2one head's own
parameters receive a non-zero one — and then does the mirror, backpropagating
the one2many losses alone and requiring the backbone gradient to be non-zero,
so "isolated" cannot be satisfied by a model where nothing trains at all.

3. NMS-free top-k decode
------------------------
Because the deployed head is one-to-one, inference is a **ranking** rather than
a suppression: take the best ``DETECTIONS_PER_IMAGE`` candidates and emit them.
``_predictions`` calls no NMS at all — there is no ``batched_nms`` import in
this file, unlike its two siblings — and the decode is the published two-stage
top-k: first the best ``max_det`` **anchors** by their strongest class score,
then the best ``max_det`` **(anchor, class) pairs** among those. The second
stage is what lets one anchor contribute two classes when it genuinely
straddles two objects, without letting a single confident anchor's whole class
row crowd out every other anchor.

4. The efficiency blocks
------------------------
YOLOv10's other half is a set of cheaper blocks, each replacing a specific
bottleneck in the YOLOv8 graph this architecture starts from:

* ``SCDown`` — spatial-channel decoupled downsampling. A 1x1 pointwise conv
  does the channel change at **full resolution**, then a depthwise kxk stride-2
  conv does the spatial reduction. A plain strided 3x3 does both at once and
  costs ``in * out * 9``; this costs ``in * out + out * 9``. ⚠️ Putting the
  stride on the **pointwise** conv instead is shape-identical AND
  parameter-identical, and reduces the block to point-sampling every other
  pixel — see ``guard_scdown_downsamples_with_the_depthwise_conv``;
* ``CIB`` — compact inverted block. Depthwise 3x3, pointwise expand, an inner
  spatial mixer, pointwise project, depthwise 3x3. Only the two pointwise convs
  are dense, so the block is nearly free at the widths where YOLOv8 spends most
  of its parameters (the stride-32 stage);
* ``RepVGGDW`` — the ``CIB``'s inner mixer when ``lk`` (large kernel) is set: a
  depthwise **7x7** and a depthwise 3x3, **summed before one shared
  activation**, which is what makes the pair collapse into a single 7x7 at
  deployment. Activating each branch separately keeps the parameter count and
  the shapes and destroys that property;
* ``PSA`` — partial self-attention. The stage splits in two and **only one
  half** attends; the other half bypasses the attention untouched and is
  concatenated back. That is what "partial" means, and it is the reason
  attention is affordable at the stride-32 stage at all.

⚠️ ``Attention``'s ``num_heads`` IS PARAMETER-INVARIANT, so it is a knob that
reaches nothing unless something checks it. ``nh_kd = num_heads *
int((dim / num_heads) * ATTN_RATIO)`` is ``dim * ATTN_RATIO`` for **any** head
count that divides ``dim``, so ``qkv`` emits ``dim + 2 * dim * ATTN_RATIO``
channels no matter what ``num_heads`` is. Hardcoding ``num_heads = 8`` here
would change no parameter, no shape, no loss key and no published count — it
would only re-factorise the attention. ``ATTENTION_HEAD_DIM`` is therefore
derived into a head count and pinned on the **built** module, with a mutation
that proves the derivation is live (see
``guard_attention_head_count_is_derived_and_reaches_the_attention``).

The published anchors, and what they actually count
===================================================
⚠️ THE v10 YAML FILES CARRY NO PARAMETER COUNTS. Unlike ``yolov9{t,s,c}.yaml``,
whose header comments state a total this repo's sibling template anchors to,
``yolov10{n,s}.yaml`` carry none, and the published docs table quotes params to
one decimal (``2.3``/``7.2`` million) — which is far too coarse to distinguish
a correct build from one a few thousand parameters wrong.

That looks disqualifying and is not, because the model summary Ultralytics
prints for each scale gives exact integers. But there are **THREE** distinct
figures per scale, not two, and reading the wrong one off makes this template
look 20% too big:

===============================  =============  ==============
graph                            yolov10n       yolov10s
===============================  =============  ==============
dual head, unfused (**TRAINED**) 2,775,520      8,128,272
  of which gradients             2,775,504      8,128,256
dual head, fused                 2,762,608      8,096,880
one2one only, fused (**DOCS**)   2,299,264      7,248,960
===============================  =============  ==============

**This template anchors to the first row**, because the dual-head unfused graph
is the one it builds and trains: the ticket asks for "NMS-free, dual
assignment", both heads exist in the module tree, both are shipped and averaged
every federated round, and GroupNorm cannot be fused into a conv the way BN can.

**The docs table's ``2.3``/``7.2`` is the LAST row** — the one2one-only graph
after ``model.fuse()``, which merges Conv+BatchNorm *and* deletes the one2many
head. Upstream says so explicitly: the Ultralytics docs page footnotes "Params
and FLOPs values are for the fused model after ``model.fuse()``, which merges
Conv and BatchNorm layers and removes the auxiliary one-to-many detection
head", and THU-MIG/yolov10#13's author answers a params question with "the
one-to-many head is not needed during inference, this part of params and FLOPs
can be ignored" — their ``flops.py`` deletes ``cv2``/``cv3`` before measuring.
So a reader comparing this template's 8.1M against the README's 7.2M is
comparing two different graphs, and that is why the table above is here rather
than a single number.

⚠️ AND MOST FIGURES IN THE WILD ARE NEITHER. A summary line quoting
``2,69x,xxx`` or ``8,0xx,xxx`` is almost always a custom-class fine-tune, with
an "Overriding model.yaml nc=80 with nc=<N>" line above it that is easy to miss
(``2,707,430`` is nc=1, ``2,708,210`` is nc=3). This head is exactly linear in
the class count at **+774 parameters per class** for any ``nc <= 100``
(``cls_hidden`` is ``max(128, min(nc, 100)) = 128`` there, so each extra class
adds one 1x1 row plus bias on three levels of two branches: ``3 * 129 * 2``),
which is the cheap way to check whether a figure found anywhere is a COCO one.
``tests/test_yolov10_s.py`` asserts that slope.

The gradient/total gap is exactly **16** at both scales, and it is the same DFL
gap as this template's two siblings: upstream stores the bin-index vector as a
frozen ``Conv2d`` weight (``requires_grad=False``, hence "16 params, 0
gradients"), while this file builds it with ``torch.arange`` inside the decode
and stores nothing. **So this template's TOTAL parameter count is the published
GRADIENT count, exactly** — 8,128,256 at 80 classes — and that gap is stated
here rather than absorbed into a tolerance.

The architecture table is a live knob, not decoration
-----------------------------------------------------
``WIDTH_MULT``/``DEPTH_MULT``/``MAX_CHANNELS`` and ``BACKBONE_P5_BLOCK`` are the
transcribed yaml. YOLOv10 scales by multipliers the way YOLOv8 does, **but the
multipliers are not the whole scale**: the n and s yamls also differ in a
*block kind*. ``yolov10n`` uses a plain ``C2f`` at the stride-32 backbone stage
while ``yolov10s`` uses a ``C2fCIB`` there, and no multiplier expresses that.
``tests/test_yolov10_s.py`` rebuilds this module with the published
**YOLOv10-N** multipliers *and* its block kind and asserts the result carries
YOLOv10-N's published gradient count exactly — which is what proves the table
reaches the model rather than sitting next to it, and is a second published
anchor measured on the built model.

Label space is MODEL space ``[1, C]``
-------------------------------------
Since backend#3062 the family handler owns the translation: it shifts dataset
labels ``[0, C-1]`` up to ``[1, C]`` on the way in (``_targets_to_model_space``)
and shifts predictions back with the background row dropped on the way out
(``_detections_to_dataset_space``, which keeps only
``labels >= BACKGROUND_LABEL_OFFSET``).

So each head allocates ``output_classes + 1`` sigmoid channels and uses the
incoming label **directly** as the channel index. Channel 0 is therefore never
a positive target — it is trained only as a negative — and ``_predictions``
**slices it off BEFORE the top-k**, not after. That ordering matters more here
than on the NMS siblings, not less: the whole decode IS a top-k budget, so a
background candidate that survives into it has taken a detection slot from a
real object with nothing downstream to recover it.

Consequence, stated plainly: this template **requires** the family handler's
shift. Fed raw 0-based dataset labels it would discard the first class.

Regression parameterisation (DFL) and where the bins live
---------------------------------------------------------
Each box edge is predicted as a ``REG_MAX``-bin discrete distribution over
distances from the anchor point, in **cell units** (multiples of that level's
stride), and decoded by its expectation. Two consequences:

* the maximum representable distance is ``REG_MAX - 1 = 15`` cells, i.e. 120px
  at stride 8 and 480px at stride 32. That is the published design and it is
  why the coarse level exists;
* every decoded box is a valid xyxy **by construction** — the distances are a
  softmax expectation and so non-negative, giving ``x2 >= x1`` without a clamp
  that would silently kill the gradient on the clamped side.

The bin index vector is built with ``torch.arange`` inside the decode rather
than stored as a parameter or a buffer, which keeps buffer count at exactly
zero and is what makes this build carry the published **gradient** count
exactly — see the table above.

Federated note (GroupNorm, not BatchNorm)
-----------------------------------------
The norm layers are GroupNorm. Upstream YOLOv10 uses BatchNorm.

BN's ``running_mean``/``running_var`` are **buffers the averaging service ships
and averages every federated round**, and they average badly across non-IID
clients (see CLAUDE.md). The rest of this family avoids that with
``norm_layer=misc_nn_ops.FrozenBatchNorm`` — and on a ``weights=None`` backbone
``FrozenBatchNorm2d`` is a **bit-exact identity** (``weight=1``, ``bias=0``,
``running_mean=0``, ``running_var=1``, verified in backend#3093), so freezing it
on a from-scratch template normalises nothing at all.

GroupNorm is preferred here over both for a specific reason: Frozen BN registers
``weight``/``bias`` as BUFFERS, which changes the parameter count and would
silently invalidate the check that compares this model against the published
YOLOv10 figures. GroupNorm keeps ``weight``/``bias`` as parameters (identical
count to BN), carries no running statistics, and actually normalises on a
from-scratch build.

``num_groups`` is **derived** from the channel count rather than fixed at 32,
and this template has more need of that than either sibling: ``CIB`` and
``SCDown`` are built from **depthwise** convolutions, so the tree carries norms
at every intermediate width the inverted blocks produce rather than only at the
round stage widths. ``guard_norm_groups_are_derived_from_the_channel_count``
rebuilds at the published YOLOv10-N multipliers, where the stem is 16 channels
and ``GroupNorm(32, 16)`` raises outright.

Verified against torch 2.11.0 / torchvision 0.26.0 (the engine pin,
``tools/requirements-engine-pin.txt``).

Reference: Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han,
Guiguang Ding, "YOLOv10: Real-Time End-to-End Object Detection",
arXiv:2405.14458 (2024), NeurIPS 2024, and its reference implementation
https://github.com/THU-MIG/yolov10; the multipliers and block table are
transcribed from ``ultralytics/cfg/models/v10/yolov10{n,s}.yaml`` and the block
shapes from ``ultralytics/nn/modules/{block,conv,head}.py``. Feng et al.,
"Generalized Focal Loss", arXiv:2006.04388 for DFL; TOOD (Feng et al.,
arXiv:2108.07755) for the task-aligned metric both assigners use. Architecture
re-implemented from those specifications; no upstream code is vendored.
"""

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import box_iou, complete_box_iou_loss

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
#
# NOTE this template has TWICE the class-shaped tensors of its siblings: the
# one2many head and the one2one head each carry a per-level class predictor,
# and BOTH are sized from output_classes. A prefix list copied from yolov8_s or
# yolov9_s would cover only half of them and a hosted seed would ship the other
# three — the exact shape mismatch backend#2642 exists to remove.
SEED_EXCLUDED_PREFIXES = (
    "head.cls_preds.0.",
    "head.cls_preds.1.",
    "head.cls_preds.2.",
    "head.one2one_cls_preds.0.",
    "head.one2one_cls_preds.1.",
    "head.one2one_cls_preds.2.",
)

framework = "pytorch"
model_type = "torchvision_detection"
main_class = "YOLOv10S"
license = "Apache-2.0"
# The resolution the backbone actually sees: the transform below is built with
# min_size == max_size == this value, so a square input scales by 1.0.
image_size = 640
batch_size = 8
output_classes = 12
category = "object_detection"

#: YOLOv10-S scaling, from the ``scales:`` table shared by the v10 yamls.
#: ``DEPTH_MULT`` multiplies the block counts, ``WIDTH_MULT`` the channel
#: counts, and ``MAX_CHANNELS`` caps a stage's width BEFORE the multiplier is
#: applied. (n = 0.33/0.25/1024, s = 0.33/0.50/1024, m = 0.67/0.75/768,
#: b = 0.67/1.00/512, l = 1.00/1.00/512, x = 1.00/1.25/512.)
DEPTH_MULT = 0.33
WIDTH_MULT = 0.50
MAX_CHANNELS = 1024

#: ⚠️ NOT EXPRESSIBLE AS A MULTIPLIER, and the reason this template needs a
#: block-kind selector at all. The stride-32 backbone stage is a plain ``C2f``
#: in ``yolov10n.yaml`` and a ``C2fCIB`` in ``yolov10s.yaml`` — the two scales
#: differ in a *block kind*, not only in width and depth. ``"c2f"`` or
#: ``"c2f_cib"``; anything else raises rather than falling back.
#:
#: This is a live knob and the test file proves it: rebuilding with YOLOv10-N's
#: multipliers AND ``"c2f"`` here reproduces YOLOv10-N's published gradient
#: count (2,775,504) exactly, while rebuilding with N's multipliers alone does
#: not. A single hardcoded kind would therefore be caught by a published figure
#: rather than by a self-measured one.
BACKBONE_P5_BLOCK = "c2f_cib"

#: Channel counts are rounded to a multiple of this, as the yaml is parsed.
CHANNEL_DIVISOR = 8

#: Full-width channel and block counts, before the multipliers. Transcribed
#: from the yaml's backbone list; strides are 4 / 8 / 16 / 32.
STEM_CHANNELS = 64
BACKBONE_STAGES = (
    #: (out channels at full width, blocks at full depth)
    (128, 3),  # stride 4
    (256, 6),  # stride 8
    (512, 6),  # stride 16
    (1024, 3),  # stride 32
)
#: Blocks at full depth in each of the four neck fusion stages.
NECK_BLOCKS = 3

#: ``SPPF``'s single 5x5 max-pool is applied this many times **in series**, for
#: an effective 5/9/13 receptive field at a third of the cost of three pools.
SPPF_REPEATS = 3
SPPF_KERNEL = 5

#: ``PSA``'s split fraction: half the channels attend, half bypass. This is
#: what "partial self-attention" names.
PSA_RATIO = 0.5

#: ``Attention``'s key width as a fraction of each head's width, and the head
#: width the head COUNT is derived from.
#:
#: ⚠️ ``ATTENTION_HEAD_DIM`` is parameter-invariant — see the module docstring.
#: ``num_heads * int((dim / num_heads) * ATTN_RATIO)`` is ``dim * ATTN_RATIO``
#: for any head count dividing ``dim``, so a wrong head count changes no
#: parameter and no shape. It has its own guard for exactly that reason.
ATTN_RATIO = 0.5
ATTENTION_HEAD_DIM = 64

#: ``CIB``'s inner spatial mixer is a ``RepVGGDW`` (depthwise 7x7 + 3x3) rather
#: than a single depthwise 3x3 when this is set. The yaml writes it as the third
#: positional argument of ``C2fCIB, [1024, True, True]``.
CIB_LARGE_KERNEL = True
#: ``RepVGGDW``'s two depthwise kernels. The 7x7 is the "large kernel"; the 3x3
#: is the re-parameterisable branch summed into it.
REPVGGDW_KERNELS = (7, 3)

#: Feature-map strides of the three head levels, smallest object first.
STRIDES = (8, 16, 32)

#: DFL bins per box edge. Each edge's distance from the anchor point is a
#: distribution over ``[0, REG_MAX - 1]`` **cell units**, decoded by expectation.
REG_MAX = 16

#: Task-aligned assigner knobs, at the published values. BOTH heads use these
#: two exponents — that shared metric is the "consistent" in "consistent dual
#: assignments" and the only reason the one2one head's single positive is the
#: anchor the one2many head also ranked first.
TAL_ALPHA = 0.5  # classification-score exponent in the alignment metric
TAL_BETA = 6.0  # IoU exponent in the alignment metric

#: Candidates per ground truth. The ONLY assigner knob that differs between the
#: two heads: the one2many head takes ten anchors per object (dense supervision,
#: and the branch that trains the backbone), the one2one head takes exactly one
#: (so its output needs no de-duplication and NMS can be dropped).
TAL_TOPK = 10
ONE2ONE_TOPK = 1

#: Loss weights, at the published values (``box`` / ``cls`` / ``dfl``). Applied
#: identically to both heads' terms, as upstream's ``E2EDetectLoss`` does — it
#: builds two identically-weighted detection losses and sums them.
BOX_LOSS_WEIGHT = 7.5
CLS_LOSS_WEIGHT = 0.5
DFL_LOSS_WEIGHT = 1.5

#: Inference post-processing. ⚠️ THERE IS NO ``NMS_THRESH`` HERE and that is the
#: architecture, not an omission: the deployed head is one-to-one, so there is
#: nothing to suppress. ``DETECTIONS_PER_IMAGE`` is upstream's ``max_det`` and
#: it is applied TWICE — once over anchors, once over (anchor, class) pairs —
#: which is the published two-stage top-k.
#:
#: ``SCORE_THRESH`` is 0.0 rather than the siblings' 0.001, also deliberately:
#: upstream's end-to-end path thresholds nothing and returns exactly ``max_det``
#: ranked candidates, because with one-to-one supervision a low-scoring
#: candidate costs a slot rather than a false positive that NMS has to clean up.
#: Kept as a named knob rather than inlined so the engine (or a guard) can raise
#: it, but the shipped value keeps the decode faithful to the published one.
SCORE_THRESH = 0.0
DETECTIONS_PER_IMAGE = 300

#: torchvision's ImageNet normalisation, matching every other CV template here.
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

_EPS = 1e-9


def _norm_groups(channels, maximum=32):
    """Largest group count ``<= maximum`` that divides ``channels``.

    GroupNorm requires ``channels % num_groups == 0``, and a hardcoded 32
    crashes as soon as ``WIDTH_MULT`` is lowered: YOLOv10-N's 0.25 puts 16
    channels in the stem. It matters more here than on the two siblings because
    ``CIB`` and ``SCDown`` are built from depthwise convolutions, so this tree
    carries norms at every intermediate width the inverted blocks produce
    rather than only at the round stage widths.

    DUPLICATED from the sibling hand-written detectors on purpose -- zoo
    templates cannot import siblings (zero relative imports repo-wide). Its
    test is duplicated alongside it for the same reason.
    """
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _round_channels(channels):
    """Width-scale a channel count the way the v10 yamls are parsed.

    ``make_divisible(min(channels, MAX_CHANNELS) * WIDTH_MULT, 8)``. The cap is
    applied BEFORE the multiplier and is what makes the m/b/l/x scales narrower
    than a naive product would give — transcribed rather than simplified away so
    the table stays comparable with the published one.
    """
    scaled = min(channels, MAX_CHANNELS) * WIDTH_MULT
    return max(
        CHANNEL_DIVISOR, int(math.ceil(scaled / CHANNEL_DIVISOR) * CHANNEL_DIVISOR)
    )


def _round_depth(blocks):
    """Depth-scale a block count: ``max(round(n * DEPTH_MULT), 1)``."""
    return max(int(round(blocks * DEPTH_MULT)), 1)


class ConvNormAct(nn.Module):
    """conv -> GroupNorm -> SiLU, the unit every block here is built from.

    ``groups`` is used heavily in this template and barely at all in its
    siblings: ``groups=in_ch=out_ch`` is a **depthwise** convolution, which is
    what ``SCDown``, ``CIB`` and ``RepVGGDW`` are built from.

    ``act=False`` drops the activation, which is what the two ``RepVGGDW``
    branches and ``SCDown``'s depthwise conv need.
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
    """3x3 -> 3x3 at full width, with an optional identity branch.

    ``expansion`` has NO DEFAULT, for the reason recorded on ``yolov8_s.py``:
    inside a ``C2f`` it must be ``1.0`` because the split has already halved the
    channel count, and squeezing again there narrows the whole backbone and neck
    while leaving every loss finite. A default invited exactly that slip once on
    a sibling and it shipped past thirty guards.
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


class RepVGGDW(nn.Module):
    """A re-parameterisable depthwise mixer: a depthwise 7x7 and a depthwise
    3x3, **summed before one shared activation**.

    The sum-then-activate order is what makes the pair collapse into a single
    7x7 depthwise kernel at deployment: both branches are conv-plus-norm and so
    affine, and the sum of two affine maps is affine. Two ways to get it wrong
    that neither the shapes nor the losses notice:

    * activating each branch separately and then summing. Identical parameter
      count, identical output shape, and no longer re-parameterisable;
    * dropping the 3x3 branch. That one DOES change the parameter count, so the
      published-architecture guard catches it.

    Depthwise throughout — ``groups=channels`` — which is what makes a 7x7
    affordable here at all: ``49 * channels`` parameters rather than
    ``49 * channels ** 2``.
    """

    def __init__(self, channels, kernels=REPVGGDW_KERNELS):
        super().__init__()
        large, small = kernels
        self.conv = ConvNormAct(
            channels, channels, large, stride=1, groups=channels, act=False
        )
        self.conv1 = ConvNormAct(
            channels, channels, small, stride=1, groups=channels, act=False
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.conv1(x))


class CIB(nn.Module):
    """Compact inverted block — YOLOv10's cheap replacement for ``Bottleneck``.

    Five convolutions, of which only two are dense:

    1. depthwise 3x3 at the input width — cheap spatial mixing;
    2. pointwise 1x1 **expanding** to ``2 * hidden`` — the "inverted" step;
    3. the inner spatial mixer at the expanded width: a ``RepVGGDW`` when
       ``large_kernel`` is set, otherwise a single depthwise 3x3;
    4. pointwise 1x1 projecting back down;
    5. depthwise 3x3 at the output width.

    ``expansion`` has NO DEFAULT here either, and for a sharper reason than on
    ``Bottleneck``: upstream calls ``CIB`` with ``e=1.0`` from inside
    ``C2fCIB``, where ``hidden`` is *already* the halved branch width, so the
    inverted expansion runs at ``2 * branch`` rather than at ``branch``. A
    default of 0.5 would look right, halve the inner mixer, keep every shape
    legal and leave every loss finite — and it would miss the published count.
    """

    def __init__(self, in_ch, out_ch, expansion, shortcut=True, large_kernel=None):
        super().__init__()
        if large_kernel is None:
            large_kernel = CIB_LARGE_KERNEL
        hidden = max(1, int(out_ch * expansion))
        expanded = 2 * hidden
        self.block = nn.Sequential(
            ConvNormAct(in_ch, in_ch, 3, stride=1, groups=in_ch),
            ConvNormAct(in_ch, expanded, 1, stride=1),
            (
                RepVGGDW(expanded)
                if large_kernel
                else ConvNormAct(expanded, expanded, 3, stride=1, groups=expanded)
            ),
            ConvNormAct(expanded, out_ch, 1, stride=1),
            ConvNormAct(out_ch, out_ch, 3, stride=1, groups=out_ch),
        )
        self.use_add = shortcut and in_ch == out_ch

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_add else y


class C2f(nn.Module):
    """YOLOv8's cross-stage stage, carried unchanged into YOLOv10.

    ``cv1`` splits into two half-width branches. One is passed straight to the
    fusion; the other is fed through ``n`` blocks **and every intermediate
    output is kept**, so ``cv2`` fuses ``2 + n`` branches rather than 2. That is
    the whole difference from YOLOv5/YOLOX's ``C3``, which concatenates only the
    final block output and the skip.

    A C3-shaped implementation trains perfectly happily at a slightly smaller
    parameter count, which is why ``guard_c2f_fuses_every_intermediate_block``
    reconstructs the expected branch list and compares it against the tensor
    ``cv2`` is actually handed, rather than reading the constructor.

    ``_block`` is the seam ``C2fCIB`` overrides — the one place the published
    scales genuinely disagree — so the split, the accumulation and the fusion
    are stated once.
    """

    def __init__(self, in_ch, out_ch, n=1, shortcut=False, expansion=0.5):
        super().__init__()
        self.hidden = int(out_ch * expansion)
        self.cv1 = ConvNormAct(in_ch, 2 * self.hidden, 1, stride=1)
        self.cv2 = ConvNormAct((2 + n) * self.hidden, out_ch, 1, stride=1)
        self.m = nn.ModuleList(
            self._block(self.hidden, shortcut) for _ in range(n)
        )

    @staticmethod
    def _block(channels, shortcut):
        # expansion=1.0, NOT the stage's own 0.5: `channels` is already the
        # halved branch, so the inner block runs at full branch width.
        return Bottleneck(channels, channels, 1.0, shortcut)

    def forward(self, x):
        branches = list(self.cv1(x).chunk(2, dim=1))
        for block in self.m:
            branches.append(block(branches[-1]))
        return self.cv2(torch.cat(branches, dim=1))


class C2fCIB(C2f):
    """``C2f`` with ``CIB`` as its computational block.

    Subclassed rather than copied, which is also what upstream does: the split,
    the accumulation and the fusion are ``C2f`` and only the block differs. A
    second copy of ``forward`` here would be a place for the two to drift apart
    — and its mutation anchors would stop being unique.
    """

    @staticmethod
    def _block(channels, shortcut):
        # expansion=1.0 for the reason recorded on CIB: `channels` is the halved
        # branch already, so the inverted expansion runs at 2 * branch.
        return CIB(channels, channels, 1.0, shortcut)


class SPPF(nn.Module):
    """Spatial pyramid pooling, fast: ONE 5x5 max-pool applied
    ``SPPF_REPEATS`` times **in series**, each on the previous output.

    Series is what gives the 5/9/13 effective receptive field from one 5x5
    kernel — the whole reason the block is "fast". Applying all three to
    ``cv1``'s output instead produces identical shapes, identical parameters and
    three identical branches, so it has a functional guard rather than a shape
    one.
    """

    def __init__(self, in_ch, out_ch, ksize=SPPF_KERNEL, repeats=SPPF_REPEATS):
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


class SCDown(nn.Module):
    """Spatial-channel decoupled downsampling.

    A pointwise 1x1 changes the channel count **at full resolution**, then a
    depthwise kxk stride-2 conv does the spatial reduction. That ordering is the
    block: a plain strided 3x3 costs ``in * out * 9`` parameters, this costs
    ``in * out + out * 9``, and at the stride-32 transition that is most of what
    YOLOv10 saves over YOLOv8.

    ⚠️ PUTTING THE STRIDE ON THE POINTWISE CONV IS SHAPE-IDENTICAL **AND**
    PARAMETER-IDENTICAL. A 1x1 conv's parameter count does not depend on its
    stride, and the two arrangements emit the same spatial size, so swapping
    them changes no shape, no count and no loss key — it just reduces the block
    to point-sampling every other pixel and throws away three quarters of the
    input before the channel transform ever sees it.
    ``guard_scdown_downsamples_with_the_depthwise_conv`` measures the tensor
    each conv receives.
    """

    def __init__(self, in_ch, out_ch, ksize=3, stride=2):
        super().__init__()
        self.cv1 = ConvNormAct(in_ch, out_ch, 1, stride=1)
        self.cv2 = ConvNormAct(
            out_ch, out_ch, ksize, stride=stride, groups=out_ch, act=False
        )

    def forward(self, x):
        return self.cv2(self.cv1(x))


class Attention(nn.Module):
    """The attention inside ``PSA``, at the published shape.

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
        attention = (query.transpose(-2, -1) @ key) * self.scale
        attention = attention.softmax(dim=-1)
        attended = (value @ attention.transpose(-2, -1)).reshape(
            batch, channels, height, width
        )
        return self.proj(attended + self.pe(value.reshape(batch, channels, height, width)))


class PSA(nn.Module):
    """Partial self-attention — the block YOLOv10 puts on its deepest stage.

    ``cv1`` splits the stage in two. **Only one half attends**; the other half
    bypasses the attention and the feed-forward entirely and is concatenated
    back at ``cv2``. That partial application is what the name means and what
    makes attention affordable at the stride-32 width at all — it halves both
    the token-mixing cost and the attention module's own parameters.

    The attending half gets attention and the feed-forward each as a
    **residual**, so the block is an identity at initialisation-scale and the
    stage does not need re-tuning to accept it. Dropping either residual keeps
    every shape and parameter identical and trains, so both are checked
    functionally.
    """

    def __init__(self, in_ch, out_ch, ratio=PSA_RATIO):
        super().__init__()
        if in_ch != out_ch:
            raise ValueError(
                f"yolov10_s: PSA preserves its width, got in_ch={in_ch} "
                f"out_ch={out_ch}"
            )
        self.hidden = int(in_ch * ratio)
        self.cv1 = ConvNormAct(in_ch, 2 * self.hidden, 1, stride=1)
        self.cv2 = ConvNormAct(2 * self.hidden, in_ch, 1, stride=1)
        self.attn = Attention(self.hidden)
        self.ffn = nn.Sequential(
            ConvNormAct(self.hidden, 2 * self.hidden, 1, stride=1),
            ConvNormAct(2 * self.hidden, self.hidden, 1, stride=1, act=False),
        )

    def forward(self, x):
        bypass, attending = self.cv1(x).split((self.hidden, self.hidden), dim=1)
        attending = attending + self.attn(attending)
        attending = attending + self.ffn(attending)
        return self.cv2(torch.cat((bypass, attending), dim=1))


def _build_stage(kind, in_ch, out_ch, blocks, shortcut):
    """A block-kind name -> an aggregation stage.

    Raises on an unknown name rather than falling back, so a typo in
    ``BACKBONE_P5_BLOCK`` is a construction failure and not a quietly different
    architecture.
    """
    if kind == "c2f":
        return C2f(in_ch, out_ch, n=blocks, shortcut=shortcut)
    if kind == "c2f_cib":
        return C2fCIB(in_ch, out_ch, n=blocks, shortcut=shortcut)
    raise ValueError(
        f"yolov10_s: block kind must be 'c2f' (the yolov10n stride-32 stage) or "
        f"'c2f_cib' (yolov10s), got {kind!r}"
    )


class YOLOv10Backbone(nn.Module):
    """The YOLOv10 backbone: a C2f trunk whose two deepest transitions are
    ``SCDown``, with ``SPPF`` then ``PSA`` on the deepest map.

    Returns the stride-8/16/32 feature maps.
    """

    def __init__(self):
        super().__init__()
        # Read from the module tables at CONSTRUCTION time, not captured in a
        # default argument -- a default is evaluated once at def time, which
        # would pin the shipped scale into the signature and make the table
        # unreachable. `guard_architecture_table_is_a_live_knob` rebuilds with
        # the published YOLOv10-N multipliers and block kind and checks the
        # result against that scale's published gradient count.
        p5_kind = BACKBONE_P5_BLOCK
        stages = BACKBONE_STAGES

        stem_ch = _round_channels(STEM_CHANNELS)
        self.stem = ConvNormAct(3, stem_ch, 3, stride=2)

        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        in_ch = stem_ch
        widths = []
        for index, (out_full, blocks_full) in enumerate(stages):
            out_ch = _round_channels(out_full)
            # ⚠️ The two DEEPEST transitions are SCDown, the two shallowest a
            # plain strided 3x3. That is the published split: SCDown's saving
            # scales with in * out, so it is worth its depthwise conv only at
            # the wide end, and using it everywhere would miss the count.
            if index >= len(stages) - 2:
                self.downsamples.append(SCDown(in_ch, out_ch))
            else:
                self.downsamples.append(ConvNormAct(in_ch, out_ch, 3, stride=2))
            kind = p5_kind if index == len(stages) - 1 else "c2f"
            self.stages.append(
                _build_stage(kind, out_ch, out_ch, _round_depth(blocks_full), True)
            )
            widths.append(out_ch)
            in_ch = out_ch

        self.sppf = SPPF(widths[-1], widths[-1])
        self.psa = PSA(widths[-1], widths[-1])
        #: (stride 8, stride 16, stride 32) channel counts, read by the neck.
        self.out_channels = (widths[1], widths[2], widths[3])

    def forward(self, x):
        x = self.stem(x)
        outputs = []
        for downsample, stage in zip(self.downsamples, self.stages):
            x = stage(downsample(x))
            outputs.append(x)
        # SPPF then PSA sit on the deepest map, after its aggregation stage.
        return outputs[1], outputs[2], self.psa(self.sppf(outputs[3]))


class YOLOv10Neck(nn.Module):
    """Path-aggregation neck: one top-down pass then one bottom-up pass, so the
    stride-8 map carries semantic context and the stride-32 map carries
    localisation detail.

    Two YOLOv10-specific choices, both of which change the parameter count and
    are therefore pinned by the published figure:

    * the bottom-up stride-32 transition is an ``SCDown``, while the stride-16
      one is a plain strided 3x3 — the same wide-end-only split as the backbone;
    * the stride-32 fusion stage is a ``C2fCIB`` **with the large kernel**,
      at every published scale. Unlike the backbone's stride-32 stage this one
      does not vary between n and s, which is why it is not a knob.
    """

    def __init__(self, in_channels):
        super().__init__()
        blocks = _round_depth(NECK_BLOCKS)
        c3, c4, c5 = in_channels
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # top-down
        self.td_p4 = C2f(c5 + c4, c4, n=blocks, shortcut=False)
        self.td_p3 = C2f(c4 + c3, c3, n=blocks, shortcut=False)
        # bottom-up
        self.bu_down3 = ConvNormAct(c3, c3, 3, stride=2)
        self.bu_p4 = C2f(c3 + c4, c4, n=blocks, shortcut=False)
        self.bu_down4 = SCDown(c4, c4)
        self.bu_p5 = C2fCIB(c4 + c5, c5, n=blocks, shortcut=True)

        #: YOLOv10 keeps the backbone's widths at the head, unlike RTMDet which
        #: projects all three levels to a common width.
        self.out_channels = (c3, c4, c5)

    def forward(self, features):
        c3, c4, c5 = features

        p4 = self.td_p4(torch.cat((self.upsample(c5), c4), dim=1))
        p3_out = self.td_p3(torch.cat((self.upsample(p4), c3), dim=1))

        p4_out = self.bu_p4(torch.cat((self.bu_down3(p3_out), p4), dim=1))
        p5_out = self.bu_p5(torch.cat((self.bu_down4(p4_out), c5), dim=1))
        return p3_out, p4_out, p5_out


class YOLOv10Head(nn.Module):
    """The dual DFL head: a one2many branch and a one2one branch.

    Both branches have the same shape — a box tower emitting ``4 * REG_MAX``
    DFL channels and a class tower emitting ``num_classes`` sigmoid logits, per
    level — and they share **no parameters**: the one2one branch is an
    independent copy, as upstream's ``copy.deepcopy`` makes it. Sharing them
    would train and would defeat the entire design, since the two branches are
    supervised with different assignments.

    ⚠️ THE CLASS TOWER IS NOT YOLOv8's. YOLOv10 replaces the two dense 3x3
    convolutions with two **depthwise-separable** pairs (depthwise 3x3 then
    pointwise 1x1, twice), which is most of the head's parameter saving and is
    the kind of drift only the published count catches — a copy of
    ``yolov8_s.py``'s tower here type-checks, trains, and is about 1.2M
    parameters too heavy at this scale.

    ⚠️ AND THE ONE2ONE BRANCH SEES DETACHED FEATURES. See ``forward`` and the
    module docstring: that ``detach`` is the single most silent thing in this
    file.

    There is **no objectness branch** in either head. Classification confidence
    is the only score, which is why the assigner's alignment metric and the
    classifier's soft target have to carry localisation quality — see
    ``assign``.
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
        # SEED_EXCLUDED_PREFIXES down to the six 1x1 predictors instead of
        # sweeping in the towers. At a narrower width (YOLOv10-N's 64) the max
        # can be won by the class term and the tower becomes class-count
        # dependent; the seed guard asserts the property this build actually
        # relies on rather than the formula.
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

        # The one2one branch: an independent copy of both towers and both
        # predictors, exactly as upstream's `copy.deepcopy(self.cv2)` /
        # `copy.deepcopy(self.cv3)`. deepcopy rather than a second construction
        # so the two branches start from IDENTICAL weights -- upstream's
        # initialisation, and it matters: the two heads are supervised with
        # different assignments from the same features, so starting them apart
        # adds a difference the design does not intend.
        self.one2one_box_convs = copy.deepcopy(self.box_convs)
        self.one2one_box_preds = copy.deepcopy(self.box_preds)
        self.one2one_cls_convs = copy.deepcopy(self.cls_convs)
        self.one2one_cls_preds = copy.deepcopy(self.cls_preds)

        self._init_prediction_biases()

    def _class_tower(self, channels):
        """YOLOv10's lightweight class tower: two depthwise-separable pairs.

        ``ConvNormAct(c, c, 3, groups=c)`` then ``ConvNormAct(c, hidden, 1)``,
        then the same shape again at ``hidden``. YOLOv8 spends two DENSE 3x3
        convolutions here; separating them is most of what the v10 head saves.
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
        """Upstream's ``Detect.bias_init``, applied to BOTH branches.

        The class bias is ``log(5 / nc / (640 / stride) ** 2)`` — "expect about
        five objects per image, spread over this level's cells" — which puts the
        prior far below any sensible score threshold. Without it the first
        batches are dominated by the ~8400 negative anchors per image.

        ⚠️ That prior is also why a freshly built model's detections are
        indistinguishable from noise, so any eval assertion taken from a forward
        pass on an untrained model is vacuous. The decode is therefore driven
        directly with synthetic above-threshold head outputs in the test file,
        at batch >= 2.

        The box bias is 1.0, not 0: it makes the initial distribution favour the
        low bins, so a fresh anchor proposes a box about one cell across at its
        own level rather than at an arbitrary scale.
        """
        for predictors in (self.cls_preds, self.one2one_cls_preds):
            for stride, module in zip(self.strides, predictors):
                cells = (640.0 / float(stride)) ** 2
                nn.init.constant_(
                    module.bias, math.log(5.0 / float(self.num_classes) / cells)
                )
        for predictors in (self.box_preds, self.one2one_box_preds):
            for module in predictors:
                nn.init.constant_(module.bias, 1.0)

    def _branch_forward(self, features, box_convs, box_preds, cls_convs, cls_preds):
        """One branch's ``(cls_logits, dist_logits, anchors)``.

        Stated once and called twice, so the two branches cannot drift into
        flattening their predictions in different orders — which would be
        invisible, since each branch is only ever compared against its own
        anchor table.

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
            cls_output = cls_preds[level](cls_convs[level](feature))
            dist_output = box_preds[level](box_convs[level](feature))

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

    def forward(self, features):
        """Return ``(one2many, one2one)``, each a ``(cls, dist, anchors)`` triple.

        ⚠️ THE ``detach`` BELOW IS THE POINT OF THIS ARCHITECTURE.

        The one2one branch is fed ``feature.detach()``, so its gradient stops at
        the head and never reaches the neck or the backbone. The one2many branch
        is what shapes the features; the one2one branch learns to read them
        one-to-one so that inference needs no NMS. Letting the one2one gradient
        into the backbone puts sparse one-to-one supervision in competition with
        the dense supervision that exists precisely because one-to-one alone
        cannot shape a backbone.

        Removing this ``detach`` changes no shape, no parameter count and no
        loss key; every loss stays finite, the losses still fall and the
        template still overfits a single object.
        ``guard_one2one_head_is_detached_from_the_backbone`` measures the
        GRADIENT GRAPH rather than any value, which is the only place the
        difference is visible.
        """
        one2many = self._branch_forward(
            features, self.box_convs, self.box_preds, self.cls_convs, self.cls_preds
        )
        one2one = self._branch_forward(
            [feature.detach() for feature in features],
            self.one2one_box_convs,
            self.one2one_box_preds,
            self.one2one_cls_convs,
            self.one2one_cls_preds,
        )
        return one2many, one2one


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


class YOLOv10S(nn.Module):
    """YOLOv10-S speaking the ``torchvision_detection`` contract."""

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
        self.backbone = YOLOv10Backbone()
        self.neck = YOLOv10Neck(self.backbone.out_channels)
        # reg_max passed EXPLICITLY, not left to the head's default. A default
        # argument is evaluated once at class-definition time, so a head reading
        # REG_MAX from its own signature would keep the value the module had at
        # import while `self.reg_max` above tracked the current one — the two
        # would disagree silently and the head's reshape would be the first
        # thing to notice. `guard_reg_max_reaches_the_head_and_the_decode`
        # rebuilds at a different REG_MAX to prove the knob is live rather than
        # decorative.
        self.head = YOLOv10Head(
            self.num_classes, self.neck.out_channels, reg_max=self.reg_max
        )

        self.score_thresh = SCORE_THRESH
        self.detections_per_image = DETECTIONS_PER_IMAGE

    # -- contract entry point ------------------------------------------------

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError(
                "yolov10_s: train mode requires targets — the engine calls "
                "model(images, targets) for the loss dict and model(images) "
                "only in eval mode"
            )

        original_image_sizes = [
            (int(img.shape[-2]), int(img.shape[-1])) for img in images
        ]
        image_list, targets = self.transform(list(images), targets)

        one2many, one2one = self.head(self.neck(self.backbone(image_list.tensors)))

        if self.training:
            return self._losses(one2many, one2one, targets)

        # ⚠️ EVAL USES THE ONE2ONE BRANCH ONLY. That is what "end-to-end" means
        # here: the one2many branch is a training-time scaffold whose ten
        # positives per object are exactly what would need NMS, and it is
        # discarded at inference. Decoding one2many instead would produce
        # duplicate-heavy predictions with no suppression anywhere in the
        # pipeline to clean them up, and mAP would collapse while every loss
        # stayed healthy.
        detections = self._predictions(*one2one, image_list.image_sizes)
        return self.transform.postprocess(
            detections, image_list.image_sizes, original_image_sizes
        )

    # -- training ------------------------------------------------------------

    def _losses(self, one2many, one2one, targets):
        """Both branches' losses, returned as one flat handler loss dict.

        Upstream's ``v10DetectLoss`` builds two identically-weighted detection
        losses — one at ``tal_topk=10`` and one at ``tal_topk=1`` — and sums
        them. Here they are kept as **separate keys** rather than summed,
        because the engine logs the dict per key and a collapsed total would
        hide which branch stopped learning. The handler sums whatever it is
        given, so the optimisation is identical.
        """
        losses = {}
        for prefix, (cls_logits, dist_logits, anchors), topk in (
            ("", one2many, TAL_TOPK),
            ("one2one_", one2one, ONE2ONE_TOPK),
        ):
            branch = self._branch_losses(cls_logits, dist_logits, anchors, targets, topk)
            for key, value in branch.items():
                losses[f"loss_{prefix}{key}"] = value
        return losses

    def _branch_losses(self, cls_logits, dist_logits, anchors, targets, topk):
        """Task-aligned losses for ONE branch, at ``topk`` candidates per object.

        Shared by both branches so the one2one head cannot drift into a
        different loss — the "consistent" in "consistent dual assignments" is
        precisely that the two differ ONLY in ``topk``.
        """
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
                gt_boxes,
                gt_labels,
                scores[index],
                boxes[index].detach(),
                anchor_points,
                topk,
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
            "box": BOX_LOSS_WEIGHT * loss_box,
            "cls": CLS_LOSS_WEIGHT * loss_cls,
            "dfl": DFL_LOSS_WEIGHT * loss_dfl,
        }

    @torch.no_grad()
    def assign(
        self, gt_boxes, gt_labels, pred_scores, pred_boxes, anchor_points, topk
    ):
        """Task-aligned assignment for one image, at ``topk`` per ground truth.

        Returns ``(fg_mask, matched_labels, matched_boxes, aligned_scores)``,
        where ``aligned_scores`` is the **normalised alignment metric** each
        positive anchor's classifier is trained towards — not a hard 1. All
        inputs are for one image: ``pred_scores`` ``(N, C)`` probabilities,
        ``pred_boxes`` ``(N, 4)`` pixel xyxy, ``anchor_points`` ``(N, 2)`` pixel
        centres.

        ``topk`` IS A REQUIRED ARGUMENT, with no default. That is deliberate and
        it is the mechanism of "consistent dual assignments": this one function
        is called twice per step, at ``TAL_TOPK`` for the one2many branch and at
        ``ONE2ONE_TOPK = 1`` for the one2one branch, and every other rule below
        is shared. A default here would let one call site quietly stop passing
        it and collapse the dual assignment into two identical ones — which
        trains, and which produces a model that still needs NMS.

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
        3. the top ``topk`` candidates per ground truth are selected, by metric;
        4. an anchor claimed by two ground truths is awarded to the one it
           overlaps **best by IoU** — not by alignment metric, which is a
           different tie-break and a plausible-looking wrong one.

        Then the target is normalised: each positive's soft label is its metric
        rescaled so the best-aligned anchor for a ground truth is trained
        towards that ground truth's best IoU. A hard 1.0 target trains happily
        and simply loses the model's ability to say "this is a car, but I have
        it badly boxed" — the score it ranks by at inference *is* this quantity,
        because there is no objectness branch to carry it.

        ``tests/test_yolov10_s.py`` registers a mutation against each and proves
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

        selection = min(topk, num_anchors)
        _, positions = torch.topk(candidate, selection, dim=1)
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
        """NMS-FREE decode: the published two-stage top-k.

        ⚠️ THERE IS NO NMS ANYWHERE IN THIS FUNCTION, and no ``batched_nms``
        import in this file. That is the architecture: the one2one branch this
        decodes was supervised with exactly ONE positive anchor per object, so
        its predictions carry no duplicates to suppress. Adding NMS back would
        not be a safety net — it would be a second, unsupervised de-duplication
        over predictions that are already one-to-one, and at the IoU thresholds
        the siblings use it would merge genuinely distinct overlapping objects.

        Two stages, both at ``detections_per_image`` (upstream's ``max_det``):

        1. the best ``max_det`` **anchors**, ranked by their strongest class
           score;
        2. the best ``max_det`` **(anchor, class) pairs** among those — which is
           what lets one anchor emit two classes when it genuinely straddles two
           objects, rather than being reduced to its argmax.

        ⚠️ THE TWO STAGES ARE AN EFFICIENCY FACTORISATION, NOT A DIFFERENT
        ANSWER, and this is recorded so nobody writes a guard against a
        difference that does not exist. The staged form returns the same score
        multiset as a single flat top-k over all ``anchors * classes`` pairs.
        Proof: if pair ``(i, c)`` scores ``s`` then anchor ``i``'s maximum is at
        least ``s``, so for anchor ``i`` to miss the top ``max_det`` anchors
        there must be ``max_det`` anchors scoring above ``s`` — each
        contributing a pair above ``s`` — and ``(i, c)`` would then be outside
        the top ``max_det`` pairs regardless. Measured too: 4000 randomised
        ``(anchors, classes, max_det)`` fixtures, zero differences in the
        returned score multiset.

        The staged form is kept because it is upstream's and because it gathers
        only ``max_det`` rows before expanding to pairs (at 8400 anchors and 80
        classes that is 300x80 instead of 8400x80). But the property worth
        guarding is not the staging — it is that **no suppression happens at
        all**, which is genuinely observable: feed several anchors that decode
        to the SAME box at the same class and all of them come back, where any
        NMS would collapse them to one. That is what
        ``guard_decode_is_nms_free`` asserts.
        """
        decoded = _decode_boxes(dist_logits, anchors)
        scores = cls_logits.sigmoid()

        results = []
        for boxes, class_scores, (height, width) in zip(decoded, scores, image_sizes):
            boxes = boxes.clone()
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=float(width))
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=float(height))

            # ⚠️ CHANNEL 0 IS DROPPED BEFORE THE TOP-K, not after.
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
            # of THIS function's top-k, so a channel-0 candidate still consumes
            # a detection slot a real object should have had. It binds harder
            # here than on the NMS siblings, not softer: the entire decode IS a
            # top-k budget, and nothing downstream can recover the slot.
            class_scores = class_scores[:, 1:]
            num_anchors, num_classes = class_scores.shape
            budget = min(self.detections_per_image, num_anchors)

            # Stage 1: the best anchors by their strongest real class.
            best_per_anchor = class_scores.amax(dim=-1)
            _, anchor_index = torch.topk(best_per_anchor, budget, dim=-1)
            anchor_scores = class_scores[anchor_index]
            anchor_boxes = boxes[anchor_index]

            # Stage 2: the best (anchor, class) pairs among those.
            pairs = min(self.detections_per_image, anchor_scores.numel())
            flat_scores, flat_index = torch.topk(
                anchor_scores.reshape(-1), pairs, dim=-1
            )
            # Channel 0 was sliced off, so the class column is offset by +1 to
            # get back to the model-space label the handler expects.
            labels = (flat_index % num_classes) + 1
            box_index = flat_index // num_classes

            keep = flat_scores > self.score_thresh
            results.append(
                {
                    "boxes": anchor_boxes[box_index][keep],
                    "scores": flat_scores[keep],
                    "labels": labels[keep],
                }
            )
        return results
