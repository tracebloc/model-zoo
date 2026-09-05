"""YOLOX-S (Ge et al., 2021) — anchor-free CSPDarknet/PAFPN detector with a
decoupled head and SimOTA label assignment, written from scratch in PyTorch.

Offline variant: nothing is fetched at construction — no hub id, no
``timm``, no ``transformers``, no torchvision pretrained enum, no
``download.pytorch.org`` (the #199 egress lockdown blocks it). Every layer is
built from the inlined width/depth multipliers below, so the template
constructs on a closed edge. No seed is hosted for this template and there is
no weight file: upload with ``weights=False``::

    user.upload_model("yolox_s", weights=False)

Hosting COCO tensors as a tracebloc model-store seed (the #1499 pattern: a
matched ``<stem>_weights.pkl`` prepped by ``tools/prep_offline_weights.py``
and strict-loaded after the architecture is built) is follow-up work, not part
of this roster addition. Until a dump is staged,
``tools/check_dump_coverage.py`` classifies this file NO_SEED and the sentence
above is what keeps that classification honest.

Why this is ONE file, and duplicated against ``rtmdet_s.py``
------------------------------------------------------------
A zoo template is uploaded as a single ``.py`` and there are zero relative
imports anywhere under ``model_zoo/`` — a template that imported a sibling
would arrive at the model checker with the sibling missing. So the backbone,
the neck, the head, the assigner and the loss all live here, and the handful
of blocks that ``rtmdet_s.py`` also needs (a conv-norm-act, a CSP stage, a
dynamic-k matcher) are **deliberately duplicated** there rather than shared.
That is not a missed refactor: the two files must each stand alone, and their
blocks differ in substance anyway — YOLOX's CSP bottleneck is 1x1/3x3 with a
SiLU, RTMDet's is 3x3 + a 5x5 depthwise-separable pair with channel attention.

Why it is NOT in the ``yolo`` family
------------------------------------
``model_type = "yolo"`` is the legacy YOLOv1 grid contract — a fixed 448px
input, a ``[7, 7, num_classes + 10]`` target tensor, one object per cell (so at
most 49 objects per image, silently overwriting co-located ones) and an
external customer ``loss.py``. That family is frozen. YOLOX is a genuine
multi-scale detector, so it declares the ``torchvision_detection`` family
instead, which is a **duck-typed contract rather than a library dependency**:

* ``model(images, targets)`` in train mode -> a dict of scalar losses
* ``model(images)`` in eval mode -> ``List[Dict]`` with ``boxes`` (pixel
  xyxy), ``scores``, ``labels``
* ``images`` is a **list** of differently-sized 3-D tensors, because
  ``_rcnn_collate`` builds tuples rather than stacking (object counts vary per
  image)

Nothing about that contract mentions torchvision, and the engine's
``TorchvisionDetectionHandler`` needs no change to train this file.

Batching and resolution — reused, not hand-rolled
-------------------------------------------------
``GeneralizedRCNNTransform`` does the resize, the normalize, the pad-to-batch
at ``size_divisible=32`` and the ``postprocess()`` that maps predicted boxes
back to each image's original coordinates. Reusing it is what makes the
variable-size list contract safe, and the 32-divisibility is a hard
requirement here twice over: the ``Focus`` stem slices on even rows/columns
and the head's three levels are at strides 8/16/32.

``image_size`` IS the resolution this model runs at
---------------------------------------------------
``min_size`` and ``max_size`` are both set to ``image_size``, so a square
``data_shape x data_shape`` image from the engine's dataset scales by exactly
1.0 and the backbone sees ``image_size x image_size``. That is stated because
the previously shipped non-yolo OD templates declared ``image_size = 448``
while their builders resized to ``min_size=800`` (backend#3058): the declared
edge was decorative. Here it is the edge, and
two tests measure it off the built model rather than trusting this docstring:
``tests/test_od_declared_resolution.py`` compares declared against the
transform's configured resolution family-wide (backend#3058), and
``test_od_hand_written_detectors.py``'s ``declared_size_measured`` guard hooks
the transform to check the tensor the backbone is actually handed is square at
exactly this edge.

Label space
-----------
The head allocates ``output_classes + 1`` sigmoid channels and uses the
incoming label **directly** as the channel index — no offset arithmetic in this
file, in either direction.

What fills those channels is the family handler's business, and as of
backend#3062 it hands this model label space ``[1, C]``: the platform's
producers emit 0-based dataset indices, and the handler shifts them up by
``BACKGROUND_LABEL_OFFSET`` before training and back down after inference. The
``+1`` channel is what makes that shift representable, so a label of ``C`` is
in range.

⚠️ The consequence, and it is a real coupling rather than a note: **channel 0
is never a positive target.** It is trained only as a negative, so emitting it
would produce detections carrying dataset class ``-1`` once the handler shifts
back. ``_predictions`` therefore drops it **before** the score threshold and
the top-k, not after — the engine does filter these out downstream, but that
filter runs after this function has already spent detection slots on them.

This template consequently REQUIRES the family handler's shift. Fed raw
0-based dataset labels it would discard the first class. That is the contract
as of backend#3062, and the zoo's own family train-step test asserts the
``[1, C]`` range (model-zoo#245).

This section previously described the opposite contract — "index 0 is a real
class", "nothing is silently folded into a background slot" — which was true
before the channel-0 drop landed and stale after it. These templates ship as a
single uploaded ``.py``, so the docstring is the spec; it was corrected in
review (model-zoo#237).

Federated note (GroupNorm, not BatchNorm)
-----------------------------------------
The norm layers are GroupNorm. Upstream YOLOX uses BatchNorm, and an earlier
revision of this file did too, defending it as "the same trade every
torchvision detector in this family already carries". That claim was FALSE and
was caught in review: at the time every torchvision template in this family
built its backbone with ``norm_layer=misc_nn_ops.FrozenBatchNorm``, i.e. FROZEN
-- the running statistics never update, so there is nothing to average. This
template carried 21,738 live buffer elements against ``efficientdet_d0``'s 0.
(Frozen BN turned out to be a bit-exact no-op on those from-scratch builds,
backend#3093, and they moved to GroupNorm in model-zoo#262. Note what that
change is NOT: moving BN -> GroupNorm here left the parameter count identical
and dropped 21,738 buffers, whereas moving FrozenBN -> GroupNorm there ADDED
2 parameters per normalised channel, because frozen BN held weight/bias as
buffers. The two migrations are not the same arithmetic.)

BN's ``running_mean``/``running_var`` are buffers the averaging service ships
and averages every round, and they average badly across non-IID clients (see
CLAUDE.md). GroupNorm is preferred here over FrozenBatchNorm2d for a specific
reason: Frozen BN registers ``weight``/``bias`` as BUFFERS, which changes the
parameter count and would silently invalidate
``guard_matches_the_published_architecture`` -- the check that compares this
model against the published YOLOX-S table, and which exists because a
self-measured parameter count already hid a ~1.15M-parameter width error here.
GroupNorm keeps ``weight``/``bias`` as parameters (identical count to BN) and
carries no running statistics, so the reference guard stays valid AND the
averaged buffers go to zero.

The COCO-seed argument does not rescue BN either: the seed for this template is
prepped by us, backbone-only, under whatever norm the template declares
(backend#3055), so there is no official checkpoint whose running statistics
must be accommodated.

Verified against torch 2.11.0 / torchvision 0.26.0 (the engine pin,
``tools/requirements-engine-pin.txt``).

Reference: Zheng Ge et al., "YOLOX: Exceeding YOLO Series in 2021",
arXiv:2107.08430. Architecture re-implemented from the paper; no upstream code
is vendored.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import batched_nms, box_iou

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("head.cls_preds.0.", "head.cls_preds.1.", "head.cls_preds.2.")

framework = "pytorch"
model_type = "torchvision_detection"
main_class = "YOLOXS"
license = "Apache-2.0"
# The resolution the backbone actually sees: the transform below is built with
# min_size == max_size == this value, so a square input scales by 1.0.
image_size = 640
batch_size = 8
output_classes = 12
category = "object_detection"

#: YOLOX-S scaling. ``depth`` multiplies the CSP block counts, ``width`` the
#: channel counts. (S = 0.33/0.50; the paper's Tiny is 0.33/0.375 and L is
#: 1.0/1.0, so this pair is the only thing that would change to rescale.)
DEPTH_MULT = 0.33
WIDTH_MULT = 0.50

#: Feature-map strides of the three head levels, smallest object first.
STRIDES = (8, 16, 32)

#: SimOTA knobs, at the paper's values.
CENTER_RADIUS = 2.5  # candidate anchors within +-2.5 strides of a GT centre
SIMOTA_TOPK = 10  # IoUs summed to pick each GT's dynamic k
SIMOTA_IOU_COST_WEIGHT = 3.0
#: Penalty added to the cost of a candidate that is outside the GT box AND
#: outside its centre region. Two-sided, and BOTH sides are load-bearing:
#:
#: * large enough that a penalised candidate never outranks an unpenalised one
#:   — so it must exceed the largest cost this matrix can attain, which is
#:   ``num_classes * -log(_EPS)`` from the classification BCE plus
#:   ``SIMOTA_IOU_COST_WEIGHT * -log(_EPS)`` from the IoU term;
#: * small enough that ``penalty + cost`` is still *representable* in float32,
#:   because the whole justification for a finite value over ``inf`` is that
#:   the matrix stays comparable — ``inf`` would make ``topk`` return arbitrary
#:   ties.
#:
#: This was ``1.0e8``, which satisfies the first and silently destroys the
#: second (model-zoo#237 review). float32's ULP at 1e8 is **8.0**, so
#: ``1e8 + x == 1e8`` for every ``x <= 4.0`` — the comment claimed
#: comparability while the arithmetic erased it. Where it bites is a ground
#: truth with no anchor centre inside it at any level: ``inside_both`` is then
#: all-False, *every* candidate carries the penalty, and selection falls to
#: 8-unit quantisation plus index order instead of to IoU and classification
#: cost. Measured on two candidates 0.373 apart in cost, at 1e8 SimOTA picked
#: the IoU-0.30 anchor over the IoU-0.34 one purely because it came first.
#:
#: 1e5 is the value: ULP 0.0078125, fine enough to preserve a cost delta of
#: 0.01 (roughly a 0.01 IoU difference near IoU 0.5), while still ~340x above
#: the attainable cost ceiling at this template's class count. Upstream YOLOX
#: has used both 1e6 (current ``main``, ULP 0.0625) and 1e5 (the long-lived
#: value still in the widely mirrored forks); 1e6's ULP is too coarse for the
#: 0.01 criterion, so this takes the finer of the two.
#: ``guard_yolox_outside_penalty_survives_float32`` asserts both sides rather
#: than trusting this comment.
SIMOTA_OUTSIDE_PENALTY = 1.0e5

#: Regression-loss weight in the total (paper: 5.0).
REG_LOSS_WEIGHT = 5.0

#: Inference post-processing, at the paper's demo values.
SCORE_THRESH = 0.01
NMS_THRESH = 0.65
DETECTIONS_PER_IMAGE = 300

#: torchvision's ImageNet normalisation, matching every other CV template here.
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

_EPS = 1e-8


def _norm_groups(channels, maximum=32):
    """Largest group count ``<= maximum`` that divides ``channels``.

    GroupNorm requires ``channels % num_groups == 0``, and a hardcoded 32
    crashes on this roster: ``rtmdet_s`` builds a 16-channel width. Deriving
    the count keeps the norm valid at any width the depth/width multipliers
    produce, not only the ones built today.

    DUPLICATED between the two templates on purpose -- zoo templates cannot
    import siblings (zero relative imports repo-wide). Its test is duplicated
    alongside it for the same reason: copied code that leaves its guard behind
    is how a duplicated assigner silently lost its centre-inside rule earlier
    in this roster.
    """
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _round_channels(channels: int) -> int:
    """Width-scale a channel count, kept a multiple of 2 so ``Focus``'s
    4x-concatenation and the CSP half-splits stay integral."""
    return max(2, int(round(channels * WIDTH_MULT / 2) * 2))


def _round_depth(blocks: int) -> int:
    return max(1, int(round(blocks * DEPTH_MULT)))


class ConvBNAct(nn.Module):
    """conv -> GroupNorm -> SiLU, the unit every block here is built from.

    Named ``ConvBNAct`` after the upstream block it stands in for; the norm
    is GroupNorm, not BatchNorm. This docstring said "BatchNorm" until
    model-zoo#237 review caught it -- the class name is a deliberate
    upstream-parity name, the docstring was simply stale, and these
    templates ship as a single uploaded ``.py`` where the docstring IS the
    spec. See the federated note in the module docstring for why GroupNorm.
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
        # eps/momentum as upstream YOLOX sets them; the defaults train more
        # slowly at this batch size.
        # GroupNorm, NOT BatchNorm. BN's running_mean/running_var are
        # buffers the averaging service ships and averages every round,
        # and they average badly across non-IID clients. The shipped
        # family avoids this with FrozenBatchNorm2d; GroupNorm is used
        # here instead because Frozen BN moves weight/bias into buffers
        # and would change the parameter count, silently invalidating
        # the published-architecture guard. GroupNorm keeps weight+bias
        # as parameters (identical count to BN) and has no running
        # statistics at all.
        self.norm = nn.GroupNorm(_norm_groups(out_ch), out_ch, eps=1e-3)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Focus(nn.Module):
    """Stride-2 stem that slices rather than convolves: the four even/odd
    row-column phases are concatenated on the channel axis, so no spatial
    information is discarded before the first conv. Needs even H and W, which
    the transform's ``size_divisible=32`` guarantees."""

    def __init__(self, in_ch, out_ch, ksize=3):
        super().__init__()
        self.conv = ConvBNAct(in_ch * 4, out_ch, ksize, stride=1)

    def forward(self, x):
        top_left = x[..., ::2, ::2]
        top_right = x[..., ::2, 1::2]
        bottom_left = x[..., 1::2, ::2]
        bottom_right = x[..., 1::2, 1::2]
        return self.conv(
            torch.cat((top_left, bottom_left, top_right, bottom_right), dim=1)
        )


class Bottleneck(nn.Module):
    """1x1 -> 3x3, with an optional identity branch.

    ``expansion`` has NO DEFAULT on purpose. Inside a ``CSPLayer`` it must be
    ``1.0``: the CSP split has already halved the channel count, and squeezing
    again there narrows the whole backbone and neck (measured: 7,788,886
    parameters instead of 8,942,326, so ~1.15M missing at every stage, and no
    official YOLOX-S checkpoint would strict-load). A default invited exactly
    that slip and it shipped past thirty guards — see
    ``tests/test_od_hand_written_detectors.py``'s ``csp_bottleneck_width``
    guard, added with the fix.
    """

    def __init__(self, in_ch, out_ch, expansion, shortcut=True):
        super().__init__()
        hidden = max(2, int(out_ch * expansion))
        self.conv1 = ConvBNAct(in_ch, hidden, 1, stride=1)
        self.conv2 = ConvBNAct(hidden, out_ch, 3, stride=1)
        self.use_add = shortcut and in_ch == out_ch

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.use_add else y


class CSPLayer(nn.Module):
    """Cross-stage partial stage: half the channels go through ``n``
    bottlenecks, half skip, and the two are concatenated and fused."""

    def __init__(self, in_ch, out_ch, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = max(2, int(out_ch * expansion))
        self.conv1 = ConvBNAct(in_ch, hidden, 1, stride=1)
        self.conv2 = ConvBNAct(in_ch, hidden, 1, stride=1)
        self.conv3 = ConvBNAct(2 * hidden, out_ch, 1, stride=1)
        # expansion=1.0, NOT the bottleneck's own 0.5: `hidden` is already the
        # halved CSP branch, so the inner 1x1 runs at full branch width. This is
        # what upstream YOLOX (and YOLOv5's C3) do, and what rtmdet_s.py's
        # CSPNeXtBlock does in this same PR.
        self.m = nn.Sequential(
            *[Bottleneck(hidden, hidden, 1.0, shortcut) for _ in range(n)]
        )

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), dim=1))


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling: three max-pools at different kernels widen the
    receptive field of the deepest stage without extra stride."""

    def __init__(self, in_ch, out_ch, kernel_sizes=(5, 9, 13)):
        super().__init__()
        hidden = in_ch // 2
        self.conv1 = ConvBNAct(in_ch, hidden, 1, stride=1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernel_sizes]
        )
        self.conv2 = ConvBNAct(hidden * (len(kernel_sizes) + 1), out_ch, 1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.m], dim=1))


class CSPDarknet(nn.Module):
    """YOLOX's backbone. Returns the stride-8/16/32 feature maps."""

    def __init__(self):
        super().__init__()
        base = _round_channels(64)
        depth = _round_depth(3)

        self.stem = Focus(3, base, ksize=3)
        self.dark2 = nn.Sequential(
            ConvBNAct(base, base * 2, 3, stride=2),
            CSPLayer(base * 2, base * 2, n=depth),
        )
        self.dark3 = nn.Sequential(
            ConvBNAct(base * 2, base * 4, 3, stride=2),
            CSPLayer(base * 4, base * 4, n=depth * 3),
        )
        self.dark4 = nn.Sequential(
            ConvBNAct(base * 4, base * 8, 3, stride=2),
            CSPLayer(base * 8, base * 8, n=depth * 3),
        )
        self.dark5 = nn.Sequential(
            ConvBNAct(base * 8, base * 16, 3, stride=2),
            SPPBottleneck(base * 16, base * 16),
            CSPLayer(base * 16, base * 16, n=depth, shortcut=False),
        )
        #: (stride 8, stride 16, stride 32) channel counts, read by the neck.
        self.out_channels = (base * 4, base * 8, base * 16)

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        c3 = self.dark3(x)
        c4 = self.dark4(c3)
        c5 = self.dark5(c4)
        return c3, c4, c5


class YOLOPAFPN(nn.Module):
    """Path-aggregation FPN: one top-down pass then one bottom-up pass, so the
    stride-8 map carries semantic context and the stride-32 map carries
    localisation detail."""

    def __init__(self, in_channels):
        super().__init__()
        c3, c4, c5 = in_channels
        depth = _round_depth(3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.lateral_conv0 = ConvBNAct(c5, c4, 1, stride=1)
        self.c3_p4 = CSPLayer(2 * c4, c4, n=depth, shortcut=False)
        self.reduce_conv1 = ConvBNAct(c4, c3, 1, stride=1)
        self.c3_p3 = CSPLayer(2 * c3, c3, n=depth, shortcut=False)

        self.bu_conv2 = ConvBNAct(c3, c3, 3, stride=2)
        self.c3_n3 = CSPLayer(2 * c3, c4, n=depth, shortcut=False)
        self.bu_conv1 = ConvBNAct(c4, c4, 3, stride=2)
        self.c3_n4 = CSPLayer(2 * c4, c5, n=depth, shortcut=False)

        self.out_channels = (c3, c4, c5)

    def forward(self, features):
        c3, c4, c5 = features

        p5 = self.lateral_conv0(c5)
        p4 = self.c3_p4(torch.cat((self.upsample(p5), c4), dim=1))
        p4_reduced = self.reduce_conv1(p4)
        p3_out = self.c3_p3(torch.cat((self.upsample(p4_reduced), c3), dim=1))

        p4_out = self.c3_n3(torch.cat((self.bu_conv2(p3_out), p4_reduced), dim=1))
        p5_out = self.c3_n4(torch.cat((self.bu_conv1(p4_out), p5), dim=1))
        return p3_out, p4_out, p5_out


class YOLOXHead(nn.Module):
    """The decoupled head — YOLOX's defining change over YOLOv5.

    Classification and regression get **separate** conv towers per level
    (``cls_convs`` and ``reg_convs`` share no parameters), which is what
    "decoupled" means. A coupled head — one tower feeding both 1x1 predictors
    — trains perfectly happily and would be silently wrong, so
    ``tests/test_od_hand_written_detectors.py`` asserts the two subtrees are
    parameter-disjoint rather than reading it off this docstring.
    """

    def __init__(self, num_classes, in_channels, strides=STRIDES):
        super().__init__()
        self.num_classes = num_classes
        self.strides = tuple(strides)
        hidden = _round_channels(256)

        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for channels in in_channels:
            self.stems.append(ConvBNAct(channels, hidden, 1, stride=1))
            self.cls_convs.append(
                nn.Sequential(
                    ConvBNAct(hidden, hidden, 3, stride=1),
                    ConvBNAct(hidden, hidden, 3, stride=1),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    ConvBNAct(hidden, hidden, 3, stride=1),
                    ConvBNAct(hidden, hidden, 3, stride=1),
                )
            )
            self.cls_preds.append(nn.Conv2d(hidden, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(hidden, 4, 1))
            self.obj_preds.append(nn.Conv2d(hidden, 1, 1))

        self._init_prediction_biases()

    def _init_prediction_biases(self, prior_prob=1e-2):
        """Start the sigmoid predictors near p=0.01. Without this the first
        batches are dominated by the ~1e4 negative anchors per image and the
        objectness loss swamps everything else.

        The regression bias is zeroed for the matching reason on the other
        branch: a zero box prediction decodes to a stride-sized box centred on
        its own anchor, so each level starts out proposing objects at the scale
        it is responsible for. Left at the ``Conv2d`` default the three levels
        start at three arbitrary scales.
        """
        bias = -math.log((1.0 - prior_prob) / prior_prob)
        for module in list(self.cls_preds) + list(self.obj_preds):
            nn.init.constant_(module.bias, bias)
        for module in self.reg_preds:
            nn.init.constant_(module.bias, 0.0)

    def forward(self, features):
        """Return ``(raw, grids)``.

        ``raw`` is ``(B, N, 5 + num_classes)`` with the box entries still in
        their per-level parameterisation (centre offset in cells, log size in
        strides); ``grids`` is ``(N, 3)`` holding each anchor point's cell
        ``x``, cell ``y`` and stride. Decoding is a separate step because the
        loss needs both forms: the assigner works in pixels, the L1-free
        regression loss works on decoded boxes.
        """
        outputs = []
        grids = []
        for level, (feature, stride) in enumerate(zip(features, self.strides)):
            x = self.stems[level](feature)
            cls_feat = self.cls_convs[level](x)
            reg_feat = self.reg_convs[level](x)

            cls_output = self.cls_preds[level](cls_feat)
            reg_output = self.reg_preds[level](reg_feat)
            obj_output = self.obj_preds[level](reg_feat)

            output = torch.cat((reg_output, obj_output, cls_output), dim=1)
            batch, channels, height, width = output.shape
            output = output.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
            outputs.append(output)

            yv, xv = torch.meshgrid(
                torch.arange(height, device=output.device, dtype=output.dtype),
                torch.arange(width, device=output.device, dtype=output.dtype),
                indexing="ij",
            )
            grid = torch.stack(
                (
                    xv.reshape(-1),
                    yv.reshape(-1),
                    torch.full_like(xv.reshape(-1), float(stride)),
                ),
                dim=1,
            )
            grids.append(grid)

        return torch.cat(outputs, dim=1), torch.cat(grids, dim=0)


def _decode_boxes(raw, grids):
    """Per-level predictions -> pixel ``cxcywh``.

    ``grids[:, 2]`` is the per-anchor stride, so this is where the head's
    multi-level structure enters the geometry: the centre offset is in cells
    and the size is in log-strides, and both are scaled by *that anchor's*
    stride. Using one stride for every level is a silent bug — the boxes stay
    finite and the model still trains.
    """
    cell_x, cell_y, stride = grids[:, 0], grids[:, 1], grids[:, 2]
    centre_x = (raw[..., 0] + cell_x) * stride
    centre_y = (raw[..., 1] + cell_y) * stride
    width = torch.exp(raw[..., 2].clamp(max=8.0)) * stride
    height = torch.exp(raw[..., 3].clamp(max=8.0)) * stride
    return torch.stack((centre_x, centre_y, width, height), dim=-1)


def _cxcywh_to_xyxy(boxes):
    centre = boxes[..., :2]
    half = boxes[..., 2:] / 2.0
    return torch.cat((centre - half, centre + half), dim=-1)


def _iou_aware_class_target(labels, ious, num_classes, dtype):
    """One-hot class target **scaled by the anchor's matched IoU**.

    YOLOX does not train its classifier against a hard 1. A positive anchor's
    target is the IoU of the box it predicts, so classification confidence and
    localisation quality are learned together and inference can rank purely by
    ``cls * obj`` with no separate centreness branch. A hard 1.0 target trains
    and detects perfectly happily — the detector just loses the ability to say
    "this is a car, but I have it badly boxed" — so this is a named function
    with its own guard rather than an inline expression nothing can point at.
    """
    one_hot = F.one_hot(labels.long(), num_classes).to(dtype)
    return one_hot * ious.to(dtype).unsqueeze(-1)


def _pairwise_iou_cxcywh(a, b):
    """IoU between two ``cxcywh`` sets, ``(len(a), len(b))``."""
    return box_iou(_cxcywh_to_xyxy(a), _cxcywh_to_xyxy(b))


class YOLOXS(nn.Module):
    """YOLOX-S speaking the ``torchvision_detection`` contract."""

    def __init__(self, num_classes=output_classes, input_size=image_size):
        super().__init__()
        # +1 sigmoid channel so the platform's 0-based labels index the head
        # directly and a 1-based producer's maximum label is still in range.
        # See "Label space" in the module docstring.
        self.num_classes = int(num_classes) + 1
        self.input_size = int(input_size)

        self.transform = GeneralizedRCNNTransform(
            min_size=self.input_size,
            max_size=self.input_size,
            image_mean=IMAGE_MEAN,
            image_std=IMAGE_STD,
            size_divisible=32,
        )
        self.backbone = CSPDarknet()
        self.neck = YOLOPAFPN(self.backbone.out_channels)
        self.head = YOLOXHead(self.num_classes, self.neck.out_channels)

        self.score_thresh = SCORE_THRESH
        self.nms_thresh = NMS_THRESH
        self.detections_per_image = DETECTIONS_PER_IMAGE

    # -- contract entry point ------------------------------------------------

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError(
                "yolox_s: train mode requires targets — the engine calls "
                "model(images, targets) for the loss dict and model(images) "
                "only in eval mode"
            )

        original_image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
        image_list, targets = self.transform(list(images), targets)

        raw, grids = self.head(self.neck(self.backbone(image_list.tensors)))

        if self.training:
            return self._losses(raw, grids, targets)

        detections = self._predictions(raw, grids, image_list.image_sizes)
        return self.transform.postprocess(
            detections, image_list.image_sizes, original_image_sizes
        )

    # -- training ------------------------------------------------------------

    def _losses(self, raw, grids, targets):
        """SimOTA-assigned losses, returned as the handler's loss dict."""
        decoded = _decode_boxes(raw, grids)
        obj_logits = raw[..., 4]
        cls_logits = raw[..., 5:]

        fg_masks = []
        cls_targets = []
        reg_targets = []
        obj_targets = []
        num_fg = 0

        for image_index, target in enumerate(targets):
            gt_boxes_xyxy = target["boxes"]
            gt_labels = target["labels"]
            num_gt = int(gt_boxes_xyxy.shape[0])
            num_anchors = decoded.shape[1]

            if num_gt == 0:
                fg_mask = torch.zeros(
                    num_anchors, dtype=torch.bool, device=decoded.device
                )
                fg_masks.append(fg_mask)
                obj_targets.append(
                    torch.zeros(num_anchors, dtype=decoded.dtype, device=decoded.device)
                )
                continue

            gt_boxes = torch.cat(
                (
                    (gt_boxes_xyxy[:, :2] + gt_boxes_xyxy[:, 2:]) / 2.0,
                    gt_boxes_xyxy[:, 2:] - gt_boxes_xyxy[:, :2],
                ),
                dim=1,
            )

            fg_mask, matched_labels, matched_boxes, matched_ious = self.assign(
                gt_boxes,
                gt_labels,
                decoded[image_index],
                obj_logits[image_index],
                cls_logits[image_index],
                grids,
            )

            num_fg += int(fg_mask.sum())
            fg_masks.append(fg_mask)
            obj_targets.append(fg_mask.to(decoded.dtype))
            if int(fg_mask.sum()) > 0:
                cls_targets.append(
                    _iou_aware_class_target(
                        matched_labels, matched_ious, self.num_classes, decoded.dtype
                    )
                )
                reg_targets.append(matched_boxes)

        fg_mask = torch.cat(fg_masks, dim=0)
        obj_target = torch.cat(obj_targets, dim=0)
        # A batch with no positive at all still has a well-defined objectness
        # loss, so clamp the divisor rather than short-circuiting: the handler
        # sums the dict and calls backward() on it.
        divisor = max(num_fg, 1)

        loss_obj = (
            F.binary_cross_entropy_with_logits(
                obj_logits.reshape(-1), obj_target, reduction="sum"
            )
            / divisor
        )

        if num_fg > 0:
            reg_target = torch.cat(reg_targets, dim=0)
            cls_target = torch.cat(cls_targets, dim=0)
            pred_boxes = decoded.reshape(-1, 4)[fg_mask]
            pred_cls = cls_logits.reshape(-1, self.num_classes)[fg_mask]

            iou = _pairwise_iou_cxcywh(pred_boxes, reg_target).diagonal()
            loss_iou = (1.0 - iou.clamp(min=0.0) ** 2).sum() / divisor
            loss_cls = (
                F.binary_cross_entropy_with_logits(
                    pred_cls, cls_target, reduction="sum"
                )
                / divisor
            )
        else:
            # Keep the dict shape and the graph: ``* 0.0`` on a real prediction
            # tensor keeps both prediction branches connected to the loss, so a
            # batch of empty images does not produce ``None`` grads.
            loss_iou = decoded.sum() * 0.0
            loss_cls = cls_logits.sum() * 0.0

        return {
            "loss_iou": REG_LOSS_WEIGHT * loss_iou,
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
        }

    @torch.no_grad()
    def assign(self, gt_boxes, gt_labels, decoded, obj_logits, cls_logits, grids):
        """SimOTA: assign each GT a dynamic number of anchors by optimal-transport cost.

        Returns ``(fg_mask, matched_labels, matched_boxes, matched_ious)`` for
        one image. ``gt_boxes`` and ``decoded`` are pixel ``cxcywh``; ``grids``
        is the head's ``(N, 3)`` cell-x / cell-y / stride table.

        Four things happen here, and every one of them fails **silently** if it
        is wrong — an all-negative image still yields a finite, small loss and a
        clean train step:

        1. a geometric pre-filter picks candidates that are inside a GT box
           **or** within ``CENTER_RADIUS`` strides of its centre;
        2. a cost matrix combines classification BCE and ``-log(IoU)``, with a
           large finite penalty on candidates that are not inside *both*
           regions;
        3. ``dynamic_k`` per GT is the rounded sum of its top-``SIMOTA_TOPK``
           IoUs — the "how many anchors does this object deserve" part;
        4. an anchor claimed by two GTs is awarded to the cheaper one.

        ``tests/test_od_hand_written_detectors.py`` registers a mutation
        against each and proves it goes red.
        """
        num_gt = int(gt_boxes.shape[0])
        num_anchors = int(decoded.shape[0])

        candidate_mask, inside_both = self._candidate_masks(gt_boxes, grids)
        if int(candidate_mask.sum()) == 0:
            return (
                torch.zeros(num_anchors, dtype=torch.bool, device=decoded.device),
                torch.zeros(0, dtype=torch.int64, device=decoded.device),
                torch.zeros((0, 4), dtype=decoded.dtype, device=decoded.device),
                torch.zeros(0, dtype=decoded.dtype, device=decoded.device),
            )

        candidate_boxes = decoded[candidate_mask]
        ious = _pairwise_iou_cxcywh(gt_boxes, candidate_boxes)
        iou_cost = -torch.log(ious + _EPS)

        # cls * obj, geometric-mean-scaled, is the score SimOTA compares
        # against the one-hot GT — the same quantity inference ranks by.
        scores = (
            cls_logits[candidate_mask].sigmoid() * obj_logits[candidate_mask, None].sigmoid()
        ).sqrt()
        one_hot = (
            F.one_hot(gt_labels.long(), self.num_classes)
            .to(scores.dtype)
            .unsqueeze(1)
            .expand(num_gt, scores.shape[0], self.num_classes)
        )
        cls_cost = F.binary_cross_entropy(
            scores.unsqueeze(0).expand_as(one_hot).clamp(_EPS, 1.0 - _EPS),
            one_hot,
            reduction="none",
        ).sum(-1)

        cost = (
            cls_cost
            + SIMOTA_IOU_COST_WEIGHT * iou_cost
            + SIMOTA_OUTSIDE_PENALTY * (~inside_both).to(cls_cost.dtype)
        )

        return self._dynamic_k_match(
            cost, ious, gt_boxes, gt_labels, candidate_mask, num_anchors
        )

    def _candidate_masks(self, gt_boxes, grids):
        """``(candidate_mask, inside_both)`` for one image.

        ``candidate_mask`` is ``(N,)`` over all anchors: inside a GT box OR
        inside its centre region. ``inside_both`` is
        ``(num_gt, num_candidates)``: inside the box AND the centre region —
        the hard constraint the cost matrix penalises.

        The centre region is ``CENTER_RADIUS`` strides wide, so it is *wider in
        pixels on the coarse levels*. Dropping the ``stride`` factor here
        collapses the per-level structure while leaving training green.
        """
        cell_x, cell_y, stride = grids[:, 0], grids[:, 1], grids[:, 2]
        # +0.5: the anchor point is the centre of its cell, not its corner.
        centre_x = ((cell_x + 0.5) * stride).unsqueeze(0)
        centre_y = ((cell_y + 0.5) * stride).unsqueeze(0)

        gt_cx = gt_boxes[:, 0:1]
        gt_cy = gt_boxes[:, 1:2]
        gt_half_w = gt_boxes[:, 2:3] / 2.0
        gt_half_h = gt_boxes[:, 3:4] / 2.0

        inside_box = torch.stack(
            (
                centre_x - (gt_cx - gt_half_w),
                (gt_cx + gt_half_w) - centre_x,
                centre_y - (gt_cy - gt_half_h),
                (gt_cy + gt_half_h) - centre_y,
            ),
            dim=-1,
        ).amin(dim=-1) > 0.0

        radius = CENTER_RADIUS * stride.unsqueeze(0)
        inside_centre = torch.stack(
            (
                centre_x - (gt_cx - radius),
                (gt_cx + radius) - centre_x,
                centre_y - (gt_cy - radius),
                (gt_cy + radius) - centre_y,
            ),
            dim=-1,
        ).amin(dim=-1) > 0.0

        candidate_mask = (inside_box | inside_centre).any(dim=0)
        inside_both = (
            inside_box[:, candidate_mask] & inside_centre[:, candidate_mask]
        )
        return candidate_mask, inside_both

    def _dynamic_k_match(
        self, cost, ious, gt_boxes, gt_labels, candidate_mask, num_anchors
    ):
        """Award each GT its ``dynamic_k`` cheapest candidates, then break ties."""
        matching = torch.zeros_like(cost, dtype=torch.uint8)
        num_gt, num_candidates = cost.shape

        topk = min(SIMOTA_TOPK, num_candidates)
        topk_ious, _ = torch.topk(ious, topk, dim=1)
        dynamic_ks = topk_ious.sum(dim=1).int().clamp(min=1)

        for gt_index in range(num_gt):
            _, positions = torch.topk(
                cost[gt_index], k=int(dynamic_ks[gt_index]), largest=False
            )
            matching[gt_index][positions] = 1

        claimed_by = matching.sum(dim=0)
        contested = claimed_by > 1
        if bool(contested.any()):
            cheapest = torch.argmin(cost[:, contested], dim=0)
            matching[:, contested] = 0
            matching[cheapest, contested] = 1

        selected = matching.sum(dim=0) > 0
        fg_mask = torch.zeros(num_anchors, dtype=torch.bool, device=cost.device)
        fg_mask[candidate_mask.nonzero(as_tuple=True)[0][selected]] = True

        matched_gt = matching[:, selected].argmax(dim=0)
        matched_labels = gt_labels.long()[matched_gt]
        matched_boxes = gt_boxes[matched_gt]
        matched_ious = (matching.to(ious.dtype) * ious).sum(dim=0)[selected]
        return fg_mask, matched_labels, matched_boxes, matched_ious

    # -- inference -----------------------------------------------------------

    def _predictions(self, raw, grids, image_sizes):
        decoded = _cxcywh_to_xyxy(_decode_boxes(raw, grids))
        scores = raw[..., 4:5].sigmoid() * raw[..., 5:].sigmoid()

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
            # The engine does drop them: `_detections_to_dataset_space` keeps
            # only `labels >= BACKGROUND_LABEL_OFFSET`. But that is downstream
            # of THIS function's score threshold and top-k, so a channel-0
            # candidate still consumes a detection slot a real object should
            # have had.
            #
            # NOT acute at initialisation on THIS template, and the sentence
            # here used to claim it was: "SCORE_THRESH is 0.001 against a 0.01
            # prior, so channel 0 clears it constantly". Both halves were the
            # rtmdet twin's, copied verbatim (model-zoo#237 review). YOLOX's
            # SCORE_THRESH is 0.01, and its score is
            # `obj.sigmoid() * cls.sigmoid()` = 1e-2 * 1e-2 = 1e-4 at the
            # prior -- two orders of magnitude BELOW its own threshold, where
            # rtmdet ranks on a single sigmoid. Measured on the built model at
            # `output_classes=12`: a forward pass at initialisation returns 2
            # detections here (random conv init scatters a few anchors over
            # the line) against rtmdet's saturated 300.
            #
            # The drop is still right to have: channel 0 can score high after
            # training, and correctness does not depend on how often it does.
            # But the justification is the sibling's and does not transfer.
            #
            # Consequence worth stating plainly: this template now REQUIRES the
            # family handler's shift. Fed raw 0-based dataset labels it would
            # discard the first class. That is the contract as of backend#3062,
            # and the zoo's own family train-step test asserts the `[1, C]`
            # range (model-zoo#245).
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
