"""EfficientDet-D0 — BiFPN with learnable weighted fusion over an EfficientNet-B0 backbone (Tan, Pang & Le, CVPR 2020). Two ideas, and both are about spending parameters where they buy accuracy. **BiFPN** replaces the FPN's one-way top-down path with a bidirectional block that is repeated, and — the part the name is about — learns a scalar weight per input edge, because a top-down P4 feature and a lateral P4 feature are not equally informative and summing them as equals throws that away. **Compound scaling** then grows depth, width and input resolution together along one coefficient, which is what turns a single design into the D0-D7 curve. This template is D0, the bottom of that curve: 64 BiFPN channels, 3 BiFPN repeats, 3 head convs, 512px input.

Offline variant: every module here is built from an inlined architecture
description with randomly initialised weights, so nothing is fetched from
``download.pytorch.org`` or any hub — the #199 egress lockdown blocks it — and
the template constructs anywhere, network or not. No seed is hosted for this
template yet, so it random-initialises and there is
no weight file — upload with ``weights=False``::

    user.upload_model("efficientdet_d0", weights=False)

Until a dump is staged, ``tools/check_dump_coverage.py`` classifies this file
NO_SEED and the statement above is what keeps that classification honest.

Built from scratch, deliberately
--------------------------------
There is no ``timm`` here and no ``torchvision.models.efficientnet``: the
backbone is written out below, stem to head. ``timm`` is being removed from the
``vision-cv`` image (backend#2974) and torchvision's EfficientNet is a
classification trunk whose intermediate taps are not exposed — extracting P3/
P4/P5 from it means a feature-extractor wrapper keyed on internal module names,
which is the kind of coupling that breaks silently on a library bump. Writing
the eight MBConv stages out is ~70 lines and pins the pyramid taps by
construction.

Fast normalized fusion, which is the whole point of the F in BiFPN
------------------------------------------------------------------
Each fusion node computes

    out = sum_i(relu(w_i) * x_i) / (eps + sum_i(relu(w_i)))

with ``w`` a **trainable** ``nn.Parameter`` initialised to ones. ``relu``
keeps the weights non-negative, and the normalisation keeps the output scale
independent of how many inputs a node has. Two failure modes this file is
written to make loud:

- **A fusion weight that is a buffer, or detached, is not fusion** — it is a
  fixed average with extra steps, and the architecture's headline claim is
  gone. ``tests/test_efficientdet_bifpn.py`` asserts every fusion weight is in
  ``model.parameters()``, has ``requires_grad``, and receives a **non-zero
  gradient** from a real backward pass through the real loss.
- **Unweighted or unnormalised summation is shape-identical.** The same test
  sets one node's weights to known constants, feeds constant tensors, and
  asserts the exact arithmetic — ``(1*1 + 3*2) / (1 + 3) = 1.75``. An
  unweighted sum gives 3.0 and an unnormalised weighted sum gives 7.0, so
  either mutation is caught by a number rather than by a shape.

GroupNorm, not BatchNorm — and this one is not just house style
---------------------------------------------------------------
The paper's backbone, BiFPN and heads are all BatchNorm. This template uses
GroupNorm throughout, for two independent reasons:

1. **Federated averaging.** ``CLAUDE.md``'s convention is explicit: BN running
   statistics average poorly across non-IID clients, so freeze BN or use
   GroupNorm/LayerNorm. GroupNorm has no running statistics to average at all.
2. **The other templates' escape hatch is not available here.** The
   torchvision family used to reach for ``FrozenBatchNorm2d``, which is
   correct *when a pretrained seed supplies the running statistics* — it
   followed this template to GroupNorm in model-zoo#259 for exactly the
   reason below (backend#3093). On a from-scratch,
   randomly-initialised trunk it supplies ``running_mean=0``/``running_var=1``,
   i.e. it does not normalise anything — measured on the ResNet-50 trunk the
   other templates share, activations reach a standard deviation of ~24 by the
   time they reach the ROI head, against ~3 with live BatchNorm. A
   from-scratch EfficientNet-B0 is deeper in *depthwise* layers and less
   forgiving of that, so frozen-BN would have been the worse of the three
   options rather than the neutral one.

A consequence worth stating: this template's state_dict has **no BN running
buffers**, so a seed prepped for it carries parameters only.

Shared heads across levels — genuinely shared
---------------------------------------------
The class and box towers are applied to all five pyramid levels, and because
the norm is GroupNorm there is nothing per-level left to hold: the official
implementation keeps per-level BN precisely because BN statistics differ by
level, and that problem does not exist here. So the towers are one module
object used five times, which makes the sharing checkable by identity and by
parameter count rather than by reading the loop.

⚠️ The flatten ordering is silent when wrong
--------------------------------------------
``AnchorGenerator`` emits anchors location-major — for each spatial position in
row-major ``(H, W)`` order, all ``A`` base anchors — and the head's conv output
is ``(N, A * K, H, W)``. The permutation to ``(N, H * W * A, K)`` must match, and
a mismatch is shape-identical: the model trains against boxes decoded at the
wrong pixels and merely learns badly. This file uses exactly torchvision's own
permutation, and the test drives the decode with **synthetic above-threshold
logits at batch >= 2 on a non-square feature map with 9 anchors per location**.
Every part of that is load-bearing. A fresh focal-loss detector initialises its
classification prior at 0.01, below the 0.05 ``score_thresh``, so it returns
**zero detections** and every eval assertion passes against a well-formed empty
list — the vacuous-eval path that shipped a real ``zip``-truncation bug
elsewhere in this roster. Square maps and one-anchor-per-location both make
anchor-major and location-major orderings indistinguishable.

What is reused
--------------
``RetinaNet`` supplies the transform, anchor generation, the per-level split of
head outputs, ``postprocess_detections`` (score filter, top-k, class-wise NMS,
box decode) and the mapping of boxes back to original image coordinates. The
backbone, the BiFPN and the head — the three things EfficientDet actually is —
are written here.

Verified against torch 2.11.0 / torchvision 0.26.0 (the engine pin,
``tools/requirements-engine-pin.txt``).
"""
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.ops import sigmoid_focal_loss

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = (
    "head.cls_logits.1.",
)

framework = "pytorch"
model_type = "torchvision_detection"
main_method = "MyModel"
license = "Apache-2.0"
# D0's compound-scaling resolution. This is a FIXED-size model: the transform is
# built with fixed_size=(512, 512), so 512 is what the model runs at whatever
# the dataset delivers — not a minimum that larger images exceed. Read off the
# built model's transform rather than asserted from here —
# tests/test_od_declared_resolution.py compares the two (backend#3058).
image_size = 512
# D0 is the small end of the curve — ~3.9M parameters against ResNet-50-FPN's
# ~34M, at 512px rather than 800px — so it fits a larger batch than the rest of
# this family.
batch_size = 8
output_classes = 12
category = "object_detection"

#: Compound-scaling coefficients for D0. Kept as named constants rather than
#: inlined so that a D1..D7 sibling is a four-line diff — that is what compound
#: scaling is for, and the paper's whole point is that these move together.
BIFPN_CHANNELS = 64
BIFPN_REPEATS = 3
HEAD_CONVS = 3

#: Base anchor edge as a multiple of the level stride, and the three scale
#: octaves and three aspect ratios per location. 9 anchors per location.
ANCHOR_SCALE = 4.0
ANCHOR_OCTAVES = (0, 1.0 / 3.0, 2.0 / 3.0)
ANCHOR_ASPECT_RATIOS = (0.5, 1.0, 2.0)

#: The pyramid this template builds: P3..P7.
PYRAMID_STRIDES = (8, 16, 32, 64, 128)

#: EfficientNet-B0. One row per stage:
#: ``(expand_ratio, out_channels, repeats, stride, kernel_size)``.
#: Rows 3, 5 and 7 (0-indexed 2, 4, 6) end at strides 8, 16 and 32 and are the
#: P3/P4/P5 taps — asserted by construction in ``_EfficientNetB0`` rather than
#: by counting strides at the call site.
EFFICIENTNET_B0_STAGES = (
    (1, 16, 1, 1, 3),
    (6, 24, 2, 2, 3),
    (6, 40, 2, 2, 5),
    (6, 80, 3, 2, 3),
    (6, 112, 3, 1, 5),
    (6, 192, 4, 2, 5),
    (6, 320, 1, 1, 3),
)
#: Output channels of the 3x3 stride-2 stem.
EFFICIENTNET_B0_STEM = 32
#: Stage indices whose outputs are the P3/P4/P5 taps.
PYRAMID_TAP_STAGES = (2, 4, 6)

#: Squeeze-excite bottleneck as a fraction of the block's *input* channels.
SE_RATIO = 0.25

#: Denominator floor in fast normalized fusion. The paper's value.
FUSION_EPS = 1e-4

#: Focal-loss constants, RetinaNet's and EfficientDet's alike.
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

#: Huber transition point for the box loss, on ENCODED deltas. EfficientDet's
#: value; RetinaNet's plain L1 is the delta -> 0 limit of this.
BOX_LOSS_DELTA = 0.1


def _norm(channels, max_groups=32):
    """GroupNorm with the largest group count that divides ``channels``.

    ``nn.GroupNorm`` requires ``channels % num_groups == 0``, and this backbone
    has stages at 16/24/40/80/112/192/320 channels plus depthwise layers at
    every expansion of those — so a fixed 32 raises on most of them.
    """
    groups = max(g for g in range(1, max_groups + 1) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


class _SqueezeExcite(nn.Module):
    """Channel attention: global pool, bottleneck, gate.

    Part of what makes an MBConv an MBConv rather than a plain inverted
    residual — dropping it is a shape-identical change to the block.
    """

    def __init__(self, channels, squeeze_channels):
        super().__init__()
        self.reduce = nn.Conv2d(channels, squeeze_channels, 1)
        self.expand = nn.Conv2d(squeeze_channels, channels, 1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.silu(self.reduce(scale))
        return x * torch.sigmoid(self.expand(scale))


class _MBConv(nn.Module):
    """Mobile inverted bottleneck: expand 1x1, depthwise kxk, squeeze-excite,
    project 1x1, with a residual when the shape allows one.

    The expansion is skipped at ``expand_ratio == 1`` (EfficientNet's first
    stage), which is not an optimisation — a 1x1 that maps 32 channels to 32 is
    a different function from the identity and the reference architecture does
    not have one there.
    """

    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size):
        super().__init__()
        hidden = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, hidden, 1, bias=False),
                _norm(hidden),
                nn.SiLU(inplace=True),
            ]
        layers += [
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=hidden,
                bias=False,
            ),
            _norm(hidden),
            nn.SiLU(inplace=True),
            _SqueezeExcite(hidden, max(1, int(in_channels * SE_RATIO))),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            _norm(out_channels),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_residual else out


class _EfficientNetB0(nn.Module):
    """EfficientNet-B0 trunk that returns the P3/P4/P5 taps.

    Returns a list rather than the whole trunk plus a lookup, so the tap
    selection is fixed here and cannot be re-derived (wrongly) by a caller.
    """

    def __init__(self, stages=EFFICIENTNET_B0_STAGES, tap_stages=PYRAMID_TAP_STAGES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, EFFICIENTNET_B0_STEM, 3, stride=2, padding=1, bias=False),
            _norm(EFFICIENTNET_B0_STEM),
            nn.SiLU(inplace=True),
        )
        self.tap_stages = tuple(tap_stages)
        blocks = []
        channels = EFFICIENTNET_B0_STEM
        tap_channels = []
        for stage_index, (expand, out_channels, repeats, stride, kernel) in enumerate(stages):
            stage = []
            for repeat in range(repeats):
                stage.append(
                    _MBConv(
                        channels,
                        out_channels,
                        expand,
                        stride if repeat == 0 else 1,
                        kernel,
                    )
                )
                channels = out_channels
            blocks.append(nn.Sequential(*stage))
            if stage_index in self.tap_stages:
                tap_channels.append(out_channels)
        self.stages = nn.ModuleList(blocks)
        #: Channel widths of the returned taps, in order — the BiFPN reads this
        #: rather than being told separately, so the two cannot disagree.
        self.tap_channels = tuple(tap_channels)

    def forward(self, x):
        x = self.stem(x)
        taps = []
        for stage_index, stage in enumerate(self.stages):
            x = stage(x)
            if stage_index in self.tap_stages:
                taps.append(x)
        return taps


class _WeightedFusion(nn.Module):
    """Fast normalized fusion of ``num_inputs`` same-shaped tensors.

    ``out = sum(relu(w_i) * x_i) / (eps + sum(relu(w_i)))``. The weights are
    trainable and initialised to ones, so the node starts as an unweighted mean
    and learns away from it — which is why "did it learn away from it?" (a
    non-zero gradient) is the property worth testing, not the initial value.
    """

    def __init__(self, num_inputs, eps=FUSION_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs))
        self.eps = eps

    def forward(self, inputs):
        if len(inputs) != self.weight.numel():
            raise ValueError(
                f"_WeightedFusion was built for {self.weight.numel()} inputs and "
                f"received {len(inputs)} — a fusion node's arity is fixed by the "
                f"BiFPN topology, so this is a wiring error, not a resizable input"
            )
        weight = F.relu(self.weight)
        normaliser = weight.sum() + self.eps
        stacked = torch.stack(inputs, dim=0)
        weighted = stacked * weight.reshape(-1, *([1] * (stacked.dim() - 1)))
        return weighted.sum(dim=0) / normaliser


def _separable_conv(channels):
    """Depthwise kxk followed by pointwise 1x1 — one conv "layer" in the BiFPN
    and in the heads. This is where EfficientDet's parameter budget goes: a
    dense 3x3 at 64 channels is 36,864 weights against this block's 4,672."""
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
        nn.Conv2d(channels, channels, 1, bias=True),
    )


class _BiFPNLayer(nn.Module):
    """One bidirectional block over five levels: top-down, then bottom-up.

    Node arities are the topology and are not uniform, which is exactly why
    they are written out. Top-down: P7 passes through, P3..P6 each fuse their
    lateral input with the level above (2 inputs). Bottom-up: P4..P6 fuse their
    lateral input, their top-down intermediate, and the level below (3 inputs);
    P7 fuses its lateral input and the level below (2 inputs); P3 is already
    final. A node built with the wrong arity raises in ``_WeightedFusion``
    rather than broadcasting into something plausible.
    """

    def __init__(self, channels=BIFPN_CHANNELS):
        super().__init__()
        self.num_levels = len(PYRAMID_STRIDES)
        # Top-down: one fusion + conv per level below the top.
        self.top_down_fusions = nn.ModuleList(
            _WeightedFusion(2) for _ in range(self.num_levels - 1)
        )
        self.top_down_convs = nn.ModuleList(
            _separable_conv(channels) for _ in range(self.num_levels - 1)
        )
        # Bottom-up: three-input nodes for the middle levels, two for the top.
        self.bottom_up_fusions = nn.ModuleList(
            _WeightedFusion(3 if level < self.num_levels - 1 else 2)
            for level in range(1, self.num_levels)
        )
        self.bottom_up_convs = nn.ModuleList(
            _separable_conv(channels) for _ in range(self.num_levels - 1)
        )
        self.norms = nn.ModuleList(
            _norm(channels) for _ in range(2 * (self.num_levels - 1))
        )

    def forward(self, features):
        if len(features) != self.num_levels:
            raise ValueError(
                f"_BiFPNLayer expects {self.num_levels} levels, got {len(features)}"
            )
        # --- top-down ------------------------------------------------------
        intermediates = [None] * self.num_levels
        intermediates[-1] = features[-1]
        for level in range(self.num_levels - 2, -1, -1):
            upper = F.interpolate(
                intermediates[level + 1], size=features[level].shape[-2:], mode="nearest"
            )
            fused = self.top_down_fusions[level]([features[level], upper])
            fused = self.norms[level](self.top_down_convs[level](F.silu(fused)))
            intermediates[level] = fused

        # --- bottom-up -----------------------------------------------------
        outputs = [None] * self.num_levels
        outputs[0] = intermediates[0]
        for level in range(1, self.num_levels):
            lower = F.adaptive_max_pool2d(outputs[level - 1], features[level].shape[-2:])
            inputs = (
                [features[level], intermediates[level], lower]
                if level < self.num_levels - 1
                else [features[level], lower]
            )
            fused = self.bottom_up_fusions[level - 1](inputs)
            fused = self.norms[self.num_levels - 1 + level - 1](
                self.bottom_up_convs[level - 1](F.silu(fused))
            )
            outputs[level] = fused
        return outputs


class _EfficientDetBackbone(nn.Module):
    """EfficientNet-B0 taps, projected to a common width, then ``repeats``
    BiFPN layers. Presents the ``out_channels`` attribute and the ``OrderedDict``
    output that ``RetinaNet`` expects of a backbone."""

    def __init__(self, channels=BIFPN_CHANNELS, repeats=BIFPN_REPEATS):
        super().__init__()
        self.body = _EfficientNetB0()
        self.out_channels = channels

        # P3..P5 come from the trunk and are projected to the BiFPN width.
        self.laterals = nn.ModuleList(
            nn.Sequential(nn.Conv2d(tap, channels, 1, bias=False), _norm(channels))
            for tap in self.body.tap_channels
        )
        # P6 and P7 are strided 3x3s on the previous level, as the paper does —
        # not max-pools, so the extra levels have capacity of their own.
        self.extra = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
                _norm(channels),
            )
            for _ in range(len(PYRAMID_STRIDES) - len(self.body.tap_channels))
        )
        self.bifpn = nn.ModuleList(_BiFPNLayer(channels) for _ in range(repeats))

    def forward(self, x):
        features = [
            lateral(tap) for lateral, tap in zip(self.laterals, self.body(x))
        ]
        for extra in self.extra:
            features.append(extra(features[-1]))
        for layer in self.bifpn:
            features = layer(features)
        return OrderedDict((str(i), f) for i, f in enumerate(features))


class _EfficientDetHead(nn.Module):
    """Shared class and box towers, applied to every pyramid level.

    Speaks ``RetinaNetHead``'s interface: ``forward(features)`` returns
    ``{"cls_logits": (N, A, K), "bbox_regression": (N, A, 4)}`` over the
    concatenated pyramid, and ``compute_loss(targets, head_outputs, anchors,
    matched_idxs)`` returns the loss dict ``RetinaNet.compute_loss`` passes
    straight through.
    """

    def __init__(self, in_channels, num_anchors, num_classes, num_convs=HEAD_CONVS):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.cls_tower = nn.ModuleList(_separable_conv(in_channels) for _ in range(num_convs))
        self.box_tower = nn.ModuleList(_separable_conv(in_channels) for _ in range(num_convs))
        self.cls_norms = nn.ModuleList(_norm(in_channels) for _ in range(num_convs))
        self.box_norms = nn.ModuleList(_norm(in_channels) for _ in range(num_convs))
        # The PREDICTION convs are depthwise-separable too, as the paper's
        # class_net/box_net are — a dense 3x3 here is a drop-in with the same
        # output shape, and at 90 classes it would be 467,370 parameters against
        # this block's 52,506. Getting that wrong is the yolox_s failure mode:
        # internally consistent, trains fine, quietly not the architecture.
        self.cls_logits = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, num_anchors * num_classes, 1, bias=True),
        )
        self.bbox_regression = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, num_anchors * 4, 1, bias=True),
        )
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Prior probability 0.01, as every focal-loss detector does: without it
        # the first steps are dominated by the ~10^4 easy negatives per image.
        # It is also why a fresh model returns no detections at all — see the
        # vacuous-eval warning in the module docstring.
        nn.init.constant_(self.cls_logits[-1].bias, -math.log((1 - 0.01) / 0.01))

    @staticmethod
    def _flatten(output, channels_per_anchor):
        """``(N, A * K, H, W)`` -> ``(N, H * W * A, K)``.

        Exactly torchvision's own permutation, and it has to be: see the
        ordering warning in the module docstring.
        """
        n, _, h, w = output.shape
        output = output.view(n, -1, channels_per_anchor, h, w)
        output = output.permute(0, 3, 4, 1, 2)
        return output.reshape(n, -1, channels_per_anchor)

    def _tower(self, feature, convs, norms):
        for conv, norm in zip(convs, norms):
            feature = F.silu(norm(conv(feature)))
        return feature

    def forward(self, features):
        all_cls, all_box = [], []
        for feature in features:
            cls_logits = self.cls_logits(
                self._tower(feature, self.cls_tower, self.cls_norms)
            )
            bbox_regression = self.bbox_regression(
                self._tower(feature, self.box_tower, self.box_norms)
            )
            all_cls.append(self._flatten(cls_logits, self.num_classes))
            all_box.append(self._flatten(bbox_regression, 4))
        return {
            "cls_logits": torch.cat(all_cls, dim=1),
            "bbox_regression": torch.cat(all_box, dim=1),
        }

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        """Focal loss on the classification, Huber on the encoded box deltas.

        Normalised by the number of foreground anchors, which is the convention
        every focal-loss detector uses: the negative count is ~10^4 per image
        and dividing by it would shrink the gradient with the image size.
        """
        cls_losses, box_losses = [], []
        cls_logits = head_outputs["cls_logits"]
        bbox_regression = head_outputs["bbox_regression"]

        for image_index, (targets_per_image, matched_per_image) in enumerate(
            zip(targets, matched_idxs)
        ):
            foreground = matched_per_image >= 0
            num_foreground = int(foreground.sum())
            # torchvision's Matcher marks anchors between the two thresholds
            # with BETWEEN_THRESHOLDS (-2); they are excluded from the
            # classification loss entirely rather than counted as negatives.
            valid = matched_per_image != det_utils.Matcher.BETWEEN_THRESHOLDS

            logits_per_image = cls_logits[image_index]
            target_classes = torch.zeros_like(logits_per_image)
            if num_foreground:
                target_classes[
                    foreground,
                    targets_per_image["labels"][matched_per_image[foreground]],
                ] = 1.0

            cls_losses.append(
                sigmoid_focal_loss(
                    logits_per_image[valid],
                    target_classes[valid],
                    alpha=FOCAL_ALPHA,
                    gamma=FOCAL_GAMMA,
                    reduction="sum",
                )
                / max(1, num_foreground)
            )

            if num_foreground:
                matched_boxes = targets_per_image["boxes"][matched_per_image[foreground]]
                target_deltas = self.box_coder.encode_single(
                    matched_boxes, anchors[image_index][foreground]
                )
                box_losses.append(
                    F.huber_loss(
                        bbox_regression[image_index][foreground],
                        target_deltas,
                        delta=BOX_LOSS_DELTA,
                        reduction="sum",
                    )
                    / num_foreground
                )
            else:
                # Keep the key present and the graph connected: an image with no
                # foreground anchor still has to contribute a differentiable
                # zero, or the returned dict changes shape with the batch.
                box_losses.append(bbox_regression[image_index].sum() * 0.0)

        return {
            "classification": torch.stack(cls_losses).mean(),
            "bbox_regression": torch.stack(box_losses).mean(),
        }


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background

    backbone = _EfficientDetBackbone()

    # Three octaves x three aspect ratios per location, each level's base edge
    # ANCHOR_SCALE times its stride — the paper's tiling. Nine anchors per
    # location is also what makes the head's flatten ordering testable: at one
    # anchor per location, anchor-major and location-major are the same thing.
    sizes = tuple(
        tuple(stride * ANCHOR_SCALE * (2.0**octave) for octave in ANCHOR_OCTAVES)
        for stride in PYRAMID_STRIDES
    )
    anchor_generator = AnchorGenerator(
        sizes=sizes, aspect_ratios=(ANCHOR_ASPECT_RATIOS,) * len(PYRAMID_STRIDES)
    )

    head = _EfficientDetHead(
        backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        num_classes,
    )

    return RetinaNet(
        backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        head=head,
        # D0 runs at a FIXED 512x512: compound scaling ties the resolution to
        # the rest of the coefficients, so this is part of the architecture
        # rather than a preference. fixed_size makes the transform resize to
        # exactly that instead of treating 512 as a lower bound.
        min_size=image_size,
        max_size=image_size,
        fixed_size=(image_size, image_size),
        # EfficientDet's assignment thresholds: 0.5 foreground, 0.4 background,
        # with the band between the two excluded from the loss.
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.4,
    )
