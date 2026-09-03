"""RTMDet-S (Lyu et al., 2022) — CSPNeXt/PAFPN detector with a shared-weight
separate-BN head and a dynamic soft-label assigner, written from scratch in
PyTorch.

Offline variant: nothing is fetched at construction — no hub id, no
``timm``, no ``transformers``, no torchvision pretrained enum, no
``download.pytorch.org`` (the #199 egress lockdown blocks it). Every layer
comes from the inlined arch table below, so the template constructs on a
closed edge. No seed is hosted for this template and there is no weight file:
upload with ``weights=False``::

    user.upload_model("rtmdet_s", weights=False)

Hosting COCO tensors as a tracebloc model-store seed (the #1499 pattern: a
matched ``<stem>_weights.pkl`` prepped by ``tools/prep_offline_weights.py``
and strict-loaded after the architecture is built) is follow-up work, not part
of this roster addition. Until a dump is staged,
``tools/check_dump_coverage.py`` classifies this file NO_SEED and the sentence
above is what keeps that classification honest.

Why this is ONE file, and duplicated against ``yolox_s.py``
------------------------------------------------------------
A zoo template is uploaded as a single ``.py`` and there are zero relative
imports anywhere under ``model_zoo/``, so backbone, neck, head, assigner and
losses all live here. The blocks that ``yolox_s.py`` also needs (a
conv-norm-act, a CSP stage, a dynamic-k matcher) are **deliberately
duplicated** rather than shared: each file must stand alone at upload time.
The duplication is also thinner than it looks — this backbone's CSP block is a
3x3 conv followed by a **5x5 depthwise-separable** pair with channel
attention, its neck adds per-level output convs, its head shares conv weights
across levels, and its assigner is soft-label rather than SimOTA. Only the
conv-norm-act wrapper is genuinely the same code.

Why it is NOT in the ``yolo`` family
------------------------------------
``model_type = "yolo"`` is the legacy YOLOv1 grid contract (fixed 448px, a
``[7, 7, num_classes + 10]`` target, one object per cell, external customer
``loss.py``) and is frozen. RTMDet declares the ``torchvision_detection``
family, which is a **duck-typed contract rather than a library dependency**:

* ``model(images, targets)`` in train mode -> a dict of scalar losses
* ``model(images)`` in eval mode -> ``List[Dict]`` with ``boxes`` (pixel
  xyxy), ``scores``, ``labels``
* ``images`` is a **list** of differently-sized 3-D tensors, because
  ``_rcnn_collate`` builds tuples rather than stacking

The engine's ``TorchvisionDetectionHandler`` trains this file unchanged.

Batching and resolution — reused, not hand-rolled
-------------------------------------------------
``GeneralizedRCNNTransform`` does the resize, normalize, pad-to-batch at
``size_divisible=32`` and the ``postprocess()`` that maps predicted boxes back
to each image's original coordinates. The 32-divisibility matters because the
head's three levels sit at strides 8/16/32.

``image_size`` IS the resolution this model runs at
---------------------------------------------------
``min_size`` and ``max_size`` are both ``image_size``, so a square
``data_shape x data_shape`` image from the engine's dataset scales by exactly
1.0 and the backbone sees ``image_size x image_size``. Stated explicitly
because the previously shipped non-yolo OD templates declared
``image_size = 448`` while their builders resized to ``min_size=800``
(backend#3058). Measured off the built model, not trusted from here:
``tests/test_od_declared_resolution.py`` covers the whole family against the
transform's configured resolution, and
``test_od_hand_written_detectors.py``'s ``declared_size_measured`` guard hooks
the transform to check the tensor the backbone is actually handed is square at
exactly this edge.

Label space
-----------
Both platform producers emit **0-based** class indices — the engine resolves
``<name>`` to ``classes.index(name)`` in ``[0, C-1]``, the SDK's dummy OD
dataset draws ``random.randint(0, num_classes - 1)``. The head allocates
``output_classes + 1`` sigmoid channels and uses the incoming label
**directly** as the channel index, so index 0 is a real class and a 1-based
producer (maximum label ``C``) is in range too. Background is not a channel at
all here: it is the sentinel index ``num_classes`` in the label vector, which
the quality-focal loss treats as "all channels target 0".

Regression parameterisation
---------------------------
The head predicts the four point-to-edge distances and multiplies them by the
level's stride. ``exp()`` is applied first — upstream's ``exp_on_reg``, which
RTMDet-m/l enable and RTMDet-s does not. It is enabled here because it makes
every decoded box a valid xyxy **by construction** (all four distances
positive), and the family contract requires ``x2 >= x1``; the alternative is a
clamp at decode time that silently kills the gradient on the clamped side. It
changes no parameter shape.

⚠️ It does, however, interact with the seed story above, and in the one way a
strict load cannot catch. An upstream RTMDet-S checkpoint was trained with
``exp_on_reg`` OFF, so its regression head predicts distances directly. Every
shape matches, the load succeeds, and the boxes come out exponentiated —
garbage, with no error anywhere. If a COCO seed is ever prepped for this
template it must be prepped against **this** build, or this flag flipped to
match the checkpoint and the decode re-verified. Shape compatibility is not
semantic compatibility here.

Federated note (GroupNorm, not BatchNorm)
-----------------------------------------
Norm layers are GroupNorm. Upstream uses BatchNorm (SyncBN when distributed),
and an earlier revision defended that as "the same trade every torchvision
detector in this family already carries" -- FALSE, and caught in review: the
shipped family uses ``norm_layer=misc_nn_ops.FrozenBatchNorm``, whose running
statistics never update. This template carried 24,978 live buffer elements,
each shipped and averaged every federated round.

GroupNorm rather than FrozenBatchNorm2d because Frozen BN moves
``weight``/``bias`` into buffers and would change the parameter count,
invalidating the published-architecture guard. GroupNorm keeps them as
parameters and carries no running statistics.

The head's per-level norms remain per-level, which is the point of RTMDet's
"SepBN" head: one conv tower's weights serving three levels, each level
keeping its own normalisation. The SEPARATION is the published design; the
norm TYPE is not. ``guard_rtmdet_head_shares_convs_and_separates_bns`` checks
the separation by storage identity and is unaffected by the type change.

Verified against torch 2.11.0 / torchvision 0.26.0 (the engine pin,
``tools/requirements-engine-pin.txt``).

Reference: Chengqi Lyu et al., "RTMDet: An Empirical Study of Designing
Real-Time Object Detectors", arXiv:2212.07784. Architecture re-implemented
from the paper; no upstream code is vendored.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import batched_nms, box_iou, generalized_box_iou_loss

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("head.rtm_cls.0.", "head.rtm_cls.1.", "head.rtm_cls.2.")

framework = "pytorch"
model_type = "torchvision_detection"
main_class = "RTMDetS"
license = "Apache-2.0"
# The resolution the backbone actually sees: the transform below is built with
# min_size == max_size == this value, so a square input scales by 1.0.
image_size = 640
batch_size = 8
output_classes = 12
category = "object_detection"

#: RTMDet-S scaling: ``deepen`` multiplies CSP block counts, ``widen`` channels.
DEEPEN_FACTOR = 0.33
WIDEN_FACTOR = 0.50

#: CSPNeXt P5 arch table: ``(in, out, blocks, add_identity, use_spp)`` per stage,
#: at full width. Strides are 4 / 8 / 16 / 32 after the stride-2 stem.
CSPNEXT_P5 = (
    (64, 128, 3, True, False),
    (128, 256, 6, True, False),
    (256, 512, 6, True, False),
    (512, 1024, 3, False, True),
)

#: Feature-map strides of the three head levels, smallest object first.
STRIDES = (8, 16, 32)

#: Head shape.
STACKED_CONVS = 2
FEAT_CHANNELS = 128

#: Dynamic soft-label assigner knobs, at the paper's values.
SOFT_CENTER_RADIUS = 3.0
ASSIGNER_TOPK = 13
ASSIGNER_IOU_WEIGHT = 3.0

#: Loss weights / shape parameters.
QFL_BETA = 2.0
BBOX_LOSS_WEIGHT = 2.0

#: Inference post-processing, at the paper's test-time values.
SCORE_THRESH = 0.001
NMS_THRESH = 0.65
DETECTIONS_PER_IMAGE = 300

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

_EPS = 1e-12


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


def _widen(channels: int) -> int:
    return max(2, int(round(channels * WIDEN_FACTOR / 2) * 2))


def _deepen(blocks: int) -> int:
    return max(1, int(round(blocks * DEEPEN_FACTOR)))


class ConvBNAct(nn.Module):
    """conv -> BatchNorm -> SiLU."""

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


class DepthwiseSeparableConv(nn.Module):
    """A large-kernel depthwise conv followed by a pointwise conv — the cheap
    way RTMDet buys a big receptive field, and what distinguishes CSPNeXt from
    YOLOX's CSPDarknet."""

    def __init__(self, in_ch, out_ch, ksize=5):
        super().__init__()
        self.depthwise = ConvBNAct(in_ch, in_ch, ksize, stride=1, groups=in_ch)
        self.pointwise = ConvBNAct(in_ch, out_ch, 1, stride=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ChannelAttention(nn.Module):
    """Squeeze-and-excite with a hard-sigmoid gate, as CSPNeXt uses it."""

    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        weight = self.act(self.fc(x.mean(dim=(2, 3), keepdim=True)))
        return x * weight


class CSPNeXtBlock(nn.Module):
    """3x3 conv then a 5x5 depthwise-separable pair, with optional identity."""

    def __init__(self, in_ch, out_ch, add_identity=True):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, 3, stride=1)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch, ksize=5)
        self.add_identity = add_identity and in_ch == out_ch

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.add_identity else y


class CSPLayer(nn.Module):
    """CSPNeXt stage: split, run ``n`` blocks on one half, gate the concatenation
    with channel attention, fuse."""

    def __init__(self, in_ch, out_ch, n=1, add_identity=True, expand_ratio=0.5):
        super().__init__()
        mid = max(2, int(out_ch * expand_ratio))
        self.main_conv = ConvBNAct(in_ch, mid, 1, stride=1)
        self.short_conv = ConvBNAct(in_ch, mid, 1, stride=1)
        self.blocks = nn.Sequential(
            *[CSPNeXtBlock(mid, mid, add_identity) for _ in range(n)]
        )
        self.attention = ChannelAttention(2 * mid)
        self.final_conv = ConvBNAct(2 * mid, out_ch, 1, stride=1)

    def forward(self, x):
        merged = torch.cat((self.blocks(self.main_conv(x)), self.short_conv(x)), dim=1)
        return self.final_conv(self.attention(merged))


class SPPFBottleneck(nn.Module):
    """Fast spatial pyramid pooling on the deepest stage."""

    def __init__(self, in_ch, out_ch, kernel_sizes=(5, 9, 13)):
        super().__init__()
        mid = in_ch // 2
        self.conv1 = ConvBNAct(in_ch, mid, 1, stride=1)
        self.poolings = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernel_sizes]
        )
        self.conv2 = ConvBNAct(mid * (len(kernel_sizes) + 1), out_ch, 1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.poolings], dim=1))


class CSPNeXt(nn.Module):
    """RTMDet's backbone. Returns the stride-8/16/32 feature maps."""

    def __init__(self):
        super().__init__()
        first = _widen(CSPNEXT_P5[0][0])
        half = max(2, first // 2)
        self.stem = nn.Sequential(
            ConvBNAct(3, half, 3, stride=2),
            ConvBNAct(half, half, 3, stride=1),
            ConvBNAct(half, first, 3, stride=1),
        )

        stages = []
        out_channels = []
        for in_ch, out_ch, blocks, add_identity, use_spp in CSPNEXT_P5:
            in_ch, out_ch = _widen(in_ch), _widen(out_ch)
            layers = [ConvBNAct(in_ch, out_ch, 3, stride=2)]
            if use_spp:
                layers.append(SPPFBottleneck(out_ch, out_ch))
            layers.append(
                CSPLayer(out_ch, out_ch, n=_deepen(blocks), add_identity=add_identity)
            )
            stages.append(nn.Sequential(*layers))
            out_channels.append(out_ch)
        self.stages = nn.ModuleList(stages)

        #: Channels of the three maps ``forward`` returns.
        self.out_channels = tuple(out_channels[1:])

    def forward(self, x):
        x = self.stem(x)
        features = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index >= 1:
                features.append(x)
        return tuple(features)


class CSPNeXtPAFPN(nn.Module):
    """Path-aggregation FPN built from CSPNeXt stages, with a 3x3 output conv
    per level so all three feed the head at the same width."""

    def __init__(self, in_channels, out_channels=FEAT_CHANNELS):
        super().__init__()
        c3, c4, c5 = in_channels
        blocks = _deepen(3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.reduce_c5 = ConvBNAct(c5, c4, 1, stride=1)
        self.top_down_c4 = CSPLayer(2 * c4, c4, n=blocks, add_identity=False)
        self.reduce_c4 = ConvBNAct(c4, c3, 1, stride=1)
        self.top_down_c3 = CSPLayer(2 * c3, c3, n=blocks, add_identity=False)

        self.downsample_p3 = ConvBNAct(c3, c3, 3, stride=2)
        self.bottom_up_p4 = CSPLayer(2 * c3, c4, n=blocks, add_identity=False)
        self.downsample_p4 = ConvBNAct(c4, c4, 3, stride=2)
        self.bottom_up_p5 = CSPLayer(2 * c4, c5, n=blocks, add_identity=False)

        self.out_convs = nn.ModuleList(
            [ConvBNAct(channels, out_channels, 3, stride=1) for channels in in_channels]
        )
        self.out_channels = (out_channels,) * len(in_channels)

    def forward(self, features):
        c3, c4, c5 = features

        p5 = self.reduce_c5(c5)
        p4 = self.top_down_c4(torch.cat((self.upsample(p5), c4), dim=1))
        p4_reduced = self.reduce_c4(p4)
        p3_out = self.top_down_c3(torch.cat((self.upsample(p4_reduced), c3), dim=1))

        p4_out = self.bottom_up_p4(
            torch.cat((self.downsample_p3(p3_out), p4_reduced), dim=1)
        )
        p5_out = self.bottom_up_p5(torch.cat((self.downsample_p4(p4_out), p5), dim=1))

        return tuple(
            conv(feature)
            for conv, feature in zip(self.out_convs, (p3_out, p4_out, p5_out))
        )


class RTMDetSepBNHead(nn.Module):
    """RTMDet's head — the "SepBN" is the distinguishing structural feature.

    Every level runs the **same** conv weights (one 128->128 3x3 tower for
    classification, one for regression) but its **own** BatchNorm statistics
    and its own 1x1 predictors. Sharing the convs is what makes the head cheap
    enough to be real-time at three levels; separating the BNs is what keeps
    the three levels' very different activation statistics from fighting.

    A head that un-shared the convs, or that shared the BNs, would train
    perfectly happily and be a different model, so
    ``tests/test_od_hand_written_detectors.py`` asserts the identity of the
    conv parameters across levels **and** the non-identity of the BN
    parameters, rather than reading it off this docstring.
    """

    def __init__(self, num_classes, in_channels, strides=STRIDES):
        super().__init__()
        self.num_classes = num_classes
        self.strides = tuple(strides)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()

        for channels in in_channels:
            cls_tower = nn.ModuleList()
            reg_tower = nn.ModuleList()
            for index in range(STACKED_CONVS):
                source = channels if index == 0 else FEAT_CHANNELS
                cls_tower.append(ConvBNAct(source, FEAT_CHANNELS, 3, stride=1))
                reg_tower.append(ConvBNAct(source, FEAT_CHANNELS, 3, stride=1))
            self.cls_convs.append(cls_tower)
            self.reg_convs.append(reg_tower)
            self.rtm_cls.append(nn.Conv2d(FEAT_CHANNELS, num_classes, 1))
            self.rtm_reg.append(nn.Conv2d(FEAT_CHANNELS, 4, 1))

        self._share_convs()
        self._init_prediction_biases()

    def _share_convs(self):
        """Point every level's conv at level 0's, leaving the BNs alone.

        Rebinding the submodule (rather than copying tensors) is what makes the
        sharing real: all three levels hold the *same* ``nn.Conv2d`` object, so
        ``named_parameters()`` — which de-duplicates by object identity, and is
        what the optimizer and the averaging service see — lists it **once**,
        under level 0's name, and one gradient accumulates from all three
        levels. ``state_dict()`` does *not* de-duplicate, so it still lists
        ``cls_convs.1.0.conv.weight`` and ``cls_convs.2.0.conv.weight`` as
        aliases of the same storage; that is harmless for a strict load and is
        exactly the identity the structural test checks with ``data_ptr()``.
        """
        for level in range(1, len(self.cls_convs)):
            for index in range(STACKED_CONVS):
                self.cls_convs[level][index].conv = self.cls_convs[0][index].conv
                self.reg_convs[level][index].conv = self.reg_convs[0][index].conv

    def _init_prediction_biases(self, prior_prob=1e-2):
        """Start the sigmoid classifier near p=0.01, or the ~1e4 negative
        priors per image dominate the first batches."""
        bias = -math.log((1.0 - prior_prob) / prior_prob)
        for module in self.rtm_cls:
            nn.init.constant_(module.bias, bias)
        for module in self.rtm_reg:
            nn.init.constant_(module.bias, 0.0)

    def forward(self, features):
        """Return ``(cls_logits, boxes, priors)``.

        ``cls_logits`` is ``(B, N, num_classes)``, ``boxes`` is ``(B, N, 4)``
        pixel xyxy, and ``priors`` is ``(N, 3)`` holding each point's pixel
        ``x``, pixel ``y`` and stride. The stride column is what carries the
        head's multi-level structure into the assigner's soft centre prior.
        """
        cls_outputs = []
        box_outputs = []
        priors = []

        for level, (feature, stride) in enumerate(zip(features, self.strides)):
            cls_feat = feature
            for layer in self.cls_convs[level]:
                cls_feat = layer(cls_feat)
            reg_feat = feature
            for layer in self.reg_convs[level]:
                reg_feat = layer(reg_feat)

            cls_score = self.rtm_cls[level](cls_feat)
            # exp() before the stride scale: every distance is positive, so
            # every decoded box is a valid xyxy. See the module docstring.
            distances = torch.exp(self.rtm_reg[level](reg_feat).clamp(max=8.0)) * stride

            batch, num_classes, height, width = cls_score.shape
            cls_outputs.append(
                cls_score.permute(0, 2, 3, 1).reshape(batch, height * width, num_classes)
            )
            distances = distances.permute(0, 2, 3, 1).reshape(batch, height * width, 4)

            # offset=0: RTMDet's points sit on the grid corners, not the cell
            # centres. Scaled by THIS level's stride.
            yv, xv = torch.meshgrid(
                torch.arange(height, device=cls_score.device, dtype=cls_score.dtype),
                torch.arange(width, device=cls_score.device, dtype=cls_score.dtype),
                indexing="ij",
            )
            point_x = xv.reshape(-1) * stride
            point_y = yv.reshape(-1) * stride
            priors.append(
                torch.stack(
                    (point_x, point_y, torch.full_like(point_x, float(stride))), dim=1
                )
            )

            box_outputs.append(
                torch.stack(
                    (
                        point_x - distances[..., 0],
                        point_y - distances[..., 1],
                        point_x + distances[..., 2],
                        point_y + distances[..., 3],
                    ),
                    dim=-1,
                )
            )

        return (
            torch.cat(cls_outputs, dim=1),
            torch.cat(box_outputs, dim=1),
            torch.cat(priors, dim=0),
        )


def _quality_focal_loss(logits, labels, quality, beta=QFL_BETA):
    """Per-prior quality focal loss (Li et al., 2020), summed over classes.

    Negatives target 0 on every channel, weighted by ``sigmoid(p) ** beta``;
    a positive targets its matched IoU on its own channel, weighted by
    ``|target - sigmoid(p)| ** beta``. Regressing the *quality* rather than a
    hard 1 is what lets RTMDet rank boxes by localisation accuracy without a
    separate centreness or objectness branch.
    """
    num_classes = logits.shape[1]
    probability = logits.sigmoid()

    loss = F.binary_cross_entropy_with_logits(
        logits, torch.zeros_like(logits), reduction="none"
    ) * probability.pow(beta)

    positive = ((labels >= 0) & (labels < num_classes)).nonzero(as_tuple=True)[0]
    if positive.numel():
        positive_labels = labels[positive].long()
        target = quality[positive]
        current = probability[positive, positive_labels]
        loss[positive, positive_labels] = F.binary_cross_entropy_with_logits(
            logits[positive, positive_labels], target, reduction="none"
        ) * (target - current).abs().pow(beta)

    return loss.sum(dim=1)


class RTMDetS(nn.Module):
    """RTMDet-S speaking the ``torchvision_detection`` contract."""

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
        self.backbone = CSPNeXt()
        self.neck = CSPNeXtPAFPN(self.backbone.out_channels)
        self.head = RTMDetSepBNHead(self.num_classes, self.neck.out_channels)

        self.score_thresh = SCORE_THRESH
        self.nms_thresh = NMS_THRESH
        self.detections_per_image = DETECTIONS_PER_IMAGE

    # -- contract entry point ------------------------------------------------

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError(
                "rtmdet_s: train mode requires targets — the engine calls "
                "model(images, targets) for the loss dict and model(images) "
                "only in eval mode"
            )

        original_image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
        image_list, targets = self.transform(list(images), targets)

        cls_logits, boxes, priors = self.head(
            self.neck(self.backbone(image_list.tensors))
        )

        if self.training:
            return self._losses(cls_logits, boxes, priors, targets)

        detections = self._predictions(cls_logits, boxes, image_list.image_sizes)
        return self.transform.postprocess(
            detections, image_list.image_sizes, original_image_sizes
        )

    # -- training ------------------------------------------------------------

    def _losses(self, cls_logits, boxes, priors, targets):
        """Soft-label-assigned losses, returned as the handler's loss dict."""
        all_labels = []
        all_quality = []
        positive_predictions = []
        positive_targets = []

        for image_index, target in enumerate(targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"].long()
            num_priors = int(priors.shape[0])

            labels = torch.full(
                (num_priors,),
                self.num_classes,
                dtype=torch.int64,
                device=priors.device,
            )
            quality = torch.zeros(num_priors, dtype=boxes.dtype, device=priors.device)

            if int(gt_boxes.shape[0]) > 0:
                positive_mask, matched_gt, matched_ious = self.assign(
                    gt_boxes, gt_labels, boxes[image_index], cls_logits[image_index], priors
                )
                if int(positive_mask.sum()) > 0:
                    labels[positive_mask] = gt_labels[matched_gt]
                    quality[positive_mask] = matched_ious
                    positive_predictions.append(boxes[image_index][positive_mask])
                    positive_targets.append(gt_boxes[matched_gt])

            all_labels.append(labels)
            all_quality.append(quality)

        labels = torch.cat(all_labels, dim=0)
        quality = torch.cat(all_quality, dim=0)
        # Normalised by the sum of assigned qualities, not by the positive
        # count: a badly localised positive should pull less weight than a
        # well localised one, which is the same idea the soft label encodes.
        avg_factor = quality.sum().clamp(min=1.0)

        loss_cls = (
            _quality_focal_loss(
                cls_logits.reshape(-1, self.num_classes), labels, quality
            ).sum()
            / avg_factor
        )

        if positive_predictions:
            predicted = torch.cat(positive_predictions, dim=0)
            expected = torch.cat(positive_targets, dim=0)
            weight = quality[labels < self.num_classes]
            loss_bbox = (
                BBOX_LOSS_WEIGHT
                * (
                    generalized_box_iou_loss(predicted, expected, reduction="none")
                    * weight
                ).sum()
                / avg_factor
            )
        else:
            # Keep the dict shape and the graph: ``* 0.0`` on a real prediction
            # tensor keeps the regression branch connected to the loss, so a
            # batch of empty images does not produce ``None`` grads.
            loss_bbox = boxes.sum() * 0.0

        return {"loss_cls": loss_cls, "loss_bbox": loss_bbox}

    @torch.no_grad()
    def assign(self, gt_boxes, gt_labels, boxes, cls_logits, priors):
        """Dynamic soft-label assignment.

        Returns ``(positive_mask, matched_gt, matched_ious)`` for one image.
        ``boxes`` and ``gt_boxes`` are pixel xyxy; ``priors`` is the head's
        ``(N, 3)`` point-x / point-y / stride table.

        Four parts, and each fails **silently** if it is wrong — an
        all-negative image still gives a finite, small quality-focal loss and a
        clean train step:

        1. only priors whose point falls **inside** a GT box are candidates;
        2. the cost adds a *soft* centre prior, ``10 ** (d / stride - 3)``,
           where ``d`` is the pixel distance from the GT centre — so it is
           scale-aware, which is the whole reason the stride column exists;
        3. ``dynamic_k`` per GT is the rounded sum of its top-``ASSIGNER_TOPK``
           IoUs;
        4. a prior claimed by two GTs is awarded to the cheaper one.

        ``tests/test_od_hand_written_detectors.py`` registers a mutation
        against each and proves it goes red.
        """
        num_gt = int(gt_boxes.shape[0])
        num_priors = int(priors.shape[0])
        empty = (
            torch.zeros(num_priors, dtype=torch.bool, device=priors.device),
            torch.zeros(0, dtype=torch.int64, device=priors.device),
            torch.zeros(0, dtype=boxes.dtype, device=priors.device),
        )

        points = priors[:, :2]
        deltas = torch.cat(
            (points[:, None, :] - gt_boxes[None, :, :2],
             gt_boxes[None, :, 2:] - points[:, None, :]),
            dim=-1,
        )
        inside_gt = deltas.amin(dim=-1) > 0.0
        candidate_mask = inside_gt.any(dim=1)
        if not bool(candidate_mask.any()):
            return empty

        candidate_boxes = boxes[candidate_mask]
        candidate_logits = cls_logits[candidate_mask]
        candidate_priors = priors[candidate_mask]

        ious = box_iou(candidate_boxes, gt_boxes)
        iou_cost = -torch.log(ious + _EPS) * ASSIGNER_IOU_WEIGHT

        gt_centres = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2.0
        distance = (
            (candidate_priors[:, None, :2] - gt_centres[None, :, :])
            .pow(2)
            .sum(dim=-1)
            .sqrt()
            / candidate_priors[:, 2:3]
        )
        soft_centre_prior = torch.pow(10.0, distance - SOFT_CENTER_RADIUS)

        one_hot = (
            F.one_hot(gt_labels, self.num_classes)
            .to(candidate_logits.dtype)
            .unsqueeze(0)
            .expand(candidate_logits.shape[0], num_gt, self.num_classes)
        )
        soft_label = one_hot * ious.unsqueeze(-1)
        expanded_logits = candidate_logits.unsqueeze(1).expand_as(soft_label)
        scale_factor = soft_label - expanded_logits.sigmoid()
        cls_cost = (
            F.binary_cross_entropy_with_logits(
                expanded_logits, soft_label, reduction="none"
            )
            * scale_factor.abs().pow(QFL_BETA)
        ).sum(dim=-1)

        # No hard outside-the-box penalty, deliberately: unlike SimOTA, DSLA
        # keeps the geometry entirely soft past the candidate filter. The
        # centre prior does the work — at ten strides from a GT centre it is
        # 1e7, which dominates any classification or IoU term — and that is
        # precisely why the stride column must be real per level.
        cost = cls_cost + iou_cost + soft_centre_prior

        return self._dynamic_k_match(cost, ious, candidate_mask, num_priors)

    def _dynamic_k_match(self, cost, ious, candidate_mask, num_priors):
        """Award each GT its ``dynamic_k`` cheapest candidates, then break ties.

        ``cost`` and ``ious`` are ``(num_candidates, num_gt)`` — transposed
        relative to SimOTA's convention, because the soft centre prior is
        naturally per-candidate.
        """
        matching = torch.zeros_like(cost, dtype=torch.uint8)
        num_candidates, num_gt = cost.shape

        topk = min(ASSIGNER_TOPK, num_candidates)
        topk_ious, _ = torch.topk(ious, topk, dim=0)
        dynamic_ks = topk_ious.sum(dim=0).int().clamp(min=1)

        for gt_index in range(num_gt):
            _, positions = torch.topk(
                cost[:, gt_index], k=int(dynamic_ks[gt_index]), largest=False
            )
            matching[positions, gt_index] = 1

        contested = matching.sum(dim=1) > 1
        if bool(contested.any()):
            cheapest = torch.argmin(cost[contested], dim=1)
            matching[contested] = 0
            matching[contested.nonzero(as_tuple=True)[0], cheapest] = 1

        selected = matching.sum(dim=1) > 0
        positive_mask = torch.zeros(
            num_priors, dtype=torch.bool, device=cost.device
        )
        positive_mask[candidate_mask.nonzero(as_tuple=True)[0][selected]] = True

        matched_gt = matching[selected].argmax(dim=1)
        matched_ious = (matching.to(ious.dtype) * ious).sum(dim=1)[selected]
        return positive_mask, matched_gt, matched_ious

    # -- inference -----------------------------------------------------------

    def _predictions(self, cls_logits, boxes, image_sizes):
        scores = cls_logits.sigmoid()

        results = []
        for image_boxes, class_scores, (height, width) in zip(
            boxes, scores, image_sizes
        ):
            image_boxes = image_boxes.clone()
            image_boxes[:, 0::2] = image_boxes[:, 0::2].clamp(min=0, max=float(width))
            image_boxes[:, 1::2] = image_boxes[:, 1::2].clamp(min=0, max=float(height))

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
            # have had. Acute here: SCORE_THRESH is 0.001 against a 0.01 prior,
            # so channel 0 clears it constantly at initialisation.
            #
            # Consequence worth stating plainly: this template now REQUIRES the
            # family handler's shift. Fed raw 0-based dataset labels it would
            # discard the first class. That is the contract as of backend#3062,
            # and the zoo's own family train-step test asserts the `[1, C]`
            # range (model-zoo#245).
            class_scores = class_scores[:, 1:]
            num_priors, num_classes = class_scores.shape
            flat_scores = class_scores.reshape(-1)
            labels = (
                torch.arange(1, num_classes + 1, device=image_boxes.device)
                .unsqueeze(0)
                .expand(num_priors, num_classes)
                .reshape(-1)
            )
            box_index = (
                torch.arange(num_priors, device=image_boxes.device)
                .unsqueeze(1)
                .expand(num_priors, num_classes)
                .reshape(-1)
            )

            keep = flat_scores > self.score_thresh
            flat_scores, labels, box_index = (
                flat_scores[keep],
                labels[keep],
                box_index[keep],
            )
            candidate_boxes = image_boxes[box_index]

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
