"""CenterNet — "Objects as Points" (Zhou et al., 2019) on a ResNet-50-FPN backbone. The simplest detector in the roster and the only NMS-free one: an object is a peak in a per-class heatmap, and its box is two more numbers read off at that peak. No anchors, no assignment step, no proposal stage, no non-maximum suppression — a 3x3 max-pool that keeps local maxima is the entire duplicate-removal mechanism. That makes it an unusually good federated baseline (few hyperparameters to disagree about across clients) and the clearest teaching example of a dense detector.

Offline variant: the architecture is built with ``weights=None`` throughout, so
nothing is fetched from ``download.pytorch.org`` — the #199 egress lockdown
blocks it — and the template constructs anywhere, network or not. No seed is
hosted for this template yet, so it random-initialises and there is
no weight file — upload with ``weights=False``::

    user.upload_model("centernet_resnet", weights=False)

Until a dump is staged, ``tools/check_dump_coverage.py`` classifies this file
NO_SEED and the statement above is what keeps that classification honest.

How it works
------------
One stride-4 feature map carries three heads:

- **heatmap**, ``num_classes`` channels — the probability an object centre of
  that class sits at this pixel
- **size**, 2 channels — the object's width and height, in feature-map units
- **offset**, 2 channels — the sub-pixel remainder lost when the true centre
  was quantised to the stride-4 grid

Training draws a 2D Gaussian at each object's centre in the heatmap and
regresses size and offset at the single integer centre pixel. Inference takes
local maxima, reads size and offset at each, and emits boxes. There is no
assignment algorithm at all: an object's positive location is simply where its
centre is.

A single high-resolution map, from the FPN
-----------------------------------------
CenterNet needs one stride-4 map rather than a pyramid. The paper reaches it
with DLA-34 or Hourglass, or with ResNet plus three transposed convolutions.
This template takes the FPN's finest level instead —
``_resnet_fpn_extractor(returned_layers=[1, 2, 3, 4])`` produces P2 at stride 4
with 256 channels, already fused with the coarser levels, which is functionally
what the paper's upsampling path is for. It reuses the same backbone assembly
as every other template in this family, so the roster stays comparable.

⚠️ **This is therefore an FPN variant, not the paper's DLA-34 CenterNet.** Its
COCO numbers will not be the published ones and no seed prepped from a DLA
checkpoint could ever load into it. Recorded plainly because the alternative —
hand-writing DLA's iterative deep aggregation in a single self-contained
template file — is a much larger job for a backbone the rest of the roster
cannot share.

⚠️ The Gaussian radius keeps the reference implementation's arithmetic
-----------------------------------------------------------------------
``_gaussian_radius`` solves three quadratics and takes the smallest root, which
is CornerNet's construction and what CenterNet inherits. The widely-copied
reference divides all three by ``2`` rather than by ``2 * a`` — correct only for
the first, where ``a == 1``. That looks like an obvious bug to fix, and fixing
it here would be wrong.

**Only the third root ever matters.** Measured across box shapes from 1x1 to
80x1 at ``min_overlap = 0.7``, ``r3`` is the minimum in every single case — so
``r1`` and ``r2`` never bind and the ``/2`` slip is irrelevant for both of them.
The whole question is ``r3``, where ``a3 = 4 * 0.7 = 2.8``, so the reference's
``/2`` differs from a general ``/(2 * a3)`` by a factor of 2.8:

    box (feature units)   reference r3   general /(2*a3)
    4x4                       1.09            0.39
    10x10                     2.73            0.98
    20x20                     5.47            1.95
    40x40                    10.93            3.90

The "corrected" radius is near-degenerate: below 1 the Gaussian target is
effectively one-hot, and the heatmap loses the soft neighbourhood that the
penalty-reduced focal loss exists to exploit. The reference value is what the
recipe was tuned around, so this file reproduces it deliberately and says so,
rather than silently shipping either the slip or a tighter target nobody has
trained with. ``tests/test_centernet.py`` pins the radii AND pins that ``r3``
is the binding root, so a change to ``min_overlap`` that promotes a different
root fails loudly instead of quietly changing every training target.

Contract
--------
``model(images, targets)`` returns ``{heatmap, size, offset}``;
``model(images)`` returns ``List[Dict]`` of pixel-xyxy ``boxes``/``scores``/
``labels``. ``GeneralizedRCNNTransform`` is reused for resize, normalisation,
padding to a batch and mapping detections back to original coordinates — the
engine hands a *list* of differently-sized images (``_rcnn_collate`` tuples,
not stacks), and that transform is what the rest of this family uses to absorb it.

Verified against torchvision 0.26.0 (the engine pin, ``tools/requirements-engine-pin.txt``).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import boxes as box_ops
from torchvision.ops import misc as misc_nn_ops

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("heatmap_head.2.",)

framework = "pytorch"
model_type = "torchvision_detection"
main_method = "MyModel"
license = "BSD-3-Clause"
# The transform below is built with min_size=IMAGE_SIZE, so the declared value
# IS the resolution this model runs at (backend#3058).
image_size = 512
# A stride-4 map at 512px is 128x128 = 16k locations with only three thin heads
# on top, so this is lighter than the 800px pyramid templates despite the higher
# resolution. OD ships no SDK shape-probe (#270), so the value is taken at face
# value and chosen conservatively.
batch_size = 8
output_classes = 12
category = "object_detection"

#: Output stride of the feature map the heads sit on. P2 of the FPN.
OUTPUT_STRIDE = 4

#: Peaks kept before scoring, per image. 100 is the paper's value.
MAX_DETECTIONS = 100

#: Peaks below this are not detections. Without it the top-k always returns
#: MAX_DETECTIONS rows, so once the real peaks are exhausted the remaining
#: slots are filled with whatever local maxima the heatmap happens to have --
#: at initialisation, noise. torchmetrics scores every returned row, so those
#: become false positives that no ground truth can match. Matches the family's
#: convention (torchvision's detectors default to 0.05).
SCORE_THRESH = 0.05

#: Minimum IoU a shifted box must retain for the Gaussian radius construction.
GAUSSIAN_MIN_OVERLAP = 0.7

#: Weights on the two L1 terms, as the paper specifies.
SIZE_WEIGHT = 0.1
OFFSET_WEIGHT = 1.0


def _gaussian_radius(height, width, min_overlap=GAUSSIAN_MIN_OVERLAP):
    """Radius whose Gaussian keeps ``min_overlap`` IoU under a centre shift.

    Reproduces the reference implementation's arithmetic, including its
    non-general ``/2``. See the module docstring for the measured reason.
    """
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(b1 * b1 - 4 * a1 * c1, 0))) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(max(b2 * b2 - 4 * a2 * c2, 0))) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(max(b3 * b3 - 4 * a3 * c3, 0))) / 2

    return max(min(r1, r2, r3), 0.0)


def _draw_gaussian(heatmap, centre_x, centre_y, radius):
    """Max-merge a 2D Gaussian into one class channel of ``heatmap`` in place.

    Max-merge rather than add: two nearby objects of the same class must each
    keep a peak of exactly 1.0, because the focal loss below treats ``== 1`` as
    the positive location. Summing would push the overlap above 1 and turn both
    into ignored pixels.
    """
    diameter = 2 * int(radius) + 1
    sigma = diameter / 6.0
    if sigma <= 0:
        return
    height, width = heatmap.shape

    grid = torch.arange(-int(radius), int(radius) + 1, device=heatmap.device, dtype=heatmap.dtype)
    gaussian = torch.exp(-(grid[:, None] ** 2 + grid[None, :] ** 2) / (2 * sigma * sigma))

    left, right = min(centre_x, int(radius)), min(width - centre_x, int(radius) + 1)
    top, bottom = min(centre_y, int(radius)), min(height - centre_y, int(radius) + 1)
    if left + right <= 0 or top + bottom <= 0:
        return

    masked_heatmap = heatmap[centre_y - top : centre_y + bottom, centre_x - left : centre_x + right]
    masked_gaussian = gaussian[
        int(radius) - top : int(radius) + bottom, int(radius) - left : int(radius) + right
    ]
    torch.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def _centernet_focal_loss(predicted, target):
    """CornerNet's penalty-reduced focal loss.

    Positives are the exact peaks (``target == 1``); every other pixel is a
    negative whose penalty is *reduced* by ``(1 - target) ** 4``, so a pixel just
    beside a centre is barely punished for firing. That reduction is the whole
    reason a Gaussian target works, and dropping it makes the loss ordinary
    focal loss against a one-hot map.

    Normalised by the number of objects, not the number of pixels — with ~16k
    locations and a handful of objects, a pixel-mean would make the positive
    term vanish.
    """
    positive_mask = target.eq(1).float()
    negative_mask = 1.0 - positive_mask
    # Clamped away from 0 and 1: log(0) is -inf, and the heads are randomly
    # initialised, so a saturated sigmoid on the first step is not unlikely.
    probabilities = predicted.clamp(min=1e-4, max=1 - 1e-4)

    positive_loss = -torch.log(probabilities) * (1 - probabilities).pow(2) * positive_mask
    negative_loss = (
        -torch.log(1 - probabilities)
        * probabilities.pow(2)
        * (1 - target).pow(4)
        * negative_mask
    )

    num_positive = positive_mask.sum()
    if float(num_positive) == 0:
        # No objects anywhere: only the negative term is defined. Still a tensor
        # on the graph, so sum(losses.values()).backward() works.
        return negative_loss.sum()
    return (positive_loss.sum() + negative_loss.sum()) / num_positive


class _Head(nn.Sequential):
    """The paper's head: one 3x3 conv to 64 channels, then a 1x1 to the output."""

    def __init__(self, in_channels, out_channels, bias_init=None):
        super().__init__(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1),
        )
        nn.init.normal_(self[0].weight, std=0.01)
        nn.init.zeros_(self[0].bias)
        nn.init.normal_(self[2].weight, std=0.01)
        if bias_init is None:
            nn.init.zeros_(self[2].bias)
        else:
            nn.init.constant_(self[2].bias, bias_init)


class CenterNet(nn.Module):
    """CenterNet over a single stride-4 feature map."""

    def __init__(self, backbone, num_classes, min_size=image_size, max_size=896):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.output_stride = OUTPUT_STRIDE
        self.max_detections = MAX_DETECTIONS
        self.score_thresh = SCORE_THRESH
        self.transform = GeneralizedRCNNTransform(
            min_size=min_size,
            max_size=max_size,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )
        channels = backbone.out_channels
        # Prior probability 0.1 for the heatmap, as the paper does: the map is
        # overwhelmingly negative, and a zero bias makes the first steps chase
        # 16k easy negatives.
        self.heatmap_head = _Head(channels, num_classes, bias_init=-2.19)
        self.size_head = _Head(channels, 2)
        self.offset_head = _Head(channels, 2)

    def _features(self, tensors):
        """One stride-4 map, fused from EVERY FPN level.

        The obvious implementation takes the finest level (P2) alone and ignores
        the rest. That trains, and it leaves the FPN's coarser output
        convolutions with **no gradient at all** — measured, six dead parameter
        tensors (``fpn.layer_blocks.{1,2,3}`` weight and bias). Dead weights are
        not merely untidy here: this is a federated platform, so every one of
        them is serialised, uploaded and averaged once per round forever, for a
        tensor that can never change.

        Averaging the upsampled levels instead uses all of them, and is closer
        to the aggregation the paper's DLA path performs than a single level is.
        ``LastLevelMaxPool``'s output is excluded deliberately — it is a
        parameter-free max-pool of the coarsest level, so it contributes nothing
        to learn and skipping it creates no dead weights.

        **Averaged, not summed**, and that is not cosmetic. Summing four levels
        multiplies the activation magnitude reaching the heads, which pushes the
        heatmap logits away from the ``-2.19`` prior-probability bias they are
        initialised to. Measured at initialisation, same seed and inputs:

            fusion      mean sigmoid(heatmap)   heatmap loss
            summed              --                  2262
            averaged           0.124                 263

        The analytic floor for a *perfectly uniform* 0.1 prior on this map is
        ~99, so averaging lands within a small factor of it — the residual is
        per-pixel variance, and the negative term is quadratic in ``p`` so any
        spread raises the mean. Summing is 8.6x worse and would spend the first
        epochs undoing its own initialisation.
        """
        features = list(self.backbone(tensors).values())
        # The FPN emits its levels finest-first; the parameter-free pool level
        # from LastLevelMaxPool is last and has no layer_block behind it.
        pyramid = features[: len(self.backbone.fpn.layer_blocks)]
        finest = pyramid[0]
        fused = finest
        for coarser in pyramid[1:]:
            fused = fused + F.interpolate(
                coarser, size=finest.shape[-2:], mode="bilinear", align_corners=False
            )
        return fused / len(pyramid)

    def _build_targets(self, targets, feature_shape, device, dtype):
        """Gaussian heatmap plus the size/offset regression targets."""
        batch, height, width = feature_shape
        heatmap = torch.zeros((batch, self.num_classes, height, width), device=device, dtype=dtype)
        size_target = torch.zeros((batch, 2, height, width), device=device, dtype=dtype)
        offset_target = torch.zeros((batch, 2, height, width), device=device, dtype=dtype)
        regression_mask = torch.zeros((batch, 1, height, width), device=device, dtype=dtype)

        for index, target in enumerate(targets):
            boxes = target["boxes"]
            labels = target["labels"]
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = (float(v) / self.output_stride for v in box)
                box_width, box_height = x2 - x1, y2 - y1
                if box_width <= 0 or box_height <= 0:
                    continue
                centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2
                grid_x = min(max(int(centre_x), 0), width - 1)
                grid_y = min(max(int(centre_y), 0), height - 1)
                radius = _gaussian_radius(box_height, box_width)
                _draw_gaussian(
                    heatmap[index, int(label)], grid_x, grid_y, max(int(radius), 1)
                )
                size_target[index, 0, grid_y, grid_x] = box_width
                size_target[index, 1, grid_y, grid_x] = box_height
                # The sub-pixel remainder the int() above threw away. Without
                # this head the box centre is quantised to the stride, which at
                # stride 4 is a 4px error on every object.
                offset_target[index, 0, grid_y, grid_x] = centre_x - grid_x
                offset_target[index, 1, grid_y, grid_x] = centre_y - grid_y
                regression_mask[index, 0, grid_y, grid_x] = 1.0

        return heatmap, size_target, offset_target, regression_mask

    def compute_loss(self, outputs, targets, device, dtype):
        heatmap_logits = outputs["heatmap"]
        batch, _, height, width = heatmap_logits.shape
        heatmap_target, size_target, offset_target, mask = self._build_targets(
            targets, (batch, height, width), device, dtype
        )

        heatmap_loss = _centernet_focal_loss(heatmap_logits.sigmoid(), heatmap_target)

        # L1 only at the object centres. Normalised by the object count, and
        # clamped so an all-background batch divides by 1 rather than 0.
        num_objects = mask.sum().clamp(min=1.0)
        size_loss = (F.l1_loss(outputs["size"] * mask, size_target * mask, reduction="sum")
                     / num_objects)
        offset_loss = (F.l1_loss(outputs["offset"] * mask, offset_target * mask, reduction="sum")
                       / num_objects)

        return {
            "heatmap": heatmap_loss,
            "size": size_loss * SIZE_WEIGHT,
            "offset": offset_loss * OFFSET_WEIGHT,
        }

    @staticmethod
    def _peaks(heatmap):
        """Keep only local maxima — CenterNet's whole substitute for NMS.

        A 3x3 max-pool equals the input exactly where the pixel is the largest
        in its neighbourhood. Everything else is zeroed, so two peaks closer
        than one pixel cannot both survive and nothing else needs suppressing.
        """
        pooled = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
        return heatmap * (pooled == heatmap).to(heatmap.dtype)

    def decode(self, outputs, image_shapes):
        """Peaks -> boxes. No NMS: see ``_peaks``."""
        heatmap = self._peaks(outputs["heatmap"].sigmoid())
        size = outputs["size"]
        offset = outputs["offset"]
        batch, num_classes, height, width = heatmap.shape

        detections = []
        for index in range(batch):
            # ⚠️ Channel 0 is the BACKGROUND slot. `MyModel` allocates
            # `output_classes + 1` channels to match this family's model space,
            # where torchvision reserves index 0 for background, so channel 0
            # carries no dataset class. It is sliced off BEFORE the top-k, not
            # filtered after: the top-k has a fixed budget, so a background
            # local maximum admitted here would consume a detection slot that a
            # real object should have had. Emitted labels are therefore in
            # [1, C], which is the model space the engine's family handler
            # translates back to dataset space (backend#3062).
            foreground = heatmap[index, 1:]
            flat = foreground.reshape(-1)
            k = min(self.max_detections, flat.numel())
            scores, topk = flat.topk(k)

            # And a threshold, because top-k always returns k rows: past the
            # real peaks it pads with noise, and every padded row is scored as
            # a false positive.
            keep = scores > self.score_thresh
            scores, topk = scores[keep], topk[keep]

            # Unflatten (class, y, x) from the (C-1, H, W) index, then +1 to
            # step back over the background channel that was sliced away.
            classes = torch.div(
                topk, height * width, rounding_mode="floor") + 1
            spatial = topk % (height * width)
            grid_y = torch.div(spatial, width, rounding_mode="floor")
            grid_x = spatial % width

            centre_x = grid_x.to(size.dtype) + offset[index, 0].reshape(-1)[spatial]
            centre_y = grid_y.to(size.dtype) + offset[index, 1].reshape(-1)[spatial]
            # ⚠️ Clamped at zero. The size head is an unconstrained 1x1
            # convolution, so nothing stops it emitting a NEGATIVE width or
            # height — and at random initialisation about half of them are.
            # A negative width produces x2 < x1, which is not a valid xyxy box:
            # the engine's torchmetrics MeanAveragePrecision and
            # IntersectionOverUnion both read these as xyxy pixels and would
            # score nonsense rather than raise. Training pushes the head
            # positive because the L1 target always is, but inference must be
            # correct on step zero too. The clamp is decode-only; the loss stays
            # on the raw output so gradient still flows from negative
            # predictions.
            box_width = size[index, 0].reshape(-1)[spatial].clamp(min=0)
            box_height = size[index, 1].reshape(-1)[spatial].clamp(min=0)

            boxes = torch.stack(
                (
                    centre_x - box_width / 2,
                    centre_y - box_height / 2,
                    centre_x + box_width / 2,
                    centre_y + box_height / 2,
                ),
                dim=1,
            ) * self.output_stride
            boxes = box_ops.clip_boxes_to_image(boxes, image_shapes[index])

            detections.append({"boxes": boxes, "scores": scores, "labels": classes.to(torch.int64)})
        return detections

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("targets should not be none when in training mode")

        original_image_sizes = [(image.shape[-2], image.shape[-1]) for image in images]
        images, targets = self.transform(images, targets)
        features = self._features(images.tensors)
        outputs = {
            "heatmap": self.heatmap_head(features),
            "size": self.size_head(features),
            "offset": self.offset_head(features),
        }

        if self.training:
            return self.compute_loss(
                outputs, targets, images.tensors.device, images.tensors.dtype
            )

        # Detections are decoded in feature coordinates scaled by the output
        # stride, i.e. in the TRANSFORMED image's pixels — which is exactly what
        # transform.postprocess expects before mapping back to the original.
        detections = self.decode(outputs, images.image_sizes)
        return self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background

    # weights=None: architecture only, no download (the #199 egress lockdown
    # blocks download.pytorch.org). FrozenBatchNorm2d and trainable_layers=3
    # match the rest of the family.
    backbone = resnet50(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    # returned_layers=[1, 2, 3, 4] so the FPN's finest level is P2 at stride 4,
    # which is the single high-resolution map CenterNet needs.
    backbone = _resnet_fpn_extractor(backbone, trainable_layers=3, returned_layers=[1, 2, 3, 4])

    return CenterNet(backbone, num_classes)
