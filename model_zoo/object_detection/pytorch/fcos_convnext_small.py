"""FCOS with a ConvNeXt-Small backbone (Tian et al., ICCV 2019; Liu et al., CVPR 2022). Anchor-free one-stage detection — per-pixel box regression with a centre-ness branch, no anchor tuning — on a modernised convolutional backbone. The one-stage counterpart to ``faster_rcnn_convnext_small``, and a much stronger baseline than the ResNet-50 FCOS the zoo already ships.

Offline variant: the architecture is built with ``weights=None`` throughout, so
nothing is fetched from ``download.pytorch.org`` — the #199 egress lockdown
blocks it — and the template constructs anywhere, network or not. No seed is
hosted for this template yet, so it random-initialises and there is
no weight file — upload with ``weights=False``::

    user.upload_model("fcos_convnext_small", weights=False)

Until a dump is staged, ``tools/check_dump_coverage.py`` classifies this file
NO_SEED and the statement above is what keeps that classification honest.

Why the backbone is assembled by hand
-------------------------------------
There is no torchvision builder for this pairing — ``fcos_resnet50_fpn`` is the
only FCOS builder — so ``BackboneWithFPN`` is the supported seam and assembly
is the only route, not a preference.

The trap that forces hand-assembly in ``fcos.py`` **does not exist here**: that
template avoids its builder because the builder swaps ``FrozenBatchNorm2d`` ->
``BatchNorm2d`` when no weights are requested, adding a ``num_batches_tracked``
buffer per norm layer. ConvNeXt carries **no BatchNorm at all** — it norms with
``LayerNorm2d`` — so there is no norm-swap branch to dodge. ConvNeXt-Small has
zero buffers in its ``state_dict``; every tensor is a parameter.

Which stages feed the pyramid — and why only three
--------------------------------------------------
``convnext_small().features`` is eight modules: a patch-embed stem, then four
stages each preceded by a downsample. Measured under torchvision 0.26.0 (the
engine pin) on a 256px input, the odd indices are the C2..C5 an FPN wants::

    features.1 ->  96ch @ stride  4
    features.3 -> 192ch @ stride  8
    features.5 -> 384ch @ stride 16
    features.7 -> 768ch @ stride 32

FCOS takes **C3..C5 only** (strides 8/16/32) and adds ``LastLevelP6P7`` for
strides 64 and 128 — the P3..P7 pyramid the paper specifies, and the same
choice ``fcos.py`` makes via ``returned_layers=[2, 3, 4]``. The stride-4 level
is deliberately dropped: an anchor-free head regresses one box per feature
location, so a stride-4 level over an 800px input is ~40k extra locations for
objects the stride-8 level already covers.

A future hosted seed needs a key remap, not a rebuild
-----------------------------------------------------
``BackboneWithFPN`` nests the backbone under ``body``, and
``IntermediateLayerGetter`` re-keys the kept stages by their ``return_layers``
values. So a torchvision ImageNet checkpoint's ``features.3.*`` lands here as
``backbone.body.3.*``: a prefix rename, with shapes untouched. Mechanical work
for whoever hosts the seed (backend#3055), recorded here so it is not
rediscovered.

Verified against torchvision 0.26.0 (the engine pin, ``tools/requirements-engine-pin.txt``).
"""
from torchvision.models import convnext_small
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.fcos import FCOS, FCOSClassificationHead
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("head.classification_head.cls_logits.",)

framework = "pytorch"
model_type = "torchvision_detection"
main_method = "MyModel"
license = "BSD-3-Clause"
# GeneralizedRCNNTransform's default is min_size=800, max_size=1333, and it
# UPSCALES anything smaller straight back to 800 — so a smaller declared edge
# would pay the resize twice and change nothing the model sees. 800 is what
# this model actually runs at.
#
# NOTE this is 800 where the existing fcos.py declares 448. That template's
# transform is also min_size=800, so 448 was never the resolution it ran at;
# not corrected here because changing a shipped template's declared shape is a
# separate change with its own blast radius.
image_size = 800
# Conservative on purpose. OD ships no SDK shape-probe (#270), so this value is
# taken at face value with nothing to correct it, and a ConvNeXt-Small backbone
# at 800px is the memory driver rather than the anchor-free head.
batch_size = 2
output_classes = 12
category = "object_detection"


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background

    # weights=None: architecture only, no download (the #199 egress lockdown
    # blocks download.pytorch.org).
    backbone = convnext_small(weights=None)

    # C3..C5 at strides 8/16/32 (see the module docstring), re-keyed 0..2, plus
    # P6/P7 from LastLevelP6P7 for the P3..P7 pyramid FCOS expects.
    # out_channels=256 is the torchvision detection convention.
    body = BackboneWithFPN(
        backbone.features,
        return_layers={"3": "0", "5": "1", "7": "2"},
        in_channels_list=[192, 384, 768],
        out_channels=256,
        extra_blocks=LastLevelP6P7(256, 256),
    )

    model = FCOS(body, num_classes=91)

    # Replace the classification head, matching the pattern fcos.py uses: build
    # at the stock 91-class COCO width, then rebuild the head from
    # output_classes so the seed contract above stays derivable.
    model.head.classification_head = FCOSClassificationHead(
        body.out_channels,
        model.head.classification_head.num_anchors,
        num_classes,
    )

    return model
