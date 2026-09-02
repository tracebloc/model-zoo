"""FCOS with a Swin Transformer Tiny backbone (Tian et al., ICCV 2019; Liu et al., ICCV 2021). Anchor-free one-stage detection — per-pixel box regression with a centre-ness branch, no anchor tuning — on a shifted-window transformer backbone. The lightest transformer-backed detector in the roster: Swin-T is 28M parameters, below the ResNet-50 the zoo's other FCOS uses, and the anchor-free head adds less than an RPN.

Offline variant: the architecture is built with ``weights=None`` throughout, so
nothing is fetched from ``download.pytorch.org`` — the #199 egress lockdown
blocks it — and the template constructs anywhere, network or not. No seed is
hosted for this template yet, so it random-initialises and there is
no weight file — upload with ``weights=False``::

    user.upload_model("fcos_swin_t", weights=False)

Until a dump is staged, ``tools/check_dump_coverage.py`` classifies this file
NO_SEED and the statement above is what keeps that classification honest.

⚠️ Swin's stages emit NHWC, and the FPN assumes NCHW
----------------------------------------------------
This is the one thing that makes a Swin detection backbone different from every
other template in this family.

``torchvision.models.swin_t().features`` returns tensors shaped
``(N, H, W, C)`` — channels last — because the shifted-window blocks operate on
a token grid. Measured under torchvision 0.26.0 on a 256px input, the stage
outputs are ``(1, 64, 64, 96)``, ``(1, 32, 32, 192)``, ``(1, 16, 16, 384)``,
``(1, 8, 8, 768)``. Both ``IntermediateLayerGetter`` and
``FeaturePyramidNetwork`` read channels from **dim 1**, which in that layout is
*H*, so the FPN's 1x1 lateral convolutions get built against a spatial extent
instead of a channel count.

Checked rather than assumed, because the interesting question was whether this
fails loudly or quietly: handing Swin's features straight to
``BackboneWithFPN`` and running one train step **raises**, at the lateral conv
for the deepest stage::

    RuntimeError: Given groups=1, weight of size [256, 768, 1, 1], expected
    input[1, 25, 32, 768] to have 768 channels, but got 25 channels instead

So the failure mode is a shape error on the first batch, not silently wrong
learning — which is the good outcome, and worth recording so nobody
re-derives the scarier version of this note. It would only pass quietly if
*H* happened to equal *C* at every pyramid level at once, which no realistic
input size produces.

``_NCHWFeatures`` below subclasses ``BackboneWithFPN`` and permutes between the
body and the FPN. Subclassing rather than wrapping is deliberate — inserting a
wrapper module would prefix every backbone key and cost the key-exactness a
hosted seed needs, whereas overriding ``forward`` leaves the ``state_dict``
byte-for-byte identical to the unpermuted assembly.

*This class is duplicated from* ``faster_rcnn_swin_t.py`` *rather than shared.*
Zoo templates are uploaded to the platform one file at a time and there is not
a single sibling or relative import anywhere in ``model_zoo/``; a shared helper
would make both files fail on upload. The duplication is the contract, not an
oversight.

Which stages feed the pyramid — and why only three
--------------------------------------------------
``swin_t().features`` is eight modules: a patch-embed stem, then four stages
each preceded by a ``PatchMerging``. The odd indices are the C2..C5 an FPN
wants — 96/192/384/768 channels at strides 4/8/16/32.

FCOS takes **C3..C5 only** (strides 8/16/32) and adds ``LastLevelP6P7`` for
strides 64 and 128 — the P3..P7 pyramid the paper specifies, and the same
choice ``fcos.py`` makes via ``returned_layers=[2, 3, 4]``. The stride-4 level
is deliberately dropped: an anchor-free head regresses one box per feature
location, so a stride-4 level over an 800px input is ~40k extra locations for
objects the stride-8 level already covers. Dropping it matters more here than
for ConvNeXt — stride 4 is where window attention is most expensive.

Unlike ConvNeXt, Swin-T's ``state_dict`` is not all parameters: it carries 12
``attn.relative_position_index`` buffers, one per transformer block
(depths 2/2/6/2). They are integer index tables, correctly receive no gradient,
and a seed must carry them — which is why the seed contract is derived by shape
diff rather than by "everything that trains".

A future hosted seed needs a key remap, not a rebuild
-----------------------------------------------------
``BackboneWithFPN`` nests the backbone under ``body`` and re-keys the kept
stages by their ``return_layers`` values, so a torchvision ImageNet
checkpoint's ``features.3.*`` lands here as ``backbone.body.3.*``: a prefix
rename with shapes untouched. Mechanical work for whoever hosts the seed
(backend#3055), recorded here so it is not rediscovered.

Verified against torchvision 0.26.0 (the engine pin, ``tools/requirements-engine-pin.txt``).
"""
from collections import OrderedDict

from torchvision.models import swin_t
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
# UPSCALES anything smaller straight back to 800, so 800 is what this model
# actually runs at. The transform also pads to a multiple of 32, which is what
# keeps the three PatchMerging stages on even dimensions.
image_size = 800
# Conservative on purpose. OD ships no SDK shape-probe (#270), so this value is
# taken at face value. Window attention at 800px is the memory driver here, not
# the parameter count.
batch_size = 2
output_classes = 12
category = "object_detection"


class _NCHWFeatures(BackboneWithFPN):
    """``BackboneWithFPN`` for a backbone whose stages emit NHWC.

    Swin returns ``(N, H, W, C)``; the FPN reads channels from dim 1. Permuting
    between body and FPN is the whole fix. See the module docstring for why this
    is a subclass and not a wrapper, and why it is duplicated rather than shared.
    """

    def forward(self, x):
        features = self.body(x)
        features = OrderedDict(
            (name, feature.permute(0, 3, 1, 2).contiguous())
            for name, feature in features.items()
        )
        return self.fpn(features)


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background

    # weights=None: architecture only, no download (the #199 egress lockdown
    # blocks download.pytorch.org).
    backbone = swin_t(weights=None)

    # C3..C5 at strides 8/16/32 (see the module docstring), re-keyed 0..2, plus
    # P6/P7 from LastLevelP6P7 for the P3..P7 pyramid FCOS expects.
    # out_channels=256 is the torchvision detection convention.
    body = _NCHWFeatures(
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
