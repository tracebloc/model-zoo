"""FCOS with a ConvNeXt-Small backbone (Tian et al., ICCV 2019; Liu et al., CVPR 2022). Anchor-free one-stage detection — per-pixel box regression with a centre-ness branch, no anchor tuning — on a modernised convolutional backbone. The one-stage counterpart to ``faster_rcnn_convnext_small``, and a much stronger baseline than the ResNet-50 FCOS the zoo already ships.

Offline variant: the architecture is built with ``weights=None`` throughout, so
nothing is fetched from ``download.pytorch.org`` — the (internal ref) egress lockdown
blocks it — and the template constructs anywhere, network or not. The pretrained
backbone is delivered from the tracebloc model store as the training seed:
upload the matched ``fcos_convnext_small_weights.pkl`` sitting next to
this file via ``weights=True``, and the platform loads it after
``MyModel()`` has built this architecture::

    user.upload_model("fcos_convnext_small", weights=True)

The seed's pretrained content is the BACKBONE ALONE (internal ref). No
checkpoint exists for this detector as a whole, so every tensor under
``backbone.body.`` is torchvision's ImageNet ``convnext_small`` checkpoint, re-
keyed by the remap recorded below, and the FPN with its P6/P7 blocks and the
FCOS classification and regression towers — which no checkpoint has — travel in
the dump at this template's own fresh initialisation. The keys under
``SEED_EXCLUDED_PREFIXES`` below are stripped from the dump by
``tools/seed_contract.py strip``, so the class head initialises fresh from
whatever ``output_classes`` the linked dataset decides and ONE dump serves
every class count — checked by ``tools/verify_backbone_seeds.py``, which builds
this template at a count no dump was ever made at.

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
stages each preceded by a downsample. Measured under the engine pin
(``tools/requirements-engine-pin.txt``) on a 256px input, the odd indices are the C2..C5 an FPN wants::

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

The hosted seed is a key remap, not a rebuild
---------------------------------------------
``BackboneWithFPN`` nests the backbone under ``body``, and
``IntermediateLayerGetter`` re-keys the kept stages by their ``return_layers``
values. So a torchvision ImageNet checkpoint's ``features.3.*`` lands here as
``backbone.body.3.*``: a prefix rename, with shapes untouched. The seed's prep
applies exactly this rename from a committed recipe (internal ref), and refuses
a checkpoint key it cannot place, a shape that differs, or a backbone key left
without a source — so a partly-seeded backbone cannot be written.

Verified against the engine pin (``tools/requirements-engine-pin.txt``).
"""
from torchvision.models import convnext_small
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.fcos import FCOS, FCOSClassificationHead
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

# (internal ref) — the task head is NOT carried by the hosted seed.
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
# NOTE `fcos.py` USED to declare 448 while its transform ran at min_size=800, so
# 448 was never the resolution it ran at. That separate change has since landed
#: `fcos.py`, `faster_rcnn_resnet` and `retinanet`
# all declare 800 now, and `KNOWN_MISMATCHES` in
# tests/test_od_declared_resolution.py is empty with its ratchet at zero.
image_size = 800
# Conservative on purpose. OD ships no SDK shape-probe (internal ref), so this value is
# taken at face value with nothing to correct it, and a ConvNeXt-Small backbone
# at 800px is the memory driver rather than the anchor-free head.
batch_size = 2
output_classes = 12
category = "object_detection"


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background

    # weights=None: architecture only, no download (the internal ref egress lockdown
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
