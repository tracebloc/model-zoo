"""Faster R-CNN with a ConvNeXt-Small backbone (Liu et al., CVPR 2022). ConvNeXt modernises the plain ResNet stack with depthwise 7x7 convolutions, an inverted bottleneck and LayerNorm, matching transformer accuracy at convolutional cost. A large accuracy step over the ResNet-50 the zoo's other two-stage templates use, at a similar parameter budget.

Offline variant: the architecture is built with ``weights=None`` throughout, so
nothing is fetched from ``download.pytorch.org`` — the (internal ref) egress lockdown
blocks it — and the template constructs anywhere, network or not. The pretrained
backbone is delivered from the tracebloc model store as the training seed:
upload the matched ``faster_rcnn_convnext_small_weights.pkl`` sitting next to
this file via ``weights=True``, and the platform loads it after
``MyModel()`` has built this architecture::

    user.upload_model("faster_rcnn_convnext_small", weights=True)

The seed's pretrained content is the BACKBONE ALONE (internal ref). No
checkpoint exists for this detector as a whole, so every tensor under
``backbone.body.`` is torchvision's ImageNet ``convnext_small`` checkpoint, re-
keyed by the remap recorded below, and the FPN, the RPN and the ROI box head —
which no checkpoint has — travel in the dump at this template's own fresh
initialisation. The keys under ``SEED_EXCLUDED_PREFIXES`` below are stripped
from the dump by ``tools/seed_contract.py strip``, so the class head
initialises fresh from whatever ``output_classes`` the linked dataset decides
and ONE dump serves every class count — checked by
``tools/verify_backbone_seeds.py``, which builds this template at a count no
dump was ever made at.

Why the backbone is assembled by hand
-------------------------------------
There is no torchvision builder for this pairing — the eleven detection
builders all wrap ResNet-50, MobileNetV3 or VGG16 — so ``BackboneWithFPN`` is
the supported seam and assembly is the only route, not a preference.

Notably the trap that forces hand-assembly in ``faster_rcnn_resnet.py`` and
``fcos.py`` **does not exist here**. Those templates avoid their builder
because it swaps ``FrozenBatchNorm2d`` -> ``BatchNorm2d`` when no weights are
requested, adding a ``num_batches_tracked`` buffer per norm layer and breaking
key-exactness with a hosted seed. ConvNeXt carries **no BatchNorm at all** —
it norms with ``LayerNorm2d`` — so there is no norm-swap branch to dodge and no
buffer divergence to reproduce. ConvNeXt-Small has zero buffers in its
``state_dict``; every tensor is a parameter.

Which stages feed the pyramid
-----------------------------
``convnext_small().features`` is eight modules: a patch-embed stem, then four
stages each preceded by a downsample. Measured under the engine pin
(``tools/requirements-engine-pin.txt``) on a 256px input, the stage outputs are::

    features.1 ->  96ch @ stride  4
    features.3 -> 192ch @ stride  8
    features.5 -> 384ch @ stride 16
    features.7 -> 768ch @ stride 32

so the odd indices are the C2..C5 an FPN wants, and the channel list is read
off the architecture rather than assumed. Faster R-CNN takes all four plus
``LastLevelMaxPool``, giving the five pyramid levels its default
``AnchorGenerator`` is built for.

The hosted seed is a key remap, not a rebuild
---------------------------------------------
``BackboneWithFPN`` nests the backbone under ``body``, and
``IntermediateLayerGetter`` re-keys the kept stages by their ``return_layers``
values. So a torchvision ImageNet checkpoint's ``features.1.*`` lands here as
``backbone.body.1.*``: a prefix rename, with shapes untouched. The seed's prep
applies exactly this rename from a committed recipe (internal ref), and refuses
a checkpoint key it cannot place, a shape that differs, or a backbone key left
without a source — so a partly-seeded backbone cannot be written.

Verified against the engine pin (``tools/requirements-engine-pin.txt``).
"""
from torchvision.models import convnext_small
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

# (internal ref) — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("roi_heads.box_predictor.bbox_pred.", "roi_heads.box_predictor.cls_score.")

framework = "pytorch"
model_type = "torchvision_detection"
main_method = "MyModel"
license = "BSD-3-Clause"
# GeneralizedRCNNTransform's default is min_size=800, max_size=1333, and it
# UPSCALES anything smaller straight back to 800 — so a smaller declared edge
# would pay the resize twice and change nothing the model sees. 800 is what
# this model actually runs at. (Not overridden here: the whole roster shares
# the torchvision default, and a per-template override is a silent divergence.)
image_size = 800
# Conservative on purpose. OD ships no SDK shape-probe (internal ref), so this value is
# taken at face value with nothing to correct it, and a ConvNeXt-Small backbone
# at 800px is the memory driver rather than the detector heads. 2 matches the
# other 800px two-stage templates in the roster.
batch_size = 2
output_classes = 12
category = "object_detection"


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background

    # weights=None: architecture only, no download (the internal ref egress lockdown
    # blocks download.pytorch.org). ConvNeXt's ImageNet weights are the only
    # thing this argument would fetch, and they arrive as the hosted seed
    # instead; the detector heads have no pretrained source either way.
    backbone = convnext_small(weights=None)

    # C2..C5 at strides 4/8/16/32 (see the module docstring), re-keyed 0..3 for
    # the FPN. out_channels=256 is the torchvision detection convention every
    # other template in this family uses.
    body = BackboneWithFPN(
        backbone.features,
        return_layers={"1": "0", "3": "1", "5": "2", "7": "3"},
        in_channels_list=[96, 192, 384, 768],
        out_channels=256,
        extra_blocks=LastLevelMaxPool(),
    )

    model = FasterRCNN(body, num_classes=91)

    # Replace the classifier head, matching the pattern the rest of the family
    # uses: build at the stock 91-class COCO width, then resize the predictor
    # from output_classes so the seed contract above stays derivable.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
