"""Faster R-CNN with a MobileNetV3-Large FPN backbone. The edge-class two-stage detector: the same Faster R-CNN head and RPN as ``faster_rcnn_resnet.py`` over a backbone roughly an order of magnitude cheaper, for federated clients whose hardware cannot carry a ResNet-50.

Offline variant: the architecture is built with ``weights=None``, so nothing
is fetched from ``download.pytorch.org`` — the #199 egress lockdown blocks it
— and the template constructs anywhere, network or not. No seed is hosted for
this template yet, so it random-initialises and there is no weight file:
upload with ``weights=False``::

    user.upload_model("faster_rcnn_mobilenet", weights=False)

Hosting the torchvision COCO tensors as a tracebloc model-store seed (the
#1499 pattern: a matched ``<stem>_weights.pkl`` prepped by
``tools/prep_offline_weights.py`` and strict-loaded after ``MyModel()`` has
built the architecture) is follow-up work, not part of this roster addition.
What makes it possible is the key-exactness recorded below — until a dump is
staged, ``tools/check_dump_coverage.py`` classifies this file NO_SEED and the
statement above is what keeps that classification honest.

The backbone is assembled explicitly instead of via the high-level
``fasterrcnn_mobilenet_v3_large_fpn(weights=None)`` builder, for exactly the
reason documented in ``faster_rcnn_resnet.py``: that builder keys its norm
layer off whether weights were requested
(``FrozenBatchNorm2d if is_trained else nn.BatchNorm2d``), and the two are not
interchangeable — ``BatchNorm2d`` contributes a ``num_batches_tracked`` buffer
per norm layer, 46 extra state_dict keys here, so a seed prepped from the COCO
checkpoint could never strict-load into a ``weights=None`` build. Diffing the
two builds under torchvision 0.26.0 (the engine pin) gives 330 tensors for the
builder's no-weights path against 284 for the checkpoint path; assembling the
backbone directly reproduces the 284 exactly — same norm layers, same three
trainable stages, same keys and shapes. The v2 templates in this directory do
NOT need this workaround; their builders have no such branch.

Unlike the ResNet R-CNN family, the mobilenet checkpoint path does not zero
``FrozenBatchNorm2d`` eps — ``overwrite_eps`` is called only from the
``*_resnet50_fpn`` v1 builders — so there is no eps overwrite here.

``_mobilenet_extractor`` is torchvision-private API, as ``_resnet_fpn_extractor``
is in the sibling templates. If a torchvision upgrade moves it, this template
fails loudly at import and the contract tests catch it.

Verified against torchvision 0.26.0 (the engine pin, ``tools/requirements-engine-pin.txt``).
"""
from torchvision.models import mobilenet_v3_large
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _mobilenet_extractor
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.ops import misc as misc_nn_ops

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("roi_heads.box_predictor.bbox_pred.", "roi_heads.box_predictor.cls_score.")

framework = "pytorch"
model_type = "torchvision_detection"
main_method = "MyModel"
license = "BSD-3-Clause"
# This variant keeps Faster R-CNN's stock transform (min_size=800,
# max_size=1333), so anything smaller is upscaled back to 800 inside the model.
# The 320 low-resolution variant is faster_rcnn_mobilenet_320.py.
image_size = 800
batch_size = 8
output_classes = 12
category = "object_detection"


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background

    # Reproduce the checkpoint-path architecture exactly, with no download:
    # frozen batch-norm backbone and the FPN over the last 3 trainable stages.
    backbone = mobilenet_v3_large(
        weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d
    )
    backbone = _mobilenet_extractor(backbone, True, 3)

    # The builder's anchor configuration and rpn_score_thresh default, restated
    # because the backbone is assembled here rather than by the builder.
    anchor_sizes = ((32, 64, 128, 256, 512),) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return FasterRCNN(
        backbone,
        num_classes,
        rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
        rpn_score_thresh=0.05,
    )
