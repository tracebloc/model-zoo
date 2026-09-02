"""Faster R-CNN with a MobileNetV3-Large FPN backbone, tuned for 320px input. The low-resolution sibling of ``faster_rcnn_mobilenet.py``: same architecture, but torchvision's small-input configuration — a 320/640 transform and a far shorter RPN proposal list at inference — for the cheapest two-stage option in the zoo.

Offline variant: the architecture is built with ``weights=None``, so nothing
is fetched from ``download.pytorch.org`` — the #199 egress lockdown blocks it
— and the template constructs anywhere, network or not. The pretrained COCO
tensors are delivered from the tracebloc model store as the training seed:
upload the matched ``faster_rcnn_mobilenet_320_weights.pkl`` sitting next to this file via
``upload_model(..., weights=True)``, and the platform loads it after
``MyModel()`` has built this architecture::

    user.upload_model("faster_rcnn_mobilenet_320", weights=True)

The seed carries the BACKBONE ALONE (backend#2642). The keys under
``SEED_EXCLUDED_PREFIXES`` below are stripped from the dump by
``tools/seed_contract.py strip``, so the class head initialises fresh from
whatever ``output_classes`` the linked dataset decides and ONE dump serves
every class count — checked by ``tools/verify_backbone_seeds.py``, which
builds this template at a count no dump was ever made at. What makes the
dump key-exact with the COCO checkpoint in the first place is the
architectural agreement recorded below.

The 320 variant differs from ``faster_rcnn_mobilenet.py`` ONLY in
``GeneralizedRCNNTransform`` and RPN inference knobs — none of which are
parameters. The two share a state_dict key for key, so one prepped dump would
serve both once a seed is staged; they are kept as separate templates because
the declared ``image_size`` is the thing a user picks between, and that is
header metadata, not a runtime argument.

The backbone is assembled explicitly instead of via the high-level
``fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)`` builder, for exactly
the reason documented in ``faster_rcnn_resnet.py``: that builder keys its norm
layer off whether weights were requested
(``FrozenBatchNorm2d if is_trained else nn.BatchNorm2d``), and ``BatchNorm2d``
contributes a ``num_batches_tracked`` buffer per norm layer — 46 extra
state_dict keys, 330 tensors against the checkpoint path's 284 — so a seed
prepped from the COCO checkpoint could never strict-load into a
``weights=None`` build. Assembling the backbone directly reproduces the 284
exactly. Verified by diffing the two builds under torchvision 0.26.0, the
engine pin.

Unlike the ResNet R-CNN family, the mobilenet checkpoint path does not zero
``FrozenBatchNorm2d`` eps — ``overwrite_eps`` is called only from the
``*_resnet50_fpn`` v1 builders — so there is no eps overwrite here.

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
# min_size=320 below, so 320 is what the model actually sees — declaring more
# would be resized away, declaring less would be upscaled back.
image_size = 320
batch_size = 16
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

    # The 320-variant defaults, restated because the backbone is assembled here
    # rather than by the builder: the small-input transform plus the shortened
    # RPN proposal list that makes this variant cheap at inference.
    anchor_sizes = ((32, 64, 128, 256, 512),) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return FasterRCNN(
        backbone,
        num_classes,
        rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
        min_size=320,
        max_size=640,
        rpn_pre_nms_top_n_test=150,
        rpn_post_nms_top_n_test=150,
        rpn_score_thresh=0.05,
    )
