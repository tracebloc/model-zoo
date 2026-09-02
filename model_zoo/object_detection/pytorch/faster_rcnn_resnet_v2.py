"""Faster R-CNN ResNet-50 FPN **v2** (Li et al., 2021 training recipe). The improved-recipe rebuild of the two-stage baseline — a deeper RPN head, a convolutional box head with normalization, and the modern augmentation/schedule — a large jump in box AP over the v1 this zoo has shipped until now.

Offline variant: the architecture is built with ``weights=None``, so nothing
is fetched from ``download.pytorch.org`` — the #199 egress lockdown blocks it
— and the template constructs anywhere, network or not. No seed is hosted for
this template yet, so it random-initialises and there is no weight file:
upload with ``weights=False``::

    user.upload_model("faster_rcnn_resnet_v2", weights=False)

Hosting the torchvision COCO tensors as a tracebloc model-store seed (the
#1499 pattern: a matched ``<stem>_weights.pkl`` prepped by
``tools/prep_offline_weights.py`` and strict-loaded after ``MyModel()`` has
built the architecture) is follow-up work, not part of this roster addition.
What makes it possible is the key-exactness recorded below — until a dump is
staged, ``tools/check_dump_coverage.py`` classifies this file NO_SEED and the
statement above is what keeps that classification honest.

Unlike ``faster_rcnn_resnet.py`` / ``retinanet.py`` / ``fcos.py``, this
template calls the high-level builder directly rather than assembling the
backbone by hand. Those three do it by hand because their v1 builders key the
architecture off whether weights were requested — with none they swap the
backbone norm from ``FrozenBatchNorm2d`` to trainable ``BatchNorm2d``, which
adds a ``num_batches_tracked`` buffer per norm layer and so breaks the
key-exact match a hosted seed needs. The **v2** recipe has no such branch: it
passes ``norm_layer=nn.BatchNorm2d`` unconditionally, so ``weights=None`` and
the COCO checkpoint path build the same 404-tensor state_dict, key for key and
shape for shape (verified by diffing the two builds under torchvision 0.26.0,
the engine pin). Hand-assembly here would only be a second, driftable copy of
the builder.

The one thing ``weights=None`` does change is ``trainable_backbone_layers``:
the checkpoint path freezes the first two ResNet stages (33 parameters), a
no-weights build trains all five. That is ``requires_grad`` only — no key, no
shape, no effect on the seed — and the builder cannot be talked out of it
(``_validate_trainable_layers`` discards the argument with a warning when no
weights are requested). Left as the builder sets it: freezing randomly
initialised stages is worse than training seeded ones.

Note for federated runs: the v2 recipe's backbone norm is trainable
``BatchNorm2d``, so its running statistics are averaged across clients — see
the batch-norm caveat in CLAUDE.md's federated-averaging section.

Verified against torchvision 0.26.0 (the engine pin, ``tools/requirements-engine-pin.txt``).
"""
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

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
# The model's own GeneralizedRCNNTransform resizes to min_size=800 before the
# backbone sees anything, so a smaller declared edge is upscaled straight back
# to 800 — paying the resize twice and gaining nothing. Declare what the
# architecture actually trains at.
image_size = 800
batch_size = 2
output_classes = 12
category = "object_detection"


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background

    # weights=None: architecture only, no download (the #199 egress lockdown
    # blocks download.pytorch.org). weights_backbone=None is passed explicitly
    # even though it is this builder's default — the mobilenet and SSD builders
    # default it to an ImageNet enum that WOULD fetch, so stating it is the
    # habit that keeps this family offline.
    return fasterrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None, num_classes=num_classes
    )
