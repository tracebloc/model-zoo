"""SSD300 with a VGG-16 backbone (Liu et al., ECCV 2016). The original single-shot detector — one forward pass, no proposal stage, multi-scale default boxes. Slower per FLOP than anything modern, but it is the reference point every later one-stage design is measured against, and at 300px it trains on very little.

Offline variant: the architecture is built with ``weights=None``, so nothing
is fetched from ``download.pytorch.org`` — the #199 egress lockdown blocks it
— and the template constructs anywhere, network or not. No seed is hosted for
this template yet, so it random-initialises and there is no weight file:
upload with ``weights=False``::

    user.upload_model("ssd_vgg16", weights=False)

Hosting the torchvision COCO tensors as a tracebloc model-store seed (the
#1499 pattern: a matched ``<stem>_weights.pkl`` prepped by
``tools/prep_offline_weights.py`` and strict-loaded after ``MyModel()`` has
built the architecture) is follow-up work, not part of this roster addition.
What makes it possible is the key-exactness recorded below — until a dump is
staged, ``tools/check_dump_coverage.py`` classifies this file NO_SEED and the
statement above is what keeps that classification honest.

This template calls the high-level builder directly. The hand-assembly the v1
R-CNN templates need is about a norm-layer swap that changes the state_dict
key set when no weights are requested; ``ssd300_vgg16`` has no such branch —
torchvision's ``vgg16`` carries no normalization layers at all — so
``weights=None`` and the COCO checkpoint path build the same 71-tensor
state_dict, key for key and shape for shape (verified by diffing the two
builds under torchvision 0.26.0, the engine pin).

``weights_backbone=None`` is NOT optional here: this builder defaults it to
the ``VGG16_Weights`` ImageNet-features enum, so the bare ``ssd300_vgg16()`` call
fetches ImageNet weights from ``download.pytorch.org`` — which the #199 egress
lockdown blocks. Passing it explicitly is what keeps the template offline.

The remaining difference from the checkpoint path is
``trainable_backbone_layers``: the checkpoint path freezes the first VGG stage
(4 parameters), a no-weights build trains all five. ``requires_grad`` only —
no key, no shape, no effect on the seed — and the builder discards the
argument with a warning when no weights are requested.

Verified against torchvision 0.26.0 (the engine pin, ``tools/requirements-engine-pin.txt``).
"""
from torchvision.models.detection import ssd300_vgg16

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("head.classification_head.module_list.0.", "head.classification_head.module_list.1.", "head.classification_head.module_list.2.", "head.classification_head.module_list.3.", "head.classification_head.module_list.4.", "head.classification_head.module_list.5.")


framework = "pytorch"
model_type = "torchvision_detection"
main_method = "MyModel"
license = "BSD-3-Clause"
# SSD's transform is FIXED-size (fixed_size=(300, 300)), not min/max like the
# R-CNN family: every input is resized to exactly 300x300 whatever is declared
# here, so 300 is the only honest value.
image_size = 300
batch_size = 16
output_classes = 12
category = "object_detection"


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background

    # weights=None AND weights_backbone=None: architecture only, no download.
    # This builder's weights_backbone default is an ImageNet enum, so omitting
    # it would fetch (see the module docstring).
    return ssd300_vgg16(weights=None, weights_backbone=None, num_classes=num_classes)
