"""SSD300 with a VGG-16 backbone (Liu et al., ECCV 2016). The original single-shot detector — one forward pass, no proposal stage, multi-scale default boxes. Slower per FLOP than anything modern, but it is the reference point every later one-stage design is measured against, and at 300px it trains on very little.

Offline variant: the architecture is built with ``weights=None``, so nothing
is fetched from ``download.pytorch.org`` — the #199 egress lockdown blocks it
— and the template constructs anywhere, network or not. The pretrained COCO
tensors are delivered from the tracebloc model store as the training seed:
upload the matched ``ssd_vgg16_weights.pkl`` sitting next to this file via
``upload_model(..., weights=True)``, and the platform loads it after
``MyModel()`` has built this architecture::

    user.upload_model("ssd_vgg16", weights=True)

The seed carries the BACKBONE ALONE (backend#2642). The keys under
``SEED_EXCLUDED_PREFIXES`` below are stripped from the dump by
``tools/seed_contract.py strip``, so the class head initialises fresh from
whatever ``output_classes`` the linked dataset decides and ONE dump serves
every class count — checked by ``tools/verify_backbone_seeds.py``, which
builds this template at a count no dump was ever made at. What makes the
dump key-exact with the COCO checkpoint in the first place is the
architectural agreement recorded below.

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
