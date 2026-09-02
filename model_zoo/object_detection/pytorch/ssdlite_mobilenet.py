"""SSDLite320 with a MobileNetV3-Large backbone (Howard et al., ICCV 2019). The smallest detector in the zoo — separable convolutions throughout the head, a reduced-tail backbone and a 320px input. The mobile end of the accuracy/latency curve, and the natural choice for a federated client with no GPU worth the name.

Offline variant: the architecture is built with ``weights=None``, so nothing
is fetched from ``download.pytorch.org`` — the #199 egress lockdown blocks it
— and the template constructs anywhere, network or not. The pretrained COCO
tensors are delivered from the tracebloc model store as the training seed:
upload the matched ``ssdlite_mobilenet_weights.pkl`` sitting next to this file via
``upload_model(..., weights=True)``, and the platform loads it after
``MyModel()`` has built this architecture::

    user.upload_model("ssdlite_mobilenet", weights=True)

The seed carries the BACKBONE ALONE (backend#2642). The keys under
``SEED_EXCLUDED_PREFIXES`` below are stripped from the dump by
``tools/seed_contract.py strip``, so the class head initialises fresh from
whatever ``output_classes`` the linked dataset decides and ONE dump serves
every class count — checked by ``tools/verify_backbone_seeds.py``, which
builds this template at a count no dump was ever made at. What makes the
dump key-exact with the COCO checkpoint in the first place is the
architectural agreement recorded below.

This template calls the high-level builder directly, and the reasoning is
worth spelling out because this builder branches on ``weights_backbone`` more
than any other in the family. It sets ``reduce_tail = weights_backbone is
None``, which is a REAL structural change to the backbone — and the COCO
checkpoint path takes that same branch, because the builder nulls
``weights_backbone`` whenever ``weights`` is given. So ``weights=None,
weights_backbone=None`` and the checkpoint path agree: both build the
reduced-tail backbone, 476 tensors, key for key and shape for shape (verified
by diffing the two builds under torchvision 0.26.0, the engine pin).
``trainable_backbone_layers`` does not diverge either — this builder's default
equals its maximum (6), so both paths train the whole backbone.

``weights_backbone=None`` is NOT optional here: the builder defaults it to
the ``MobileNet_V3_Large_Weights`` ImageNet enum, so a bare
``ssdlite320_mobilenet_v3_large()`` call both fetches from
``download.pytorch.org`` — which the #199 egress lockdown blocks — and, by
flipping ``reduce_tail`` to False, silently builds a DIFFERENT architecture
from the one the COCO checkpoint (and so any future seed) matches.

Verified against torchvision 0.26.0 (the engine pin, ``tools/requirements-engine-pin.txt``).
"""
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("head.classification_head.module_list.0.1.", "head.classification_head.module_list.1.1.", "head.classification_head.module_list.2.1.", "head.classification_head.module_list.3.1.", "head.classification_head.module_list.4.1.", "head.classification_head.module_list.5.1.")


framework = "pytorch"
model_type = "torchvision_detection"
main_method = "MyModel"
license = "BSD-3-Clause"
# SSD's transform is FIXED-size (fixed_size=(320, 320)), not min/max like the
# R-CNN family: every input is resized to exactly 320x320 whatever is declared
# here, so 320 is the only honest value.
image_size = 320
batch_size = 24
output_classes = 12
category = "object_detection"


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background

    # weights=None AND weights_backbone=None: architecture only, no download.
    # weights_backbone additionally selects the reduced-tail backbone here, so
    # it is load-bearing beyond the offline rule (see the module docstring).
    return ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=num_classes
    )
