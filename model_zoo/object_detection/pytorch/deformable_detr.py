"""Deformable DETR (SenseTime, ICLR 2021). Multi-scale deformable attention — 10x faster convergence than vanilla DETR and stronger on small objects. The bridge between DETR (2020) and RT-DETR (2024).

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not (the timm ResNet-50 backbone is built
randomly initialized — ``use_pretrained_backbone`` is False — so timm makes
no download either). The pretrained detector tensors
(``SenseTime/deformable-detr``, backbone included; the class head is sized
to ``output_classes`` and freshly initialized) are delivered from the
tracebloc model store as the training seed: upload the matched
``deformable_detr_weights.pkl`` sitting next to this file via
``weights=True``::

    user.upload_model("deformable_detr", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file.
"""
from transformers import DeformableDetrForObjectDetection, DeformableDetrConfig

framework = "pytorch"
model_type = "detr"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 800
batch_size = 4
output_classes = 12
category = "object_detection"

# Architecture config for SenseTime/deformable-detr (DeformableDetrConfig,
# model_type "deformable_detr"), inlined so the model builds with no config
# fetch. The nested timm backbone config is inlined in full so the backbone
# architecture cannot drift with library defaults. The SDK uploads the .py
# plus its named weight sibling — there is no config.json path — so the
# config lives here in the template.
CONFIG = {
    "backbone_config": {
        "model_type": "timm_backbone",
        "backbone": "resnet50",
        "features_only": True,
        "freeze_batch_norm_2d": False,
        "num_channels": 3,
        "out_features": None,
        "out_indices": [2, 3, 4],
        "output_stride": None,
        "use_pretrained_backbone": False,
    },
    "decoder_layerdrop": 0.0,
}


def MyModel(num_classes=output_classes):
    config = DeformableDetrConfig(**CONFIG, num_labels=num_classes)
    return DeformableDetrForObjectDetection(config)
