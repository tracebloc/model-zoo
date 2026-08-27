"""DETR (Meta, ECCV 2020). The original transformer detector — set-prediction with Hungarian matching. Historical reference; pairs with RT-DETR (efficient) and Grounding DINO (open-vocabulary).

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not (the timm ResNet-50 backbone is built
randomly initialized — ``use_pretrained_backbone`` is False — so timm makes
no download either). The pretrained detector tensors
(``facebook/detr-resnet-50``, backbone included; the class head is sized to
``output_classes`` and freshly initialized) are delivered from the tracebloc
model store as the training seed: upload the matched ``detr_weights.pkl``
sitting next to this file via ``weights=True``::

    user.upload_model("detr", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file.
"""
from transformers import DetrForObjectDetection, DetrConfig

framework = "pytorch"
model_type = "detr"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 800
batch_size = 4
output_classes = 12
category = "object_detection"

# Architecture config for facebook/detr-resnet-50 (DetrConfig, model_type
# "detr"), inlined so the model builds with no config fetch. The nested timm
# backbone config is inlined in full so the backbone architecture cannot
# drift with library defaults. The SDK uploads the .py plus its named weight
# sibling — there is no config.json path — so the config lives here in the
# template.
CONFIG = {
    "backbone_config": {
        "model_type": "timm_backbone",
        "backbone": "resnet50",
        "features_only": True,
        "freeze_batch_norm_2d": False,
        "num_channels": 3,
        "out_features": None,
        "out_indices": [1, 2, 3, 4],
        "output_stride": None,
        "use_pretrained_backbone": False,
    },
    "classifier_dropout": 0.0,
    "max_position_embeddings": 1024,
    "scale_embedding": False,
}


def MyModel(num_classes=output_classes):
    config = DetrConfig(**CONFIG, num_labels=num_classes)
    return DetrForObjectDetection(config)
