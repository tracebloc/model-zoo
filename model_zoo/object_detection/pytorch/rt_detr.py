"""RT-DETR (Baidu, CVPR 2024). First real-time DETR; Apache-2.0; ~53 AP COCO. Fills the entire transformer-detector gap left by the YOLO + Faster R-CNN lineup.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained detector tensors
(``PekingU/rtdetr_r50vd``; the class head is sized to ``output_classes`` and
freshly initialized) are delivered from the tracebloc model store as the
training seed: upload the matched ``rt_detr_weights.pkl`` sitting next to
this file via ``weights=True``::

    user.upload_model("rt_detr", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file.
"""
from transformers import RTDetrForObjectDetection, RTDetrConfig

framework = "pytorch"
model_type = "detr"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 640
batch_size = 4
output_classes = 12
category = "object_detection"

# Architecture config for PekingU/rtdetr_r50vd (RTDetrConfig, model_type
# "rt_detr"), inlined so the model builds with no config fetch. The nested
# RTDetrResNet backbone config is inlined in full so the backbone
# architecture cannot drift with library defaults. The SDK uploads the .py
# plus its named weight sibling — there is no config.json path — so the
# config lives here in the template.
CONFIG = {
    "backbone": None,
    "backbone_config": {
        "model_type": "rt_detr_resnet",
        "depths": [3, 4, 6, 3],
        "downsample_in_bottleneck": False,
        "downsample_in_first_stage": False,
        "embedding_size": 64,
        "hidden_act": "relu",
        "hidden_sizes": [256, 512, 1024, 2048],
        "layer_type": "bottleneck",
        "num_channels": 3,
        "out_features": ["stage2", "stage3", "stage4"],
        "out_indices": [2, 3, 4],
        "stage_names": ["stem", "stage1", "stage2", "stage3", "stage4"],
    },
}


def MyModel(num_classes=output_classes):
    config = RTDetrConfig(**CONFIG, num_labels=num_classes)
    return RTDetrForObjectDetection(config)
