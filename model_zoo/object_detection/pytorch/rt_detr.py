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

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("model.decoder.class_embed.0.", "model.decoder.class_embed.1.", "model.decoder.class_embed.2.", "model.decoder.class_embed.3.", "model.decoder.class_embed.4.", "model.decoder.class_embed.5.", "model.denoising_class_embed.", "model.enc_score_head.")

framework = "pytorch"
model_type = "hf_transformer"
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
