"""D-FINE (USTC, ICLR 2025). DETR variant with fine-grained distribution refinement on bbox regression; ~55 AP COCO at S scale while keeping RT-DETR-class latency.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained detector tensors
(``ustc-community/dfine-small-coco``; the class head is sized to
``output_classes`` and freshly initialized) are delivered from the tracebloc
model store as the training seed: upload the matched
``d_fine_weights.pkl`` sitting next to this file via ``weights=True``::

    user.upload_model("d_fine", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file.
"""
from transformers import DFineForObjectDetection, DFineConfig

framework = "pytorch"
model_type = "detr"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 640
batch_size = 4
output_classes = 12
category = "object_detection"

# Architecture config for ustc-community/dfine-small-coco (DFineConfig,
# model_type "d_fine"), inlined so the model builds with no config fetch.
# The nested HGNetV2 backbone config is inlined in full so the backbone
# architecture cannot drift with library defaults. The SDK uploads the .py
# plus its named weight sibling — there is no config.json path — so the
# config lives here in the template.
CONFIG = {
    "backbone": None,
    "backbone_config": {
        "model_type": "hgnet_v2",
        "depths": [3, 4, 6, 3],
        "downsample_in_bottleneck": False,
        "downsample_in_first_stage": False,
        "embedding_size": 32,
        "hidden_act": "relu",
        "hidden_sizes": [128, 256, 512, 1024],
        "initializer_range": 0.02,
        "layer_type": "basic",
        "num_channels": 3,
        "out_features": ["stage2", "stage3", "stage4"],
        "out_indices": [2, 3, 4],
        "stage_downsample": [False, True, True, True],
        "stage_downsample_strides": [2, 2, 2, 2],
        "stage_in_channels": [16, 64, 256, 512],
        "stage_kernel_size": [3, 3, 5, 5],
        "stage_light_block": [False, False, True, True],
        "stage_mid_channels": [16, 32, 64, 128],
        "stage_names": ["stem", "stage1", "stage2", "stage3", "stage4"],
        "stage_num_blocks": [1, 1, 2, 1],
        "stage_numb_of_layers": [3, 3, 3, 3],
        "stage_out_channels": [64, 256, 512, 1024],
        "stem_channels": [3, 16, 16],
        "stem_strides": [2, 1, 1, 2, 1],
        "use_learnable_affine_block": True,
    },
    "decoder_layers": 3,
    "decoder_n_points": [3, 6, 3],
    "depth_mult": 0.34,
    "encoder_in_channels": [256, 512, 1024],
    "hidden_expansion": 0.5,
}


def MyModel(num_classes=output_classes):
    config = DFineConfig(**CONFIG, num_labels=num_classes)
    return DFineForObjectDetection(config)
