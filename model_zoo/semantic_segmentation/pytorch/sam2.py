"""SAM 2 (Meta, Aug 2024). Segment-anything foundation model — image + video, promptable. Used here as a frozen encoder with a trainable linear segmentation head so federated averaging only sees the head.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. Only the vision encoder is built (the
prompt encoder / mask decoder are unused here — the pre-migration template
already kept just ``.vision_encoder``, and the inlined vision sub-config was
verified equal to the composite checkpoint's ``vision_config``). The
pretrained encoder tensors are delivered from the tracebloc model store as
the training seed: upload the matched ``sam2_weights.pkl`` sitting next to
this file via ``weights=True``::

    user.upload_model("sam2", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file (vision-encoder weights of ``facebook/sam2-hiera-tiny``).
"""
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

framework = "pytorch"
main_class = "MyModel"
license = "Apache-2.0"
image_size = 1024
batch_size = 2
output_classes = 2
category = "semantic_segmentation"

# Architecture config for the vision encoder of facebook/sam2-hiera-tiny
# (Sam2VisionModel, model_type "sam2_vision_model") with its nested Hiera
# backbone sub-config, inlined so the model builds with no config fetch.
# The SDK uploads the .py plus its named weight sibling — there is no
# config.json path — so the config lives here in the template.
CONFIG = {
    "model_type": "sam2_vision_model",
    "backbone_config": {
        "model_type": "sam2_hiera_det_model",
        "hidden_size": 96,
        "num_attention_heads": 1,
        "num_channels": 3,
        "image_size": [1024, 1024],
        "patch_kernel_size": [7, 7],
        "patch_stride": [4, 4],
        "patch_padding": [3, 3],
        "mlp_ratio": 4.0,
        "blocks_per_stage": [1, 2, 7, 2],
        "embed_dim_per_stage": [96, 192, 384, 768],
        "num_attention_heads_per_stage": [1, 2, 4, 8],
        "window_size_per_stage": [8, 4, 14, 7],
        "window_positional_embedding_background_size": [7, 7],
        "global_attention_blocks": [5, 7, 9],
        "num_query_pool_stages": 3,
        "query_stride": [2, 2],
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-06,
        "initializer_range": 0.02,
    },
    "backbone_channel_list": [768, 384, 192, 96],
    "backbone_feature_sizes": [[256, 256], [128, 128], [64, 64]],
    "fpn_hidden_size": 256,
    "fpn_kernel_size": 1,
    "fpn_stride": 1,
    "fpn_padding": 0,
    "fpn_top_down_levels": [2, 3],
    "num_feature_levels": 3,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-06,
    "initializer_range": 0.02,
}


class MyModel(nn.Module):
    def __init__(self, num_classes=output_classes):
        super().__init__()
        config = AutoConfig.for_model(**CONFIG)
        self.backbone = AutoModel.from_config(config)
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Hiera image encoder outputs C=256 feature map at stride 16
        self.head = nn.Conv2d(256, num_classes, kernel_size=1)
        self._out_size = image_size

    def forward(self, x):
        # HF SAM2 vision_encoder returns last_hidden_state in channels-last
        # layout (B, H', W', C); Conv2d expects channels-first.
        feats = self.backbone(x).last_hidden_state.permute(0, 3, 1, 2).contiguous()
        logits = self.head(feats)
        return F.interpolate(logits, size=self._out_size, mode="bilinear", align_corners=False)
