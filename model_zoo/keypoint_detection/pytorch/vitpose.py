"""ViTPose (NeurIPS 2022). First transformer pose model in the zoo — ViT backbone + simple decoder; 81+ AP COCO whole-body at Huge scale, fine-tunes well at Base.

Offline variant: the architecture is built from the inlined config below
(ViTPose backbone sub-config included) — no hub model id, no config fetch,
no download at build time, so the template constructs anywhere, network or
not. The pretrained tensors are delivered from the tracebloc model store as
the training seed: upload the matched ``vitpose_weights.pkl`` sitting next
to this file via ``weights=True``::

    user.upload_model("vitpose", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file (weights of ``usyd-community/vitpose-base-simple``; the
position embeddings are sized to the declared 256x256 input, freshly
initialized — same as before via the checkpoint's size-mismatch reinit).
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, VitPoseForPoseEstimation

framework = "pytorch"
model_type = "transformer"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 256
batch_size = 32
output_classes = 1
category = "keypoint_detection"
num_feature_points = 17

# Architecture config for usyd-community/vitpose-base-simple
# (VitPoseForPoseEstimation, model_type "vitpose") with its nested ViTPose
# backbone sub-config, inlined so the model builds with no config fetch.
# The SDK uploads the .py plus its named weight sibling — there is no
# config.json path — so the config lives here in the template. The
# checkpoint's native input grid is 256x192; ``MyModel`` overrides it to
# the declared square ``image_size`` below, exactly as before.
CONFIG = {
    "model_type": "vitpose",
    "backbone_config": {
        "model_type": "vitpose_backbone",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "mlp_ratio": 4,
        "num_channels": 3,
        "image_size": [256, 192],
        "patch_size": [16, 16],
        "num_experts": 1,
        "part_features": 0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "qkv_bias": True,
        "out_features": ["stage12"],
        "out_indices": [12],
        "stage_names": [
            "stem",
            "stage1",
            "stage2",
            "stage3",
            "stage4",
            "stage5",
            "stage6",
            "stage7",
            "stage8",
            "stage9",
            "stage10",
            "stage11",
            "stage12",
        ],
    },
    "initializer_range": 0.02,
    "scale_factor": 4,
    "use_simple_decoder": True,
    "edges": [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
        [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2],
        [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
    ],
}


class _VitPoseWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, *args, **kwargs):
        batch, _, in_h, in_w = pixel_values.shape
        heatmaps = self.model(pixel_values=pixel_values).heatmaps
        b, k, h, w = heatmaps.shape
        flat = heatmaps.view(b, k, -1)
        probs = torch.softmax(flat, dim=-1)
        # Soft-argmax in *pixel* space (same fix as vitpose_plus.py). The
        # platform's keypoint targets are pixel coordinates in the input
        # image; emitting normalized ``[0, 1]`` coords here would scale the
        # per-pixel MSE loss by ``image_size ** 2`` and explode gradients —
        # observed as loss ~1e11 with ``val_loss = NaN`` on the first cycle.
        ys = torch.linspace(0, in_h - 1, h, device=heatmaps.device).view(1, 1, h, 1)
        xs = torch.linspace(0, in_w - 1, w, device=heatmaps.device).view(1, 1, 1, w)
        probs2d = probs.view(b, k, h, w)
        x_coord = (probs2d * xs).sum(dim=(2, 3))
        y_coord = (probs2d * ys).sum(dim=(2, 3))
        # Confidence — squashed to ``[0, 1]`` so it can't dominate the loss
        # against per-keypoint visibility flags. Raw ``flat.max`` is an
        # unbounded transformer logit (easily 1e2+ at init).
        conf = torch.sigmoid(flat.max(dim=-1).values)
        return torch.stack([x_coord, y_coord, conf], dim=-1)


def MyModel(num_feature_points=num_feature_points):
    # VitPoseSimpleDecoder reads config.num_labels for its heatmap-output
    # channel count; passed through for_model so the SDK override lands.
    config = AutoConfig.for_model(**CONFIG, num_labels=num_feature_points)
    config.image_size = [image_size, image_size]
    if getattr(config, "backbone_config", None) is not None:
        config.backbone_config.image_size = [image_size, image_size]
    model = VitPoseForPoseEstimation(config)
    return _VitPoseWrapper(model)
