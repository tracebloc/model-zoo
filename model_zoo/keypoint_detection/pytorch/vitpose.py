"""ViTPose (NeurIPS 2022). First transformer pose model in the zoo — ViT backbone + simple decoder; 81+ AP COCO whole-body at Huge scale, fine-tunes well at Base."""
import torch
import torch.nn as nn
from transformers import VitPoseForPoseEstimation, VitPoseConfig

framework = "pytorch"
model_type = "transformer"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 256
batch_size = 32
output_classes = 1
category = "keypoint_detection"
num_feature_points = 17

_PRETRAINED_ID = "usyd-community/vitpose-base-simple"


class _VitPoseWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, *args, **kwargs):
        heatmaps = self.model(pixel_values=pixel_values).heatmaps
        b, k, h, w = heatmaps.shape
        flat = heatmaps.view(b, k, -1)
        probs = torch.softmax(flat, dim=-1)
        ys = torch.linspace(0, 1, h, device=heatmaps.device).view(1, 1, h, 1)
        xs = torch.linspace(0, 1, w, device=heatmaps.device).view(1, 1, 1, w)
        probs2d = probs.view(b, k, h, w)
        x_coord = (probs2d * xs).sum(dim=(2, 3))
        y_coord = (probs2d * ys).sum(dim=(2, 3))
        conf = flat.max(dim=-1).values
        return torch.stack([x_coord, y_coord, conf], dim=-1)


def MyModel(num_feature_points=num_feature_points):
    config = VitPoseConfig.from_pretrained(
        _PRETRAINED_ID,
        num_labels=num_feature_points,
        image_size=[image_size, image_size],
    )
    # VitPoseSimpleDecoder reads config.num_labels for its heatmap-output
    # channel count; set it explicitly so the SDK override always lands.
    config.num_labels = num_feature_points
    if getattr(config, "backbone_config", None) is not None:
        config.backbone_config.image_size = [image_size, image_size]
    model = VitPoseForPoseEstimation.from_pretrained(
        _PRETRAINED_ID, config=config, ignore_mismatched_sizes=True
    )
    return _VitPoseWrapper(model)
