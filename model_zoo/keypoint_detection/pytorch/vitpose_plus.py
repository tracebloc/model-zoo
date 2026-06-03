"""ViTPose++ (NeurIPS 2023). MoE-style task-shared / task-specific experts on top of ViTPose; current top of the COCO whole-body leaderboard at Huge scale. Base variant is the practical fine-tune default."""
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

_PRETRAINED_ID = "usyd-community/vitpose-plus-base"


class _Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # ViTPose++ ships a Mixture-of-Experts head with one expert per
        # training dataset (COCO / AIC / MPII / …).
        # ``VitPoseForPoseEstimation.forward`` therefore requires a
        # ``dataset_index`` kwarg on every call to pick which expert
        # routes the features — without it, ``model_func_checks`` rejects
        # the file with:
        #
        #     dataset_index must be provided when using multiple experts
        #     (num_experts=6). Please provide dataset_index to the
        #     forward pass.
        #
        # We always select expert 0 because the platform's keypoint
        # pipeline is a single-dataset fine-tune — there's no per-batch
        # routing decision to make.
        self._expert_index = 0

    def forward(self, pixel_values, *args, **kwargs):
        batch, _, in_h, in_w = pixel_values.shape
        dataset_index = torch.full(
            (batch,),
            self._expert_index,
            dtype=torch.long,
            device=pixel_values.device,
        )
        heatmaps = self.model(
            pixel_values=pixel_values, dataset_index=dataset_index
        ).heatmaps
        b, k, h, w = heatmaps.shape
        flat = heatmaps.view(b, k, -1)
        probs = torch.softmax(flat, dim=-1)
        # Soft-argmax in *pixel* space. The platform's keypoint targets
        # are pixel coordinates in the input image; emitting normalized
        # ``[0, 1]`` coords here would scale the per-pixel MSE loss by
        # ``image_size ** 2`` and explode gradients — we observed loss
        # ~1e11 with ``val_loss = NaN`` on the first cycle before this
        # change.
        ys = torch.linspace(0, in_h - 1, h, device=heatmaps.device).view(1, 1, h, 1)
        xs = torch.linspace(0, in_w - 1, w, device=heatmaps.device).view(1, 1, 1, w)
        probs2d = probs.view(b, k, h, w)
        x_coord = (probs2d * xs).sum(dim=(2, 3))
        y_coord = (probs2d * ys).sum(dim=(2, 3))
        # Confidence — squashed to ``[0, 1]`` so it can't dominate the
        # loss against per-keypoint visibility flags. Raw ``flat.max``
        # is an unbounded transformer logit (easily 1e2+ at init).
        conf = torch.sigmoid(flat.max(dim=-1).values)
        return torch.stack([x_coord, y_coord, conf], dim=-1)


def MyModel(num_feature_points=num_feature_points):
    config = VitPoseConfig.from_pretrained(
        _PRETRAINED_ID, num_labels=num_feature_points,
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
    return _Wrapper(model)
