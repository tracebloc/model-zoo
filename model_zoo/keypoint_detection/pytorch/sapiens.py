"""Sapiens (Meta, ECCV 2024). Human-centric foundation model — pose, depth, normals, segmentation in one family. The official `facebook/sapiens-pose-*` Hub repos ship as TorchScript artifacts (not loadable via AutoModel), so this template loads the MAE-pretrained ViT backbone `facebook/sapiens-pretrain-0.3b` and attaches a fresh pose head. Fine-tuned LoRA-only so federated averaging only syncs the adapter + final regressor."""
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

framework = "pytorch"
model_type = "transformer"
main_method = "MyModel"
license = "CC-BY-NC-4.0"
image_size = 256
batch_size = 16
output_classes = 1
category = "keypoint_detection"
num_feature_points = 17

_BACKBONE_ID = "facebook/sapiens-pretrain-0.3b"


class _SapiensWrapper(nn.Module):
    def __init__(self, backbone, num_feature_points):
        super().__init__()
        self.backbone = backbone
        # PEFT-wrapped backbones expose the underlying config via `.config`
        hidden = backbone.config.hidden_size
        self.head = nn.Linear(hidden, num_feature_points * 3)
        self.num_feature_points = num_feature_points

    def forward(self, pixel_values, *args, **kwargs):
        out = self.backbone(pixel_values=pixel_values)
        feat = out.last_hidden_state.mean(dim=1)
        coords = self.head(feat).view(-1, self.num_feature_points, 3)
        return coords


def MyModel(num_feature_points=num_feature_points):
    base = AutoModel.from_pretrained(_BACKBONE_ID, trust_remote_code=True)
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
        target_modules=["qkv", "proj"],
    )
    base = get_peft_model(base, lora_config)
    return _SapiensWrapper(base, num_feature_points)
