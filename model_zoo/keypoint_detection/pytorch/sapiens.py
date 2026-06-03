"""Sapiens (Meta, ECCV 2024). Human-centric foundation model — pose, depth,
normals, segmentation in one family.

Backbone-loading note
---------------------
The official ``facebook/sapiens-pose-*`` Hub repos ship as TorchScript
artifacts (not loadable via ``AutoModel``), and the MAE-pretrained
``facebook/sapiens-pretrain-0.3b`` repo ships a ``config.json`` that
lacks a ``model_type`` key — so ``AutoModel.from_pretrained`` rejects
it during ``model_func_checks`` with:

    Unrecognized model in facebook/sapiens-pretrain-0.3b. Should have a
    `model_type` key in its config.json.

even with ``trust_remote_code=True`` (the repo doesn't ship a custom
modeling file either). Until Meta publishes an HF-AutoModel-loadable
sapiens checkpoint, we substitute the upstream ViT-MAE base backbone
sapiens is architecturally built on (``facebook/vit-mae-base``). The
ViT geometry matches; only the human-centric pretraining is lost.
Switch ``_BACKBONE_ID`` back to ``facebook/sapiens-pretrain-0.3b`` once
that repo's config carries ``model_type: vit_mae``.

Fine-tuned LoRA-only so federated averaging only syncs the adapter +
the final regressor.
"""
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

_BACKBONE_ID = "facebook/vit-mae-base"


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
    # ViT-MAE base ships at 224x224 by default; we declare 256 above to
    # match the rest of the keypoint model family. ``image_size`` is a
    # config-level override (the position embeddings get interpolated to
    # fit), and ``ignore_mismatched_sizes=True`` is required because the
    # checkpoint's position-embedding tensor stays at the 224-grid shape
    # — HF reinitializes those positions rather than raising. The patch
    # projection and attention weights still load.
    base = AutoModel.from_pretrained(
        _BACKBONE_ID, image_size=image_size, ignore_mismatched_sizes=True,
    )
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
        # ViT-MAE exposes attention as ``query`` / ``key`` / ``value`` /
        # ``output.dense`` (not the fused ``qkv`` block used by some ViT
        # variants), so the LoRA target list has to match that naming
        # for the adapter wrap to land.
        target_modules=["query", "value"],
    )
    base = get_peft_model(base, lora_config)
    return _SapiensWrapper(base, num_feature_points)
