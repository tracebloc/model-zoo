"""Mask2Former (Meta, CVPR 2022). Dominant universal segmentation architecture —
same model handles semantic / instance / panoptic via masked attention; ~57 mIoU
ADE20K at Swin-L.

This wrapper exposes the HF `Mask2FormerForUniversalSegmentation` as a plain
semantic-segmentation module: `forward(x)` returns `[B, num_classes, H, W]`
logits, matching the contract used by the other models in this folder
(deeplab.py, fcn.py, ...). The platform trainer then applies its standard
per-pixel CrossEntropyLoss against the `[B, H, W]` mask targets.

We do NOT use Mask2Former's internal MaskFormerLoss here because the platform
trainer does not pass `mask_labels` / `class_labels`; instead we derive
semantic logits from the (mask_queries, class_queries) outputs via the standard
collapse:

    seg_logits = einsum("bqc,bqhw->bchw",
                        softmax(class_logits)[..., :-1],
                        sigmoid(mask_logits))

then upsample to the input resolution. This is the same formula HF's
`image_processor.post_process_semantic_segmentation` uses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

# Module-level metadata contract
framework = "pytorch"
main_class = "Mask2Former"
license = "Apache-2.0"
image_size = 384
batch_size = 4
output_classes = 2
category = "semantic_segmentation"

_PRETRAINED_ID = "facebook/mask2former-swin-tiny-ade-semantic"


class Mask2Former(nn.Module):
    def __init__(self, num_classes: int = output_classes, img_size: int = image_size):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        config = Mask2FormerConfig.from_pretrained(_PRETRAINED_ID, num_labels=num_classes)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            _PRETRAINED_ID, config=config, ignore_mismatched_sizes=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We bypass the HF loss path by only passing pixel_values; this returns
        # masks_queries_logits [B, Q, H', W'] and class_queries_logits
        # [B, Q, num_classes + 1] (last channel is the "no object" class).
        outputs = self.model(pixel_values=x)

        mask_logits = outputs.masks_queries_logits          # [B, Q, H', W']
        class_logits = outputs.class_queries_logits          # [B, Q, C+1]

        # Drop the "no object" class and collapse queries into per-class
        # pixel logits.
        class_probs = class_logits.softmax(dim=-1)[..., :-1]  # [B, Q, C]
        mask_probs = mask_logits.sigmoid()                    # [B, Q, H', W']

        seg_logits = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

        # Upsample to the input spatial size so the per-pixel CE loss against
        # [B, H, W] targets lines up.
        seg_logits = F.interpolate(
            seg_logits,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return seg_logits
