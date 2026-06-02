"""Mask2Former (Meta, CVPR 2022). Dominant universal segmentation architecture — same model handles semantic / instance / panoptic via masked attention; ~57 mIoU ADE20K at Swin-L."""
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig

framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 384
batch_size = 4
output_classes = 2
category = "semantic_segmentation"

_PRETRAINED_ID = "facebook/mask2former-swin-tiny-ade-semantic"


def MyModel(num_classes=output_classes):
    config = Mask2FormerConfig.from_pretrained(_PRETRAINED_ID, num_labels=num_classes)
    return Mask2FormerForUniversalSegmentation.from_pretrained(
        _PRETRAINED_ID, config=config, ignore_mismatched_sizes=True
    )
