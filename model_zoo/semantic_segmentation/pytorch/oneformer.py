"""OneFormer (CVPR 2023). Single set of weights for semantic / instance / panoptic segmentation; consistently matches or beats Mask2Former with one checkpoint instead of three."""
from transformers import OneFormerForUniversalSegmentation, OneFormerConfig

framework = "pytorch"
main_method = "MyModel"
license = "MIT"
image_size = 512
batch_size = 4
output_classes = 2
category = "semantic_segmentation"

_PRETRAINED_ID = "shi-labs/oneformer_ade20k_swin_tiny"


def MyModel(num_classes=output_classes):
    config = OneFormerConfig.from_pretrained(_PRETRAINED_ID, num_labels=num_classes)
    return OneFormerForUniversalSegmentation.from_pretrained(
        _PRETRAINED_ID, config=config, ignore_mismatched_sizes=True
    )
