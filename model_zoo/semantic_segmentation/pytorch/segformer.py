"""SegFormer (NVIDIA, NeurIPS 2021). Efficient transformer segmenter — hierarchical MiT encoder + lightweight MLP decoder; ~5x fewer params than prior SOTA at similar mIoU. Pairs with Mask2Former: SegFormer for speed, Mask2Former for accuracy."""
from transformers import SegformerForSemanticSegmentation, SegformerConfig

framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 512
batch_size = 8
output_classes = 2
category = "semantic_segmentation"

_PRETRAINED_ID = "nvidia/segformer-b0-finetuned-ade-512-512"


def MyModel(num_classes=output_classes):
    config = SegformerConfig.from_pretrained(_PRETRAINED_ID, num_labels=num_classes)
    return SegformerForSemanticSegmentation.from_pretrained(
        _PRETRAINED_ID, config=config, ignore_mismatched_sizes=True
    )
