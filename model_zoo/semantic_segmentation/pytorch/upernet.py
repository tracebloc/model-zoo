"""UPerNet (ECCV 2018). PPM + FPN decoder paired here with a ConvNeXt backbone. The canonical decoder used in modern segmentation papers when reporting ConvNeXt / Swin results — strong, well-understood baseline."""
from transformers import UperNetForSemanticSegmentation, UperNetConfig

framework = "pytorch"
main_method = "MyModel"
license = "MIT"
image_size = 512
batch_size = 4
output_classes = 2
category = "semantic_segmentation"

_PRETRAINED_ID = "openmmlab/upernet-convnext-small"


def MyModel(num_classes=output_classes):
    config = UperNetConfig.from_pretrained(_PRETRAINED_ID, num_labels=num_classes)
    return UperNetForSemanticSegmentation.from_pretrained(
        _PRETRAINED_ID, config=config, ignore_mismatched_sizes=True
    )
