"""DPT — Dense Prediction Transformer (Intel ISL, ICCV 2021). ViT backbone with a dense prediction head; primarily known for depth but the semantic-segmentation variant is a strong reference baseline on ADE20K."""
from transformers import DPTForSemanticSegmentation, DPTConfig

framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 480
batch_size = 4
output_classes = 2
category = "semantic_segmentation"

_PRETRAINED_ID = "Intel/dpt-large-ade"


def MyModel(num_classes=output_classes):
    config = DPTConfig.from_pretrained(_PRETRAINED_ID, num_labels=num_classes)
    return DPTForSemanticSegmentation.from_pretrained(
        _PRETRAINED_ID, config=config, ignore_mismatched_sizes=True
    )
