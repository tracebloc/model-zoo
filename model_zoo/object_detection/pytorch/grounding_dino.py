"""Grounding DINO (IDEA, ECCV 2024). Open-vocabulary object detection — detects classes given as text rather than fixed integer labels. 52.5 AP zero-shot COCO; fine-tunes well on private class names."""
from transformers import AutoModelForZeroShotObjectDetection

framework = "pytorch"
model_type = "detr"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 800
batch_size = 4
output_classes = 12
category = "object_detection"

_PRETRAINED_ID = "IDEA-Research/grounding-dino-tiny"


def MyModel(num_classes=output_classes):
    # Grounding DINO is open-vocabulary: classes are passed as text queries at
    # inference time, not as a fixed integer head size. num_classes is accepted
    # for SDK signature uniformity but intentionally not wired into the model.
    del num_classes
    return AutoModelForZeroShotObjectDetection.from_pretrained(_PRETRAINED_ID)
