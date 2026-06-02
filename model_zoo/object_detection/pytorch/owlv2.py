"""OWLv2 (Google, NeurIPS 2023). Open-vocabulary detector — like Grounding DINO but built on OWL-ViT lineage with self-training on web-scale pseudo-labels. Complements grounding_dino.py; useful when class vocabulary is fluid or zero-shot."""
from transformers import Owlv2ForObjectDetection

framework = "pytorch"
model_type = "detr"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 960
batch_size = 4
output_classes = 12
category = "object_detection"

_PRETRAINED_ID = "google/owlv2-base-patch16-ensemble"


def MyModel(num_classes=output_classes):
    # OWLv2 is open-vocabulary: classes are passed as text queries at inference
    # time, not as a fixed integer head size. num_classes is accepted for SDK
    # signature uniformity but intentionally not wired into the model.
    del num_classes
    return Owlv2ForObjectDetection.from_pretrained(_PRETRAINED_ID)
