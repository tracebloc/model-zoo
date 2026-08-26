"""Faster R-CNN with a ResNet backbone. Two-stage detector; strong accuracy, slower than YOLO variants.

Offline variant: the architecture is built with ``weights=None`` — no hub
download at build time, so the template constructs anywhere, network or not.
The pretrained ResNet50-FPN tensors are delivered from the tracebloc model
store as the training seed: upload the matched ``faster_rcnn_resnet_weights.pkl``
sitting next to this file via ``upload_model(..., weights=True)``, and the
platform loads it with ``load_state_dict(strict=True)`` after ``MyModel()``
builds this architecture. See ``tools/prep_offline_weights.py`` for producing
and verifying that matched weight file.
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


framework = "pytorch"
model_type = "rcnn"
main_class = "MyModel"
image_size = 448
batch_size = 16
output_classes = 12
category = "object_detection"


def MyModel(num_classes=output_classes):
    num_classes = num_classes + 1  # 1 for background

    # Build the architecture only; pretrained tensors arrive as the
    # tracebloc-hosted seed weights.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None
    )

    # Replace the classifier head (identical to the pretrained-path build, so
    # the hosted seed state_dict keys/shapes match this module exactly).
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
