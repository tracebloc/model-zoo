"""YOLO11 (Ultralytics, Oct 2024). Current Ultralytics flagship — C3k2 + C2PSA blocks, ~22% fewer params than YOLOv8m at higher mAP. Federated-friendly: no foundation backbone, BN layers replaced with SyncBN at training time recommended."""
import torch.nn as nn
from ultralytics import YOLO

framework = "pytorch"
model_type = "yolo"
main_class = "MyModel"
license = "AGPL-3.0"
image_size = 640
batch_size = 16
output_classes = 80
category = "object_detection"


class MyModel(nn.Module):
    def __init__(self, num_classes=output_classes):
        super().__init__()
        self.model = YOLO("yolo11n.yaml").model
        # Rebuild detection head for custom class count
        self.model.nc = num_classes
        for m in self.model.modules():
            if hasattr(m, "nc"):
                m.nc = num_classes

    def forward(self, x):
        return self.model(x)
