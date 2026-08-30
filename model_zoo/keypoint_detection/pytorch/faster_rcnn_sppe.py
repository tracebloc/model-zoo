"""Single-Person Pose Estimator on a Faster R-CNN ResNet-50 backbone. Reuses a strong detection backbone for keypoints.

Offline variant: the architecture is built without any checkpoint download,
so the template constructs anywhere, network or not. The pretrained backbone
tensors are delivered from the tracebloc model store as the training seed:
upload the matched ``faster_rcnn_sppe_weights.pkl`` sitting next to this
file via ``upload_model(..., weights=True)``, and the platform loads it with
``load_state_dict(strict=True)`` after the model builds. See
``tools/prep_offline_weights.py`` for producing and verifying that matched
weight file.

The ResNet50-FPN backbone is assembled explicitly instead of via
``fasterrcnn_resnet50_fpn(weights=None)``, for the reasons documented in
``object_detection/pytorch/faster_rcnn_resnet.py``: with no weights
requested that builder swaps the backbone norm layers from
``FrozenBatchNorm2d`` to trainable ``BatchNorm2d`` (which changes the
state_dict key set) and unfreezes all five backbone stages instead of the
last three. Building the backbone directly reproduces the checkpoint-path
backbone exactly — same norm layers, same three trainable stages, same
state_dict keys and shapes. Verified against torchvision 0.27.
"""
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops import misc as misc_nn_ops

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("fc.",)


# Configuration
framework = "pytorch"
model_type = ""
main_class = "FasterRCNNSPPE"
image_size = 64
batch_size = 128
output_classes = 1
category = "keypoint_detection"
num_feature_points = 16

class FasterRCNNSPPE(nn.Module):
    def __init__(self, num_feature_points=num_feature_points):
        super(FasterRCNNSPPE, self).__init__()
        self.num_feature_points = num_feature_points

        # Build the Faster R-CNN ResNet50-FPN backbone directly, with no
        # download — checkpoint-path architecture: frozen batch-norm
        # backbone, FPN with the last 3 stages trainable.
        resnet = resnet50(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        backbone = _resnet_fpn_extractor(resnet, trainable_layers=3)

        # The checkpoint path zeroes FrozenBatchNorm2d eps across the whole
        # detector, backbone included (torchvision's overwrite_eps); match
        # it so numerics are identical, not just the parameter set.
        for module in backbone.modules():
            if isinstance(module, misc_nn_ops.FrozenBatchNorm2d):
                module.eps = 0.0

        # Assume the feature extractor provides a feature map, which is what we use here
        self.feature_extractor = backbone

        # Create a pooling layer compatible with the Faster R-CNN backbone output
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Adjust the feature map size to the input of the final fully connected layer
        num_features = (
            256  # Feature size; confirm actual dimensions from the backbone output
        )
        self.fc = nn.Linear(num_features, num_feature_points * 3)

    def forward(self, x):
        # Ensure x is a tensor and process through the backbone
        features = self.feature_extractor(x)

        # Depending on the backbone structure, you may need to specify the output layer or pick one feature
        if isinstance(features, dict):
            # Pick a particular layer output (e.g., '0') based on your feature extractor
            x = features["0"]  # Replace with the appropriate key

        # Apply adaptive average pooling to match expected fully connected layer input size
        x = self.global_avg_pool(x)

        # Flatten pooled output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Pass through the fully connected layer for keypoint prediction
        x = self.fc(x)

        # Reshape to (batch_size, num_feature_points, 3)
        x = x.view(-1, self.num_feature_points, 3)
        return x
