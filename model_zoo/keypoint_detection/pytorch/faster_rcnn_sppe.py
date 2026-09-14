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
``fasterrcnn_resnet50_fpn(weights=None)``, because that builder keys its
architecture off whether weights were requested: with no weights it unfreezes
all five backbone stages instead of the last three, and it picks the backbone
norm off the same flag. Building the backbone directly keeps the three
trainable stages under explicit control. The norm is no longer the
checkpoint's — see below. Verified against torchvision 0.27.

Backbone norm: GroupNorm, and it is NOT the checkpoint's norm
----------------------------------------------------------------------------
This template used to build ``norm_layer=misc_nn_ops.FrozenBatchNorm2d`` to
reproduce torchvision's checkpoint-path backbone key-exactly. That was wrong
in the regime the platform actually runs: frozen BN at construction holds
``weight=1``, ``bias=0``, ``running_mean=0``, ``running_var=1``, so on a
``weights=None`` build it computes ``(x - 0) / sqrt(1 + eps) * 1 + 0`` -- a
bit-exact identity. Its buffers are meaningful only after a checkpoint loads
real statistics into them, so every from-scratch run of this template to date
trained with no backbone normalisation at all. Measured downstream on the OD
roster, which shared this exact line: activations reach sigma ~= 24 against
~= 3 with a live norm, and ``centernet_resnet`` diverged 1032.6 -> 3.19e+29
with grad norm going to inf, where GroupNorm on the same script and seeds gave
165.9 -> 15.87 -> 14.61.

GroupNorm normalises per sample, so it is correct with no checkpoint and adds
no running statistics for the averaging service to ship each federated round
-- both halves of the constraint that produced frozen BN in the first place.
This is the same conversion model-zoo#262 applied to twelve OD templates; the
two keypoint templates carrying the identical line were outside that PR's
directory scan.

⚠️ WHAT THIS COSTS, EXPLICITLY, AND IT COSTS MORE HERE THAN ON OD. A
torchvision COCO checkpoint's BN running statistics have nowhere to go in a
GroupNorm tree, so this template can no longer strict-load a seed prepped from
``download.pytorch.org`` weights. Unlike the OD twelve -- where no seed was
ever staged (an internal ticket is blocked on the store decision in the hosting decision) --
``faster_rcnn_sppe_weights.pkl`` IS a live entry in the backend dump manifest,
so this change invalidates a dump that exists and must be re-prepped.
``tools/prep_offline_weights.py`` fails loudly on the strict load rather than
producing a mismatched dump, which is the right place for it to fail. The
re-prep itself lives in ``backend``, not here. The trade taken is a real
defect on every from-scratch run today against a seed path whose store is
still undecided.

Parameter arithmetic, for any published count: frozen BN holds weight/bias as
BUFFERS and GroupNorm holds them as PARAMETERS, so this is +53,120 parameters
/ -106,240 buffers for the ResNet-50 trunk. It is NOT parameter-neutral.
"""
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor

# (internal ref) — the task head is NOT carried by the hosted seed.
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

def _group_norm(channels):
    """GroupNorm with the largest group count ``<= 32`` that divides ``channels``.

    The backbone norm for a from-scratch build — see the module
    docstring for why frozen BN was wrong here.

    The group count is derived, not hardcoded to 32: ``nn.GroupNorm`` requires
    ``channels % num_groups == 0``, which ResNet-50's 64..2048 all satisfy at
    32 (the canonical Wu & He setting), but hardcoding it would break the
    moment this template is retargeted at a backbone with stages that do not
    divide (MobileNetV3's 16/24/40/72/120/184 do not).

    Duplicated per template on purpose -- a zoo template is uploaded as ONE
    file and cannot import a sibling (no relative imports anywhere in this
    repo). ``keypoint_rcnn._group_norm`` and the OD family's ``_group_norm`` /
    ``_norm`` / ``_norm_groups`` are the same helper for the same reason.
    """
    groups = max(g for g in range(1, 33) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


class FasterRCNNSPPE(nn.Module):
    def __init__(self, num_feature_points=num_feature_points):
        super(FasterRCNNSPPE, self).__init__()
        self.num_feature_points = num_feature_points

        # Build the Faster R-CNN ResNet50-FPN backbone directly, with no
        # download: GroupNorm backbone (an internal ticket -- frozen BN is a
        # bit-exact identity from scratch), FPN with the last 3 stages
        # trainable.
        #
        # There is no `overwrite_eps` loop any more: it existed to zero
        # FrozenBatchNorm2d's eps so numerics matched the checkpoint path, and
        # with no FrozenBatchNorm2d in the tree it would iterate every module
        # and match none. Dead code that reads as coverage is worse than none.
        resnet = resnet50(weights=None, norm_layer=_group_norm)
        backbone = _resnet_fpn_extractor(resnet, trainable_layers=3)

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
