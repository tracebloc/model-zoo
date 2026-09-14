"""DeepLab semantic segmentation. Atrous convolutions give multi-scale context without losing resolution.

Builds RANDOM-INIT: both `weights` and `weights_backbone` are None, so nothing is
fetched at construction (internal ref). Accuracy comes from training or from a seed loaded
after construction -- this file promises the architecture, not a starting quality.
"""
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

# Configuration
framework = "pytorch"
main_class = "DeepLabV3"
image_size = 256
batch_size = 8
output_classes = 2
category = "semantic_segmentation"


class DeepLabV3(nn.Module):
    def __init__(self, backbone="resnet50"):
        super(DeepLabV3, self).__init__()

        # weights=None: no segmentation-head download. weights_backbone=None is the
        # load-bearing one: these builders default it to ResNet50/101_Weights.IMAGENET1K_V1,
        # so `pretrained=False` alone leaves the BACKBONE downloading through
        # load_state_dict_from_url — which also unpickles what it fetches, and which
        # the egress lockdown blocks at the edge. `pretrained` is not even a parameter of
        # these builders any more; the legacy shim maps it to
        # `weights` only and never touches `weights_backbone`. Same habit as
        # object_detection/pytorch/faster_rcnn_resnet_v2.py.
        #
        # The two branches stay `if/elif` rather than collapsing into a
        # `BUILDERS[backbone](...)` dict lookup, and that is deliberate. The guard that
        # keeps this file honest -- tests/test_zoo_backbone_weights_explicit.py -- matches
        # calls by CALLEE IDENTIFIER, so a subscript callee has no name for it to match
        # and this template, the one the guard exists for, would silently drop out of the
        # check. Two call sites the guard can see beat one tidier call site it cannot.
        if backbone == "resnet50":
            self.model = deeplabv3_resnet50(
                weights=None,
                weights_backbone=None,
                num_classes=output_classes,
            )
        elif backbone == "resnet101":
            self.model = deeplabv3_resnet101(
                weights=None,
                weights_backbone=None,
                num_classes=output_classes,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.model(x)["out"]
