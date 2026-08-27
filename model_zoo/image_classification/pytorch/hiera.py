"""Hiera (Meta, ICML 2023). Hierarchical ViT that strips per-stage tricks (relative position, conv stem, etc.) while matching Swin/MViTv2 accuracy. Simpler architecture, MAE-pretrained, available via transformers.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained encoder tensors are
delivered from the tracebloc model store as the training seed: upload the
matched ``hiera_weights.pkl`` sitting next to this file via ``weights=True``::

    user.upload_model("hiera", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file (weights of ``facebook/hiera-tiny-224-in1k-hf``; the
classification head is sized to ``output_classes`` and freshly initialized).
"""
import torch.nn as nn
from transformers import AutoConfig, AutoModelForImageClassification

framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
image_size = 224
batch_size = 32
output_classes = 2
category = "image_classification"

# Architecture config for facebook/hiera-tiny-224-in1k-hf
# (HieraForImageClassification, model_type "hiera"), inlined so the model
# builds with no config fetch. The SDK uploads the .py plus its named weight
# sibling — there is no config.json path — so the config lives here in the
# template.
CONFIG = {
    "model_type": "hiera",
    "embed_dim": 96,
    "image_size": [224, 224],
    "patch_size": [7, 7],
    "patch_stride": [4, 4],
    "patch_padding": [3, 3],
    "mlp_ratio": 4.0,
    "depths": [1, 2, 7, 2],
    "num_layers": 4,
    "num_heads": [1, 2, 4, 8],
    "use_separate_position_embedding": False,
    "embed_dim_multiplier": 2.0,
    "num_query_pool": 3,
    "query_stride": [2, 2],
    "masked_unit_size": [8, 8],
    "masked_unit_attention": [True, True, False, False],
    "drop_path_rate": 0.0,
    "num_channels": 3,
    "hidden_act": "gelu",
    "initializer_range": 0.02,
    "layer_norm_init": 1.0,
    "layer_norm_eps": 1e-06,
}


class Hiera(nn.Module):
    """Wraps the HF classifier so ``forward`` returns a plain logits tensor.

    ``HieraForImageClassification`` returns a
    ``HieraForImageClassificationOutput`` dataclass, which the training loop
    hands straight to the loss — ``cross_entropy`` then rejects it
    ("argument 'input' must be Tensor, not HieraForImageClassificationOutput").
    Every sibling HF template here (vit_google, aimv2, dinov3, siglip2) already
    unwraps to a tensor; this one did not.
    """

    def __init__(self, num_classes=output_classes):
        super().__init__()
        config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
        self.hiera = AutoModelForImageClassification.from_config(config)

    def forward(self, pixel_values):
        return self.hiera(pixel_values=pixel_values).logits


def MyModel(num_classes=output_classes):
    return Hiera(num_classes=num_classes)
