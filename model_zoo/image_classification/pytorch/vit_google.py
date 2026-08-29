"""ViT-B/16 via HuggingFace ViTForImageClassification, google/vit-base-patch16-224 weights. Pretrained classifier; fine-tune the head for your classes.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained encoder tensors are
delivered from the tracebloc model store as the training seed: upload the
matched ``vit_google_weights.pkl`` sitting next to this file via
``weights=True``::

    user.upload_model("vit_google", weights=True)

See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file (weights of ``google/vit-base-patch16-224``; the
classification head is sized to ``output_classes`` and freshly initialized).
"""
from torch import nn
from transformers import AutoConfig, AutoModelForImageClassification

# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
SEED_EXCLUDED_PREFIXES = ("vit.classifier.",)


framework = "pytorch"
main_class = "VisionTransformer"
image_size = 224
batch_size = 16
category = "image_classification"
output_classes = 2

# Architecture config for google/vit-base-patch16-224
# (ViTForImageClassification, model_type "vit"), inlined so the model builds
# with no config fetch. The SDK uploads the .py plus its named weight
# sibling — there is no config.json path — so the config lives here in the
# template.
CONFIG = {
    "model_type": "vit",
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "image_size": 224,
    "patch_size": 16,
    "num_channels": 3,
    "qkv_bias": True,
    "encoder_stride": 16,
}


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Build the classifier with the declared number of output labels
        config = AutoConfig.for_model(**CONFIG, num_labels=output_classes)
        self.vit = AutoModelForImageClassification.from_config(config)

    def forward(self, pixel_values):
        # The model will output a dictionary with various keys.
        outputs = self.vit(pixel_values=pixel_values)
        # The logits are now directly available from the output's 'logits' key.
        logits = outputs.logits
        return logits
