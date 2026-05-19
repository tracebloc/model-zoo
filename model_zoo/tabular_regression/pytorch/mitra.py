"""Mitra regressor (Amazon, NeurIPS 2025). Tabular foundation model for regression — pretrained on synthetic priors. Repo id is hard-coded as a string literal so the model-upload security gate (TBT001) recognises it against its vetted-repos allowlist."""
from transformers import AutoModel

framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
category = "tabular_regression"
model_type = ""
batch_size = 256
num_feature_points = 50
output_classes = 1


def MyModel():
    return AutoModel.from_pretrained("autogluon/mitra-regressor", trust_remote_code=True)
