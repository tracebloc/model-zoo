"""Mitra (Amazon, NeurIPS 2025). Tabular foundation model — pretrained on synthetic priors; competitive with TabPFN v2 / CatBoost. Repo id is hard-coded as a string literal so the model-upload security gate (TBT001) recognises it against its vetted-repos allowlist."""
from transformers import AutoModel

framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
category = "tabular_classification"
model_type = ""
batch_size = 256
num_feature_points = 50
output_classes = 2


def MyModel(num_classes=output_classes):
    return AutoModel.from_pretrained("autogluon/mitra-classifier", trust_remote_code=True)
