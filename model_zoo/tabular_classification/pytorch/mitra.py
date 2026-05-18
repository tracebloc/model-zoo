"""Mitra (Amazon, NeurIPS 2025). Tabular foundation model — pretrained on synthetic priors; competitive with TabPFN v2 / CatBoost. Loaded via trust_remote_code so the HF repo supplies the architecture."""
from transformers import AutoModel

model_id = "autogluon/mitra-classifier"
framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
category = "tabular_classification"
model_type = ""
batch_size = 256
num_feature_points = 50
output_classes = 2


def MyModel(num_classes=output_classes):
    return AutoModel.from_pretrained(model_id, trust_remote_code=True)
