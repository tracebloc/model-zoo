"""TimesFM 2.0 (Google, 2024-2025). Decoder-only time-series foundation model; Chronos's main competitor and the leading zero-shot forecaster on GIFT-Eval as of 2025. LoRA-only fine-tune so federated averaging only syncs the adapter."""
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

framework = "pytorch"
model_type = ""
main_method = "MyModel"
license = "Apache-2.0"
category = "time_series_forecasting"
batch_size = 16
num_feature_points = 1
sequence_length = 512
forecast_horizon = 128

_PRETRAINED_ID = "google/timesfm-2.0-500m-pytorch"


def MyModel():
    base = AutoModel.from_pretrained(_PRETRAINED_ID, trust_remote_code=True)
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(base, lora_config)
