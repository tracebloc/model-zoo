"""TimesFM 2.0 (Google, 2024-2025). Decoder-only time-series foundation model; Chronos's main competitor and the leading zero-shot forecaster on GIFT-Eval as of 2025. LoRA-only fine-tune so federated averaging only syncs the adapter. A thin wrapper exposes the zoo's (B, L, N) → (B, H, N) tensor contract by flattening multivariate input to per-channel univariate calls."""
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import TimesFmModelForPrediction

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


class _TimesFMWrapper(nn.Module):
    """Adapts univariate TimesFM to the zoo's multivariate tensor contract.

    Forward accepts `past_values` of shape (B, L, N) and returns predictions
    of shape (B, H, N). Each of the N channels is forecast independently —
    TimesFM is a univariate model, so multivariate input is processed
    channel-by-channel and the outputs are restacked.
    """

    def __init__(self, base, forecast_horizon):
        super().__init__()
        self.base = base
        self.h = forecast_horizon

    def forward(self, past_values, *args, **kwargs):
        b, L, n = past_values.shape
        # (B, L, N) → (B*N, L)
        flat = past_values.permute(0, 2, 1).reshape(b * n, L)
        # Standard differentiable forward through PEFT → TimesFmModelForPrediction.
        # `TimesFmOutputForPrediction.mean_predictions` has shape (B*N, H), so
        # LoRA adapters receive gradients in both training and inference.
        out = self.base(past_values=flat)
        pred = out.mean_predictions  # (B*N, H)
        # If the model emitted more horizon than requested, trim to self.h.
        if pred.shape[-1] != self.h:
            pred = pred[..., : self.h]
        pred = pred.reshape(b, n, self.h).permute(0, 2, 1)
        return pred


def MyModel(forecast_horizon=forecast_horizon):
    # `TimesFmModelForPrediction` exposes a standard differentiable forward
    # returning `mean_predictions`. The base `TimesFmModel` only returns
    # `last_hidden_state`, which would be hidden representations rather than
    # forecast values — silently corrupting fine-tuning gradients.
    base = TimesFmModelForPrediction.from_pretrained(_PRETRAINED_ID)
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    base = get_peft_model(base, lora_config)
    return _TimesFMWrapper(base, forecast_horizon)
