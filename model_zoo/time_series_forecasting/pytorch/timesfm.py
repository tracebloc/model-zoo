"""TimesFM 2.0 (Google, 2024-2025). Decoder-only time-series foundation model; Chronos's main competitor and the leading zero-shot forecaster on GIFT-Eval as of 2025. LoRA-only fine-tune so federated averaging only syncs the adapter. A thin wrapper exposes the zoo's (B, L, N) → (B, H, N) tensor contract by flattening multivariate input to per-channel univariate calls."""
import torch
import torch.nn as nn
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

        # Training path: use the module's standard differentiable `__call__`
        # (which goes through PEFT → base.forward) so LoRA adapters receive
        # gradients during federated fine-tuning. `forecast` is wrapped in
        # `torch.no_grad()` internally on the official TimesFM impl, so
        # using it for training would silently skip autograd.
        if self.training:
            out = self.base(flat)
            pred = out.predictions if hasattr(out, "predictions") else (
                out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            )
            # Take the last `h` steps along the time axis.
            if pred.ndim == 3:
                pred = pred[..., -self.h:, 0] if pred.shape[-1] == 1 else pred.mean(-1)[..., -self.h:]
            elif pred.ndim == 2:
                pred = pred[..., -self.h:]
            pred = pred.reshape(b, n, self.h).permute(0, 2, 1)
            return pred

        # Inference path: use TimesFM's optimized `forecast` (no_grad).
        inner = self.base
        for attr in ("base_model", "model"):
            if hasattr(inner, "forecast"):
                break
            if hasattr(inner, attr):
                inner = getattr(inner, attr)
        if not hasattr(inner, "forecast"):
            raise AttributeError(
                "Wrapped TimesFM model does not expose a `forecast` method; "
                "the zoo's wrapper relies on TimesFM's native forecast API."
            )
        pred = inner.forecast(flat, horizon=self.h)
        pred = torch.as_tensor(pred).reshape(b, n, self.h).permute(0, 2, 1)
        return pred


def MyModel(forecast_horizon=forecast_horizon):
    base = AutoModel.from_pretrained(_PRETRAINED_ID, trust_remote_code=True)
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    base = get_peft_model(base, lora_config)
    return _TimesFMWrapper(base, forecast_horizon)
