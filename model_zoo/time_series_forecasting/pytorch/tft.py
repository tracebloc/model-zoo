"""Temporal Fusion Transformer (Google, 2019). De-facto standard for interpretable multi-horizon forecasting with exogenous covariates — variable-selection networks, gated residual blocks, attention over time."""
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

framework = "pytorch"
model_type = ""
main_method = "MyModel"
license = "MIT"
category = "time_series_forecasting"
batch_size = 64
num_feature_points = 9
sequence_length = 96
forecast_horizon = 24


def MyModel():
    return TemporalFusionTransformer(
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        loss=QuantileLoss(),
        output_size=7,
    )
