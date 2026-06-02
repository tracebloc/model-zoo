"""PatchTST (ICLR 2023). Patches + channel-independence — the architecture that broke 'transformers don't work for time series'. Standard reference baseline in every 2025 forecasting paper."""
from transformers import PatchTSTConfig, PatchTSTForPrediction

framework = "pytorch"
model_type = ""
main_method = "MyModel"
license = "Apache-2.0"
category = "time_series_forecasting"
batch_size = 64
num_feature_points = 9
sequence_length = 96
forecast_horizon = 24


def MyModel(num_feature_points=num_feature_points,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon):
    config = PatchTSTConfig(
        num_input_channels=num_feature_points,
        context_length=sequence_length,
        prediction_length=forecast_horizon,
        patch_length=16,
        patch_stride=8,
        d_model=128,
        num_attention_heads=4,
        num_hidden_layers=3,
        ffn_dim=256,
        dropout=0.2,
    )
    return PatchTSTForPrediction(config)
