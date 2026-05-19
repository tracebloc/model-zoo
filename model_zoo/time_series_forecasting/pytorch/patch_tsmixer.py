"""PatchTSMixer (IBM, KDD 2023). MLP-Mixer architecture for time series — patches + channel mixing without attention. Competitive with PatchTST at a fraction of the compute; the strong "do you even need a transformer" baseline."""
from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction

framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
category = "time_series_forecasting"
batch_size = 64
num_feature_points = 9
sequence_length = 96
forecast_horizon = 24


def MyModel():
    config = PatchTSMixerConfig(
        num_input_channels=num_feature_points,
        context_length=sequence_length,
        prediction_length=forecast_horizon,
        patch_length=8,
        patch_stride=8,
        d_model=64,
        expansion_factor=2,
        num_layers=3,
        dropout=0.2,
    )
    return PatchTSMixerForPrediction(config)
