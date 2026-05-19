"""Chronos-Bolt (Amazon, 2024). T5-based time-series foundation model — Apache-2.0, fine-tunes via standard seq2seq. Loaded directly through transformers to avoid pulling in chronos-forecasting."""
from transformers import AutoModelForSeq2SeqLM

model_id = "amazon/chronos-bolt-base"
framework = "pytorch"
main_method = "MyModel"
license = "Apache-2.0"
category = "time_series_forecasting"
model_type = ""
batch_size = 32
num_feature_points = 1
sequence_length = 512
forecast_horizon = 64


def MyModel():
    return AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True)
