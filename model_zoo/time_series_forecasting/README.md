# Time series forecasting

Predict future values of a sequence from its past values. All models expect sequential input and output a `forecast_horizon` of future predictions.

## Start here

**New to time-series forecasting?** Use [`pytorch/lstm.py`](pytorch/lstm.py). Solid general-purpose sequence model — start here before reaching for transformers.

For long-horizon forecasting, try [`pytorch/tcn.py`](pytorch/tcn.py) (Temporal Convolutional Network) — dilated causal convolutions often beat RNNs on long sequences.

## Models

| Model | When to pick |
|---|---|
| [`lstm.py`](pytorch/lstm.py) | Default sequence model; reliable baseline |
| [`bidirectional_lstm.py`](pytorch/bidirectional_lstm.py) | Reads each sequence both directions; future context during training |
| [`gru.py`](pytorch/gru.py) | Lighter than LSTM; faster training on short sequences |
| [`rnn.py`](pytorch/rnn.py) | Vanilla RNN; weak for long sequences |
| [`tcn.py`](pytorch/tcn.py) | Dilated causal convs; strong for long-horizon forecasting |
| [`transformer.py`](pytorch/transformer.py) | Self-attention; needs a lot of data to justify |

## Dataset expectations

- **Input**: `(B, sequence_length, num_features)`.
- **Output**: `(B, forecast_horizon, num_features)` (or similar, per model).
- **`sequence_length`**, **`forecast_horizon`**: declared per model file, override for your use case.
