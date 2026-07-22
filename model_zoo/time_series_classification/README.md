# Time series classification

Assign one label to each whole sequence (e.g. classify a patient's vital-sign history, a machine's sensor trace, or an activity recording). One dataset item = one sequence; the label is constant within a sequence.

## Start here

**New to time-series classification?** Use [`pytorch/lstm.py`](pytorch/lstm.py). Solid general-purpose sequence classifier — start here before reaching for transformers.

For a fast sanity-check baseline, try [`pytorch/masked_pool_mlp.py`](pytorch/masked_pool_mlp.py) — a per-timestep MLP with masked pooling. If it matches your recurrent models, temporal order is not carrying much signal.

## Models

| Model | When to pick |
|---|---|
| [`lstm.py`](pytorch/lstm.py) | Default sequence classifier; reliable baseline |
| [`gru.py`](pytorch/gru.py) | Lighter than LSTM; faster training on short sequences |
| [`tcn.py`](pytorch/tcn.py) | Dilated causal convs; strong on long sequences |
| [`transformer.py`](pytorch/transformer.py) | Self-attention; needs a lot of data to justify |
| [`masked_pool_mlp.py`](pytorch/masked_pool_mlp.py) | Order-agnostic baseline; fastest to train |

## Dataset expectations

- **Input**: `(B, sequence_length, num_feature_points)` float32. Sequences are scaled first, then zero **post**-padded (or tail-keep truncated) to `sequence_length`.
- **Output**: `(B, output_classes)` raw logits — no softmax in the model.
- **Padding**: models take a single tensor and derive the padding mask internally via `(x.abs().sum(-1) > 0)`. A timestep whose features are all exactly zero counts as padding, so scale before padding — never the other way around.
- **Labels**: one integer class per sequence.

## Writing your own

Heads must respect the padding mask. In particular:

- Recurrent models must read the last **valid** timestep, not `out[:, -1, :]` — that reads hidden state computed on zero padding.
- Transformers should pass `src_key_padding_mask` and guard the all-padding edge case (a fully masked row makes attention output NaN).
- Pooling heads must exclude padded timesteps (masked mean/max), not average over the full padded length.
- Federated averaging constraint: no BatchNorm running stats, no EMA buffers — use LayerNorm instead.
