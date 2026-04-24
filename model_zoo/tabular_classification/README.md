# Tabular classification

Classify structured, tabular records into discrete categories.

## Start here

**New to tabular classification?** Use [`sklearn/xgboost.py`](sklearn/xgboost.py). On small and medium tabular datasets, gradient-boosted trees typically beat deep models — start here before trying PyTorch or TensorFlow variants.

## Models

### sklearn (usually the right choice for tabular)

| Model | When to pick |
|---|---|
| [`xgboost.py`](sklearn/xgboost.py) | Strong default; usually wins on small/medium data |
| [`lightgbm.py`](sklearn/lightgbm.py) | Faster training than XGBoost, often near-tied accuracy |
| [`catboost.py`](sklearn/catboost.py) | Best handling of categorical features with minimal preprocessing |

### PyTorch (pick when features have sequential/spatial structure)

| Model | When to pick |
|---|---|
| [`fcn.py`](pytorch/fcn.py) | MLP baseline; default deep-learning choice |
| [`cnn.py`](pytorch/cnn.py) | 1D CNN; use when feature order matters |
| [`lstm.py`](pytorch/lstm.py) | LSTM; rows as sequences |
| [`rnn.py`](pytorch/rnn.py) | Simple RNN; short sequences |

### TensorFlow

Same architectures as PyTorch, under [`tensorflow/`](tensorflow/). Pick the framework your team prefers.

## Dataset expectations

- **Input**: tabular rows with `num_feature_points` numeric features.
- **Labels**: integer class IDs.
- **Batch size**: default 4096 (PyTorch/TF), 512 (sklearn).
