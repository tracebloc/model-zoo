# Tabular classification

Classify structured, tabular records into discrete categories.

## Start here

**New to tabular classification?** Use [`sklearn/xgboost_classifier.py`](sklearn/xgboost_classifier.py). On small and medium tabular datasets, gradient-boosted trees typically beat deep models — start here before trying PyTorch variants.

## Models

### sklearn (usually the right choice for tabular)

| Model | When to pick |
|---|---|
| [`xgboost_classifier.py`](sklearn/xgboost_classifier.py) | Strong default; usually wins on small/medium data |
| [`lightgbm_classifier.py`](sklearn/lightgbm_classifier.py) | Faster training than XGBoost, often near-tied accuracy |
| [`catboost_classifier.py`](sklearn/catboost_classifier.py) | Best handling of categorical features with minimal preprocessing |

### sklearn: biomarker discovery with stability selection

| Model | When to pick |
|---|---|
| [`logistic_regression_stability.py`](sklearn/logistic_regression_stability.py) | Many candidate features, few rows: which features truly predict the label, and which only look predictive by chance |

`fit` fits an L1-penalised logistic regression on 1,000 random halves of the
training rows, at each of 10 penalties. A feature chosen in at least 75% of the
fits at some penalty is stable, and one logistic regression refit on the stable
set makes the predictions. The report gives each feature's stability, the stable
set with odds ratios per standard deviation, and the Meinshausen-Buehlmann bound
on the expected number of false selections.

Run it with **1 epoch and 1 cycle**: the resampling happens inside one `fit`. It
runs on one CPU core; on 400 rows and 910 features it takes about 20 seconds.

The weights artifact is a plain scikit-learn `Pipeline` (scaler, then logistic
regression with zero weights off the stable set), so your run's artifact loads
without the model file:

```python
import pickle

with open("trained_weights.pkl", "rb") as f:
    model = pickle.load(f)
report = model.stability_report_
report["stable_set"]   # feature, index, stability, coefficient and odds ratio per SD
report["stability"]    # every feature, highest stability first, with its share per penalty
report["expected_false_selections_bound"]
```

On the platform the model sees the features as numbers, so the report names them
`x0`, `x1`, ... in the dataset's column order, without the label.

### PyTorch (pick when features have sequential/spatial structure)

| Model | When to pick |
|---|---|
| [`fcn.py`](pytorch/fcn.py) | MLP baseline; default deep-learning choice |
| [`cnn.py`](pytorch/cnn.py) | 1D CNN; use when feature order matters |
| [`lstm.py`](pytorch/lstm.py) | LSTM; rows as sequences |
| [`rnn.py`](pytorch/rnn.py) | Simple RNN; short sequences |

## Dataset expectations

- **Input**: tabular rows with `num_feature_points` numeric features.
- **Labels**: integer class IDs.
- **Batch size**: default 4096 (PyTorch), 512 (sklearn).
