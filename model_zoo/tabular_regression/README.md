# Tabular regression

Predict continuous target values from structured, tabular features.

## Start here

**New to tabular regression?** Use [`sklearn/xgboost.py`](sklearn/xgboost.py). Gradient-boosted trees typically dominate on tabular data — start here before trying deep models.

For a fast, zero-tuning baseline, use [`sklearn/random_forest.py`](sklearn/random_forest.py) or [`sklearn/linear_regression.py`](sklearn/linear_regression.py).

## Models

### sklearn (usually the right choice for tabular)

| Model | When to pick |
|---|---|
| [`xgboost.py`](sklearn/xgboost.py) | Strongest default |
| [`lightgbm.py`](sklearn/lightgbm.py) | Fast training; near-XGBoost accuracy |
| [`random_forest.py`](sklearn/random_forest.py) | Robust out-of-the-box baseline |
| [`gradient_boosting.py`](sklearn/gradient_boosting.py) | Classical sklearn boosting |
| [`decision_tree.py`](sklearn/decision_tree.py) | Interpretable single tree; high variance |
| [`knn.py`](sklearn/knn.py) | Small datasets with meaningful distance |
| [`mlp.py`](sklearn/mlp.py) | Small feed-forward net inside sklearn |
| [`svm.py`](sklearn/svm.py) | Moderate data with kernel methods |
| [`linear_regression.py`](sklearn/linear_regression.py) | Always run this first as a floor |
| [`ridge.py`](sklearn/ridge.py) | L2 regularization; correlated features |
| [`lasso.py`](sklearn/lasso.py) | L1 regularization; feature selection |
| [`elastic_net.py`](sklearn/elastic_net.py) | L1 + L2; sparsity with stability |

### PyTorch

| Model | When to pick |
|---|---|
| [`fcn.py`](pytorch/fcn.py) | MLP baseline |
| [`cnn.py`](pytorch/cnn.py) | 1D CNN; ordered features |
| [`rnn.py`](pytorch/rnn.py) | Short-sequence RNN |

## Dataset expectations

- **Input**: tabular rows with `num_feature_points` numeric features.
- **Labels**: continuous target values.
- **Batch size**: default 4096 (PyTorch), 512 (sklearn).
