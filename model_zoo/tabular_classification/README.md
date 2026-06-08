# Tabular classification

Classify structured, tabular records into discrete categories.

## Start here

**New to tabular classification?** Use [`sklearn/xgboost_classifier.py`](sklearn/xgboost_classifier.py). On small and medium tabular datasets, gradient-boosted trees typically beat deep models — start here before trying PyTorch or TensorFlow variants.

## Models

### sklearn (usually the right choice for tabular)

| Model | When to pick |
|---|---|
| [`xgboost_classifier.py`](sklearn/xgboost_classifier.py) | Strong default; usually wins on small/medium data |
| [`lightgbm_classifier.py`](sklearn/lightgbm_classifier.py) | Faster training than XGBoost, often near-tied accuracy |
| [`catboost_classifier.py`](sklearn/catboost_classifier.py) | Best handling of categorical features with minimal preprocessing |

### sklearn — feature stability selection

For biomarker-panel / feature-selection studies: these models run repeated stratified
cross-validation **inside** `fit` and report how often each feature is selected, plus
held-out AUROC per fold.

| Model | When to pick |
|---|---|
| [`logistic_regression_stability.py`](sklearn/logistic_regression_stability.py) | LASSO (L1) selection frequency + signed direction of effect |
| [`random_forest_stability.py`](sklearn/random_forest_stability.py) | Random-forest importance-based selection; consensus comparator to LASSO |

From the platform's point of view both are ordinary sklearn classifiers
(`fit` / `predict` / `predict_proba`), so they upload and train through the normal
tracebloc flow with **no special configuration**. Internally `fit` runs
`n_splits` × `n_repeats` CV (default 10 × 10 = 100 models), building a fresh
`impute → scale → estimator` pipeline per fold so per-fold statistics never leak
across the split. Class imbalance is handled with `class_weight='balanced'`.

Run the experiment with **1 epoch, 1 cycle** — the resampling happens inside a single
`fit`; extra epochs just repeat the whole CV. Defaults assume binary classification
(`output_classes = 2`) with ~12 features (`num_feature_points = 12`); edit those
module-level values to match your dataset before uploading.

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

## Reading stability-selection results

The two `*_stability.py` models save the fitted estimator as the experiment's weights
artifact, carrying the selection results as attributes. Download the artifact and call
`get_stability_report()`. Because the artifact stores a custom class, point the loader at
the model file so the class resolves (the shim below works regardless of the artifact's
internal module name):

```python
import pickle, importlib.util

# 1) make the model class importable
spec = importlib.util.spec_from_file_location("stab", "logistic_regression_stability.py")
stab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stab)

class _Loader(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "StabilityFeatureSelector":
            return stab.StabilityFeatureSelector
        return super().find_class(module, name)

# 2) load the downloaded weights artifact
with open("trained_weights.pkl", "rb") as f:
    model = _Loader(f).load()

report = model.get_stability_report()
#   report["features"]   per-feature: selection_frequency, mean_effect, direction (sorted)
#   report["fold_aurocs"]  held-out AUROC for every fold
#   report["summary"]      auroc_mean / std / min / max / n_folds
```

For the LASSO model, `features[i]["direction"]` is the sign of the mean L1 coefficient
(direction of effect). For the random-forest model, `mean_effect` is the mean importance
magnitude and direction is not meaningful — use the LASSO model for direction.
