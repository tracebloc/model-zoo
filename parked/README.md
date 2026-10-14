# Parked templates

A template here is **not offered**: it is outside `model_zoo/`, so it has no row in
[`model_zoo/index.v1.json`](../model_zoo/index.v1.json), and none of the zoo's
template walkers (the index build, the contract tests, the upload-scan test) see it.
It is kept, unchanged, because it comes back once the reason it was parked is gone.

A template is parked when the platform's upload scan refuses it and the fix is a
platform change, not a template change. Each one is listed in `PARKED` in
[`tests/test_upload_import_allowlist.py`](../tests/test_upload_import_allowlist.py)
with the imports the scan refuses and the issue that owns its return. That test goes
red when a parked template would pass the scan, so un-parking cannot be forgotten.

**To un-park:** `git mv` the file back to the same path under `model_zoo/`, delete
its `PARKED` row, move its README section back, and run `make index`.

| Template | Refused for | Returns with |
|---|---|---|
| [`tabular_classification/sklearn/logistic_regression_stability.py`](tabular_classification/sklearn/logistic_regression_stability.py) | `numpy` is not on the upload scan's import allowlist | model-zoo#143 |
| [`tabular_classification/sklearn/random_forest_stability.py`](tabular_classification/sklearn/random_forest_stability.py) | same | model-zoo#143 |

When the two stability templates come back, the `try/except/pass` in each file's
fold-AUROC loop goes too (Bandit B110 refuses it).

## Tabular classification: feature stability selection

The section below is the template documentation, kept for when they return.

For biomarker-panel / feature-selection studies: these models run repeated stratified
cross-validation **inside** `fit` and report how often each feature is selected, plus
held-out AUROC per fold.

| Model | When to pick |
|---|---|
| [`logistic_regression_stability.py`](tabular_classification/sklearn/logistic_regression_stability.py) | LASSO (L1) selection frequency + signed direction of effect |
| [`random_forest_stability.py`](tabular_classification/sklearn/random_forest_stability.py) | Random-forest importance-based selection; consensus comparator to LASSO |

From the platform's point of view both are ordinary sklearn classifiers
(`fit` / `predict` / `predict_proba`), so once
the upload scan admits them they train through the normal tracebloc flow. Internally `fit` runs
`n_splits` × `n_repeats` CV (default 10 × 10 = 100 models), building a fresh
`impute → scale → estimator` pipeline per fold so per-fold statistics never leak
across the split. Class imbalance is handled with `class_weight='balanced'`.

Run the experiment with **1 epoch, 1 cycle** — the resampling happens inside a single
`fit`; extra epochs just repeat the whole CV. Defaults assume binary classification
(`output_classes = 2`) with ~12 features (`num_feature_points = 12`); edit those
module-level values to match your dataset before uploading.

### Reading stability-selection results

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
