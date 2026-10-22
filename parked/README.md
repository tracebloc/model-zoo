# Parked templates

A template here is **not offered**: it is outside `model_zoo/`, so it has no row in
[`model_zoo/index.v1.json`](../model_zoo/index.v1.json), and none of the zoo's
template walkers (the index build, the contract tests, the upload-scan test) see it.
It is kept, unchanged, because it comes back once the reason it was parked is gone.

A template is parked when the platform's upload scan refuses it. Each one is listed
in `PARKED` in [`tests/test_upload_import_allowlist.py`](../tests/test_upload_import_allowlist.py)
with what the scan refuses: imports by top-level name, Bandit findings by test id.
That test goes red when a parked template would pass the scan, so un-parking cannot
be forgotten.

**To un-park:** `git mv` the file back to the same path under `model_zoo/`, delete
its `PARKED` row, move its README section back, and run `make index`.

| Template | Refused for | Returns with |
|---|---|---|
| [`tabular_classification/sklearn/random_forest_stability.py`](tabular_classification/sklearn/random_forest_stability.py) | Bandit B110: the `try/except/pass` in its fold-AUROC loop | A rewrite in the shape of `logistic_regression_stability.py` |

Beyond the scan, its fitted model pickles as its own class, and the platform loads
weights with a loader that admits scikit-learn and numpy classes only. The logistic
template pickles as a plain scikit-learn `Pipeline` for that reason.

## Tabular classification: random-forest stability selection

The section below is the template documentation, kept for when it returns.

| Model | When to pick |
|---|---|
| [`random_forest_stability.py`](tabular_classification/sklearn/random_forest_stability.py) | Random-forest importance-based selection; consensus comparator to the logistic template |

Internally `fit` runs `n_splits` × `n_repeats` stratified CV (default 10 × 10 = 100
models), building a fresh `impute → scale → estimator` pipeline per fold so per-fold
statistics never leak across the split, and records how often each feature's
importance clears the threshold, plus held-out AUROC per fold. Class imbalance is
handled with `class_weight='balanced'`. Run the experiment with **1 epoch, 1 cycle**.

`get_stability_report()` returns `features` (per feature: `selection_frequency`,
`mean_effect`, sorted), `fold_aurocs` and a `summary` of them. `mean_effect` is the
mean importance; direction is not meaningful for a forest.
