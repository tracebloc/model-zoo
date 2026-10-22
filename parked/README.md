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

No template is parked today. The last one, `random_forest_stability.py`, returned
to `model_zoo/tabular_classification/sklearn/` once its Bandit B110 handler was
gone and its fitted model pickled as a plain scikit-learn `Pipeline` (the
platform loads weights with a loader that admits scikit-learn and numpy classes
only).
