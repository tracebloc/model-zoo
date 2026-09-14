"""Behavioural tests for the stability-selection tabular models.

Fast + deterministic (fixed seeds, reduced CV). Exercises the binary
classification path: is_classifier compliance, per-feature selection
frequency, held-out AUROC, predict/predict_proba, and the tiny-n guard
that lets the SDK's dummy-data validation fit succeed.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.base import is_classifier  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
SKL = ROOT / "model_zoo" / "tabular_classification" / "sklearn"
FILES = ["logistic_regression_stability.py", "random_forest_stability.py"]


def _load(fname):
    spec = importlib.util.spec_from_file_location(fname[:-3], SKL / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _data(seed=0, n_per=30, p=8, nan_frac=0.1):
    rng = np.random.default_rng(seed)
    n = 2 * n_per
    X = rng.normal(size=(n, p))
    y = np.array([1] * n_per + [0] * n_per)
    X[:n_per, 0] += 2.0   # informative
    X[:n_per, 1] -= 2.0   # informative
    X[rng.random((n, p)) < nan_frac] = np.nan  # missing values, like real proteomics
    idx = rng.permutation(n)
    return X[idx], y[idx]


@pytest.mark.parametrize("fname", FILES)
def test_is_classifier(fname):
    assert is_classifier(_load(fname).MyModel())


@pytest.mark.parametrize("fname", FILES)
def test_fit_reports_stability(fname):
    mod = _load(fname)
    base = mod.MyModel().base_estimator
    model = mod.StabilityFeatureSelector(
        base_estimator=base, n_splits=3, n_repeats=2, random_state=0
    )
    X, y = _data()
    model.fit(X, y)
    report = model.get_stability_report()

    assert len(model.selection_frequency_) == X.shape[1]
    assert len(report["features"]) == X.shape[1]
    assert model.fold_aurocs_ and all(0.0 <= a <= 1.0 for a in model.fold_aurocs_)

    freq = {f["feature_index"]: f["selection_frequency"] for f in report["features"]}
    # informative features are selected often; more often than the noisy tail on average
    assert freq[0] > 0.5 and freq[1] > 0.5
    noise = [freq[i] for i in range(2, X.shape[1])]
    assert min(freq[0], freq[1]) > sum(noise) / len(noise)

    assert model.predict(X).shape == (len(y),)
    assert model.predict_proba(X).shape == (len(y), 2)


@pytest.mark.parametrize("fname", FILES)
def test_tiny_n_does_not_raise(fname):
    # mirrors the SDK's dummy-data validation fit on a handful of rows
    mod = _load(fname)
    model = mod.MyModel()
    X = np.random.RandomState(0).normal(size=(4, 12))
    y = np.array([0, 1, 0, 1])
    model.fit(X, y)
    assert model.predict(X).shape == (4,)
