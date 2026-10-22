"""Behavioural tests for the stability-selection tabular templates.

The logistic template is checked here on a small planted problem (seeded, a few
seconds), and on the platform's first-run example data in the CI job that checks
out the ingestor (the ``ingest_producer`` marker, deselected everywhere else).
The random-forest template is parked (``parked/README.md``) and keeps the
behaviour tests it was parked with.
"""
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import pathlib
import pickle
import sys
import time

import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.base import clone, is_classifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
LOGISTIC = ROOT / "model_zoo" / "tabular_classification" / "sklearn" / "logistic_regression_stability.py"
PARKED_SKL = ROOT / "parked" / "tabular_classification" / "sklearn"


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the logistic template on a planted problem ---------------------------

PLANTED = (0, 3, 7)


def _planted(seed=0, n=240, p=200):
    """Three features drive the outcome; the other ``p - 3`` are noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * rng.uniform(0.5, 20.0, size=p) + rng.normal(size=p)
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    eta = 1.6 * z[:, 0] - 1.4 * z[:, 3] + 1.2 * z[:, 7]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    return X, y


@pytest.fixture(scope="module")
def planted_fit():
    mod = _load(LOGISTIC)
    X, y = _planted()
    model = mod.StabilitySelectionClassifier(n_subsamples=60).fit(X, y)
    return mod, X, y, model


def test_is_classifier():
    assert is_classifier(_load(LOGISTIC).MyModel())


def test_the_planted_features_rank_first_and_are_stable(planted_fit):
    _, _, _, model = planted_fit
    report = model.get_stability_report()
    assert {row["index"] for row in report["stability"][: len(PLANTED)]} == set(PLANTED)
    stable = {row["index"] for row in report["stable_set"]}
    assert set(PLANTED) <= stable
    assert {row["feature"] for row in report["stable_set"]} == {f"x{i}" for i in stable}


def test_the_report_covers_every_feature_and_derives_the_stable_set(planted_fit):
    _, X, _, model = planted_fit
    report = model.get_stability_report()
    rows = report["stability"]
    assert sorted(row["index"] for row in rows) == list(range(X.shape[1]))
    assert [row["stability"] for row in rows] == sorted(
        (row["stability"] for row in rows), reverse=True
    )
    penalties = len(report["method"]["penalty_path_c"])
    for row in rows:
        assert len(row["share_by_penalty"]) == penalties
        assert row["stability"] == max(row["share_by_penalty"])
    threshold = report["method"]["threshold"]
    assert {row["index"] for row in report["stable_set"]} == {
        row["index"] for row in rows if row["stability"] >= threshold
    }
    q, p = report["mean_selected_q"], report["method"]["features"]
    assert report["expected_false_selections_bound"] == pytest.approx(
        q**2 / ((2 * threshold - 1) * p)
    )
    for row in report["stable_set"]:
        assert row["odds_ratio_per_sd"] == pytest.approx(math.exp(row["coefficient_per_sd"]))


def test_predictions_are_the_refit_on_the_stable_set(planted_fit):
    _, X, y, model = planted_fit
    stable = sorted(row["index"] for row in model.get_stability_report()["stable_set"])
    scaler = StandardScaler().fit(X)
    refit = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(X)[:, stable], y)
    X_new = _planted(seed=1)[0]
    expected = refit.predict_proba(scaler.transform(X_new)[:, stable])
    np.testing.assert_allclose(model.predict_proba(X_new), expected, rtol=1e-6, atol=1e-9)
    coef = model.model_.named_steps["model"].coef_
    assert not np.any(np.delete(coef, stable, axis=1))
    assert model.predict(X_new).shape == (len(X_new),)


def test_a_dataframe_names_the_features():
    pd = pytest.importorskip("pandas")
    X, y = _planted()
    frame = pd.DataFrame(X, columns=[f"marker_{i}" for i in range(X.shape[1])])
    model = _load(LOGISTIC).StabilitySelectionClassifier(n_subsamples=20).fit(frame, y)
    names = {row["feature"] for row in model.get_stability_report()["stability"]}
    assert names == set(frame.columns)


def test_nothing_stable_predicts_the_base_rate():
    X, y = _planted()
    model = _load(LOGISTIC).StabilitySelectionClassifier(n_subsamples=0).fit(X, y)
    report = model.get_stability_report()
    assert report["stable_set"] == []
    assert not np.any(model.model_.named_steps["model"].coef_)
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], y.mean(), atol=0.01)


def test_one_class_is_refused():
    X, _ = _planted()
    with pytest.raises(ValueError, match="at least two classes"):
        _load(LOGISTIC).MyModel().fit(X, np.zeros(len(X), dtype=int))


def test_tiny_n_does_not_raise():
    # mirrors the SDK's dummy-data validation fit on a handful of rows
    model = _load(LOGISTIC).StabilitySelectionClassifier(n_subsamples=20)
    X = np.random.RandomState(0).normal(size=(4, 12))
    y = np.array([0, 1, 0, 1])
    model.fit(X, y)
    assert model.predict_proba(X).shape == (4, 2)


def _globals_in(blob):
    """Every (module, name) the pickle resolves, read by loading it."""
    seen = set()

    class Recorder(pickle.Unpickler):
        def find_class(self, module, name):
            seen.add((module, name))
            return super().find_class(module, name)

    return Recorder(io.BytesIO(blob)).load(), seen


def test_the_fitted_model_pickles_as_a_plain_sklearn_pipeline(planted_fit):
    """The platform's weight loader admits scikit-learn and numpy classes, never
    a class from the uploaded file, so the artifact must carry none."""
    mod, X, _, model = planted_fit
    loaded, seen = _globals_in(pickle.dumps(model))
    assert type(loaded) is Pipeline
    assert not any(module == mod.__name__ for module, _ in seen), seen
    assert {module.split(".")[0] for module, _ in seen} <= {"sklearn", "numpy"}, seen
    assert loaded.stability_report_ == model.get_stability_report()
    np.testing.assert_array_equal(loaded.predict_proba(X), model.predict_proba(X))


def test_an_unfitted_model_keeps_its_class(monkeypatch):
    # The SDK pickles the model before training, with the file imported as a module.
    mod = _load(LOGISTIC)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    model = mod.MyModel()
    assert type(pickle.loads(pickle.dumps(model))) is type(model)
    assert clone(model).get_params() == model.get_params()


# --- the logistic template on the first-run example data -------------------

#: Env var naming a checkout of the ingestor; set only by the CI job that
#: selects the marked tests below.
PRODUCER_CHECKOUT_ENV = "INGEST_PRODUCER_CHECKOUT"
BUNDLE = pathlib.Path("templates") / "examples" / "biomarker_discovery"
LABEL, ID_COLUMN = "recurrence_5y", "patient_id"

#: Calibrated from this template's defaults on the bundle: 7 of its 8 planted
#: CpGs, no false CpG, test AUC 0.909, about 20 s on one core. Seeds 1-10 of
#: the bundle's reference analysis spread 5-8 recovered and 0-2 false.
MIN_PLANTED_RECOVERED = 7
MAX_FALSE_CPGS = 1
AUC_BAND = (0.85, 0.95)
MAX_SECONDS = 300


def _bundle():
    root = os.environ.get(PRODUCER_CHECKOUT_ENV)
    assert root, f"{PRODUCER_CHECKOUT_ENV} is unset: these tests need the ingestor's checkout"
    bundle = pathlib.Path(root) / BUNDLE
    assert bundle.is_dir(), f"no example bundle at {bundle}"
    return bundle


def _verified(bundle, name):
    """The file's bytes, held to the bundle's own SHA256SUMS."""
    sums = {}
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        if line and not line.startswith("#"):
            digest, file = line.split()
            sums[file] = digest
    data = (bundle / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == sums[name], f"{name} disagrees with SHA256SUMS"
    return data


def _table(bundle, name, drop):
    rows = list(csv.reader(io.StringIO(_verified(bundle, name).decode())))
    header = rows[0]
    features = [c for c in header if c not in (ID_COLUMN, LABEL, *drop)]
    index = [header.index(c) for c in features]
    X = np.array([[float(r[i]) for i in index] for r in rows[1:]])
    y = np.array([int(r[header.index(LABEL)]) for r in rows[1:]])
    return features, X, y


def _acceptance(bundle, drop=()):
    """Fit the template's defaults on the bundle's train table as the engine feeds
    it (numbers in column order), score the test table once, and return the
    failed criteria with the report."""
    features, X, y = _table(bundle, "train.csv", drop)
    _, X_test, y_test = _table(bundle, "test.csv", drop)
    truth = json.loads(_verified(bundle, "ground_truth.json"))
    planted = {row["column"] for row in truth["true_cpgs"]}

    started = time.perf_counter()
    model = _load(LOGISTIC).MyModel().fit(X, y)
    seconds = time.perf_counter() - started
    report = model.get_stability_report()
    stable = {features[row["index"]] for row in report["stable_set"]}
    cpgs = {c for c in stable if c.startswith("cpg_")}
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    failures = []
    if len(cpgs & planted) < MIN_PLANTED_RECOVERED:
        failures.append(
            f"planted CpGs recovered: {len(cpgs & planted)}, expected at least {MIN_PLANTED_RECOVERED}"
        )
    if len(cpgs - planted) > MAX_FALSE_CPGS:
        failures.append(f"false CpGs: {len(cpgs - planted)}, expected at most {MAX_FALSE_CPGS}")
    if not AUC_BAND[0] <= auc <= AUC_BAND[1]:
        failures.append(f"test AUC {auc:.3f} outside {AUC_BAND}")
    if seconds > MAX_SECONDS:
        failures.append(f"fit took {seconds:.0f} s, more than {MAX_SECONDS} s")
    return failures, report, features


@pytest.fixture(scope="module")
def bundle_run():
    bundle = _bundle()
    return bundle, _acceptance(bundle)


@pytest.mark.ingest_producer
def test_the_example_bundle_meets_the_calibrated_result(bundle_run):
    _, (failures, report, features) = bundle_run
    assert features and report["method"]["features"] == len(features)
    assert failures == []


@pytest.mark.ingest_producer
def test_dropping_a_recovered_planted_cpg_fails_the_check(bundle_run):
    bundle, (_, report, features) = bundle_run
    truth = json.loads(_verified(bundle, "ground_truth.json"))
    planted = {row["column"] for row in truth["true_cpgs"]}
    strongest = next(
        features[row["index"]] for row in report["stable_set"] if features[row["index"]] in planted
    )
    failures, mutated, kept = _acceptance(bundle, drop=(strongest,))
    assert strongest in features and strongest not in kept
    assert mutated["method"]["features"] == len(features) - 1
    assert failures[0] == (
        f"planted CpGs recovered: {MIN_PLANTED_RECOVERED - 1}, "
        f"expected at least {MIN_PLANTED_RECOVERED}"
    )


# --- the parked random-forest template -------------------------------------

FOREST = PARKED_SKL / "random_forest_stability.py"


def _forest_data(seed=0, n_per=30, p=8, nan_frac=0.1):
    rng = np.random.default_rng(seed)
    n = 2 * n_per
    X = rng.normal(size=(n, p))
    y = np.array([1] * n_per + [0] * n_per)
    X[:n_per, 0] += 2.0   # informative
    X[:n_per, 1] -= 2.0   # informative
    X[rng.random((n, p)) < nan_frac] = np.nan  # missing values, like real proteomics
    idx = rng.permutation(n)
    return X[idx], y[idx]


def test_forest_is_classifier():
    assert is_classifier(_load(FOREST).MyModel())


def test_forest_fit_reports_stability():
    mod = _load(FOREST)
    base = mod.MyModel().base_estimator
    model = mod.StabilityFeatureSelector(
        base_estimator=base, n_splits=3, n_repeats=2, random_state=0
    )
    X, y = _forest_data()
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


def test_forest_tiny_n_does_not_raise():
    # mirrors the SDK's dummy-data validation fit on a handful of rows
    model = _load(FOREST).MyModel()
    X = np.random.RandomState(0).normal(size=(4, 12))
    y = np.array([0, 1, 0, 1])
    model.fit(X, y)
    assert model.predict(X).shape == (4,)
