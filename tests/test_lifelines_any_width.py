"""Every lifelines template fits a time-to-event table of any covariate width.

lifelines fits the DataFrame it is given and estimates one coefficient per
covariate column, so a lifelines template's ``num_feature_points`` is not an
input width (the category README's "Dataset expectations"). That only holds
while no lifelines template pins a width; one that did goes red here.

Runs where lifelines is installed (the survival CI job); skips elsewhere.
"""

import contextlib
import importlib.util
import io
import pathlib

import pytest

pytest.importorskip("lifelines")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_index = _load("build_index_for_lifelines_width", ROOT / "tools" / "build_index.py")

#: Every template the zoo index lists under the lifelines framework.
TEMPLATES = sorted(
    (path, consts)
    for path, consts in build_index.discover()
    if consts["framework"] == "lifelines"
)

#: Covariate counts the templates are fitted on. None is a template's declared
#: ``num_feature_points``, which ``test_no_width_is_the_declared_one`` holds.
WIDTHS = (1, 4, 23)

ROWS = 300


def _table(width: int) -> pd.DataFrame:
    """Weibull survival times with a known effect per covariate, about a quarter
    of them censored, in the ``time`` / ``event`` layout lifelines fits."""
    rng = np.random.default_rng(width)
    covariates = rng.normal(size=(ROWS, width))
    effects = rng.uniform(-0.5, 0.5, size=width)
    scale = np.exp(2.0 + covariates @ effects)
    time = scale * rng.weibull(1.5, size=ROWS)
    censor_at = np.quantile(time, 0.75)
    frame = pd.DataFrame(covariates, columns=[f"x{i}" for i in range(width)])
    frame["time"] = np.minimum(time, censor_at)
    frame["event"] = (time <= censor_at).astype(int)
    return frame


def _fitted_covariates(model) -> set[str]:
    """The covariate names a fitted lifelines model holds a coefficient for. An
    AFT model's index is ``(parameter, covariate)`` and carries an intercept."""
    return set(model.params_.index.get_level_values(-1)) - {"Intercept"}


def _id(entry) -> str:
    return entry[0].stem


def test_the_lifelines_templates_are_found():
    # An emptied discovery would pass the parametrised cases below vacuously.
    assert TEMPLATES, "no template under model_zoo/ declares framework = 'lifelines'"


@pytest.mark.parametrize("entry", TEMPLATES, ids=_id)
def test_no_width_is_the_declared_one(entry):
    _, consts = entry
    assert consts.get("num_feature_points") not in WIDTHS


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("entry", TEMPLATES, ids=_id)
def test_a_lifelines_template_fits_any_covariate_width(entry, width):
    path, _ = entry
    frame = _table(width)
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_index._load(path)()
        model.fit(frame, duration_col="time", event_col="event")
    covariates = {f"x{i}" for i in range(width)}
    assert _fitted_covariates(model) == covariates
    expected = model.predict_expectation(frame[sorted(covariates)])
    risk = np.asarray(expected, dtype=float).ravel()
    assert risk.shape[0] == ROWS and np.isfinite(risk).all()
