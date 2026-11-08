"""The onboarding rows of ``model_zoo/index.v1.json``: the ready-made models a first
run trains, read from the index and held to ``onboarding_findings`` in
tools/build_index.py (exactly three, one task, three families, each CPU-capable).

The behaviour tests run on every pull request. The ``test_MUTATION_*`` cases are
mutation rows (``.github/mutation-coverage.yml``): each starts from the committed
index, asserts it passes, applies one change, asserts the change applied and
asserts the exact finding. They need nothing but the standard library.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "build_index", ROOT / "tools" / "build_index.py"
)
build_index = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_index)

#: Rows of the synthetic table each onboarding template is trained on: the size
#: of a first run's training table.
FIT_ROWS = 4000


def committed() -> dict:
    return json.loads(build_index.INDEX_PATH.read_text())


def picked(index: dict) -> list[dict]:
    return [row for row in index["templates"] if row["onboarding"]]


def _passes(index: dict) -> dict:
    assert build_index.onboarding_findings(index) == [], "the committed index fails"
    return index


# --- behaviour ---------------------------------------------------------------


def test_the_committed_index_holds_a_first_runs_models() -> None:
    index = committed()
    assert build_index.onboarding_findings(index) == []
    declared = {
        build_index.static_row(path, consts)["id"]
        for path, consts in build_index.discover()
        if consts.get("onboarding") is True
    }
    assert declared == {row["id"] for row in picked(index)}


def test_a_declared_mark_must_be_a_bool(tmp_path, monkeypatch) -> None:
    path = tmp_path / "tabular_classification" / "sklearn" / "probe.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        'framework = "sklearn"\ncategory = "tabular_classification"\n'
        'batch_size = 4\nonboarding = "yes"\n'
    )
    monkeypatch.setattr(build_index, "MODEL_ROOT", tmp_path)
    with pytest.raises(ValueError, match="probe.py: `onboarding` must be True or False"):
        build_index.static_row(path, build_index._module_constants(path))


@pytest.mark.parametrize(
    ("framework", "params", "cpu"),
    [
        ("sklearn", None, True),
        ("lifelines", None, True),
        ("scikit_survival", None, True),
        ("pytorch", 1, True),
        ("pytorch", build_index.CPU_PARAMS_LIMIT, True),
        ("pytorch", build_index.CPU_PARAMS_LIMIT + 1, False),
        ("pytorch", None, False),
        ("pytorch", 0, False),
        ("pytorch", True, False),
    ],
)
def test_cpu_capable(framework, params, cpu) -> None:
    assert build_index.cpu_capable({"framework": framework, "params": params}) is cpu


def test_every_family_estimator_resolves() -> None:
    """A misspelt entry would leave its estimator in no family, and an onboarding
    template using it could not be indexed."""
    missing = []
    for family, entries in build_index.ESTIMATOR_FAMILIES.items():
        for module_name, cls_name in entries:
            if not build_index._importable(module_name):
                continue
            if getattr(importlib.import_module(module_name), cls_name, None) is None:
                missing.append(f"{family}: {module_name}.{cls_name}")
    assert not missing, "not found in the installed library:\n" + "\n".join(missing)


def _buildable_onboarding_rows() -> list[tuple[dict, pathlib.Path, dict]]:
    out = []
    for path, consts in build_index.discover():
        row = build_index.static_row(path, consts)
        if row["onboarding"] and build_index.can_build(path, row["framework"]):
            out.append((row, path, consts))
    if not out:
        pytest.skip("no onboarding template's framework is installed in this job")
    return out


def test_family_of_refuses_an_estimator_no_family_names(monkeypatch) -> None:
    rows = [r for r in _buildable_onboarding_rows() if r[0]["framework"] == "sklearn"]
    if not rows:
        pytest.skip("no sklearn onboarding template is buildable in this job")
    row, path, consts = rows[0]
    family = build_index.family_of(path, consts)
    in_index = {r["id"]: r["family"] for r in committed()["templates"]}
    assert family == in_index[row["id"]]
    monkeypatch.setattr(
        build_index,
        "ESTIMATOR_FAMILIES",
        {k: v for k, v in build_index.ESTIMATOR_FAMILIES.items() if k != family},
    )
    with pytest.raises(ValueError, match="in 0 families of ESTIMATOR_FAMILIES"):
        build_index.family_of(path, consts)


def test_each_onboarding_row_trains_on_a_cpu() -> None:
    """One epoch (one fit) on a synthetic table of a first run's size, shaped by
    the template's own feature and class counts, on the CPU."""
    import numpy as np

    for row, path, consts in _buildable_onboarding_rows():
        features, classes = consts["num_feature_points"], consts["output_classes"]
        rng = np.random.default_rng(0)
        X = rng.normal(size=(FIT_ROWS, features)).astype("float32")
        y = np.arange(FIT_ROWS) % classes
        entry = build_index._load(path)
        if row["framework"] == "pytorch":
            import torch

            model = entry().to("cpu")
            optimiser = torch.optim.Adam(model.parameters())
            batch = consts["batch_size"]
            for start in range(0, FIT_ROWS, batch):
                xb = torch.from_numpy(X[start : start + batch])
                yb = torch.from_numpy(y[start : start + batch])
                loss = torch.nn.functional.cross_entropy(model(xb), yb)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            assert torch.isfinite(loss), row["id"]
            assert model(torch.from_numpy(X[:3])).shape == (3, classes), row["id"]
        else:
            model = entry().fit(X, y)
            assert model.predict(X[:3]).shape == (3,), row["id"]


# --- mutation rows -------------------------------------------------------------


def test_MUTATION_an_onboarding_row_dropped() -> None:
    index = _passes(committed())
    first, *rest = picked(index)
    first["onboarding"] = False
    assert len(picked(index)) == build_index.ONBOARDING_ROWS - 1
    assert build_index.onboarding_findings(index) == [
        f"2 onboarding rows, not 3: {', '.join(row['id'] for row in rest)}"
    ]


def test_MUTATION_a_fourth_onboarding_row() -> None:
    index = _passes(committed())
    task = picked(index)[0]["category"]
    extra = next(
        row
        for row in index["templates"]
        if not row["onboarding"] and row["category"] == task and row["cpu"]
    )
    extra.update(onboarding=True, family="a_fourth_family")
    assert len(picked(index)) == build_index.ONBOARDING_ROWS + 1
    assert build_index.onboarding_findings(index) == [
        f"4 onboarding rows, not 3: {', '.join(row['id'] for row in picked(index))}"
    ]


def test_MUTATION_two_rows_of_one_family() -> None:
    index = _passes(committed())
    one, two, _ = picked(index)
    two["family"] = one["family"]
    assert len({row["family"] for row in picked(index)}) == 2
    assert build_index.onboarding_findings(index) == [
        f"a family more than one onboarding row has: {one['family']}"
    ]


def test_MUTATION_a_row_with_no_family() -> None:
    index = _passes(committed())
    row = picked(index)[1]
    row["family"] = None
    assert sum(r["family"] is None for r in picked(index)) == 1
    assert build_index.onboarding_findings(index) == [
        f"onboarding rows with no family: {row['id']}"
    ]


def test_MUTATION_a_network_too_large_for_a_cpu() -> None:
    index = _passes(committed())
    row = next(r for r in picked(index) if r["framework"] == "pytorch")
    row["params"] = build_index.CPU_PARAMS_LIMIT + 1
    row["cpu"] = build_index.cpu_capable(row)
    assert row["cpu"] is False
    assert build_index.onboarding_findings(index) == [
        f"onboarding rows a laptop CPU cannot train: {row['id']}"
    ]


def test_MUTATION_rows_of_two_tasks() -> None:
    index = _passes(committed())
    row = picked(index)[2]
    task = row["category"]
    row["category"] = "time_series_classification"
    assert row["category"] != task
    assert build_index.onboarding_findings(index) == [
        "onboarding rows span 2 tasks: "
        + ", ".join(sorted({task, "time_series_classification"}))
    ]


def test_MUTATION_an_index_with_no_templates() -> None:
    index = _passes(committed())
    index["templates"] = []
    assert build_index.onboarding_findings(index) == [
        "the index lists no templates: cannot tell"
    ]


def test_MUTATION_a_template_loses_its_mark(tmp_path, monkeypatch) -> None:
    index = _passes(committed())
    row = picked(index)[0]
    source = build_index.MODEL_ROOT / f"{row['id']}.py"
    text = source.read_text(encoding="utf-8")
    assert text.count("\nonboarding = True\n") == 1
    copy = tmp_path / f"{row['id']}.py"
    copy.parent.mkdir(parents=True)
    copy.write_text(text.replace("\nonboarding = True\n", "\n"), encoding="utf-8")
    monkeypatch.setattr(build_index, "MODEL_ROOT", tmp_path)
    rebuilt = build_index.static_row(copy, build_index._module_constants(copy))
    assert rebuilt["id"] == row["id"] and rebuilt["onboarding"] is False
    row["onboarding"] = rebuilt["onboarding"]
    rest = [r["id"] for r in picked(index)]
    assert build_index.onboarding_findings(index) == [
        f"2 onboarding rows, not 3: {', '.join(rest)}"
    ]
