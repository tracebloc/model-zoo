"""``model_zoo/index.v1.json`` is what ``tools/build_index.py`` builds, byte for byte.

The index is committed so a consumer can list the zoo without importing it,
which makes it a second copy of facts the templates own: a template's
``batch_size``, its architecture (its parameter count), its estimator (whether
it averages across edges). This test is what keeps the copy honest. It
regenerates the index and fails on any difference, so a PR that moves one of
those facts has to carry the index change with it.

Each CI job recomputes the rows its framework can build and carries the rest
from the committed file (see the builder's docstring). The static fields --
id, category, framework, model_type, label, batch -- and the set of templates
are checked in every job.
"""

from __future__ import annotations

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


def _diff(committed: dict, built: dict) -> list[str]:
    old = {row["id"]: row for row in committed["templates"]}
    new = {row["id"]: row for row in built["templates"]}
    lines = [f"  missing from the index: {i}" for i in sorted(new.keys() - old.keys())]
    lines += [f"  no longer a template: {i}" for i in sorted(old.keys() - new.keys())]
    for i in sorted(old.keys() & new.keys()):
        for field in sorted(old[i].keys() | new[i].keys()):
            if old[i].get(field) != new[i].get(field):
                lines.append(
                    f"  {i}: {field} is {old[i].get(field)!r} in the index, "
                    f"{new[i].get(field)!r} in the template"
                )
    return lines


def test_the_committed_index_is_current() -> None:
    assert build_index.INDEX_PATH.is_file(), (
        "model_zoo/index.v1.json is missing. Run: python tools/build_index.py"
    )
    text = build_index.INDEX_PATH.read_text()
    committed = json.loads(text)
    built, _carried = build_index.build(
        {row["id"]: row for row in committed["templates"]}
    )
    if build_index.render(built) != text:
        detail = _diff(committed, built) or ["  (formatting or row order only)"]
        pytest.fail(
            "model_zoo/index.v1.json is stale. Run `python tools/build_index.py` "
            "and commit the result.\n" + "\n".join(detail)
        )


def test_the_index_has_one_row_per_template_with_every_field() -> None:
    index = json.loads(build_index.INDEX_PATH.read_text())
    assert index["version"] == build_index.INDEX_VERSION
    ids = [row["id"] for row in index["templates"]]
    assert ids == sorted(ids) and len(ids) == len(set(ids))
    assert len(ids) == len(build_index.discover())
    fields = {"id", "category", "framework", "model_type", "label", "batch"}
    for row in index["templates"]:
        assert set(row) == fields | {"params", "averageable"}, row["id"]
        assert isinstance(row["averageable"], bool), row["id"]
        if row["framework"] == "pytorch":
            assert type(row["params"]) is int and row["params"] > 0, row["id"]
            assert row["averageable"] is True, row["id"]
        else:
            assert row["params"] is None, row["id"]


def test_every_averageable_estimator_resolves() -> None:
    """A misspelt entry would silently drop a family from the allowlist, and
    every template using it would read as not averageable."""
    missing = []
    for framework, entries in build_index.AVERAGEABLE_ESTIMATORS.items():
        for module_name, cls_name in entries:
            # Absent here (another framework job) or unloadable (a native
            # library without its runtime): the builder skips it too.
            if not build_index._importable(module_name):
                continue
            module = importlib.import_module(module_name)
            if getattr(module, cls_name, None) is None:
                missing.append(f"{framework}: {module_name}.{cls_name}")
    assert not missing, "not found in the installed library:\n" + "\n".join(missing)


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("resnet_18", "ResNet 18"),
        ("faster_rcnn_mobilenet_320", "Faster R-CNN MobileNet 320"),
        ("efficientdet_d0", "EfficientDet D0"),
        ("logistic_regression", "Logistic Regression"),
        ("yolov8_s", "YOLOv8 S"),
        ("qwen2_5_0_5b", "Qwen2.5 0.5B"),
    ],
)
def test_label_for(name: str, label: str) -> None:
    assert build_index.label_for(name) == label


def test_a_packaged_template_is_named_by_its_directory() -> None:
    rel = pathlib.PurePosixPath("object_detection/pytorch/yolo_v8/model.py")
    assert build_index.template_name(rel) == "yolo_v8"
    assert build_index.template_name(
        pathlib.PurePosixPath("image_classification/pytorch/resnet_18.py")
    ) == "resnet_18"
