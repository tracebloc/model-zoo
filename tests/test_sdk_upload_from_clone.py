"""Every template uploads the way a user uploads it from a clone of this repo.

A user who clones the zoo passes one path to the SDK's ``upload_model``: a
one-file template's ``.py``, or a folder template's flat ``.zip``
(``tools/build_folder_zips.py``). ``model_zoo/index.v1.json`` names that file
per template as ``upload``. This runs the SDK's own local upload checks on it --
the code ``upload_model`` runs before its POST, with the network shut
(``tests/conftest.py``) -- so a template that a user cannot upload as cloned
goes red here instead of in their notebook. That is how the YOLO folders
shipped: ``yolo_v8/model.py`` uploaded alone fails with "loss.py file missing
in the zip", and nothing in this repo ran the SDK.

THE SDK IS PINNED in ``.github/requirements/sdk.txt`` (a released PyPI version,
installed beside the engine-pinned ``pytorch.txt``) and only the ``sdk-upload``
CI job installs it and sets ``MODEL_ZOO_SDK_UPLOAD=1``. Without that variable
this module skips as a whole, even where the SDK happens to be installed: the
full checks below are heavy, and ``make check`` must stay fast. With it, a
missing SDK is a collection error, never a skip: that job exists to look.
Run it locally with::

    pip install -r .github/requirements/sdk.txt
    MODEL_ZOO_SDK_UPLOAD=1 pytest tests/test_sdk_upload_from_clone.py

Three depths, by cost:

* every template: the SDK's ``get_paths`` resolves its ``upload`` path, with
  and without the extension, to that file (stdlib-cheap, every framework);
* every PyTorch template this job can build: the SDK's general model checks
  (``ModelChecker.model_func_checks``: extract or copy, parse, build), on a
  copy of the file in a temp directory, as ``upload_model`` makes them;
* every folder template, plus the smallest PyTorch template of each category:
  the whole local upload check, ``validate_model_file(upload=False)`` -- the
  task handler's checks and its training step on dummy data. Running it on
  every template is not affordable on a CI runner (a ViT-scale template peaks
  near 15 GB in the training step), so one per category stands in for the rest.

And the reason the zips exist: a folder template's ``model.py`` uploaded on its
own is refused for its missing ``loss.py``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
from unittest.mock import MagicMock

import pytest

if os.environ.get("MODEL_ZOO_SDK_UPLOAD") != "1":
    pytest.skip(
        "runs in the sdk-upload CI job (MODEL_ZOO_SDK_UPLOAD=1, SDK from "
        ".github/requirements/sdk.txt)",
        allow_module_level=True,
    )

from tracebloc.user import User  # noqa: E402
from tracebloc.utils.general_utils import get_paths  # noqa: E402
from tracebloc.validation.checker import ModelChecker  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_index = _load("build_index", ROOT / "tools" / "build_index.py")
build_folder_zips = _load("build_folder_zips", ROOT / "tools" / "build_folder_zips.py")
# The RAM ceiling test_model_instantiates already skips on, read from there.
_TOO_LARGE_FOR_CI_RAM = _load(
    "model_contract_for_sdk", ROOT / "tests" / "test_model_contract.py"
)._TOO_LARGE_FOR_CI_RAM

MODEL_ROOT = build_index.MODEL_ROOT
ROWS = json.loads(build_index.INDEX_PATH.read_text())["templates"]
CONSTS = {
    str(path.relative_to(MODEL_ROOT).with_suffix("")): (path, consts)
    for path, consts in build_index.discover()
}
FOLDERS = build_folder_zips.folders()


def _ids(row: dict) -> str:
    return row["id"]


def _pytorch_rows() -> list[dict]:
    return [r for r in ROWS if r["framework"] == "pytorch"]


def _full_check_rows() -> list[dict]:
    """Every folder template, and the smallest PyTorch template per category."""
    chosen = {r["id"]: r for r in _pytorch_rows() if r["upload"].endswith(".zip")}
    smallest: dict[str, dict] = {}
    for row in _pytorch_rows():
        best = smallest.get(row["category"])
        if best is None or (row["params"], row["id"]) < (best["params"], best["id"]):
            smallest[row["category"]] = row
    chosen.update({r["id"]: r for r in smallest.values()})
    return sorted(chosen.values(), key=_ids)


def _clone_copy(row: dict, tmp_path: pathlib.Path) -> tuple[pathlib.Path, str | None]:
    """The upload file copied into ``tmp_path`` (the SDK extracts beside the
    file it is given, which must not be this checkout), and the tokenizer the
    template declares (``tokenizer_file``), copied beside it, as the README
    snippets pass it."""
    source = MODEL_ROOT / row["upload"]
    target = tmp_path / source.name
    shutil.copy2(source, target)
    _path, consts = CONSTS[row["id"]]
    tokenizer = consts.get("tokenizer_file")
    if isinstance(tokenizer, str):
        shutil.copy2(source.parent / tokenizer, tmp_path / tokenizer)
        return target, str(tmp_path / tokenizer)
    return target, None


def _skip_unbuildable(row: dict) -> None:
    if row["id"] + ".py" in _TOO_LARGE_FOR_CI_RAM:
        pytest.skip("random-init construction exceeds CI runner RAM")
    path, _consts = CONSTS[row["id"]]
    if not build_index.can_build(path, row["framework"]):
        pytest.skip(f"{row['framework']} or an optional library is not installed here")


def test_the_index_names_every_template_and_every_folder_zip() -> None:
    # Fails closed: an empty index or a rule that found no folder would make
    # every parametrized test below vanish while this module stayed green.
    assert len(ROWS) == len(CONSTS) > 0
    zips = {r["upload"] for r in ROWS if r["upload"].endswith(".zip")}
    assert zips == {
        build_folder_zips.zip_path(f).relative_to(MODEL_ROOT).as_posix() for f in FOLDERS
    }
    assert zips, "no folder template in the index"


@pytest.mark.parametrize("row", ROWS, ids=_ids)
def test_the_sdk_resolves_the_upload_path(row: dict) -> None:
    upload = MODEL_ROOT / row["upload"]
    stem, suffix = upload.stem, upload.suffix
    for given in (str(upload), str(upload.with_suffix(""))):
        name, model_path, _weights, ext = get_paths(path=given)
        assert (name, model_path, ext) == (stem, str(upload), suffix), given


@pytest.mark.parametrize("row", _pytorch_rows(), ids=_ids)
def test_the_sdk_general_checks_pass(row: dict, tmp_path: pathlib.Path) -> None:
    _skip_unbuildable(row)
    target, _tokenizer = _clone_copy(row, tmp_path)
    name, model_path, _weights, _ext = get_paths(path=str(target))
    checker = ModelChecker(MagicMock(), model_name=name, model_path=model_path)
    status, message, _name, _bar = checker.model_func_checks()
    assert status, message


def _validate(target: pathlib.Path, tokenizer: str | None):
    """``upload_model``'s checks, stopped before the POST."""
    handler = User._for_local_checks()._prepare_model_check(
        str(target.with_suffix("")), False, tokenizer
    )
    handler.validate_model_file(upload=False)
    return handler


@pytest.mark.parametrize("row", _full_check_rows(), ids=_ids)
def test_the_whole_local_upload_check_passes(row: dict, tmp_path: pathlib.Path) -> None:
    _skip_unbuildable(row)
    target, tokenizer = _clone_copy(row, tmp_path)
    handler = _validate(target, tokenizer)
    assert handler.validated, getattr(handler, "failure_reason", None)


@pytest.mark.parametrize("folder", FOLDERS, ids=lambda f: f.relative_to(MODEL_ROOT).as_posix())
def test_a_folder_model_py_alone_is_refused_for_its_loss(
    folder: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    # Why the zips exist. A folder template's model.py passed on its own never
    # brings its loss.py, and the SDK refuses it -- so the README and the index
    # name the zip, and this goes red if that ever stops being true.
    copy = tmp_path / folder.name
    shutil.copytree(folder, copy, ignore=shutil.ignore_patterns("__pycache__"))
    handler = _validate(copy / "model.py", None)
    assert not handler.validated
    assert "loss.py file missing" in (handler.failure_reason or ""), handler.failure_reason
