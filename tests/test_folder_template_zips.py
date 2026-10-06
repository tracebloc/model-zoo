"""Every folder template ships a ready-to-upload zip, and the zip is its folder.

A folder template (``object_detection/pytorch/yolo_v8/``: ``model.py`` and
``loss.py``) is uploaded as the flat zip beside it, ``yolo_v8.zip``, because the
SDK's ``upload_model`` takes one ``.py`` or ``.zip`` and a YOLO needs its
``loss.py``. ``tools/build_folder_zips.py`` builds those zips reproducibly; the
zips are committed so a clone can upload them as they are.

A committed zip is a second copy of its folder, so this rebuilds every zip and
fails on any difference: a ``loss.py`` fix that did not rebuild the zip would
otherwise ship the old loss to everyone who uploads the zip. The mutation rows
below hold ``tools/build_folder_zips.py`` itself to each finding it reports:
each starts from a passing tree, applies one change, asserts the change applied,
and asserts the exact finding.
"""

from __future__ import annotations

import importlib.util
import io
import os
import pathlib
import shutil
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "build_folder_zips.py"  # tools/build_folder_zips.py

_spec = importlib.util.spec_from_file_location("build_folder_zips", TOOL)
build_folder_zips = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_folder_zips)

MODEL_ROOT = build_folder_zips.MODEL_ROOT
FOLDERS = build_folder_zips.folders()


def _rel(path: pathlib.Path) -> str:
    return path.relative_to(MODEL_ROOT).as_posix()


# --- the committed tree ------------------------------------------------------


def test_folder_templates_are_found() -> None:
    # Written down independently of the tool's rule: the three YOLO folders
    # the zoo ships today. A new folder template adds itself to FOLDERS; this
    # floor only stops the rule from quietly matching nothing.
    found = {_rel(f) for f in FOLDERS}
    assert {
        "object_detection/pytorch/yolo_v1",
        "object_detection/pytorch/yolo_v5",
        "object_detection/pytorch/yolo_v8",
    } <= found, found


def test_every_folder_zip_is_committed_and_current() -> None:
    problems = build_folder_zips.problems()
    assert not problems, (
        "the committed folder-template zips disagree with their folders. Run "
        "`python tools/build_folder_zips.py` and commit the result.\n  "
        + "\n  ".join(problems)
    )


@pytest.mark.parametrize("folder", FOLDERS, ids=_rel)
def test_a_folder_zip_is_flat_and_holds_every_py(folder: pathlib.Path) -> None:
    target = build_folder_zips.zip_path(folder)
    assert target == folder.parent / f"{folder.name}.zip"
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        assert names == sorted(p.name for p in folder.glob("*.py")), names
        assert all("/" not in n for n in names), f"not flat: {names}"
        assert {"model.py", "loss.py"} <= set(names), names
        for name in names:
            assert archive.read(name) == (folder / name).read_bytes(), name


# --- mutation rows for tools/build_folder_zips.py -----------------------------


@pytest.fixture
def zoo(tmp_path: pathlib.Path) -> pathlib.Path:
    """A one-template copy of model_zoo/: yolo_v8's folder and its built zip,
    plus a one-file template beside it that must not get a zip."""
    root = tmp_path / "model_zoo"
    source = MODEL_ROOT / "object_detection" / "pytorch"
    target = root / "object_detection" / "pytorch"
    shutil.copytree(source / "yolo_v8", target / "yolo_v8")
    shutil.copy2(source / "faster_rcnn_resnet.py", target / "faster_rcnn_resnet.py")
    assert build_folder_zips.write_all(root) == [target / "yolo_v8.zip"]
    assert build_folder_zips.problems(root) == []
    return root


def test_row_a_changed_source_reports_the_zip_stale(zoo: pathlib.Path) -> None:
    loss = zoo / "object_detection" / "pytorch" / "yolo_v8" / "loss.py"
    before = loss.read_bytes()
    loss.write_bytes(before + b"\n# changed\n")
    assert loss.read_bytes() != before
    assert build_folder_zips.problems(zoo) == [
        "object_detection/pytorch/yolo_v8.zip: stale (does not match yolo_v8/)"
    ]


def test_row_a_deleted_zip_reports_missing(zoo: pathlib.Path) -> None:
    target = zoo / "object_detection" / "pytorch" / "yolo_v8.zip"
    target.unlink()
    assert not target.exists()
    assert build_folder_zips.problems(zoo) == [
        "object_detection/pytorch/yolo_v8.zip: missing"
    ]


def test_row_a_zip_with_no_folder_reports_orphaned(zoo: pathlib.Path) -> None:
    stray = zoo / "object_detection" / "pytorch" / "yolo_v9.zip"
    shutil.copy2(zoo / "object_detection" / "pytorch" / "yolo_v8.zip", stray)
    assert stray.is_file()
    assert build_folder_zips.problems(zoo) == [
        "object_detection/pytorch/yolo_v9.zip: no folder template builds it"
    ]


def test_row_no_folder_template_at_all_is_a_finding(zoo: pathlib.Path) -> None:
    folder = zoo / "object_detection" / "pytorch" / "yolo_v8"
    shutil.rmtree(folder)
    (zoo / "object_detection" / "pytorch" / "yolo_v8.zip").unlink()
    assert not folder.exists()
    assert build_folder_zips.problems(zoo) == [f"no folder template found under {zoo}"]


def test_row_a_one_file_template_gets_no_zip(zoo: pathlib.Path) -> None:
    pytorch = zoo / "object_detection" / "pytorch"
    assert not (pytorch / "faster_rcnn_resnet.zip").exists()
    assert build_folder_zips.folders(zoo) == [pytorch / "yolo_v8"]


def test_row_the_zip_skips_caches_and_non_py_files(zoo: pathlib.Path) -> None:
    folder = zoo / "object_detection" / "pytorch" / "yolo_v8"
    clean = build_folder_zips.build(folder)
    (folder / "__pycache__").mkdir()
    (folder / "__pycache__" / "model.cpython-311.pyc").write_bytes(b"\0")
    (folder / "README.txt").write_text("notes")
    assert (folder / "__pycache__").is_dir()
    assert build_folder_zips.build(folder) == clean
    with zipfile.ZipFile(io.BytesIO(clean)) as archive:
        assert archive.namelist() == ["loss.py", "model.py"]


def test_row_the_build_ignores_mtimes(zoo: pathlib.Path) -> None:
    folder = zoo / "object_detection" / "pytorch" / "yolo_v8"
    first = build_folder_zips.build(folder)
    for path in folder.glob("*.py"):
        os.utime(path, (2_000_000_000, 2_000_000_000))
    assert os.stat(folder / "model.py").st_mtime == 2_000_000_000
    assert build_folder_zips.build(folder) == first
