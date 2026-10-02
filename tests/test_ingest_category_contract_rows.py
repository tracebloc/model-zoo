"""Mutation rows for ``tools/ingest_category_contract.py``.

``tests/test_zoo_category_contract.py`` holds the zoo to the vendored category
enum, but it drives the tool's helpers on synthetic input and recomputes the
real-tree comparison inline, so nothing showed that the tool itself still fires
on the zoo. Each row here starts from the REAL input -- a copy of the committed
``model_zoo/`` tree and the committed vendored enum -- asserts that it passes,
makes one change and asserts the change applied, then asserts the exact finding
the tool reports for it.

The copy holds only what the tool reads (the category directories and the
``.py`` modules under them), not the weights and data files beside them.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Named by its repo-relative path: the mutation-coverage gate reads the literal path.
TOOL = "tools/ingest_category_contract.py"


def _load():
    spec = importlib.util.spec_from_file_location("ingest_category_contract", ROOT / TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load()
OURS = "model_zoo/"
VENDORED_NAME = "the vendored enum"


@pytest.fixture
def zoo(tmp_path):
    """A copy of the real ``model_zoo/`` with only the directories and modules."""
    def keep_only_py(directory, names):
        return [n for n in names if not (pathlib.Path(directory, n).is_dir() or n.endswith(".py"))]

    copy = tmp_path / "model_zoo"
    shutil.copytree(gate.MODEL_ROOT, copy, ignore=keep_only_py)
    return copy


@pytest.fixture
def vendored(tmp_path):
    """A copy of the committed vendored enum."""
    copy = tmp_path / gate.VENDORED.name
    shutil.copyfile(gate.VENDORED, copy)
    return copy


def findings(zoo_root, vendored_path):
    return gate.compare(gate.load_vendored(vendored_path), gate.zoo_directories(zoo_root),
                        VENDORED_NAME, OURS)


def mismatches(zoo_root):
    return {rel: (d, c) for rel, (d, c) in gate.declared_categories(zoo_root).items() if c != d}


def test_the_real_tree_and_the_real_enum_agree(zoo, vendored):
    """The known-good state every row below starts from."""
    assert gate.zoo_directories(zoo), "the copy holds category directories"
    assert findings(zoo, vendored) == []
    assert gate.declared_categories(zoo), "the copy holds model modules"
    assert mismatches(zoo) == {}


def test_a_published_category_with_no_zoo_directory_is_named(zoo, vendored):
    victim = sorted(gate.zoo_directories(zoo))[0]
    shutil.rmtree(zoo / victim)
    assert victim not in gate.zoo_directories(zoo), "the directory is gone"

    assert findings(zoo, vendored) == [f"only {VENDORED_NAME} has: {[victim]}"]


def test_a_zoo_directory_the_producer_does_not_publish_is_named(zoo, vendored):
    (zoo / "zombie_category").mkdir()
    assert "zombie_category" in gate.zoo_directories(zoo), "the directory is there"

    assert findings(zoo, vendored) == [f"only {OURS} has: ['zombie_category']"]


def test_a_hidden_or_private_directory_is_not_a_category(zoo, vendored):
    (zoo / "__pycache__").mkdir(exist_ok=True)
    (zoo / ".cache").mkdir()

    assert findings(zoo, vendored) == []


def test_a_category_the_vendored_enum_lists_twice_is_named(zoo, vendored):
    doc = json.loads(vendored.read_text(encoding="utf-8"))
    first = doc["categories"][0]
    doc["categories"].append(first)
    vendored.write_text(json.dumps(doc), encoding="utf-8")
    assert gate.load_vendored(vendored).count(first) == 2, "the duplicate is there"

    assert findings(zoo, vendored) == [f"{VENDORED_NAME} lists {[first]} more than once"]


def test_a_module_declaring_another_directorys_category_is_named(zoo):
    modules = gate.declared_categories(zoo)
    rel, (directory, category) = next((r, v) for r, v in sorted(modules.items()) if v[1] is not None)
    path = zoo / rel
    source = path.read_text(encoding="utf-8")
    anchor = f'category = "{category}"'
    assert source.count(anchor) == 1, f"the anchor in {rel} must match exactly once"
    path.write_text(source.replace(anchor, 'category = "not_this_directory"'), encoding="utf-8")

    assert mismatches(zoo) == {rel: (directory, "not_this_directory")}
