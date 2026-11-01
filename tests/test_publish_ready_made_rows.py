# mutates: tools/publish_ready_made.py
"""Mutation rows for ``tools/publish_ready_made.py``.

Each row starts from a copy of the REAL ``model_zoo/`` (its index and every
file the index names), asserts the tool reads it in full, makes one change,
asserts the change applied, and asserts the exact finding the tool reports.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import shutil

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Named by its repo-relative path: the mutation-coverage gate reads the literal path.
TOOL = "tools/publish_ready_made.py"


def _load():
    spec = importlib.util.spec_from_file_location("publish_ready_made", ROOT / TOOL)
    mod = importlib.util.module_from_spec(spec)
    # Registered first: its dataclass resolves its annotations through it.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tool = _load()


@pytest.fixture
def zoo(tmp_path):
    """A copy of the real ``model_zoo/``: the index and the files it names."""
    copy = tmp_path / "zoo"
    shutil.copytree(ROOT / "model_zoo", copy / "model_zoo")
    return copy


def index(zoo):
    return json.loads((zoo / tool.INDEX).read_text())


def write(zoo, document):
    (zoo / tool.INDEX).write_text(json.dumps(document))


def read_all(zoo):
    return [tool.template(zoo, row) for row in tool.index_rows(zoo)]


def test_the_real_zoo_reads_in_full(zoo):
    """The known-good state every row below starts from."""
    items = read_all(zoo)
    assert len(items) == len(index(zoo)["templates"]) > 0
    assert any(i.tokenizer is not None for i in items)


def test_row_an_index_of_another_version_is_unreadable(zoo):
    document = index(zoo)
    document["version"] = 2
    write(zoo, document)
    assert index(zoo)["version"] == 2
    with pytest.raises(tool.Unreadable, match=r"is not version 1"):
        tool.index_rows(zoo)


def test_row_an_index_with_no_templates_is_unreadable(zoo):
    document = index(zoo)
    document["templates"] = []
    write(zoo, document)
    assert index(zoo)["templates"] == []
    with pytest.raises(tool.Unreadable, match=r"lists no templates"):
        tool.index_rows(zoo)


def test_row_a_declared_tokenizer_that_is_gone_is_named(zoo):
    victim = next(i for i in read_all(zoo) if i.tokenizer is not None)
    victim.tokenizer.unlink()
    assert not victim.tokenizer.exists()
    with pytest.raises(tool.Unreadable, match=rf"declares a missing {victim.tokenizer.name}"):
        tool.template(zoo, victim.row)


def test_row_an_upload_outside_the_zoo_is_refused(zoo):
    (zoo / "outside.py").write_text("x = 1\n")
    row = dict(tool.index_rows(zoo)[0], upload="../outside.py")
    assert (zoo / "model_zoo" / row["upload"]).resolve().is_file()
    with pytest.raises(tool.Unreadable, match=r"is not in the zoo"):
        tool.template(zoo, row)


def test_row_a_row_missing_a_fact_is_named(zoo):
    row = dict(tool.index_rows(zoo)[0])
    del row["batch"]
    assert "batch" not in row
    with pytest.raises(tool.Unreadable, match=r"missing batch"):
        tool.template(zoo, row)
