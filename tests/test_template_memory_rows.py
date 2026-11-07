# mutates: tools/template_memory.py
"""Mutation rows for ``tools/template_memory.py``.

The SDK's estimate is not installed where the rows run, so each row gives the
tool a recorded estimator: the figure and the warning the SDK returns for the
template's real defaults. The rows break the tool's own half, reading a
template's defaults and reporting, on a copy of the real ``model_zoo/``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Named by its repo-relative path: the mutation-coverage gate reads the literal path.
TOOL = "tools/template_memory.py"


def _load():
    spec = importlib.util.spec_from_file_location("template_memory", ROOT / TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load()
TEMPLATE = tool.CHECKED[0]


def recorded(defaults, path):
    """The SDK's verdict at the template's committed defaults (512 x 2: 3.42 GiB,
    no warning), and its warning for any larger run."""
    if (defaults.batch_size, defaults.input_size) == (2, 512):
        return int(3.42 * 1024**3), None
    return int(3.79 * 1024**3), "Memory warning: this run may need ~3.8 GiB"


@pytest.fixture
def zoo(tmp_path):
    shutil.copytree(ROOT / tool.ZOO, tmp_path / tool.ZOO)
    return tmp_path


def _rewrite(zoo, old, new):
    path = zoo / tool.ZOO / f"{TEMPLATE}.py"
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))
    assert new in path.read_text()


def test_row_the_real_zoo_fits(zoo):
    lines, findings = tool.check(zoo, recorded)
    assert findings == [] and len(lines) == len(tool.CHECKED)


def test_row_raised_batch_is_a_finding(zoo):
    _rewrite(zoo, "batch_size = 2\n", "batch_size = 8\n")
    _, findings = tool.check(zoo, recorded)
    assert findings == [
        f"{TEMPLATE}: ~3.79 GiB at batch 8, size 512, over the 4 GiB budget "
        "with the SDK's margin; lower its defaults"
    ]


def test_row_raised_sequence_length_is_a_finding(zoo):
    _rewrite(zoo, "sequence_length = 512\n", "sequence_length = 1024\n")
    _, findings = tool.check(zoo, recorded)
    assert findings == [
        f"{TEMPLATE}: ~3.79 GiB at batch 2, size 1024, over the 4 GiB budget "
        "with the SDK's margin; lower its defaults"
    ]


def test_row_unreadable_batch_is_a_finding(zoo):
    _rewrite(zoo, "batch_size = 2\n", "batch_size = int('2')\n")
    _, findings = tool.check(zoo, recorded)
    assert findings == [f"{TEMPLATE}: batch_size None is not a positive int"]


def test_row_missing_template_is_a_finding(zoo):
    (zoo / tool.ZOO / f"{TEMPLATE}.py").unlink()
    _, findings = tool.check(zoo, recorded)
    assert findings == [
        f"{TEMPLATE}: no template at {tool.ZOO}/{TEMPLATE}.py",
    ]


def test_row_nothing_checked_is_a_finding(zoo):
    _, findings = tool.check(zoo, recorded, templates=())
    assert findings == ["no template is checked"]
