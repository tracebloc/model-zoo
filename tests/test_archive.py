"""A template under ``archive/`` is offered nowhere.

``archive/`` holds templates at the path they had under ``model_zoo/``. Every
template walker, the index build among them, reads ``model_zoo/`` alone, so an
archived template has no row in ``model_zoo/index.v1.json`` and is never
published. These tests go red when one comes back by another route: a copy
under ``model_zoo/`` again, or a row in the index. Stdlib only.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ZOO = ROOT / "model_zoo"


def _publish_tool():
    spec = importlib.util.spec_from_file_location(
        "publish_ready_made", ROOT / "tools/publish_ready_made.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Registered first: its dataclass resolves its annotations through it.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tool = _publish_tool()
ARCHIVED = sorted(tool.archived(ROOT))


def test_the_archive_holds_templates() -> None:
    """An empty archive would make every test below pass on nothing: delete the
    directory and this file together instead."""
    assert ARCHIVED, "archive/ holds no template"


@pytest.mark.parametrize("template_id", ARCHIVED)
def test_an_archived_template_is_not_under_model_zoo(template_id: str) -> None:
    assert not (ZOO / f"{template_id}.py").exists(), (
        f"{template_id} is archived and also under model_zoo/"
    )


@pytest.mark.parametrize("template_id", ARCHIVED)
def test_an_archived_template_is_not_in_the_index(template_id: str) -> None:
    rows = json.loads((ZOO / "index.v1.json").read_text())["templates"]
    assert template_id not in {row["id"] for row in rows}, (
        f"{template_id} is archived and listed in model_zoo/index.v1.json"
    )


def test_the_publish_tool_reads_the_real_index_past_the_archive() -> None:
    tool.refuse_archived(ROOT, tool.index_rows(ROOT))
