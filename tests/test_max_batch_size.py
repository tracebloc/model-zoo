"""Every ``max_batch_size`` a ready-made model declares is one the SDK can use.

``batch_size`` is sized for the smallest secure environment (see
``tools/template_memory.py``). ``max_batch_size`` is the largest batch the SDK
may raise it to when a training gets more memory than that; a model without it
keeps its ``batch_size``. So a declared value must be a positive int literal at
least ``batch_size``, and the SDK's header reader must find it.

Stdlib only, like ``tests/test_readme_batch_size.py``.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "model_zoo"
#: The SDK's reader of this header line: a line that starts with the name.
SDK_READER = re.compile(r"^max_batch_size\s*=\s*[0-9]", re.MULTILINE)


def _constants(path: pathlib.Path) -> dict:
    found = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    found[target.id] = node.value
    return found


def _declaring() -> list[pathlib.Path]:
    return sorted(
        path
        for path in MODEL_ROOT.rglob("*.py")
        if "max_batch_size" in path.read_text(encoding="utf-8")
        and not any(part.startswith("tmpmodel_") for part in path.parts)
    )


DECLARING = _declaring()


def test_some_ready_made_model_declares_one():
    assert DECLARING, "no model file declares max_batch_size: the reader found nothing"


@pytest.mark.parametrize(
    "path", DECLARING, ids=lambda p: str(p.relative_to(MODEL_ROOT))
)
def test_max_batch_size_is_an_int_at_least_batch_size(path):
    constants = _constants(path)
    batch, ceiling = constants.get("batch_size"), constants.get("max_batch_size")
    assert type(ceiling) is int, f"max_batch_size must be an int literal, got {ceiling!r}"
    assert type(batch) is int and 1 <= batch <= ceiling, (batch, ceiling)
    assert len(SDK_READER.findall(path.read_text(encoding="utf-8"))) == 1
