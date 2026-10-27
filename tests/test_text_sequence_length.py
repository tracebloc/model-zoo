"""A text template's ``sequence_length`` is long enough to hold text.

The engine truncates every text to the declared ``sequence_length`` without
saying so, so a template that declares a handful of tokens trains on the first
few words of each example and reports a loss all the same.

WHAT IS ASSERTED
----------------
* the templates are DERIVED from ``model_zoo/index.v1.json``, never listed: a
  text template is an index row whose file declares a ``tokenizer_file``;
* each such template declares ``sequence_length`` as an int literal, and it is
  at least ``MIN_SEQUENCE_LENGTH``;
* the zero-templates case fails: a derivation that finds nothing is not a pass.

All of it is stdlib: this test runs in every CI framework job.
"""
from __future__ import annotations

import ast
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "model_zoo"

#: Below the shortest legitimate template (64); well above anything that cuts typical text to a fragment.
MIN_SEQUENCE_LENGTH = 32


def _literals(source: str) -> dict:
    """Module-level ``name = <literal>`` assignments, read by AST."""
    out = {}
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        ):
            out[node.targets[0].id] = node.value.value
    return out


def too_short(source: str) -> str | None:
    """Why a text template's declared length is unusable, or None when it is fine or not a text template."""
    consts = _literals(source)
    if "tokenizer_file" not in consts:
        return None
    length = consts.get("sequence_length")
    if type(length) is not int:
        return f"`sequence_length` must be an int literal, got {length!r}"
    if length < MIN_SEQUENCE_LENGTH:
        return f"`sequence_length = {length}` is below {MIN_SEQUENCE_LENGTH}: every text is cut to {length} tokens"
    return None


def _text_templates() -> list[str]:
    index = json.loads((MODEL_ROOT / "index.v1.json").read_text())
    uploads = [row["upload"] for row in index["templates"] if row["upload"].endswith(".py")]
    return [u for u in uploads if "tokenizer_file" in _literals((MODEL_ROOT / u).read_text())]


TEMPLATES = _text_templates()


def test_some_text_template_is_derived():
    assert TEMPLATES, "no index row declares a tokenizer_file -- every test below is vacuous"


@pytest.mark.parametrize("upload", TEMPLATES, ids=TEMPLATES)
def test_text_template_does_not_cut_typical_text(upload: str):
    reason = too_short((MODEL_ROOT / upload).read_text())
    assert reason is None, f"model_zoo/{upload}: {reason}"


@pytest.mark.parametrize("planted,expect_fail", [(5, True), (31, True), (32, False), (512, False)])
def test_the_floor_catches_a_planted_length(planted: int, expect_fail: bool):
    source = f'tokenizer_file = "t.json"\nsequence_length = {planted}\n'
    assert (too_short(source) is not None) is expect_fail


def test_a_non_literal_length_is_refused():
    assert "int literal" in too_short('tokenizer_file = "t.json"\nsequence_length = LENGTH\n')
