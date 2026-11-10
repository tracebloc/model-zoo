# mutates: tools/contract_copies.py
"""Mutation rows for ``tools/contract_copies.py``.

Each row starts from the REAL committed copy, asserts it is held, makes one
change, asserts the change applied, asserts the exact finding, and restores it.
The producer-side rows start from a synthetic producer checkout
(``contract_copies_fixtures``), because the real sources are private.
"""

from __future__ import annotations

import copy as copylib
import json
import pathlib

import contract_copies_fixtures as fx
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Named by its repo-relative path: the mutation-coverage gate reads the literal path.
TOOL = "tools/contract_copies.py"
tool = fx.load_tool()
assert (ROOT / TOOL).is_file()


def _real(copy):
    doc = json.loads((ROOT / "tests" / "contracts" / copy.file).read_text(encoding="utf-8"))
    assert tool.pin_findings(doc, copy) == [], "the committed copy is not held to start with"
    return doc


def _drift(value):
    """One change to a contract value, of whatever shape."""
    if isinstance(value, list):
        return value[:-1] if len(value) > 1 else value + ["zz_drift"]
    if isinstance(value, dict):
        return {k: _drift(v) for k, v in value.items()}
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    return f"{value}zz"


FIELD_FINDING = "are not the ones --write derived (fields_sha256)"


@pytest.mark.parametrize("copy", tool.COPIES, ids=lambda c: c.file)
def test_a_hand_edited_contract_field_is_caught_and_restoring_it_clears(copy):
    doc = _real(copy)
    for field in copy.fields:
        mutant = copylib.deepcopy(doc)
        mutant[field] = _drift(mutant[field])
        assert tool.fields_of(mutant, copy) != tool.fields_of(doc, copy), f"the {field} mutation did not apply"
        findings = tool.pin_findings(mutant, copy)
        assert len(findings) == 1 and FIELD_FINDING in findings[0], (field, findings)
    assert tool.pin_findings(doc, copy) == []


@pytest.mark.parametrize("copy", tool.COPIES, ids=lambda c: c.file)
@pytest.mark.parametrize(
    "key, value, finding",
    [
        ("repo", "someone/else", "generated_from.repo"),
        ("paths", ["moved.py"], "generated_from.paths"),
        ("ref", "develop", "generated_from.ref is not a 40-hex commit"),
        ("ref_branch", "feature", "generated_from.ref_branch"),
        ("blobs", {}, "generated_from.blobs"),
    ],
)
def test_a_pin_that_pins_nothing_is_caught(copy, key, value, finding):
    doc = _real(copy)
    mutant = copylib.deepcopy(doc)
    mutant["generated_from"][key] = value
    assert mutant != doc
    findings = tool.pin_findings(mutant, copy)
    assert len(findings) == 1 and findings[0].startswith(finding), findings


@pytest.mark.parametrize("copy", tool.COPIES, ids=lambda c: c.file)
def test_a_copy_with_no_pin_is_caught(copy):
    mutant = _real(copy)
    del mutant["generated_from"]
    assert tool.pin_findings(mutant, copy) == ["generated_from is missing"]


@pytest.mark.parametrize("copy", tool.COPIES, ids=lambda c: c.file)
def test_a_producer_that_moved_under_a_copy_is_drift_on_that_field(copy, tmp_path):
    trees = fx.producer_tree(tmp_path / "p")
    copies = fx.rendered_copies(tool, trees, tmp_path / "c")
    doc = tool.load_doc(copies / copy.file)
    sources = tool.fetch_tree(copy, trees[copy.repo])
    assert tool.check(copy, doc, sources) == []
    field = copy.fields[-1]
    mutant = copylib.deepcopy(doc)
    mutant[field] = _drift(mutant[field])
    assert tool.check(copy, mutant, sources) == [f"{copy.file}: {field} differs from the producer"]
