"""No template declares a hub reference — the SDK rejects the upload if one does.

WHY THIS TEST EXISTS
--------------------
`tests/conftest.py` shuts the HuggingFace hub for the whole suite, so a template
that *fetches* at build time already fails here. This test covers the other half,
which nothing did: a template that merely *declares* a hub id. That never reaches
a fetch, because the SDK refuses the upload before the model is ever built —
`tracebloc/validation/rewriter.py` collects the module-level `model_id`,
`tokenizer_id` and `hf_token` assignments out of every uploaded file and raises

    HuggingFace / hub-referenced models are no longer supported (declared: ...)

when any of them carries a value (internal ref). So a declaration is not inert
documentation and not a fallback: it is the difference between a template that
uploads and one that cannot ((internal ref), where
`masked_language_modeling/pytorch/wide_mini_mlm.py` shipped a correct
`tokenizer.json` and still could not be uploaded, because it also declared
`tokenizer_id = "bert-base-uncased"` beside it).

An empty value does not soften it. The SDK's own check is `value is not None`
after `ast.literal_eval`, so `tokenizer_id = ""` parses to `""` and is rejected
exactly like a real id. The only accepted state is no assignment at all, and
that is what this test asserts.

WHAT A TEMPLATE DECLARES INSTEAD
--------------------------------
`tokenizer_file` — the tokenizer that travels with the model, checked by
`tests/test_tokenizer_declaration.py`. A pretrained architecture is built from
an inlined config with its weights delivered from the tracebloc model store
(internal ref), so no hub id is needed to name either the model or its tokenizer.

Stdlib only: runs in every CI framework job, including the ones with no torch.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "model_zoo"

#: The fields the SDK treats as a hub reference, mirrored from
#: ``tracebloc/validation/rewriter.py``'s ``_HF_HUB_FIELDS``. Refresh this tuple
#: if that one moves — the SDK is the authority on what it refuses.
_HUB_FIELDS = ("model_id", "tokenizer_id", "hf_token")

#: The SDK reads each field with ``^<field>\s*=\s*[a-zA-Z_\-0-9'"]`` — anchored at
#: column zero, so an indented or commented mention is invisible to it. Matching
#: the same shape keeps this guard from failing an upload that would succeed, and
#: from passing one that would not.
_DECLARATION = {
    field: re.compile(rf"^{field}\s*=\s*[a-zA-Z_\-0-9'\"]", re.MULTILINE)
    for field in _HUB_FIELDS
}

_FRAMEWORK = re.compile(r'^\s*framework\s*=\s*["\'](\w+)["\']', re.MULTILINE)


def _templates() -> list[pathlib.Path]:
    """Every uploadable model file: a `.py` under model_zoo/ declaring a framework.

    Derived from the tree rather than from a list, so a new template is covered
    the day it lands.
    """
    out = sorted(p for p in MODEL_ROOT.rglob("*.py")
                 if _FRAMEWORK.search(p.read_text(encoding="utf-8")))
    assert len(out) > 50, f"only {len(out)} template(s) found — this suite would be vacuous"
    return out


def _ids(path: pathlib.Path) -> str:
    return str(path.relative_to(ROOT))


@pytest.mark.parametrize("path", _templates(), ids=_ids)
def test_no_template_declares_a_hub_reference(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")
    declared = [field for field, pattern in _DECLARATION.items() if pattern.search(text)]
    assert not declared, (
        f"{path}: declares {declared} at module level. The SDK's upload rewriter "
        f"raises 'HuggingFace / hub-referenced models are no longer supported' on "
        f"any of {list(_HUB_FIELDS)}, so this template cannot be uploaded at all — "
        f"the declaration is not documentation. Build the architecture from an "
        f"inlined config and name the tokenizer that ships with the model in "
        f"`tokenizer_file` instead."
    )


def test_the_declaration_reader_sees_what_it_should() -> None:
    """Non-vacuity for the regexes: they must read the declarations the SDK reads
    and ignore what it ignores, or the test above passes over an empty list."""
    def seen(text: str) -> list[str]:
        return [f for f, p in _DECLARATION.items() if p.search(text)]

    # What the SDK refuses — including an empty value, which literal-evals to
    # ``""`` and is still ``is not None`` on its side.
    assert seen('tokenizer_id = "bert-base-uncased"\n') == ["tokenizer_id"]
    assert seen('tokenizer_id=""\n') == ["tokenizer_id"]
    assert seen("model_id = 'distilgpt2'\nhf_token = HF\n") == ["model_id", "hf_token"]
    # What the SDK never sees: not anchored at column zero, or not an assignment.
    assert seen('# tokenizer_id = "bert-base-uncased"\n') == []
    assert seen('    tokenizer_id = "bert-base-uncased"\n') == []
    assert seen('``tokenizer_id`` names the tokenizer\n') == []
    # ``tokenizer_file`` is the attribute templates DO declare; it must not be
    # mistaken for ``tokenizer_id``'s prefix or every NLP template fails here.
    assert seen('tokenizer_file = "tokenizer.json"\n') == []
