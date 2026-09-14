"""Every sklearn tabular template declares a model_type the SDK will TRAIN.

WHY THIS IS A SECOND FILE and not a case in ``test_zoo_model_type_contract.py``:
that file asserts a value is **storable** by the backend, against
``model_type_choices.v1.json``. Storable is not trainable. The two gates
disagree, and the disagreement is exactly how the EBM declaration incident shipped:

    model_type = ""   ->  backend:  stored, coerced to 'default'   (accepted)
                      ->  SDK:      coerced to None                (REFUSED)

``tracebloc/validation/rewriter.py::_parse_constant_rhs`` ends with an explicit
``if parsed == "": return None``, and ``None`` is in no accepted set, so
``average_estimators`` raises *"model type None is not supported for Sklearn"*
and ``model_func_checks`` fails. Both ebm templates declared ``""`` and were
untrainable end to end while passing the storable check. ``'default'`` is not in
any accepted set either, so the blank could not have worked by that route.

The accepted sets are vendored in
``tests/contracts/tracebloc_sdk/sklearn_model_types.v1.json``, derived by AST
from the SDK at a pinned ref — never hand-transcribed (the OD vocabulary-drift incident).

WHAT THIS FILE DOES NOT CLAIM: that the estimator is federated-averageable.
Membership only means the SDK's upload-time smoke fit routes. The SDK says so
itself in ``SklTabularBase._ensemble_prefixes`` (internal ref).
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
ZOO = REPO / "model_zoo"
CONTRACT = REPO / "tests" / "contracts" / "tracebloc_sdk" / "sklearn_model_types.v1.json"

#: category directory -> the SDK family whose accepted set applies.
_FAMILY_BY_CATEGORY = {
    "tabular_classification": "classifier",
    "tabular_regression": "regression",
}


def _contract() -> dict:
    return json.loads(CONTRACT.read_text())


def _module_constants(path: pathlib.Path) -> dict:
    """Module-level literal assignments, by AST. Text matching would read the
    prose in a declaration comment as a declaration."""
    out: dict = {}
    for node in ast.parse(path.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                out[node.targets[0].id] = "<non-literal>"
    return out


def _sklearn_templates() -> list[tuple[pathlib.Path, dict]]:
    rows = []
    for path in sorted(ZOO.rglob("*.py")):
        consts = _module_constants(path)
        if consts.get("framework") == "sklearn":
            rows.append((path, consts))
    return rows


# --- the SDK's own coercion, mirrored so the shape table can be asserted -----
# Verbatim behaviour of rewriter.py::_parse_constant_rhs for the shapes below,
# plus the model_type canonicaliser (.strip().lower(), the canonicaliser change) the reader
# applies afterwards. Kept tiny and pinned by
# test_shape_table_matches_the_vendored_mirror -- which pins THIS mirror, not
# the SDK; see that test's docstring.
def _as_sdk_sees_it(raw: str | None):
    if raw is None:
        return None
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        parsed = raw.split("#", 1)[0].strip().replace("'", "").replace('"', "")
    if parsed == "":
        return None
    if isinstance(parsed, str):
        parsed = parsed.strip().lower()
    return parsed


def test_the_scan_found_the_sklearn_templates() -> None:
    """Positive control: an empty scan would pass every assertion below."""
    rows = _sklearn_templates()
    assert len(rows) >= 25, f"only {len(rows)} sklearn templates found -- scan broken"


def test_the_contract_carries_both_families() -> None:
    accepted = _contract()["accepted_by_family"]
    assert set(accepted) == {"classifier", "regression"}
    for family, values in accepted.items():
        assert values, f"{family} accepted set is empty -- derivation broken"
        assert "tree" in values, f"{family} lost 'tree' -- derivation suspect"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('""', None),          # the EBM incident: the shipped ebm declaration
        ("''", None),          # same, single-quoted
        ('"  "', ""),          # survives coercion, then .strip() -> ''
        (None, None),          # no declaration at all
        ('"default"', "default"),   # storable, and in no accepted set
        ('"tree"', "tree"),
        ('"TREE"', "tree"),    # canonicaliser lowercases
    ],
)
def test_shape_table_matches_the_vendored_mirror(raw, expected) -> None:
    """A TABLE of shapes, not just the one in the ticket. Empty, whitespace,
    absent and 'default' all arrive outside the accepted set, by three different
    routes -- so a fix that only handles the empty literal is incomplete.

    NAMED FOR WHAT IT ACTUALLY PINS. ``_as_sdk_sees_it`` is a hand-written
    MIRROR of the SDK's ``_parse_constant_rhs`` plus the canonicaliser, not the
    SDK itself, so this asserts the mirror still behaves as documented -- it
    does NOT detect the SDK changing underneath us. Mutating the mirror reddens
    this; mutating the SDK does not. The ref the mirror was verified against is
    recorded in the contract's ``generated_from.ref``.

    The real gate is ``test_every_sklearn_template_declares_a_trainable_type``,
    which reads the templates and the derived contract and never calls the
    mirror."""
    assert _as_sdk_sees_it(raw) == expected


@pytest.mark.parametrize(
    "shape", ['""', "''", '"  "', None, '"default"']
)
def test_every_refused_shape_is_outside_every_accepted_set(shape) -> None:
    accepted = _contract()["accepted_by_family"]
    seen = _as_sdk_sees_it(shape)
    for family, values in accepted.items():
        assert seen not in values, (
            f"{shape!r} reaches the SDK as {seen!r}, which IS in the {family} "
            "accepted set -- this contract file is stale"
        )


def test_every_sklearn_template_declares_a_trainable_model_type() -> None:
    """The regression this file exists for. A storable-but-untrainable value
    (notably ``\"\"``) fails here while passing the storable contract."""
    accepted = _contract()["accepted_by_family"]
    offenders = []
    unmapped = []
    for path, consts in _sklearn_templates():
        rel = path.relative_to(ZOO).as_posix()
        family = _FAMILY_BY_CATEGORY.get(consts.get("category"))
        if family is None:
            # A sklearn template outside the two tabular families. Collected
            # rather than skipped silently: a new sklearn category would
            # otherwise leave this guard passing while checking nothing about
            # it, which is the same shape as the defect this file exists for.
            unmapped.append(f"{rel}: category={consts.get('category')!r}")
            continue
        declared = consts.get("model_type", "<ABSENT>")
        canonical = (
            declared.strip().lower() if isinstance(declared, str) else declared
        )
        if canonical in ("", "<absent>", None) or canonical not in accepted[family]:
            offenders.append(f"{rel}: model_type={declared!r} (family={family})")
    assert not offenders, (
        "sklearn templates declaring a model_type the SDK's sklearn upload path "
        "refuses -- model_func_checks would fail and the template is untrainable "
        "end to end:\n  " + "\n  ".join(offenders)
    )
    assert not unmapped, (
        "sklearn template(s) in a category _FAMILY_BY_CATEGORY does not map, so "
        "this guard checked NOTHING for them. Add the category to the mapping "
        "(and to the contract) or state why it has no sklearn upload family:\n  "
        + "\n  ".join(unmapped)
    )
