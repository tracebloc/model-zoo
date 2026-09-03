"""Contract test: every object-detection template's ``model_type`` is a value
the training engine can route — and, by construction, a valid backend choice.

Background (backend#1829)
-------------------------
An OD experiment's ``model_type`` is resolved by the engine's
``resolve_family()`` (tracebloc-engine ``core/utils/object_detection_utils.py``),
which accepts exactly the family vocabulary published in
``object_detection_families.v2.json`` — ``torchvision_detection`` (+ its legacy
alias ``rcnn``) and ``yolo`` — and raises ``ValueError`` on anything else
*before training starts*. Architecture-level names
(``detr``/``retinanet``/``fcos``/…) are NOT accepted: a template declares the
FAMILY, not the architecture.

Nothing crossed the repo boundary, so the zoo drifted: nine OD templates
declared architecture names that ``resolve_family`` rejects. Because the engine
reads ``model_type`` from the experiment parameters (the backend field), not from
the model file, such a model could only run mislabelled — routed onto the
torchvision path whose loss-dict contract it does not speak.

SCHEMA v2 — the DETR templates are GONE (backend#2973)
-------------------------------------------------------
The seven templates that declared ``hf_transformer`` (``detr``, ``rt_detr``,
``rt_detr_v2``, ``deformable_detr``, ``d_fine``, ``owlv2``, ``grounding_dino``)
were deleted with the family. The platform stopped supporting HuggingFace
models, and the engine handler that value routed to never trained anything — it
raised ``NotImplementedError`` at every entry point. This repo adopts the
narrowed vocabulary by bumping the vendored schema to v2, in the same commit as
the deletions: v2 lists ``hf_transformer`` under ``not_accepted.examples``, so a
surviving template declaring it would now fail here by name.

What this test pins
-------------------
Every ``model_type`` an OD template declares is in the engine's
``accepted_model_type_values`` — DERIVED from the vendored schema
(``tests/contracts/tracebloc_engine/``), not transcribed (a hand-copied list is
how the drift shipped). Matching mirrors ``resolve_family``: whitespace-stripped,
case-insensitive.

"AND a valid backend choice": the same ``accepted_model_type_values`` is, by the
backend's ``global_meta/tests/test_od_families_contract.py``, exactly the OD
subset of ``Experiment.MODEL_TYPE_CHOICES``. So a value that is engine-accepted
is thereby a storable backend ``model_type`` — the two producer-side checks,
each anchored to the one engine SSoT, together prove a template's declaration
resolves in the registry *and* is a valid backend choice, without either repo
importing the other.

Scope: object-detection only. ``model_type`` is a generic field used across task
types (the keypoint templates declare ``transformer``/``heatmap`` resolved by a
different engine path); those are out of scope for the OD registry.
"""

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parent.parent
OD_ROOT = ROOT / "model_zoo" / "object_detection"
SCHEMA_PATH = (
    pathlib.Path(__file__).parent
    / "contracts"
    / "tracebloc_engine"
    / "object_detection_families.v2.json"
)

_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
# resolve_family lowercases + strips before lookup, so normalize the accepted
# set the same way and compare normalized declarations against it.
ACCEPTED = frozenset(v.strip().lower() for v in _SCHEMA["accepted_model_type_values"])
NOT_ACCEPTED_EXAMPLES = frozenset(
    v.strip().lower() for v in _SCHEMA.get("not_accepted", {}).get("examples", [])
)


def _read_model_type(path: pathlib.Path) -> str | None:
    """Return the declared module-level ``model_type`` (possibly ``""``), or
    ``None`` if the file declares none. Read statically — no import — so the
    check runs in every CI job regardless of which framework is installed."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(r'^\s*model_type\s*=\s*["\'](\w*)["\']', text, re.MULTILINE)
    return match.group(1) if match else None


def _read_framework(path: pathlib.Path) -> str | None:
    """The module-level ``framework``, or ``None`` for a support module.

    Every model file declares it (the metadata contract in CLAUDE.md); the
    ``yolo_*/loss.py`` helpers do not. A SECOND, INDEPENDENT regex from
    ``_read_model_type`` on purpose — the guard below compares the two readers'
    verdicts, and one reader answering for both would make that comparison
    vacuous the moment it broke.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(r'^\s*framework\s*=\s*["\'](\w*)["\']', text, re.MULTILINE)
    return match.group(1) if match else None


def _od_model_files() -> list[pathlib.Path]:
    return sorted(OD_ROOT.rglob("*.py"))


def _od_templates() -> list[pathlib.Path]:
    """The OD files that are template entry points, not support modules."""
    return [p for p in _od_model_files() if _read_framework(p)]


def test_accepted_vocabulary_is_not_empty() -> None:
    """Guard the guard: an empty accepted set would make every membership
    assertion below vacuously pass."""
    assert ACCEPTED, f"{SCHEMA_PATH}: accepted_model_type_values is empty"


def test_od_templates_declaring_model_type_were_found() -> None:
    """The parametrized test skips support files and empty declarations; if that
    left it with nothing to check, an OD template rename could hide real drift
    behind an all-skipped run (the backend#1859 silent-green shape).

    THIS USED TO BE A FLOOR (``len(declared) >= 6``), hand-recomputed on every
    change: lowered 10 -> 7 by backend#2973's DETR deletions, 7 -> 6 by
    backend#2988's mask_rcnn deletion, with a comment telling the next author to
    raise it again for backend#2982's roster. A literal every roster PR edits is
    a serialisation point — the same cost the census literal had, measured on
    backend#2982 — and a floor that trails the tree is also a guard that has
    stopped guarding.

    It is now derived: EVERY OD TEMPLATE MUST DECLARE ``model_type``, and a
    template is a file declaring ``framework``. That is strictly stronger than
    the floor, because the parametrized test below SKIPS a file with no
    declaration as a "support file" — so an OD template that simply forgot
    ``model_type`` was silently uncovered at any floor value. Adding a template
    moves both sides at once; there is nothing to raise.
    """
    templates = _od_templates()
    assert templates, (
        f"no file under {OD_ROOT} declares `framework` — the scan lost the tree, "
        f"and every assertion in this file would pass by checking nothing"
    )
    undeclared = sorted(
        str(p.relative_to(ROOT)) for p in templates if _read_model_type(p) is None
    )
    assert not undeclared, (
        "OD template(s) declaring `framework` but no `model_type` — the "
        "parametrized test below SKIPS these as support files, so they are "
        "covered by nothing. A template declares the FAMILY it routes to "
        f"(torchvision_detection / yolo): {undeclared}"
    )
    declared = [p for p in templates if (_read_model_type(p) or "") != ""]
    assert declared, (
        f"every OD template under {OD_ROOT} declares an EMPTY `model_type`, so "
        f"the registry check below skips all of them — the vocabulary assertion "
        f"is exercising nothing"
    )


def test_the_two_readers_are_independent_and_discriminate(tmp_path) -> None:
    """Guard the guard above: it compares two readers' verdicts, and if both
    collapsed to "always None" the template list and the undeclared list would go
    empty together and it would pass on nothing."""
    support = tmp_path / "loss.py"
    support.write_text("import torch\n\n\ndef loss(a, b):\n    return a - b\n", "utf-8")
    assert _read_framework(support) is None
    assert _read_model_type(support) is None

    template = tmp_path / "model.py"
    template.write_text('framework = "pytorch"\nmodel_type = "yolo"\n', "utf-8")
    assert _read_framework(template) == "pytorch"
    assert _read_model_type(template) == "yolo"


@pytest.mark.parametrize(
    "path", _od_model_files(), ids=lambda p: str(p.relative_to(ROOT))
)
def test_od_model_type_resolves_in_engine_registry(path: pathlib.Path) -> None:
    model_type = _read_model_type(path)
    if model_type is None:
        pytest.skip("support file (no `model_type` declaration)")
    if model_type == "":
        # An empty value is defaulted upstream by the engine's _infer_model_type
        # (to rcnn, or yolo by name) before resolve_family — not rejected. See
        # the schema's matching.empty_or_absent.
        pytest.skip("empty `model_type` — inferred upstream, not resolved here")

    normalized = model_type.strip().lower()
    hint = ""
    if normalized in NOT_ACCEPTED_EXAMPLES:
        hint = (
            " — the engine explicitly rejects this value; a template declares "
            "the FAMILY, not the architecture (retinanet/fcos -> "
            "torchvision_detection). Note hf_transformer is NOT a way out: the "
            "DETR family was retired in backend#2973 and is itself rejected."
        )
    assert normalized in ACCEPTED, (
        f"{path.relative_to(ROOT)}: model_type {model_type!r} is not in the "
        f"engine's accepted OD vocabulary {sorted(ACCEPTED)} (backend#1829){hint}"
    )
