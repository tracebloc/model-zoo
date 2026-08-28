"""Contract test: every object-detection template's ``model_type`` is a value
the training engine can route — and, by construction, a valid backend choice.

Background (backend#1829)
-------------------------
An OD experiment's ``model_type`` is resolved by the engine's
``resolve_family()`` (tracebloc-engine ``core/utils/object_detection_utils.py``),
which accepts exactly the family vocabulary published in
``object_detection_families.v1.json`` — ``torchvision_detection`` (+ its legacy
alias ``rcnn``), ``yolo``, ``hf_transformer`` — and raises ``ValueError`` on
anything else *before training starts*. Architecture-level names
(``detr``/``retinanet``/``fcos``/…) are NOT accepted: a template declares the
FAMILY, not the architecture.

Nothing crossed the repo boundary, so the zoo drifted: nine OD templates
declared architecture names (seven ``detr``, plus ``retinanet`` and ``fcos``)
that ``resolve_family`` rejects. Because the engine reads ``model_type`` from the
experiment parameters (the backend field), not from the model file, such a model
could only run mislabelled — routed onto the torchvision path where a DETR
forward dies opaquely inside ``_rcnn_loss_forward``.

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
    / "object_detection_families.v1.json"
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


def _od_model_files() -> list[pathlib.Path]:
    return sorted(OD_ROOT.rglob("*.py"))


def test_accepted_vocabulary_is_not_empty() -> None:
    """Guard the guard: an empty accepted set would make every membership
    assertion below vacuously pass."""
    assert ACCEPTED, f"{SCHEMA_PATH}: accepted_model_type_values is empty"


def test_od_templates_declaring_model_type_were_found() -> None:
    """The parametrized test skips support files and empty declarations; if that
    left it with nothing to check, an OD template rename could hide real drift
    behind an all-skipped run (the backend#1859 silent-green shape)."""
    declared = [
        p for p in _od_model_files() if (_read_model_type(p) or "") != ""
    ]
    assert len(declared) >= 10, (
        f"expected the OD templates to declare model_type, found {len(declared)} "
        f"under {OD_ROOT} — did the tree move?"
    )


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
            " — this is an ARCHITECTURE name the engine explicitly rejects; "
            "declare the FAMILY instead (e.g. detr-family -> hf_transformer, "
            "retinanet/fcos -> torchvision_detection)"
        )
    assert normalized in ACCEPTED, (
        f"{path.relative_to(ROOT)}: model_type {model_type!r} is not in the "
        f"engine's accepted OD vocabulary {sorted(ACCEPTED)} (backend#1829){hint}"
    )
