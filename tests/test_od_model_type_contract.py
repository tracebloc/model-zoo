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
    # Floor lowered 10 -> 7 by backend#2973 (deleting the seven DETR templates),
    # then 7 -> 6 by backend#2988 (deleting the unusable mask_rcnn), leaving six
    # declaring templates (faster_rcnn_resnet, fcos, retinanet, and the three
    # yolo model.py entry points). The floor tracks the live tree, not a
    # historical high-water mark — left too high it would have failed on a
    # deletion that is the point of the change, which teaches people to lower it
    # reflexively and defeats the guard.
    #
    # It rose again with the roster work, as that note said it would: 6 -> 12
    # with backend#2982's Tier 0 (six torchvision builders), 12 -> 16 with its
    # Tier 1 (four modern-backbone detectors) and 16 -> 18 with its Tier 2
    # (yolox_s, rtmdet_s). Like the census in test_check_dump_coverage.py this
    # is a RUNNING TOTAL: a rebase re-counts the tree rather than keeping this
    # branch's literal.
    assert len(declared) >= 18, (
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
            " — the engine explicitly rejects this value; a template declares "
            "the FAMILY, not the architecture (retinanet/fcos -> "
            "torchvision_detection). Note hf_transformer is NOT a way out: the "
            "DETR family was retired in backend#2973 and is itself rejected."
        )
    assert normalized in ACCEPTED, (
        f"{path.relative_to(ROOT)}: model_type {model_type!r} is not in the "
        f"engine's accepted OD vocabulary {sorted(ACCEPTED)} (backend#1829){hint}"
    )
