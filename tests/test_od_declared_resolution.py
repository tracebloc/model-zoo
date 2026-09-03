"""Every OD template's declared ``image_size`` must be the resolution it runs at
(backend#3058, filed from backend#2982).

The defect this pins
--------------------
``image_size`` is not decoration: the SDK hands it to the edge to size the
dataset. A torchvision detector then applies its own
``GeneralizedRCNNTransform``, which **upscales anything below ``min_size``
straight back to ``min_size``**. So a template declaring 448 while its transform
runs at 800 produces this pipeline:

    dataset delivers 448x448  ->  transform upscales to 800  ->  model trains

The model trains on 448-resolution content stretched to 800. It pays the resize
twice and throws away the detail it would have had if the dataset had delivered
800 natively. Nothing errors, nothing warns, and the only symptom is accuracy
that is worse than the architecture should give — which is indistinguishable
from "detection is hard".

Three shipped templates have it (``faster_rcnn_resnet``, ``fcos``,
``retinanet``, all 448 against a transform at 800). **They are deliberately not
fixed here.** Changing a declared shape changes what the edge is asked to
deliver, so backend#3058 wants a before/after mAP measurement rather than a
blind edit. This file is the guard, landing first: it stops a *new* template
acquiring the same defect, and it makes the three known ones explicit instead of
folklore.

Asserted in both directions
---------------------------
``KNOWN_MISMATCHES`` is not a skip list. A template in it that has been *fixed*
fails too, with an instruction to delete its row. That is the RFC's
``EXPECTED_RED`` discipline: a list that quietly tolerates a fixed entry decays
into a list nobody trusts, and the next person cannot tell which rows are real.

Scope: the ``torchvision_detection`` family only. The ``yolo`` family speaks a
different contract — a hard 7x7 grid at ``YOLO_DATA_SHAPE = 448`` — where 448 is
the correct declared value and there is no ``transform`` to compare against.
"""

import importlib.util
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parent.parent
OD_ROOT = ROOT / "model_zoo" / "object_detection"
CONTRACTS = pathlib.Path(__file__).parent / "contracts" / "tracebloc_engine"

FAMILY = "torchvision_detection"

#: Templates whose declared ``image_size`` is known NOT to match the resolution
#: they run at, tracked as backend#3058. Value is the declared/effective pair, so
#: a partial change is as loud as no change.
#:
#: ⚠️ Asserted in BOTH directions — fixing one of these fails this file until
#: its row is deleted. Do not add a row to silence a new template; a new
#: mismatch is a bug in that template, not a new known issue.
KNOWN_MISMATCHES = {
    "faster_rcnn_resnet": (448, 800),
    "fcos": (448, 800),
    "retinanet": (448, 800),
}

#: The list is a RATCHET: it may only ever shrink, and its length is pinned by
#: EQUALITY rather than by an upper bound.
#:
#: Added after a fail-ability sweep, then tightened after review. Four scenarios
#: were checked first: a new template acquiring the defect, a listed one being
#: fixed but left on the list, a listed one changing to a different wrong value
#: (all caught), and adding a row for a newly-broken template (SURVIVED). An
#: exception list that can grow is not a guard, it is a habit, so the count was
#: pinned.
#:
#: ⚠️ Pinning it with `<=` was NOT sufficient, and the hole is worth naming
#: because the weaker version reads as correct. `<=` blocks growth above the
#: high-water mark but not RE-GROWTH after a fix: fix a template, delete its row,
#: and the length drops to 2 while the cap stays 3 — both assertions still pass,
#: and a later commit can drop a brand-new mismatch into the freed slot and stay
#: green. That is exactly the evasion the ratchet exists to stop. The sweep
#: missed it because it tested ADDING a row above the cap and never
#: DELETE-then-RE-ADD.
#:
#: With equality, deleting a row forces this number down in the same commit, and
#: the `MAX_KNOWN_MISMATCHES == <n>` pin below keeps that a conscious,
#: reviewable edit rather than a silent one. The legal edit is therefore
#: *enforced* rather than merely documented.
MAX_KNOWN_MISMATCHES = 3


def _schema_path() -> pathlib.Path:
    paths = sorted(
        CONTRACTS.glob("object_detection_families.v*.json"),
        key=lambda p: int(re.search(r"\.v(\d+)\.json$", p.name).group(1)),
    )
    assert paths, f"no vendored OD families schema under {CONTRACTS}"
    return paths[-1]


def _family_values() -> frozenset[str]:
    """The family name plus every alias.

    Resolved by family, never by the literal ``"torchvision_detection"``:
    ``faster_rcnn_resnet.py`` declares the legacy ``rcnn`` alias, and it is one
    of the three templates this file exists to pin — so a literal-keyed scan
    would silently drop the most important row.
    """
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    entries = [f for f in schema["families"] if f["family"] == FAMILY]
    assert entries, f"{_schema_path().name}: no {FAMILY!r} family entry"
    return frozenset(v.strip().lower() for v in {FAMILY, *entries[0].get("aliases", [])})


FAMILY_VALUES = _family_values()


def _read_model_type(path: pathlib.Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(r'^\s*model_type\s*=\s*["\'](\w*)["\']', text, re.MULTILINE)
    return match.group(1) if match else None


def _other_family_values() -> frozenset[str]:
    """Accepted values routing to a family that is NOT this one.

    The schema says object detection has exactly two families, so this plus
    ``FAMILY_VALUES`` partitions the vocabulary — which is what lets the guard
    below assert COVERAGE without a floor to recompute (backend#2982).
    """
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    values: set[str] = set()
    for entry in schema["families"]:
        if entry["family"] == FAMILY:
            continue
        values |= {entry["family"], *entry.get("aliases", [])}
    return frozenset(v.strip().lower() for v in values)


OTHER_FAMILY_VALUES = _other_family_values()


def _declares_framework(path: pathlib.Path) -> str | None:
    """The module-level ``framework``, or ``None`` for a support module.

    A SECOND, INDEPENDENT regex from ``_read_model_type``: the partition below
    compares the two readers' verdicts, and one reader answering for both would
    make that comparison vacuous the moment it broke.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(r'^\s*framework\s*=\s*["\'](\w*)["\']', text, re.MULTILINE)
    return match.group(1) if match else None


def _od_templates() -> list[pathlib.Path]:
    return [p for p in sorted(OD_ROOT.rglob("*.py")) if _declares_framework(p)]


FAMILY_TEMPLATES = [
    p for p in sorted(OD_ROOT.rglob("*.py"))
    if (_read_model_type(p) or "").strip().lower() in FAMILY_VALUES
]


def _build(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(
        re.sub(r"\W", "_", f"resolution_{path.stem}"), path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = getattr(module, "main_class", None) or getattr(module, "main_method", None)
    assert entry, f"{path}: neither main_class nor main_method is defined"
    return module, getattr(module, entry)(3)


def _effective_resolution(model):
    """What the model's own transform actually resizes to.

    Two shapes: the SSDs are genuinely fixed-size (``fixed_size=(300, 300)``),
    while the R-CNN/RetinaNet/FCOS family uses ``min_size``/``max_size`` and
    upscales below the minimum.
    """
    transform = getattr(model, "transform", None)
    if transform is None:
        return None
    fixed = getattr(transform, "fixed_size", None)
    if fixed:
        return int(fixed[0])
    min_size = transform.min_size
    if isinstance(min_size, (list, tuple)):
        return int(min_size[0])
    return int(min_size)


def test_family_templates_were_found():
    """Guard the guard: an empty scan would make this file pass by checking
    nothing, and it is driven by a file scan plus a schema lookup."""
    assert "rcnn" in FAMILY_VALUES, (
        f"{_schema_path().name}: {FAMILY!r} lost its legacy 'rcnn' alias — "
        f"faster_rcnn_resnet declares it and is one of the KNOWN_MISMATCHES, so "
        f"this scan just stopped covering the row that matters most"
    )
    # This was `len(FAMILY_TEMPLATES) >= 4` — a floor every roster PR is
    # invited to raise, i.e. a shared literal with the same serialisation cost
    # the census literal had (backend#2982). It is a PARTITION now: the schema
    # publishes exactly two OD families, so every OD template belongs to this
    # one or to `yolo`, and adding a template moves both sides at once.
    templates = _od_templates()
    assert templates, (
        f"no file under {OD_ROOT} declares `framework` — the scan lost the tree, "
        f"and this file would pass by checking nothing"
    )
    other = {
        p
        for p in templates
        if (_read_model_type(p) or "").strip().lower() in OTHER_FAMILY_VALUES
    }
    assert other, (
        f"no OD template routes to a family other than {FAMILY!r}; the yolo "
        f"roster is part of the tree, so this means model_type reading broke"
    )
    uncovered = sorted(
        str(p.relative_to(ROOT)) for p in set(templates) - other - set(FAMILY_TEMPLATES)
    )
    assert not uncovered, (
        f"OD template(s) in neither the {FAMILY!r} roster this file checks nor "
        f"any other family — they declare a model_type outside the schema's "
        f"vocabulary, or none at all, so no resolution check reaches them: "
        f"{uncovered}"
    )
    unexpected = sorted(
        str(p.relative_to(ROOT)) for p in set(FAMILY_TEMPLATES) - set(templates)
    )
    assert not unexpected, (
        f"the {FAMILY!r} roster contains files that do not declare `framework` — "
        f"a support module cannot be a template: {unexpected}"
    )
    missing = set(KNOWN_MISMATCHES) - {p.stem for p in FAMILY_TEMPLATES}
    assert not missing, (
        f"KNOWN_MISMATCHES names templates the scan did not find: {sorted(missing)} "
        f"— if they were deleted, delete their rows too"
    )


@pytest.mark.parametrize(
    "path", FAMILY_TEMPLATES, ids=lambda p: str(p.relative_to(ROOT))
)
def test_declared_image_size_is_the_resolution_the_model_runs_at(path):
    torch = pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")
    del torch

    module, model = _build(path)
    declared = int(module.image_size)
    effective = _effective_resolution(model)
    assert effective is not None, (
        f"{path.name}: no .transform to read an effective resolution from — if "
        f"this template resizes some other way, this guard needs to learn about it"
    )

    known = KNOWN_MISMATCHES.get(path.stem)
    if known is None:
        assert declared == effective, (
            f"{path.name} declares image_size = {declared} but its transform "
            f"runs at {effective}. The SDK hands image_size to the edge to size "
            f"the dataset, and the transform then rescales to {effective} — so "
            f"the model trains on {declared}-resolution content resized to "
            f"{effective}, paying the resize twice and losing detail. Declare "
            f"{effective}, or set the transform's min_size/fixed_size to "
            f"{declared} if that is really the intent. See backend#3058."
        )
        return

    # A known mismatch: assert it is STILL exactly what was recorded.
    assert (declared, effective) == known, (
        f"{path.name} is listed in KNOWN_MISMATCHES as declared={known[0]} / "
        f"effective={known[1]}, but is now declared={declared} / "
        f"effective={effective}.\n"
        f"  - If this was FIXED (declared == effective), delete its row from "
        f"KNOWN_MISMATCHES; the guard then holds it correct forever.\n"
        f"  - If it changed some other way, backend#3058 needs updating before "
        f"this row does."
    )


def test_known_mismatches_are_all_still_mismatched():
    """The other direction, stated once rather than per template.

    A row that has quietly become correct makes the whole list untrustworthy —
    the next reader cannot tell which entries are real defects and which are
    stale. This fails loudly with an instruction to prune.
    """
    pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")

    stale = []
    for stem, (recorded_declared, recorded_effective) in sorted(KNOWN_MISMATCHES.items()):
        matches = [p for p in FAMILY_TEMPLATES if p.stem == stem]
        assert matches, f"KNOWN_MISMATCHES row {stem!r} matches no template"
        module, model = _build(matches[0])
        declared = int(module.image_size)
        effective = _effective_resolution(model)
        if declared == effective:
            stale.append(f"{stem} (now consistently {declared})")
        else:
            assert (declared, effective) == (recorded_declared, recorded_effective), (
                f"{stem}: recorded {recorded_declared}/{recorded_effective}, "
                f"found {declared}/{effective}"
            )

    assert not stale, (
        f"these KNOWN_MISMATCHES entries are no longer mismatched: "
        f"{', '.join(stale)}. Delete their rows — the guard will then hold them "
        f"correct, which is the point. Leaving a fixed row in place makes the "
        f"list decay into folklore. See backend#3058."
    )


def test_the_known_mismatch_list_only_ever_shrinks():
    """The ratchet.

    Without this, the cheapest way to make a new declared/effective mismatch go
    green is to add its name to ``KNOWN_MISMATCHES`` — which is exactly the
    failure the list exists to prevent, performed on the list itself. Verified
    by mutation: adding a row for a newly-broken template silenced every other
    assertion in this file.

    Asserted by EQUALITY rather than ``<=``. The weaker form reads as correct
    and is not: it blocks growth above the high-water mark but not *re-growth
    after a fix*. Fix a template, delete its row, and the length drops below the
    cap — both a ``<=`` bound and the ``MAX == n`` pin still pass, leaving a free
    slot a later commit can refill with a brand-new mismatch and stay green.

    Legal edits are: delete a row, lower ``MAX_KNOWN_MISMATCHES``, and update
    the pin below — all in one commit. Adding a row fails here, so a new
    mismatch has to be argued for rather than absorbed.
    """
    assert len(KNOWN_MISMATCHES) == MAX_KNOWN_MISMATCHES, (
        f"KNOWN_MISMATCHES holds {len(KNOWN_MISMATCHES)} entries "
        f"({sorted(KNOWN_MISMATCHES)}) against a pinned {MAX_KNOWN_MISMATCHES}.\n"
        f"  - GREW? A declared/effective mismatch in a NEW template is a bug in "
        f"that template — fix its image_size instead of listing it.\n"
        f"  - SHRANK? Good: backend#3058 fixed one. Lower MAX_KNOWN_MISMATCHES "
        f"to {len(KNOWN_MISMATCHES)} in this same commit, and update the "
        f"equality pin below.\n"
        f"This is asserted by EQUALITY, not `<=`, on purpose: an upper bound "
        f"would let a fix free a slot that a later commit could quietly refill "
        f"with a brand-new mismatch."
    )
    assert MAX_KNOWN_MISMATCHES == 3, (
        f"MAX_KNOWN_MISMATCHES is {MAX_KNOWN_MISMATCHES}, not the 3 recorded "
        f"when this guard landed. Raising it defeats the ratchet; lowering it "
        f"is correct once backend#3058 fixes a template, and this assertion "
        f"should be updated in the same commit that lowers it."
    )



def test_the_two_readers_are_independent_and_discriminate(tmp_path):
    """Guard the guard above: the partition compares two readers' verdicts, and
    if both collapsed to "always None" the roster and the expected roster would
    go empty together and it would pass on nothing (backend#2982)."""
    support = tmp_path / "loss.py"
    support.write_text("import torch\n\n\ndef loss(a, b):\n    return a - b\n", "utf-8")
    assert _declares_framework(support) is None
    assert _read_model_type(support) is None

    template = tmp_path / "model.py"
    template.write_text(
        'framework = "pytorch"\nmodel_type = "torchvision_detection"\n', "utf-8"
    )
    assert _declares_framework(template) == "pytorch"
    assert _read_model_type(template) == "torchvision_detection"
