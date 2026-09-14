"""Every template's declared ``model_type`` is one the PLATFORM accepts
(model-zoo#273).

The defect this pins
--------------------
``metaApi.models.Experiment.model_type`` is a Django ``CharField`` with
``choices=MODEL_TYPE_CHOICES``. A value outside that set is refused by the
ChoiceField when the experiment is created — a **400 before the model file is
looked at**, with nothing in the error naming the template as the cause. So a
template declaring an unaccepted value is not degraded, it is *unusable*, and
the user is told nothing that would let them work it out.

Three keypoint templates shipped that way: ``sapiens``, ``vitpose`` and
``vitpose_plus`` all declared ``model_type = "transformer"``, which has never
been a member of ``MODEL_TYPE_CHOICES``. Nothing in either repo compared the
zoo's declarations against the platform's choice set, so all three shipped and
stayed broken.

Why this is a THIRD file and not an extension of the OD one
-----------------------------------------------------------
``test_od_model_type_contract.py`` answers a narrower question against a
different source of truth: whether an OD template's value resolves in the
ENGINE's OD family registry (``torchvision_detection`` / ``rcnn`` / ``yolo``).
Its own docstring scopes keypoint out by name. That vocabulary is a *subset* of
this one, and the two questions are genuinely different:

* engine-routable (OD) — 3 values, ``object_detection_families.v2.json``
* backend-storable (all task types) — 16 values, this file's contract

An OD template must satisfy BOTH, and ``test_od_accepted_values_are_a_subset``
below asserts the containment so the two cannot drift into contradiction.
A keypoint template has no published routing schema, so storable is the
strongest statement this repo can make about it — which is exactly the
statement model-zoo#273 needed and nobody was making.

The accepted set is DERIVED, not transcribed
--------------------------------------------
``tests/contracts/tracebloc_backend/model_type_choices.v1.json`` is generated
from the backend's ``MODEL_TYPE_CHOICES`` by ``ast``, at a pinned ref, with the
generator recorded in the sibling README. A hand-copied list is how the OD
vocabulary drift shipped in the first place (the OD vocabulary-drift incident), and it would rot
here the same way.

Note what the refresh caught, because it is the reason the pin is recorded: a
locally stale backend checkout still offered ``hf_transformer``, which
a later canonicaliser change has since REMOVED. Generating from a stale tree would have
vendored a 17-value set that accepts a value the platform now refuses.

Reading is by AST, and the reason is specific
---------------------------------------------
All three of the templates this ticket fixes carry a ``CONFIG`` dict holding a
nested ``"model_type"`` key — the *HuggingFace architecture id*
(``"vitpose"``, ``"vit_mae"``), which has nothing to do with the platform
field. A text scan for ``model_type`` finds four hits in ``vitpose.py`` and
only one of them is the declaration.

``_declared_model_type`` therefore walks ``tree.body`` — MODULE LEVEL ONLY,
not ``ast.walk`` — so a nested dict entry, a class attribute and a local
variable are all structurally out of reach rather than excluded by a pattern
that has to be got right. That is also what makes the match whole-identifier by
construction: an AST target name is the whole name, so no substring can match
and there is no word-boundary regex to get subtly wrong.

A declaration that is present but NOT a plain string literal (an f-string, a
concatenation, a name) is a hard failure rather than a ``None``: this file
cannot say what such a template declares, and "cannot say" must not read as
"fine".
"""

from __future__ import annotations

import ast
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parent.parent
MODEL_ROOT = ROOT / "model_zoo"

CONTRACT_PATH = (
    pathlib.Path(__file__).parent
    / "contracts"
    / "tracebloc_backend"
    / "model_type_choices.v1.json"
)
OD_SCHEMA_PATH = (
    pathlib.Path(__file__).parent
    / "contracts"
    / "tracebloc_engine"
    / "object_detection_families.v2.json"
)

_CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

#: The storable vocabulary. Normalised the way the engine's resolvers do
#: (strip + lowercase) so a declaration is compared on the same footing; every
#: value in the contract is already lowercase, so this is no weaker.
ACCEPTED = frozenset(v.strip().lower() for v in _CONTRACT["accepted_model_type_values"])

#: ``""`` is STORED, not refused — the field is ``blank=True`` and
#: ``ExperimentSerializer.validate_model_type`` coerces falsy to ``default``.
#: Most templates in this zoo declare it. Kept as a contract flag rather than a
#: constant here so the vendored file stays the single source of that fact.
BLANK_ACCEPTED = bool(_CONTRACT["blank_is_accepted"])

#: Values a producer has actually shipped that the ChoiceField refuses, so a
#: failure can name the value's history instead of reporting an anonymous miss.
NOT_ACCEPTED_EXAMPLES = frozenset(
    v.strip().lower() for v in _CONTRACT.get("not_accepted", {}).get("examples", [])
)
NOT_ACCEPTED_NOTES = {
    k.strip().lower(): v
    for k, v in _CONTRACT.get("not_accepted", {}).get("notes", {}).items()
}

#: A COLLAPSE DETECTOR, not a census. See
#: ``test_the_zoo_roster_was_found`` for why this is a loose floor and the
#: exact-count assertion is derived instead of written down.
_MIN_ZOO_TEMPLATES = 100

#: The same, for the templates that actually declare a ``model_type`` — the set
#: this file's rule is about. 164 of 198 do today; this fires only if the AST
#: reader stops reading, which ``violations == []`` cannot see because an
#: all-``None`` result is indistinguishable from an all-clean one.
_MIN_DECLARING = 120

#: Task types where NO template declares a ``model_type``, with the reason.
#:
#: This is not an exemption from the rule — a template here that DID declare
#: one would still have to declare an accepted value. It records that the
#: absence is a categorical convention rather than 33 individual omissions, so
#: that a routing-sensitive template silently LOSING its declaration is loud
#: instead of joining a crowd.
#:
#: ⚠️ Asserted in BOTH directions by ``test_undeclared_model_types_are_confined``:
#: a listed category that acquires a declaration fails (delete its row), and an
#: unlisted category with an undeclared template fails rather than passing
#: quietly.
CATEGORIES_WITHOUT_MODEL_TYPE = {
    "image_classification": (
        "no family routing exists for image classification — the engine picks "
        "its strategy from the category alone, and all 20 templates are "
        "consistent in declaring nothing."
    ),
    "semantic_segmentation": (
        "same as image_classification: no per-family routing, and all 13 "
        "templates are consistent."
    ),
}

#: Individual templates that declare no ``model_type`` while their category's
#: siblings all do, with the reason. NOT a category-wide convention, so it is
#: recorded per file and kept deliberately small.
#:
#: Keyed on the ``model_zoo/``-relative posix path, never the stem: zoo-wide,
#: stems collide across categories (``tools/seed_index.py`` documents the trap
#: — ``cnn``, ``mlp``, ``lstm``, ``hrnet`` and a dozen more name two or three
#: templates each), so a stem-keyed row would exempt the wrong file.
UNDECLARED_EXCEPTIONS = {
    "time_series_forecasting/pytorch/patch_tsmixer.py": (
        "the other 11 time_series_forecasting templates all declare "
        "`model_type = \"\"` and this one declares nothing. Harmless today — "
        "the backend field defaults to `default` and time-series has no family "
        "routing — but it is an inconsistency rather than a convention, and "
        "adding the declaration is outside the scope of model-zoo#273 (which "
        "is about declarations that are WRONG, not absent). Recorded here so "
        "the both-directions check above stays honest; filed separately."
    ),
}


def _declared_model_type(path: pathlib.Path) -> str | None:
    """The MODULE-LEVEL ``model_type`` string, or ``None`` if none is declared.

    Walks ``tree.body`` only — see the module docstring. A syntax error
    propagates: an unparseable template is a loud failure, never a skip.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: str | None = None
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "model_type" for t in targets):
            continue
        value = node.value
        assert value is not None, (
            f"{_rel(path)}: `model_type` is annotated but never "
            f"assigned, so nothing declares what this template routes as"
        )
        assert isinstance(value, ast.Constant) and isinstance(value.value, str), (
            f"{_rel(path)}: `model_type` is not a plain string "
            f"literal (found {type(value).__name__}). This file cannot say what "
            f"the template declares, and the platform's ChoiceField compares a "
            f"literal string — declare one."
        )
        found = value.value
    return found


def _declares_framework(path: pathlib.Path) -> str | None:
    """The module-level ``framework``, or ``None`` for a support module.

    A SECOND, INDEPENDENT reader from ``_declared_model_type`` on purpose, and
    a regex rather than a second AST walk so the independence is real: the
    roster assertion below compares the two readers' verdicts, and one
    implementation answering for both would make that comparison vacuous the
    moment it broke. Same argument, and same regex, as ``tests/_od.py``.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(r'^\s*framework\s*=\s*["\'](\w*)["\']', text, re.MULTILINE)
    return match.group(1) if match else None


def _model_files() -> list[pathlib.Path]:
    return sorted(MODEL_ROOT.rglob("*.py"))


#: A template is a file declaring ``framework``; the ``yolo_*/loss.py`` helpers
#: and other support modules declare none.
TEMPLATES = [p for p in _model_files() if _declares_framework(p)]


def _rel(path: pathlib.Path) -> str:
    """Repo-relative path for a message, degrading to the absolute path.

    The readers are driven against ``tmp_path`` fixtures in this file's
    guard-the-guard tests, and ``Path.relative_to`` RAISES on a path outside
    the repo — so formatting a failure message this way turned a clean
    assertion into a confusing ``ValueError`` from ``pathlib``.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def test_the_accepted_vocabulary_is_not_empty() -> None:
    """Guard the guard: an empty accepted set makes every membership assertion
    below vacuously true, which is the failure mode this whole file exists to
    remove."""
    assert ACCEPTED, (
        f"{CONTRACT_PATH}: accepted_model_type_values is empty — every "
        f"membership assertion in this file would pass by checking nothing"
    )
    assert len(ACCEPTED) == len(_CONTRACT["accepted_model_type_values"]), (
        f"{CONTRACT_PATH}: accepted_model_type_values contains duplicates "
        f"(or values differing only by case/whitespace), so the vendored "
        f"count overstates the vocabulary"
    )


def test_the_contract_and_its_choice_table_agree() -> None:
    """The vendored file carries both a flat ``accepted_model_type_values`` and
    the richer ``choices`` table it was generated from. They are two
    serialisations of one ``ast`` pass, so a hand-edit to either that did not
    touch the other is a corrupted contract — and the flat list is the one this
    file reads, so the corruption would be silent."""
    from_choices = frozenset(c["value"].strip().lower() for c in _CONTRACT["choices"])
    assert from_choices == ACCEPTED, (
        f"{CONTRACT_PATH} disagrees with itself: `choices` yields "
        f"{sorted(from_choices)} but `accepted_model_type_values` says "
        f"{sorted(ACCEPTED)}. Regenerate the file — see the sibling README."
    )


def test_the_zoo_roster_was_found() -> None:
    """Guard the guard: this file is driven by a directory walk, and a walk that
    visits nothing passes exactly as quietly as one that finds nothing wrong.
    That is how the sibling defect in a sibling guard defect survived — a guard reading as
    roster-complete while the files it was about sat outside its scan.

    THE EXACT COUNT IS DERIVED, NOT WRITTEN DOWN, and that is deliberate.
    ``test_od_model_type_contract.py`` made this argument first and it applies
    unchanged: a literal every roster PR has to bump is a serialisation point,
    and a floor that trails the tree is a guard that has stopped guarding. So
    the exact assertion is an EQUALITY between two independently-derived counts
    — every file declaring ``framework`` also declares ``model_type``, checked
    by a different reader — which is strictly stronger than any literal,
    because it moves on its own when the roster does.

    ``_MIN_ZOO_TEMPLATES`` is kept alongside it as a pure collapse detector: it
    is far below the real count and only fires if the walk loses the tree
    entirely (a rename, a moved root, a bad ``rglob``), the one failure the
    derived equality cannot see because both sides would go empty together.
    """
    assert TEMPLATES, (
        f"no file under {MODEL_ROOT} declares `framework` — the walk lost the "
        f"tree, and every assertion in this file would pass on an empty roster"
    )
    assert len(TEMPLATES) >= _MIN_ZOO_TEMPLATES, (
        f"the zoo walk found only {len(TEMPLATES)} template(s) under "
        f"{MODEL_ROOT}, below the {_MIN_ZOO_TEMPLATES} collapse floor. Either "
        f"the tree moved (fix the walk) or the zoo really did shrink that far "
        f"(lower the floor deliberately, in the same commit as the deletions)."
    )
    declaring = [p for p in TEMPLATES if _declared_model_type(p) is not None]
    assert len(declaring) >= _MIN_DECLARING, (
        f"only {len(declaring)} of {len(TEMPLATES)} templates yielded a "
        f"`model_type` declaration, below the {_MIN_DECLARING} floor. The rule "
        f"below asserts a list of violations is empty, and an all-`None` read "
        f"produces exactly that — so this is the assertion standing between a "
        f"broken reader and a green run."
    )


def test_undeclared_model_types_are_confined() -> None:
    """A template with NO ``model_type`` is covered by nothing in this file, so
    where that is allowed is pinned rather than left to accumulate.

    Both directions, because a one-way list rots. 34 of 198 templates declare
    nothing: two whole categories by convention, plus one recorded outlier. The
    risk being managed is a routing-sensitive template — an OD or keypoint one,
    where ``model_type`` decides which engine path runs — silently losing its
    declaration and being invisible here rather than red.
    """
    by_category: dict[str, list[str]] = {}
    for path in TEMPLATES:
        relative = path.relative_to(MODEL_ROOT).as_posix()
        category = path.relative_to(MODEL_ROOT).parts[0]
        if _declared_model_type(path) is None:
            by_category.setdefault(category, []).append(relative)

    unexpected: list[str] = []
    for category, files in sorted(by_category.items()):
        if category in CATEGORIES_WITHOUT_MODEL_TYPE:
            continue
        unexpected.extend(f for f in files if f not in UNDECLARED_EXCEPTIONS)
    assert not unexpected, (
        f"template(s) declare `framework` but no module-level `model_type`, in "
        f"a category where their siblings do: {sorted(unexpected)}.\n"
        f"If `model_type` decides routing for this category — object_detection "
        f"and keypoint_detection both — this is a real gap: the engine's "
        f"keypoint `_infer_model_type` falls an absent declaration through to "
        f"an `RCNN_FAMILY` fallback, which mis-routes anything that is not an "
        f"R-CNN. Declare the family. If the category genuinely has no routing, "
        f"add it to CATEGORIES_WITHOUT_MODEL_TYPE with the reason."
    )

    # The other direction: a listed category that now declares one, and a
    # recorded per-file exception that has been fixed, are both stale rows.
    for category, reason in sorted(CATEGORIES_WITHOUT_MODEL_TYPE.items()):
        in_category = [
            p for p in TEMPLATES if p.relative_to(MODEL_ROOT).parts[0] == category
        ]
        assert in_category, (
            f"CATEGORIES_WITHOUT_MODEL_TYPE names {category!r} but the walk "
            f"found no template in it — if the category was deleted or "
            f"renamed, delete its row too. Recorded reason: {reason}"
        )
        declaring = sorted(
            p.relative_to(MODEL_ROOT).as_posix()
            for p in in_category
            if _declared_model_type(p) is not None
        )
        assert not declaring, (
            f"{category!r} is listed in CATEGORIES_WITHOUT_MODEL_TYPE as "
            f"declaring none, but {declaring} now do. Delete its row — the "
            f"convention has changed, and a stale row would hide the next "
            f"template in this category that forgets a declaration. Recorded "
            f"reason: {reason}"
        )

    stale = sorted(
        key
        for key in UNDECLARED_EXCEPTIONS
        if _declared_model_type(MODEL_ROOT / key) is not None
    )
    assert not stale, (
        f"{stale} are listed in UNDECLARED_EXCEPTIONS but now declare a "
        f"`model_type`. Delete their rows in this commit — a list that "
        f"tolerates a fixed entry decays into folklore nobody can audit."
    )
    missing = sorted(
        key for key in UNDECLARED_EXCEPTIONS if not (MODEL_ROOT / key).is_file()
    )
    assert not missing, (
        f"UNDECLARED_EXCEPTIONS names files the walk did not find: {missing} — "
        f"if they were deleted or renamed, delete their rows too"
    )


def test_every_template_declares_a_storable_model_type() -> None:
    """The rule, asserted as one aggregate rather than parametrized per file.

    Aggregated on purpose: the useful failure for a vocabulary drift is the
    whole list of offenders at once (all three of model-zoo#273's templates in
    one message), not three separate reds that each look like an isolated typo.

    Both halves are asserted — the violations are empty AND the scan actually
    examined every template. ``len(scanned) == len(TEMPLATES)`` is the count
    assertion: a walk that silently visited zero files, or a reader that
    silently returned ``None`` for all of them, would satisfy
    ``violations == []`` perfectly.
    """
    scanned: list[str] = []
    declared_count = 0
    violations: list[tuple[str, str]] = []
    for path in TEMPLATES:
        declared = _declared_model_type(path)
        scanned.append(_rel(path))
        if declared is None:
            # Covered by test_undeclared_model_types_are_confined, which pins
            # WHERE this is allowed in both directions. Not silently ignored:
            # the declared_count assertion below is what stops an all-`None`
            # read from satisfying this test by examining nothing.
            continue
        declared_count += 1
        normalized = declared.strip().lower()
        if normalized == "":
            assert BLANK_ACCEPTED, (
                f"{_rel(path)}: declares an empty `model_type`, and the "
                f"vendored contract says blank is NOT accepted"
            )
            continue
        if normalized not in ACCEPTED:
            violations.append((_rel(path), declared))

    assert len(scanned) == len(TEMPLATES), (
        f"the scan examined {len(scanned)} template(s) but the roster holds "
        f"{len(TEMPLATES)} — the loop skipped files, so `violations` is an "
        f"answer about a subset of the zoo and says nothing about the rest. "
        f"A roster walk that silently visits fewer files than it found passes "
        f"as quietly as a clean one; that is how a sibling guard defect happened."
    )
    assert declared_count >= _MIN_DECLARING, (
        f"only {declared_count} of {len(scanned)} scanned templates yielded a "
        f"declaration, below the {_MIN_DECLARING} floor. `violations == []` is "
        f"satisfied just as well by a reader that returns `None` for "
        f"everything, so this is the half that proves the rule was applied."
    )

    hints = []
    for rel, declared in violations:
        note = NOT_ACCEPTED_NOTES.get(declared.strip().lower())
        hints.append(f"  - {rel}: model_type={declared!r}" + (f"\n      {note}" if note else ""))
    assert not violations, (
        f"{len(violations)} template(s) declare a `model_type` the platform "
        f"does not accept. `Experiment.model_type` is a ChoiceField, so an "
        f"experiment built from one of these is refused with a 400 BEFORE the "
        f"model is looked at, and the error names nothing that would point at "
        f"the template (model-zoo#273).\n"
        + "\n".join(hints)
        + f"\n\nAccepted values ({len(ACCEPTED)}): {sorted(ACCEPTED)}\n"
        f"Pick the accepted value that matches what the model EMITS — for "
        f"keypoint, `heatmap` routes to the heatmap step path, `rcnn` to the "
        f"detection path, and anything else (`direct`) to direct coordinate "
        f"regression. Do NOT widen the vocabulary here to make a declaration "
        f"fit: this file is generated from the backend's choice set, and "
        f"editing it green would only move the 400 out of CI and back onto a "
        f"user."
    )


def test_the_transformer_regression_stays_caught() -> None:
    """The specific value model-zoo#273 was filed for.

    Pinned by name because "not in a 16-element set" is a weak thing to trust
    on its own: a future contract refresh that ADDED ``transformer`` would make
    the rule above green on the three templates this ticket fixed, and nothing
    else would notice. If the platform ever genuinely accepts it, this
    assertion is the deliberate place that decision has to be taken.
    """
    assert "transformer" not in ACCEPTED, (
        "`transformer` is now in the vendored accepted set. If the platform "
        "really did add it, delete this test in the same commit as the "
        "contract refresh and say so in the PR — model-zoo#273 exists because "
        "three keypoint templates declared it while it was refused."
    )
    assert "transformer" in NOT_ACCEPTED_EXAMPLES, (
        "`transformer` dropped out of the contract's not_accepted examples, so "
        "a template reintroducing it would fail with an anonymous 'not in set' "
        "message instead of the history of why"
    )


def test_od_accepted_values_are_a_subset() -> None:
    """The two vocabularies must not contradict each other.

    The engine's OD registry is narrower than the backend's choice set, and OD
    templates are checked against both (here, and in
    ``test_od_model_type_contract.py``). Containment is what makes satisfying
    the engine schema sufficient to be storable — the argument the backend's
    own ``test_od_families_contract.py`` docstring relies on. A backend
    narrowing that stranded an engine-accepted value would otherwise leave
    every OD template passing one file and failing the other with no
    explanation of which is right.
    """
    od_schema = json.loads(OD_SCHEMA_PATH.read_text(encoding="utf-8"))
    od_accepted = frozenset(
        v.strip().lower() for v in od_schema["accepted_model_type_values"]
    )
    assert od_accepted, f"{OD_SCHEMA_PATH}: accepted_model_type_values is empty"
    stranded = sorted(od_accepted - ACCEPTED)
    assert not stranded, (
        f"the engine accepts OD model_type(s) the backend no longer offers: "
        f"{stranded}. An OD template declaring one passes "
        f"test_od_model_type_contract.py and is then refused by the "
        f"ChoiceField. Reconcile the two vendored contracts — and note which "
        f"side moved, because that decides whether a template or the backend "
        f"is what needs the change."
    )


def test_the_readers_are_independent_and_discriminate(tmp_path) -> None:
    """Guard the guard: ``test_the_zoo_roster_was_found`` compares two readers'
    verdicts, and if both collapsed to "always None" the roster and the
    undeclared list would go empty together and it would pass on nothing.

    Also pins the trap this file's AST reader exists for — a nested
    ``"model_type"`` dict key must NOT be read as the declaration. All three
    templates model-zoo#273 fixes carry one.
    """
    support = tmp_path / "loss.py"
    support.write_text("import torch\n\n\ndef loss(a, b):\n    return a - b\n", "utf-8")
    assert _declares_framework(support) is None
    assert _declared_model_type(support) is None

    template = tmp_path / "model.py"
    template.write_text('framework = "pytorch"\nmodel_type = "direct"\n', "utf-8")
    assert _declares_framework(template) == "pytorch"
    assert _declared_model_type(template) == "direct"

    # The CONFIG trap, in the shape the three keypoint templates actually have:
    # a nested HF architecture id that a text scan would pick up, and — worse —
    # a nested key whose value is NOT accepted while the real declaration is.
    nested = tmp_path / "nested.py"
    nested.write_text(
        'framework = "pytorch"\n'
        'model_type = "direct"\n'
        'CONFIG = {\n'
        '    "model_type": "vitpose",\n'
        '    "backbone_config": {"model_type": "vitpose_backbone"},\n'
        '}\n',
        "utf-8",
    )
    assert _declared_model_type(nested) == "direct", (
        "the reader picked up a nested CONFIG['model_type'] instead of the "
        "module-level declaration"
    )

    # A class attribute and a local are equally out of reach.
    scoped = tmp_path / "scoped.py"
    scoped.write_text(
        'framework = "pytorch"\n'
        'model_type = "heatmap"\n'
        'class M:\n'
        '    model_type = "transformer"\n'
        'def f():\n'
        '    model_type = "transformer"\n'
        '    return model_type\n',
        "utf-8",
    )
    assert _declared_model_type(scoped) == "heatmap"

    # Whole-identifier matching, not substring: a DIFFERENT module-level name
    # that merely contains `model_type` is not the declaration.
    lookalike = tmp_path / "lookalike.py"
    lookalike.write_text(
        'framework = "pytorch"\n'
        'hf_model_type = "transformer"\n'
        'model_type_note = "transformer"\n',
        "utf-8",
    )
    assert _declared_model_type(lookalike) is None, (
        "a name merely containing `model_type` was read as the declaration"
    )


def test_a_non_literal_declaration_is_a_loud_failure(tmp_path) -> None:
    """A declaration this file cannot evaluate must fail, not read as absent.

    ``None`` means "no declaration", which
    ``test_the_zoo_roster_was_found`` reports as its own error. A computed
    value returning ``None`` would instead be indistinguishable from a support
    file and drop out of the roster silently — the vacuous-pass shape this file
    is built to refuse.
    """
    computed = tmp_path / "computed.py"
    computed.write_text(
        'framework = "pytorch"\n'
        'SUFFIX = "former"\n'
        'model_type = "trans" + SUFFIX\n',
        "utf-8",
    )
    with pytest.raises(AssertionError, match="not a plain string literal"):
        _declared_model_type(computed)

    annotated = tmp_path / "annotated.py"
    annotated.write_text('framework = "pytorch"\nmodel_type: str\n', "utf-8")
    with pytest.raises(AssertionError, match="never assigned"):
        _declared_model_type(annotated)


def test_an_unparseable_template_is_a_loud_failure(tmp_path) -> None:
    """No ``except SyntaxError: pass``, and no skip.

    A template that does not parse is an open question about what it declares.
    Swallowing that would make this file quietly stop covering the one template
    most likely to be broken.
    """
    broken = tmp_path / "broken.py"
    broken.write_text('framework = "pytorch"\nmodel_type = "direct\n', "utf-8")
    with pytest.raises(SyntaxError):
        _declared_model_type(broken)


#: The boosting-classifier declarations model-zoo#272 corrected, PINNED PER FILE
#: so the semantic mismatch cannot silently return. ``test_every_template_
#: declares_a_storable_model_type`` only proves a value is STORABLE, and ``tree``
#: is storable — so it never caught the classification boosting models being
#: folded into ``tree`` while their regression siblings (``xgboost_regressor`` /
#: ``lightgbm_regressor``) declared their library. xgboost/lightgbm/catboost now
#: match that convention; hist_gradient_boosting has no histgb-specific value in
#: the platform vocabulary and stays ``tree``; ebm is a glass-box GAM with no
#: exact value and now ALSO declares ``tree`` -- it was ``""``, which the backend
#: stores as ``default`` but which the SDK coerces to ``None``, and ``None`` is in
#: no accepted set, so both ebm templates were untrainable end to end
#: (the EBM declaration incident). Each file's declaration comment records the reason.
#: BOTH halves of the disagreement model-zoo#272 fixed, pinned per file. The
#: classifiers are what the ticket corrected; the REGRESSORS are the baseline the
#: correction was measured against ("their regression siblings declare their
#: library"). Pinning only the classifiers would leave the other half free to
#: silently recreate the split — e.g. xgboost_regressor drifting to ``tree`` — so
#: both are asserted here (@shujaatTracebloc on #288). ``test_every_template_
#: declares_a_storable_model_type`` only proves a value is STORABLE, and ``tree``
#: is storable, so only a per-file pin catches a regression back to it.
#: hist_gradient_boosting has no histgb value in the platform vocabulary and stays
#: ``tree`` on both sides; ebm now matches it for the same reason (was ``""``,
#: the EBM declaration incident); there is no catboost_regressor. Each file records its own reason.
_BOOSTING_MODEL_TYPES = {
    "tabular_classification/sklearn/xgboost_classifier.py": "xgboost",
    "tabular_classification/sklearn/lightgbm_classifier.py": "lightgbm",
    "tabular_classification/sklearn/catboost_classifier.py": "catboost",
    "tabular_classification/sklearn/hist_gradient_boosting_classifier.py": "tree",
    "tabular_classification/sklearn/ebm_classifier.py": "tree",
    "tabular_regression/sklearn/xgboost_regressor.py": "xgboost",
    "tabular_regression/sklearn/lightgbm_regressor.py": "lightgbm",
    "tabular_regression/sklearn/hist_gradient_boosting_regressor.py": "tree",
    "tabular_regression/sklearn/ebm_regressor.py": "tree",
}


def test_boosting_siblings_declare_their_library() -> None:
    """Each boosting classifier AND regressor declares the ``model_type`` #272 pinned.

    A per-file assertion, not set membership: the defect was a value that is
    accepted (``tree``) yet wrong for the file, so only pinning the exact expected
    declaration catches a regression back to it — on either side of the
    classifier/regressor pair.

    Collect every mismatch and report them together, rather than aborting on the
    first: same reporting convention ``test_every_template_declares_a_storable_
    model_type`` argues for — one run should name all the wrong declarations, not
    just the earliest in dict order.
    """
    violations = []
    for rel, expected in _BOOSTING_MODEL_TYPES.items():
        path = MODEL_ROOT / rel
        if not path.exists():
            violations.append(f"{rel}: pinned template is missing")
            continue
        actual = _declared_model_type(path)
        if actual != expected:
            violations.append(f"{rel}: model_type={actual!r}, expected {expected!r}")
    assert not violations, (
        "boosting model_type declarations drifted (model-zoo#272):\n  "
        + "\n  ".join(violations)
    )
