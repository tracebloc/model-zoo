"""No template anywhere in the zoo may BUILD a ``FrozenBatchNorm2d``
.

The defect this pins
--------------------
``FrozenBatchNorm2d`` at construction holds ``weight=1``, ``bias=0``,
``running_mean=0``, ``running_var=1``, so it computes::

    (x - 0) / sqrt(1 + eps) * 1 + 0   ==   x

the identity, up to ``eps``. Those four buffers mean something only once a
pretrained checkpoint loads real statistics into them; that is what the layer
is *for*. Every template in this repo builds ``weights=None`` (the hub is a
closed door, design note D6), so there is nothing to freeze and the layer
degenerates to a no-op — the backbone trains with no normalisation at all.

Not a quality question. On the OD roster it broke training outright:
``centernet_resnet`` diverged 1032.6 -> **3.19e+29** with grad norm going to
**inf**, where GroupNorm on the same script and seeds gave
165.9 -> 15.87 -> 14.61 with grad norm 648 -> 60 -> 21.

WHY THIS FILE EXISTS — the guard that read as complete
------------------------------------------------------
(internal ref) converted twelve OD templates off this line, and
``test_od_norm_layers_normalise.py`` holds them correct. That guard is an
**object-detection directory scan**. Its ``NON_NORMALISING`` set is empty and
its ratchet is at 0, and both facts are true *and say nothing* about anything
outside ``model_zoo/object_detection/``.

Two keypoint templates carried the identical line the whole time —
``keypoint_detection/pytorch/keypoint_rcnn.py`` and
``.../faster_rcnn_sppe.py`` — invisible to it, because the defect was never
OD-specific and the scan was. That is the shape of gap (internal ref) was filed
for, and a scan-shaped gap cannot be closed by tightening what the scan
checks. Hence a **sibling guard with the widest possible scope**, which
an internal ticket's definition of done names as an acceptable form ("the guard's
scan covers them, or a sibling guard does").

WHAT THIS FILE IS AND IS NOT
----------------------------
It is a STRUCTURAL scan of every template in the zoo for one exact
construction, and its scope is the whole tree rather than one task directory.
That is the half ``test_od_norm_layers_normalise.py`` cannot have: that file
BUILDS each model and probes whether its norms really normalise, which is a
far stronger statement and affordable only on one roster (a from-scratch
ResNet-50 detector is several hundred modules, and the OD sweep is already the
slow part of CI). Extending a build-and-probe sweep to all ~180 zoo templates
would trade a cheap check that covers everything for an expensive one that
would have to be narrowed again.

So the two are complementary and neither subsumes the other:

* this file  — every template, one known-bad construction, no imports, runs in
  all three CI jobs (pytorch / sklearn / survival) because it needs no
  framework at all;
* the OD file — one roster, every norm module, behaviourally probed.

⚠️ AND HERE IS WHAT NEITHER CATCHES, recorded rather than papered over. A norm
that is *constructed but never applied*, and *partial* norm removal, are both
invisible to a structural check — and (internal ref) measured that they survive
even the corrected behavioural guard. Dropping a module from ``forward`` while
leaving it constructed does not move parameter count, tensor shapes,
``state_dict`` keys or loss keys. This file closes the scan-shaped hole; it
does not claim to close those.

Matching is by AST, which is what makes it exact
------------------------------------------------
The name appears in PROSE in some thirty template docstrings and comments
across this repo — the OD conversions all explain at length what they moved
away from, and the two keypoint fixes now do the same. A text scan would flag
every one of them, and the obvious repair (drop the prose) would delete the
explanations that stop the line coming back.

So the scan parses each file and inspects ``ast.Name`` / ``ast.Attribute`` /
``ast.alias`` nodes. Docstrings and comments are structurally out of reach, not
filtered out by a pattern that has to be right. It is also whole-identifier by
construction: an AST identifier is the whole name, so ``substring`` matching is
not merely avoided but impossible — no ``\\b`` regex to get subtly wrong, and a
hypothetical ``MyFrozenBatchNorm2dWrapper`` is a different identifier rather
than a near-miss. Both directions are pinned in
``test_the_detector_discriminates``.

A syntax error propagates. An unparseable template is a loud failure — this
file cannot say what it builds, and "cannot say" must never read as "clean".
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parent.parent
MODEL_ROOT = ROOT / "model_zoo"

#: The one identifier this file is about. torchvision exposes it as
#: ``torchvision.ops.misc.FrozenBatchNorm2d`` and re-exports it from
#: ``torchvision.ops``; every spelling reaches it by this attribute or name.
_FROZEN_BN = "FrozenBatchNorm2d"

#: Templates permitted to construct a frozen BN, with the reason.
#:
#: EMPTY, and the ratchet below pins it at empty. A template that genuinely
#: loaded a pretrained checkpoint would have a real argument for frozen BN —
#: running statistics average badly across non-IID federated clients, which is
#: why it was reached for in the first place — but no template in this repo
#: does: they all build ``weights=None``, and a hosted seed arrives through
#: ``load_state_dict`` AFTER construction, by which point the layer has already
#: been an identity for the whole from-scratch path.
#:
#: ⚠️ Asserted in BOTH directions: a listed template that no longer builds one
#: fails too, with an instruction to delete its row. A list that quietly
#: tolerates a fixed entry decays into folklore nobody can audit.
#:
#: Do not add a row to silence a new template. Fourteen templates once shared
#: this one wrong default; a fifteenth is the same defect, not a new one. Use a
#: norm that is correct from scratch — GroupNorm normalises per sample, needs no
#: checkpoint, and adds no running statistics for the averaging service to ship
#: every federated round.
FROZEN_BN_ALLOWED: dict[str, str] = {}

#: The ratchet, pinned by EQUALITY rather than an upper bound — the same shape
#: and reasoning as ``MAX_NON_NORMALISING`` in
#: ``test_od_norm_layers_normalise.py``, where ``<=`` was found insufficient:
#: it blocks growth above the high-water mark but not RE-GROWTH after a fix.
#: At 0 it is the floor, so any row at all is a regression.
MAX_FROZEN_BN_ALLOWED = 0

#: A COLLAPSE DETECTOR, not a census — see ``test_the_zoo_roster_was_found``.
_MIN_ZOO_TEMPLATES = 100


def _declares_framework(path: pathlib.Path) -> str | None:
    """The module-level ``framework``, or ``None`` for a support module.

    What separates a TEMPLATE ENTRY POINT from a helper: the metadata contract
    in CLAUDE.md requires it of every model file, and the ``yolo_*/loss.py``
    helpers declare none. Read statically so file SELECTION costs nothing.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(r'^\s*framework\s*=\s*["\'](\w*)["\']', text, re.MULTILINE)
    return match.group(1) if match else None


def _model_files() -> list[pathlib.Path]:
    return sorted(MODEL_ROOT.rglob("*.py"))


TEMPLATES = [p for p in _model_files() if _declares_framework(p)]


def _template_key(path: pathlib.Path) -> str:
    """Row key for a template: its ``model_zoo/``-relative posix path.

    NOT the file stem, and that is a correctness requirement rather than a
    preference. Zoo-wide, stems collide heavily — ``cnn``, ``mlp``, ``lstm``,
    ``rnn``, ``svm``, ``tcn``, ``knn``, ``random_forest``, ``decision_tree``,
    ``ft_transformer``, ``tabm``, ``fcn``, ``gru``, ``cox_ph``, ``hrnet``,
    ``transformer`` and ``bert_base_uncased`` each name two or three different
    templates in different categories. ``tools/seed_index.py`` documents this
    exact trap: keying by stem means one silently wins, so a row meant to
    exempt one template would silence a DIFFERENT template's defect.

    A relative path is unique by construction, which is why the roster check
    below asserts allowlist membership rather than key collisions — with path
    keys a collision is not possible, and a test that can never fail is noise.
    """
    return path.relative_to(MODEL_ROOT).as_posix()


def _rel(path: pathlib.Path) -> str:
    """Repo-relative path for a message, degrading to the absolute path.

    The detector is driven against ``tmp_path`` fixtures in this file's
    guard-the-guard tests, and ``Path.relative_to`` RAISES on a path outside
    the repo — so formatting a failure message this way turned a clean
    assertion into a confusing ``ValueError`` from ``pathlib``.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _frozen_bn_code_references(path: pathlib.Path) -> list[str]:
    """Every CODE reference to ``FrozenBatchNorm2d``, as ``"line N: <what>"``.

    Docstrings and comments are not code and cannot appear here. A
    ``SyntaxError`` propagates on purpose.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == _FROZEN_BN:
            hits.append(f"line {node.lineno}: bare name `{_FROZEN_BN}`")
        elif isinstance(node, ast.Attribute) and node.attr == _FROZEN_BN:
            hits.append(f"line {node.lineno}: attribute `....{_FROZEN_BN}`")
        elif isinstance(node, ast.alias) and _FROZEN_BN in (node.name, node.asname):
            # `from torchvision.ops import FrozenBatchNorm2d`, or an alias of
            # it. Caught at the import so a template cannot smuggle it in
            # under another name and reference it as an unrelated identifier.
            hits.append(
                f"line {getattr(node, 'lineno', 0)}: imported "
                f"`{node.name}`" + (f" as `{node.asname}`" if node.asname else "")
            )
    return hits


def test_the_zoo_roster_was_found() -> None:
    """Guard the guard: this file is driven by a directory walk, and a walk that
    visits nothing passes exactly as quietly as one that finds nothing wrong.

    That is not a hypothetical here — it is the mechanism of the bug this file
    was written for. ``test_od_norm_layers_normalise.py`` was green, its
    ratchet at 0, while two templates with the defect sat outside its scan.

    The exact-count assertion is the EQUALITY in
    ``test_no_template_builds_a_frozen_bn_identity`` (scanned == roster), which
    moves with the roster instead of being a literal every zoo PR must bump.
    The floor here is the separate failure that equality cannot see, because
    both sides would go empty together: the walk losing the tree.
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
    unknown = sorted(set(FROZEN_BN_ALLOWED) - {_template_key(p) for p in TEMPLATES})
    assert not unknown, (
        f"FROZEN_BN_ALLOWED names templates the walk did not find: {unknown} — "
        f"if they were deleted, delete their rows (and lower "
        f"MAX_FROZEN_BN_ALLOWED) in the same commit"
    )


def test_no_template_builds_a_frozen_bn_identity() -> None:
    """The rule, in both directions.

    Unlisted template: it must not reference ``FrozenBatchNorm2d`` in code at
    all. Listed template: it must still reference one, or its row is stale.

    Aggregated rather than parametrized per file: the useful failure for a
    roster-wide defect is every offender in one message — both of
    an internal ticket's keypoint templates together — not two reds that each read
    as an isolated typo.

    THE COUNT IS ASSERTED, not just the absence. ``scanned == roster`` is the
    whole point of this file's existence: a walk that visited zero templates,
    or a detector that silently returned ``[]`` for every one of them, would
    satisfy ``violations == []`` perfectly and reproduce (internal ref) exactly.
    """
    scanned: list[str] = []
    violations: list[tuple[str, list[str]]] = []
    stale_rows: list[str] = []

    for path in TEMPLATES:
        refs = _frozen_bn_code_references(path)
        scanned.append(_rel(path))
        key = _template_key(path)
        if key in FROZEN_BN_ALLOWED:
            if not refs:
                stale_rows.append(key)
            continue
        if refs:
            violations.append((_rel(path), refs))

    assert len(scanned) == len(TEMPLATES), (
        f"the scan examined {len(scanned)} template(s) but the roster holds "
        f"{len(TEMPLATES)} — the loop skipped files, so this result is an "
        f"answer about a subset of the zoo and says nothing about the rest. "
        f"A roster walk that silently visits fewer files than it found passes "
        f"as quietly as a clean one; that is how (internal ref) happened."
    )

    assert not stale_rows, (
        f"{stale_rows} are listed in FROZEN_BN_ALLOWED but no longer build a "
        f"{_FROZEN_BN}. Delete their rows and lower MAX_FROZEN_BN_ALLOWED to "
        f"{len(FROZEN_BN_ALLOWED) - len(stale_rows)} in this commit — a list "
        f"that tolerates a fixed entry decays into folklore nobody can audit."
    )

    detail = "\n".join(
        f"  - {rel}\n" + "\n".join(f"      {r}" for r in refs)
        for rel, refs in violations
    )
    assert not violations, (
        f"{len(violations)} template(s) build a {_FROZEN_BN} on a "
        f"`weights=None` backbone, where it is a BIT-EXACT IDENTITY and the "
        f"trunk therefore trains with no normalisation at all "
        f":\n{detail}\n\n"
        f"Use a norm that is correct from scratch. GroupNorm normalises per "
        f"sample, so it needs no checkpoint and adds no running statistics for "
        f"the averaging service to ship every federated round — which is the "
        f"whole reason frozen BN was reached for. The twelve OD templates in "
        f"(internal ref) and the two keypoint templates in (internal ref) were "
        f"all fixed this way; copy `_group_norm` from any of them (a template "
        f"is uploaded as ONE file and cannot import a sibling, so it is "
        f"duplicated per template on purpose).\n\n"
        f"Do NOT add a row to FROZEN_BN_ALLOWED to silence this — the ratchet "
        f"refuses it, which is the point."
    )


def test_the_allowlist_only_ever_shrinks() -> None:
    """The ratchet.

    Without it the cheapest way to green a newly-broken template is to add its
    name to ``FROZEN_BN_ALLOWED`` — the exact failure the list exists to
    prevent, performed on the list itself.

    Legal edits: delete a row and lower ``MAX_FROZEN_BN_ALLOWED`` in one
    commit. Adding a row fails here, so a template that wants frozen BN has to
    be argued for rather than absorbed.
    """
    assert len(FROZEN_BN_ALLOWED) == MAX_FROZEN_BN_ALLOWED, (
        f"FROZEN_BN_ALLOWED holds {len(FROZEN_BN_ALLOWED)} entries "
        f"({sorted(FROZEN_BN_ALLOWED)}) against a pinned "
        f"{MAX_FROZEN_BN_ALLOWED}.\n"
        f"  - GREW? A frozen BN on a from-scratch build is that template's "
        f"bug. Pick a norm that works from scratch instead of listing it.\n"
        f"  - SHRANK? It cannot: 0 is the floor.\n"
        f"Asserted by EQUALITY, not `<=`: an upper bound would let a fix free "
        f"a slot a later commit could quietly refill."
    )
    assert MAX_FROZEN_BN_ALLOWED == 0, (
        f"MAX_FROZEN_BN_ALLOWED is {MAX_FROZEN_BN_ALLOWED}, not the 0 this "
        f"file records. 0 is the floor and the whole zoo is at it; raising it "
        f"is a decision that belongs in (internal ref), not here."
    )


def test_the_detector_discriminates(tmp_path) -> None:
    """Guard the guard, in both directions at once.

    The rule above is an assertion that a list is empty, and an empty list is
    also what a detector that can no longer detect anything returns. So the
    detector is driven against a template that DOES build a frozen BN — in
    each spelling a template could reach it by — and against one that only
    discusses it in prose, which is what every fixed template in this repo now
    looks like.
    """
    # The defect, in the exact shape both keypoint templates had.
    offender = tmp_path / "offender.py"
    offender.write_text(
        'framework = "pytorch"\n'
        "from torchvision.models import resnet50\n"
        "from torchvision.ops import misc as misc_nn_ops\n"
        "def MyModel():\n"
        "    return resnet50(weights=None, "
        "norm_layer=misc_nn_ops.FrozenBatchNorm2d)\n",
        "utf-8",
    )
    assert _frozen_bn_code_references(offender), (
        "the detector missed `norm_layer=misc_nn_ops.FrozenBatchNorm2d`, the "
        "exact line (internal ref) is about"
    )

    # A direct import, and an aliased one — a template must not be able to
    # smuggle the class in under another name.
    imported = tmp_path / "imported.py"
    imported.write_text(
        'framework = "pytorch"\n'
        "from torchvision.ops import FrozenBatchNorm2d\n"
        "def MyModel():\n"
        "    return FrozenBatchNorm2d(64)\n",
        "utf-8",
    )
    assert len(_frozen_bn_code_references(imported)) >= 2, (
        "a bare `from ... import FrozenBatchNorm2d` plus its use should be "
        "reported at both sites"
    )

    aliased = tmp_path / "aliased.py"
    aliased.write_text(
        'framework = "pytorch"\n'
        "from torchvision.ops import FrozenBatchNorm2d as _FBN\n"
        "def MyModel():\n"
        "    return _FBN(64)\n",
        "utf-8",
    )
    assert _frozen_bn_code_references(aliased), (
        "an aliased import hid the construction — the import site must be "
        "caught, because the use site is an unrelated identifier"
    )

    # An `isinstance` check counts too. It is how both keypoint templates'
    # dead `overwrite_eps` loops referenced the class after the norm swap, and
    # dead code that reads as coverage is worse than none.
    isinstance_only = tmp_path / "isinstance_only.py"
    isinstance_only.write_text(
        'framework = "pytorch"\n'
        "from torchvision.ops import misc as misc_nn_ops\n"
        "def f(m):\n"
        "    return isinstance(m, misc_nn_ops.FrozenBatchNorm2d)\n",
        "utf-8",
    )
    assert _frozen_bn_code_references(isinstance_only)

    # PROSE ONLY — a fixed template explaining what it moved away from. This is
    # the direction a text scan gets wrong, and getting it wrong would pressure
    # authors to delete the explanations that stop the line coming back.
    documented = tmp_path / "documented.py"
    documented.write_text(
        '"""This template used to build norm_layer=FrozenBatchNorm2d, which on\n'
        "a weights=None backbone is a bit-exact identity.\n"
        '"""\n'
        'framework = "pytorch"\n'
        "from torch import nn\n"
        "# FrozenBatchNorm2d held weight/bias as buffers; GroupNorm holds them\n"
        "# as parameters, so this is not parameter-neutral.\n"
        "def _group_norm(c):\n"
        "    return nn.GroupNorm(max(g for g in range(1, 33) if c % g == 0), c)\n",
        "utf-8",
    )
    assert _frozen_bn_code_references(documented) == [], (
        "prose in a docstring or comment was read as a construction — every "
        "fixed template in this repo would be a false positive"
    )

    # Whole-identifier, not substring: a different class whose name merely
    # CONTAINS the identifier is not this defect.
    lookalike = tmp_path / "lookalike.py"
    lookalike.write_text(
        'framework = "pytorch"\n'
        "from torch import nn\n"
        "class MyFrozenBatchNorm2dWrapper(nn.Module):\n"
        "    pass\n"
        "def MyModel():\n"
        "    return MyFrozenBatchNorm2dWrapper()\n",
        "utf-8",
    )
    assert _frozen_bn_code_references(lookalike) == [], (
        "`MyFrozenBatchNorm2dWrapper` was matched as `FrozenBatchNorm2d` — the "
        "scan is matching substrings, not whole identifiers"
    )


def test_an_unparseable_template_is_a_loud_failure(tmp_path) -> None:
    """No ``except SyntaxError: pass``, and no skip.

    A template that does not parse is an open question about what it builds.
    Swallowing that would make this file quietly stop covering the one template
    most likely to be broken — and it would report a clean scan while doing it.
    """
    broken = tmp_path / "broken.py"
    broken.write_text(
        'framework = "pytorch"\ndef MyModel(\n    return None\n', "utf-8"
    )
    with pytest.raises(SyntaxError):
        _frozen_bn_code_references(broken)


def test_the_keypoint_templates_backend_3182_names_are_on_the_roster() -> None:
    """The two files (internal ref) was filed about are actually covered here.

    The bug was a scan that did not reach them, so "the rule passes" is only
    meaningful alongside "the rule was applied to these two". Without this, a
    future change to the walk could drop the keypoint tree and every assertion
    above would stay green.

    Named by path rather than derived on purpose: this is the regression pin
    for one specific scan-shaped hole, and deriving it from the same walk it is
    meant to check would make it circular.
    """
    expected = {
        "model_zoo/keypoint_detection/pytorch/keypoint_rcnn.py",
        "model_zoo/keypoint_detection/pytorch/faster_rcnn_sppe.py",
    }
    on_roster = {str(p.relative_to(ROOT)) for p in TEMPLATES}
    missing = sorted(expected - on_roster)
    assert not missing, (
        f"an internal ticket's templates are not on this file's roster: {missing}. "
        f"Either they were renamed or moved (update this pin in the same "
        f"commit) or the walk stopped reaching the keypoint tree — which is "
        f"the exact failure this file exists to prevent, reintroduced."
    )
