"""Re-derive, from the tree, which templates ``check_dump_coverage.py`` must survey.

WHY THIS MODULE EXISTS (backend#2982)
-------------------------------------
The census used to be a hand-maintained literal in
``tests/test_check_dump_coverage.py``::

    MIGRATED_TEMPLATE_CENSUS = 61

That is a real guard — it is what stops a template joining or leaving the survey
without anyone noticing — but it lived in a file EVERY template PR had to edit,
so *n* open template PRs produced *O(n)* conflicts per merge. With four OD roster
PRs in flight the census went 55 -> 59 -> 60 -> 61 in one day and three of the
four went ``DIRTY`` simultaneously; a conflicted PR runs no ``pull_request``
workflows at all, so each one's green checks described a tree that no longer
existed. Roughly 20-30 more templates are coming and none of them depend on each
other: the literal was the only thing serialising them.

So the count is DERIVED here instead of restated there. Adding a template moves
both sides at once and no literal needs editing.

THE FAILURE MODE THIS MODULE IS SHAPED AROUND
---------------------------------------------
A derivation that counts the same files the tool counts is worth nothing: both
sides go wrong together and the assertion passes while the gate is blind. That
is the whole risk of replacing a literal with a derivation, and it is why this is
a SECOND, DELIBERATELY DIFFERENT IMPLEMENTATION rather than a call into
``check_dump_coverage``. Three differences, each of which is a place the tool can
go blind and this cannot:

1. **Recursion.** ``survey()`` does ``(category / "pytorch").glob("*.py")`` —
   ONE level. This walks ``rglob``, so a migrated template placed in a
   subdirectory (the shape ``object_detection/pytorch/yolo_v1/model.py`` already
   has) is seen here and missed there. Keys are the path under ``pytorch/``
   without its suffix, so a flat template's key is identical to the tool's and a
   nested one produces a key the tool can never emit.

2. **Whitespace.** The tool substring-matches the RAW source, so a line wrap
   that splits ``no weight file`` across two lines silently declassifies a
   template — and ``classify()`` needs both that phrase and ``weights=False``
   for its NO_SEED verdict, so the wrap turns a scratch template into an
   UNCLASSIFIED or, worse, flips a status. Here the source is normalised to
   single spaces first, so the phrase survives the wrap and the two sides
   DISAGREE, which is what reddens.

3. **Membership.** The tool surveys a file if and only if it contains
   ``"Offline variant"``. This one surveys it on ANY of several independent
   footprints (see ``footprints()``) — most usefully ``SEED_EXCLUDED_PREFIXES``,
   which is written into the file by ``tools/seed_contract.py apply`` and is not
   prose at all. So deleting a migrated template's docstring paragraph does not
   quietly remove it from the census: 53 of the 61 still carry that constant.

``tests/test_zoo_census.py`` proves all three divergences on synthetic trees. If
someone later "simplifies" this module into a wrapper around the tool, those
tests fail.

THE HOLE THE DERIVATION LEFT, AND THE TRIPWIRE THAT CLOSES IT
-------------------------------------------------------------
Eight templates carry no ``SEED_EXCLUDED_PREFIXES`` — they are headless
(encoders, LMs), so ``seed_contract.py apply`` has no head to exclude — and for
those the docstring is the ONLY footprint in the file. Deleting the whole
migration paragraph from one of them removes it from BOTH sides at once, which
the old literal caught (61 -> 60) and a pure derivation cannot. Measured, not
assumed: that mutation survived the first version of this module.

``PROSE_ONLY`` below is that direction's tripwire, and it is deliberately the
ONE literal left. It is not in ``tests/test_check_dump_coverage.py`` and it does
not move when a template is added: every template with a task head — which is
every detector, classifier and segmenter, i.e. all of backend#2982's roster —
carries the constant and is covered without being listed. Only a new HEADLESS
template joins the list, and ``test_zoo_census.py`` makes that failure loud at
authoring time rather than leaving it to be noticed later.

The list is self-policing in both directions, which is what keeps it from
rotting the way a hand-maintained census does: an entry naming a file that no
longer exists is red, an entry for a template that has since gained the constant
is red (drop it), and a surveyed template that needs an entry and has none is
red (add it).
"""

from __future__ import annotations

import pathlib
import re
from typing import Dict, List

#: A migrated template's docstring says this. Same phrase ``check_dump_coverage``
#: keys on, matched against normalised text rather than the raw source.
MARKER = "offline variant"

#: A template that random-initialises by design says BOTH of these. Matched the
#: way the tool matches them (loose substring) so the two sides do not disagree
#: gratuitously about a status — the divergence that matters is the whitespace
#: one, and it is the normalisation above that produces it.
NO_SEED_PHRASE = "no weight file"
NO_SEED_KWARG = "weights=false"

#: ``weights=True`` / ``weights=False`` as an UPLOAD INSTRUCTION, not as any
#: keyword argument that happens to end in ``weights``.
#:
#: The lookbehind is load-bearing. ``time_series_forecasting/tft.py`` calls
#: ``self.attn(h, h, h, need_weights=False)``, which a bare ``weights=false``
#: substring matches — that one line would enrol an unmigrated from-scratch
#: template in the census and fail this repo's tests on the real tree. The tool
#: has the same looseness in ``classify()`` and gets away with it only because it
#: never looks at a file without the marker.
UPLOAD_KWARG = re.compile(r"(?<![\w.])weights\s*=\s*(?:true|false)")

#: Written into each template with a head by ``tools/seed_contract.py apply``.
#: The one footprint here that is CODE rather than prose, which is what makes it
#: worth having: prose rots and this does not.
SEED_CONSTANT = "seed_excluded_prefixes"

EXPECTS_SEED = "EXPECTS_SEED"
NO_SEED = "NO_SEED"
UNCLASSIFIED = "UNCLASSIFIED"

#: Migrated templates whose ONLY footprint is their docstring — see "THE HOLE
#: THE DERIVATION LEFT" above. All eight are headless, so no head declaration
#: pins them down; deleting the migration paragraph would otherwise remove them
#: from the census and from the tool together, silently.
#:
#: DO NOT EDIT THIS TO MAKE A TEST PASS. Every way it can be wrong is checked in
#: ``tests/test_zoo_census.py``: a name that is not a file, an entry for a
#: template that has since gained ``SEED_EXCLUDED_PREFIXES``, and a surveyed
#: template that belongs here and is absent. If the completeness test sends you
#: here, add your template — that is the one edit this mechanism still asks of an
#: author, and only for a template with no task head.
PROSE_ONLY = (
    "causal_language_modeling/distilgpt2",
    "embeddings/minilm",
    "image_classification/vit",
    "masked_language_modeling/netmedgpt_style_warmstart",
    "seq2seq/t5_small",
    "text_classification/bert_base_uncased_scratch",
    "text_classification/distilbert_scratch",
    "time_series_forecasting/timesfm",
)


def normalise(source: str) -> str:
    """Lowercased source with every whitespace run collapsed to ONE SPACE.

    Not stripped of whitespace entirely: ``need_weights=False`` and
    ``weights=False`` would still be distinguishable, but ``no weight file``
    would collapse into a token that also matches inside unrelated prose, and
    word boundaries are what ``UPLOAD_KWARG`` relies on.
    """
    return re.sub(r"\s+", " ", source).lower()


def template_files(zoo: pathlib.Path) -> List[pathlib.Path]:
    """Every ``.py`` under any ``<category>/pytorch/`` directory, RECURSIVELY.

    Recursion is the point (see this module's docstring): ``survey()`` globs one
    level, so a template in a subdirectory is invisible to it and visible here.
    """
    root = zoo / "model_zoo" if (zoo / "model_zoo").is_dir() else zoo
    files: List[pathlib.Path] = []
    for category in sorted(p for p in root.iterdir() if p.is_dir()):
        pytorch = category / "pytorch"
        if not pytorch.is_dir():
            continue
        files.extend(sorted(pytorch.rglob("*.py")))
    return files


def key_for(zoo: pathlib.Path, path: pathlib.Path) -> str:
    """``"<category>/<path under pytorch, no suffix>"``.

    Identical to ``survey()``'s ``f"{category}/{stem}"`` for a flat template, and
    deliberately NOT identical for a nested one — ``object_detection/yolo_v1/model``
    is a key the tool cannot produce, so a migrated template hidden one directory
    down shows up as a one-sided difference instead of as nothing at all.
    """
    root = zoo / "model_zoo" if (zoo / "model_zoo").is_dir() else zoo
    relative = path.relative_to(root)
    category = relative.parts[0]
    under_pytorch = pathlib.Path(*relative.parts[2:]).with_suffix("")
    return f"{category}/{under_pytorch.as_posix()}"


def footprints(source: str, stem: str) -> List[str]:
    """Every reason the tree gives for believing this file is a migrated template.

    A LIST rather than a bool so a failing test can say WHICH footprint the tool
    is disagreeing with — "it declares SEED_EXCLUDED_PREFIXES but the survey does
    not see it" is actionable in a way that "count mismatch" never was.
    """
    text = normalise(source)
    found: List[str] = []
    if MARKER in text:
        found.append(f"docstring says {MARKER!r}")
    if f"{stem}_weights.pkl" in text:
        found.append(f"names {stem}_weights.pkl")
    if NO_SEED_PHRASE in text:
        found.append(f"says {NO_SEED_PHRASE!r}")
    if UPLOAD_KWARG.search(text):
        found.append("gives a weights= upload instruction")
    if SEED_CONSTANT in text:
        found.append("declares SEED_EXCLUDED_PREFIXES")
    return found


def status_of(source: str, stem: str) -> str:
    """``classify()``'s tri-state verdict, re-derived on normalised text.

    Same rule, different input: ``names a dump`` XOR ``declares it needs none``,
    and agreeing on both or neither is UNCLASSIFIED. Re-deriving the STATUS and
    not merely the membership is what makes a line wrap inside ``no weight
    file`` red — it flips the verdict here while the tool keeps the old one.
    """
    text = normalise(source)
    names_dump = f"{stem}_weights.pkl" in text
    declares_none = NO_SEED_KWARG in text and NO_SEED_PHRASE in text
    if names_dump == declares_none:
        return UNCLASSIFIED
    return EXPECTS_SEED if names_dump else NO_SEED


def census(zoo: pathlib.Path) -> Dict[str, str]:
    """``{"<category>/<stem>": status}`` for every template the tree says is migrated."""
    found: Dict[str, str] = {}
    for path in template_files(zoo):
        source = path.read_text(encoding="utf-8")
        if not footprints(source, path.stem):
            continue
        found[key_for(zoo, path)] = status_of(source, path.stem)
    return found


def has_code_footprint(source: str) -> bool:
    """Whether this file's migration is recorded in CODE, not only in prose.

    ``SEED_EXCLUDED_PREFIXES`` is the one footprint a docstring rewrite cannot
    remove, so it is the one that decides whether a template needs a
    ``PROSE_ONLY`` entry.
    """
    return SEED_CONSTANT in normalise(source)


def path_for(zoo: pathlib.Path, key: str) -> pathlib.Path:
    """The template file a census key names. Inverse of ``key_for``."""
    root = zoo / "model_zoo" if (zoo / "model_zoo").is_dir() else zoo
    category, relative = key.split("/", 1)
    return root / category / "pytorch" / f"{relative}.py"


def why(zoo: pathlib.Path) -> Dict[str, List[str]]:
    """``{key: footprints}`` — the evidence behind each ``census`` entry."""
    reasons: Dict[str, List[str]] = {}
    for path in template_files(zoo):
        source = path.read_text(encoding="utf-8")
        marks = footprints(source, path.stem)
        if marks:
            reasons[key_for(zoo, path)] = marks
    return reasons


def prose_only_candidates(zoo: pathlib.Path) -> List[str]:
    """Surveyed templates with no code footprint — what ``PROSE_ONLY`` must hold."""
    return sorted(
        key
        for key, path in ((k, path_for(zoo, k)) for k in census(zoo))
        if not has_code_footprint(path.read_text(encoding="utf-8"))
    )
