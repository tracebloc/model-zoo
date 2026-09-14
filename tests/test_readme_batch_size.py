"""Every batch size a category README states for a named template is the value
that template declares -- and every README that states a batch size is either
checked or a listed, visible exemption.

WHY A GUARD
-----------
``batch_size`` is a module-level literal the SDK reads and ships with the model
(the upload rewriter regexes the value; the submitter copies it into the
experiment as ``batchSize``), and the training plan has no setter for it -- a
user cannot override what the template says. So the category README is the only
other place the number lives, and it is the place a user sizing a run reads
first. When ``seq2seq/pytorch/simple_seq2seq.py`` moved from 16 to 4 because 16
was OOM-killed inside the training envelope (internal ref), the seq2seq README
kept saying 16 -- documenting exactly the value that had just been removed as
OOM-inducing -- and nothing in ``tests/`` referenced ``batch_size`` at all.

WHAT IS ASSERTED, and where each rule comes from
------------------------------------------------
* the README surface is DERIVED, not listed: every ``model_zoo/*/README.md`` is
  read, and every ``N (`stem`)`` pair on its ``**Batch size**`` line is a claim
  about the template whose file stem is ``stem``;
* the named template exists exactly once under that category. SDK scratch
  copies (``tmpmodel_<stem>/<stem>.py``, left beside a template by
  ``upload_model()`` and gitignored) are excluded from the lookup -- they are
  verbatim copies that exist in any checkout that has run an upload and in no
  CI clone, so without the exclusion this suite is green in CI and red on every
  developer's machine ((internal ref) review). The exclusion prefix is held
  equal to the ``.gitignore`` pattern so the two cannot drift apart;
* its single module-level ``batch_size`` literal equals ``N``;
* PER README, not globally: every README that has a ``**Batch size**`` line
  either yields at least one attributable claim or is named in
  ``PROSE_ONLY_BATCH_READMES``. A global "at least one claim somewhere" check
  stops discriminating the moment a second category adds a claim -- rewording
  one README's bullet out of the grammar's reach would then drop its claims
  silently and free it to drift back ((internal ref) review). Per README,
  coverage can only be removed by editing the allowlist, never by rewording;
* the allowlist is exact in both directions: a listed category must still have
  a Batch size line and must yield no attributable claim. A category that
  gains a backticked claim, or loses its Batch size line, fails with an
  instruction to delete its row -- a list that quietly tolerates a stale entry
  decays into a list nobody trusts.

WHAT IS NOT COVERED, on purpose
-------------------------------
Only the backticked ``N (`stem`)`` form is attributable. The eight categories in
``PROSE_ONLY_BATCH_READMES`` state their batch in prose (``default 512``,
``32 (simple/medium)``, ``default 4096 (PyTorch), 512 (sklearn)``), and a
survey against the templates found five of them already wrong -- each with
three to five distinct declared values behind a single "default". Widening the
grammar to those forms would land a red gate on five READMEs; the content
decision (a per-template table? a default plus exceptions?) belongs to the docs
ticket that carries the survey. Until each is rewritten in the attributable
shape and deleted from the allowlist, it is a listed exemption here, not
silence.

All of it is stdlib: this test runs in every CI framework job, including the
ones with no torch installed.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "model_zoo"

_BATCH_LINE = re.compile(r"^- \*\*Batch size\*\*:(?P<rest>.*)$", re.MULTILINE)
_CLAIM = re.compile(r"(?P<batch>\d+) \(`(?P<stem>[A-Za-z0-9_]+)`\)")

#: Directories the SDK's ``upload_model()`` leaves beside a template, holding a
#: verbatim copy of it. Gitignored (``tmpmodel_*/``) and therefore absent from
#: every CI clone and present in most developer checkouts.
SCRATCH_DIR_PREFIX = "tmpmodel_"

#: Categories whose README states a batch size in prose that names no template
#: (``default 512``, ``32 (simple/medium)`` ...), so this guard cannot attribute
#: the claim. Each row is a KNOWN, VISIBLE gap -- five of these READMEs disagree
#: with their templates today (see the module docstring). Delete a row when its
#: README is rewritten in the attributable ``N (`stem`)`` shape; the exactness
#: test below fails until you do.
PROSE_ONLY_BATCH_READMES = frozenset({
    "causal_language_modeling",
    "embeddings",
    "image_classification",
    "semantic_segmentation",
    "sentence_pair_classification",
    "tabular_classification",
    "tabular_regression",
    "text_classification",
})


def _readmes_with_a_batch_line() -> dict[str, str]:
    """``{category: text after '**Batch size**:'}`` for every README stating one."""
    out: dict[str, str] = {}
    for readme in sorted(MODEL_ROOT.glob("*/README.md")):
        lines = _BATCH_LINE.findall(readme.read_text(encoding="utf-8"))
        assert len(lines) <= 1, f"{readme.relative_to(ROOT)}: more than one **Batch size** line"
        if lines:
            out[readme.parent.name] = lines[0]
    return out


def _claims_in(rest: str) -> list[tuple[str, int]]:
    return [(m.group("stem"), int(m.group("batch"))) for m in _CLAIM.finditer(rest)]


BATCH_LINES = _readmes_with_a_batch_line()
CLAIMS = [
    (category, stem, batch)
    for category, rest in BATCH_LINES.items()
    for stem, batch in _claims_in(rest)
]


def _is_scratch(path: pathlib.Path) -> bool:
    return any(part.startswith(SCRATCH_DIR_PREFIX) for part in path.relative_to(MODEL_ROOT).parts)


def _templates_named(category: str, stem: str) -> list[pathlib.Path]:
    """Every ``<stem>.py`` under the category, minus the SDK's scratch copies."""
    return sorted(p for p in (MODEL_ROOT / category).rglob(f"{stem}.py") if not _is_scratch(p))


def _declared_batch_size(path: pathlib.Path) -> object:
    """The template's module-level ``batch_size`` literal -- exactly one."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "batch_size"
        and isinstance(node.value, ast.Constant)
    ]
    assert len(values) == 1, (
        f"{path.relative_to(ROOT)}: expected exactly one module-level "
        f"`batch_size = <literal>`, found {len(values)}"
    )
    return values[0]


def test_the_scratch_prefix_is_the_gitignored_one():
    """The lookup excludes what ``.gitignore`` excludes -- held equal, not restated."""
    patterns = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert f"{SCRATCH_DIR_PREFIX}*/" in patterns, (
        f".gitignore no longer ignores `{SCRATCH_DIR_PREFIX}*/`; if the SDK's scratch "
        "directory was renamed, rename SCRATCH_DIR_PREFIX with it"
    )


def test_some_readme_states_a_batch_size():
    assert BATCH_LINES, "no model_zoo/*/README.md has a **Batch size** line -- every test below is vacuous"


@pytest.mark.parametrize("category", sorted(BATCH_LINES), ids=sorted(BATCH_LINES))
def test_every_readme_batch_line_is_checked_or_a_listed_exemption(category: str):
    if _claims_in(BATCH_LINES[category]):
        return
    assert category in PROSE_ONLY_BATCH_READMES, (
        f"model_zoo/{category}/README.md states a batch size but names no template in the "
        "attributable `N (`stem`)` form, and is not listed in PROSE_ONLY_BATCH_READMES. If the "
        "line was reworded, restore the backticked stems; if it is deliberately prose-only, list "
        "it -- an unchecked README must be a visible exemption, not silence."
    )


@pytest.mark.parametrize("category", sorted(PROSE_ONLY_BATCH_READMES), ids=sorted(PROSE_ONLY_BATCH_READMES))
def test_every_listed_exemption_is_still_one(category: str):
    assert category in BATCH_LINES, (
        f"`{category}` is in PROSE_ONLY_BATCH_READMES but model_zoo/{category}/README.md has no "
        "**Batch size** line (or no README) -- delete the row"
    )
    assert not _claims_in(BATCH_LINES[category]), (
        f"model_zoo/{category}/README.md now names templates in the attributable form, so the "
        "guard checks it -- delete `{category}` from PROSE_ONLY_BATCH_READMES"
    )


@pytest.mark.parametrize(
    "category,stem,readme_batch", CLAIMS, ids=[f"{c}/{s}" for c, s, _ in CLAIMS]
)
def test_readme_batch_size_matches_the_template(category: str, stem: str, readme_batch: int):
    matches = _templates_named(category, stem)
    assert len(matches) == 1, (
        f"model_zoo/{category}/README.md names `{stem}` but {len(matches)} templates "
        f"match model_zoo/{category}/**/{stem}.py outside {SCRATCH_DIR_PREFIX}*/ scratch dirs"
    )
    declared = _declared_batch_size(matches[0])
    assert declared == readme_batch, (
        f"model_zoo/{category}/README.md says `{stem}` trains at batch {readme_batch}; "
        f"{matches[0].relative_to(ROOT)} declares batch_size = {declared!r}. Fix whichever "
        "is wrong: the SDK ships the template's literal, and the README is what a user reads."
    )
