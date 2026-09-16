"""The vendored engine contract carries no reference into a private repo.

WHY THIS TEST EXISTS
--------------------
``tests/contracts/tracebloc_engine/`` is a copy of a file that lives in a
**private** repository, vendored into a **public** one. Upstream's prose cites
internal issues by number; this copy replaces each citation with
``(internal ref)``.

Nothing enforced that. The copy's own README told the next person to refresh it
with

    gh api "repos/.../object_detection_families.v2.json?ref=<sha>" \
      -H "Accept: application/vnd.github.raw" > <the vendored file>

— a straight overwrite with raw upstream bytes, which re-introduces every
citation the scrub removed, into a public repo, with no step in between saying
to take them out again. The recipe now names the scrub; this test is what makes
the recipe's claim checkable rather than another line of prose (CLAUDE.md
rule 7).

It reads the file's BYTES, not its parsed fields: a citation can appear in any
string, in a key this test has never heard of, or in a comment-shaped value, and
the leak is the same.

WHAT COUNTS AS A REFERENCE
--------------------------
GitHub's own cross-repo and same-repo issue syntaxes, which is what upstream
writes and what a reader would follow:

  * ``owner/repo#123`` and ``repo#123`` — a cross-repo reference; the repo name
    alone identifies a private repository even when the number means nothing to
    a stranger.
  * ``#123`` — a bare issue reference. It resolves against whatever repo the
    reader is in, so in a public repo it silently points at a public issue that
    is not the one meant. Wrong for a second reason, caught by the same rule.

A version like ``v2`` or a colour like ``#fff`` is not an issue reference and
must not fail this: the number is required, and a hex colour is excluded by
requiring the match to be digits only.

Stdlib only, no parsing, no network: this runs in every CI framework env.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDORED = ROOT / "tests" / "contracts" / "tracebloc_engine"

#: ``owner/repo#123``, ``repo#123`` or a bare ``#123``. ``[A-Za-z0-9._-]*`` before
#: the ``#`` makes the repo half optional, so both shapes are one pattern rather
#: than two that could disagree. The number must be digits to the end of the
#: run, which is what keeps ``#fff`` out.
ISSUE_REF = re.compile(r"(?:[A-Za-z0-9._-]+/)?[A-Za-z0-9._-]*#\d+\b")

#: What the scrub leaves behind. Present in the file today; named here so a
#: reader of a failure knows what the fix looks like.
PLACEHOLDER = "(internal ref)"


def vendored_files():
    """Every file in the vendored directory. Derived by listing, never named:
    a second contract added beside this one is covered the day it lands."""
    return sorted(p for p in VENDORED.rglob("*") if p.is_file())


def test_the_vendored_directory_is_not_empty():
    """Fail closed. An empty listing makes every test below vacuously true, and
    that is indistinguishable in a log from a clean scrub."""
    assert vendored_files(), f"no files under {VENDORED} -- nothing was checked"


@pytest.mark.parametrize("path", vendored_files(), ids=lambda p: p.name)
def test_no_internal_issue_reference_survives_the_scrub(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    hits = sorted(set(ISSUE_REF.findall(text)))
    assert not hits, (
        f"{path.relative_to(ROOT)} cites {', '.join(hits)}. This directory is a copy of a "
        f"file from a PRIVATE repo, vendored into a PUBLIC one: replace each citation with "
        f"{PLACEHOLDER!r} and leave every other byte as upstream wrote it. If you just ran the "
        f"refresh recipe in this directory's README, this is the scrub step it tells you to do next."
    )
