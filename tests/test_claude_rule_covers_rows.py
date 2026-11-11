"""The mutation-rows naming rule must load for every file that holds rows.

``.claude/rules/mutation-rows.md`` tells a session how to name a row, and Claude Code
loads it only when a file matching its ``paths:`` globs is read or edited. A rows file
the globs miss is one where a mis-named row silently runs on every pull request.
"""

from __future__ import annotations

import re
from pathlib import Path

import mutation_selection

ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT / ".claude" / "rules" / "mutation-rows.md"


def rule_globs(text: str) -> list[str]:
    """The ``paths:`` entries of a rule's front matter; [] when it has none."""
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert match, f"{RULE} has no front matter"
    block = re.search(r"^paths:\n((?:  - .*\n?)+)", match.group(1) + "\n", re.M)
    assert block, f"{RULE} front matter has no `paths:` list"
    return [line.split("-", 1)[1].strip().strip("\"'") for line in block.group(1).splitlines()]


def glob_matches(glob: str, path: str) -> bool:
    """Gitignore-style: ``**`` crosses ``/``, ``*`` does not."""
    pattern = "".join(
        ".*" if tok == "**" else "[^/]*" if tok == "*" else re.escape(tok)
        for tok in re.split(r"(\*\*|\*)", glob)
    )
    return re.fullmatch(pattern, path) is not None


def uncovered(globs: list[str], files: list[str]) -> list[str]:
    return [f for f in files if not any(glob_matches(g, f) for g in globs)]


def test_every_rows_file_loads_the_rule():
    files = mutation_selection.row_files(ROOT)
    assert files, "the declaration lists no rows files: nothing to check"
    missing = uncovered(rule_globs(RULE.read_text(encoding="utf-8")), files)
    assert not missing, (
        f"{RULE.relative_to(ROOT)} does not load for rows files {missing}: "
        "add a `paths:` glob that matches them"
    )


def test_every_glob_matches_a_tracked_path():
    for glob in rule_globs(RULE.read_text(encoding="utf-8")):
        assert any(p.is_file() for p in ROOT.glob(glob)), f"`paths:` glob {glob!r} matches no file"


def test_a_dropped_glob_is_reported():
    """Break the real rule: drop the glob that covers the mixed guard files."""
    text = RULE.read_text(encoding="utf-8")
    broken = text.replace('  - "tests/test_*_guard.py"\n', "")
    assert broken != text, "mutation did not apply"
    files = mutation_selection.row_files(ROOT)
    assert uncovered(rule_globs(text), files) == []
    assert "tests/test_engine_pin_drift_guard.py" in uncovered(rule_globs(broken), files)


def test_star_does_not_cross_a_slash():
    assert glob_matches("tests/*_rows.py", "tests/a_rows.py")
    assert not glob_matches("tests/*_rows.py", "tests/sub/a_rows.py")
    assert glob_matches("tools/**", "tools/sub/x.py")
