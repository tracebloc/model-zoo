"""The mutation rows run in the weekly run and never on a pull request.

``tests/mutation_selection.py`` holds the rule. These tests hold it, its hooks and
its wiring to what they claim, on every pull request:

  * the reader takes the ``rows:`` block list and refuses every other shape;
  * in a child pytest over a small tree, the hooks (through the real
    ``tests/conftest.py``) drop exactly the rows without ``--mutation-rows``, keep
    exactly the rows with it, and refuse each weekly run that would be narrower;
  * the weekly workflow's entry runs pytest with ``--mutation-rows`` on every rows
    file the declaration lists, after installing the pinned pytest into its venv;
  * no workflow passes ``--mutation-rows`` or runs the weekly target, and the weekly
    workflow runs on no pull request or push.

Each judge returns findings, so ``tests/test_mutation_selection_rows.py`` can plant
one fault and assert the exact finding.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import mutation_selection
import workflow_text

REPO_ROOT = Path(__file__).resolve().parents[1]
OPTION = mutation_selection.OPTION
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
WEEKLY = WORKFLOWS / "mutation-nightly.yml"
TEST_REQUIREMENTS = ".github/requirements/test.txt"
#: What the real conftest needs to load in a child tree.
CONFTEST_FILES = (
    "tests/conftest.py",
    "tests/ingest_producer_marker.py",
    "tests/mutation_selection.py",
    "tools/prep_offline_weights.py",
)
USAGE_ERROR = 4

# --- the child tree: two rows files and one that is not ------------------------

DECLARED = "guards:\n  - tools/*.py\nrows:\n  - tests/test_alpha_rows.py\n  - tests/test_mixed.py\n"
ROWS_FILES = ("tests/test_alpha_rows.py", "tests/test_mixed.py")
TESTS = {
    "tests/test_alpha_rows.py": "def test_one():\n    pass\n\n\ndef test_two():\n    pass\n",
    "tests/test_mixed.py": (
        "import pytest\n\n\n"
        "def test_row_a():\n    pass\n\n\n"
        "def test_MUTATION_b():\n    pass\n\n\n"
        "@pytest.mark.parametrize('n', [1, 2])\n"
        "def test_row_c(n):\n    pass\n\n\n"
        "def test_plain():\n    pass\n\n\n"
        "def test_rowboat():\n    pass\n"
    ),
    "tests/test_other.py": (
        "import pytest\n\n\n"
        "def test_row_lookalike():\n    pass\n\n\n"
        "def test_other():\n    pass\n\n\n"
        "@pytest.mark.ingest_producer\n"
        "def test_needs_producer():\n    pass\n"
    ),
}
#: Written down by hand, never derived from the rule.
PR_RUN = {
    "test_mixed.py::test_plain",
    "test_mixed.py::test_rowboat",
    "test_other.py::test_row_lookalike",
    "test_other.py::test_other",
}
WEEKLY_RUN = {
    "test_alpha_rows.py::test_one",
    "test_alpha_rows.py::test_two",
    "test_mixed.py::test_row_a",
    "test_mixed.py::test_MUTATION_b",
    "test_mixed.py::test_row_c[1]",
    "test_mixed.py::test_row_c[2]",
}
RUNS = (
    ("the pull request run", ("tests/",), PR_RUN),
    ("the weekly run of the rows files", (OPTION, *ROWS_FILES), WEEKLY_RUN),
    ("the weekly run of tests/", (OPTION, "tests/"), WEEKLY_RUN),
)
#: (label, declaration, extra files, pytest args, the refusal's own words)
REFUSALS = (
    (
        "a rows file that gives the weekly run no test",
        DECLARED + "  - tests/test_plain_only.py\n",
        {"tests/test_plain_only.py": "def test_plain():\n    pass\n"},
        (OPTION, *ROWS_FILES, "tests/test_plain_only.py"),
        "rows file(s) ['tests/test_plain_only.py'] gave the weekly run no test",
    ),
    (
        "a weekly run that leaves out a rows file",
        DECLARED,
        {},
        (OPTION, "tests/test_mixed.py"),
        "rows file(s) ['tests/test_alpha_rows.py'] were not collected",
    ),
    (
        "a weekly run when `rows:` is empty",
        "guards:\n  - tools/*.py\nrows: []\n",
        {},
        (OPTION, "tests/"),
        f"{OPTION} selected no mutation row",
    ),
    (
        "a rows entry that names no file, on a pull request",
        DECLARED + "  - tests/test_gone.py\n",
        {},
        ("tests/",),
        "`rows:` names ['tests/test_gone.py'], which is not a file",
    ),
    (
        "a rows entry that names no file, weekly",
        DECLARED + "  - tests/test_gone.py\n",
        {},
        (OPTION, "tests/"),
        "`rows:` names ['tests/test_gone.py'], which is not a file",
    ),
    (
        "a declaration the reader refuses, on a pull request",
        "guards:\n  - tools/*.py\nrows: [tests/test_mixed.py]\n",
        {},
        ("tests/",),
        "has the inline value",
    ),
)


def _tree(root: Path, declaration: str = DECLARED, files: dict | None = None,
          sources: dict | None = None) -> Path:
    """A repo of ``files`` under ``declaration``, hooked by the real conftest.
    ``sources`` replaces a conftest file's text (a planted fault)."""
    for rel in CONFTEST_FILES:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text((sources or {}).get(rel) or (REPO_ROOT / rel).read_text())
    written = {".github/mutation-coverage.yml": declaration, **TESTS, **(files or {})}
    for rel, text in written.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def _collect(root: Path, *args: str) -> tuple[int, set, str]:
    """A child ``pytest --collect-only`` in ``root``: exit code, selected ids by
    file name (the rootdir varies with the arguments), and everything it printed."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *args],
        cwd=root, env=env, capture_output=True, text=True, timeout=300, check=False,
    )
    ids = {
        Path(line.split("::", 1)[0]).name + "::" + line.split("::", 1)[1]
        for line in proc.stdout.splitlines()
        if "::" in line
    }
    return proc.returncode, ids, proc.stdout + proc.stderr


def _selection_findings(root: Path) -> list[str]:
    findings = []
    for label, args, expected in RUNS:
        code, got, _ = _collect(root, *args)
        if code != 0:
            findings.append(f"{label} exited {code}")
            continue
        if got - expected:
            findings.append(f"{label} selected {sorted(got - expected)}")
        if expected - got:
            findings.append(f"{label} did not select {sorted(expected - got)}")
    return findings


def _refusal_findings(base: Path, sources: dict | None = None) -> list[str]:
    findings = []
    for n, (label, declaration, files, args, words) in enumerate(REFUSALS):
        root = _tree(base / f"refusal{n}", declaration, files, sources)
        code, _, output = _collect(root, *args)
        if code != USAGE_ERROR:
            findings.append(f"accepted {label} (exit {code})")
        elif words not in output:
            findings.append(f"refused {label} for another reason")
    return findings


# --- the reader ----------------------------------------------------------------

READS = (
    (
        (
            "guards:\n  - tools/*.py\n# the row files\nrows:   # by name\n"
            "  - tests/a.py   # a comment\n\n  - .github/b_rows.py\n"
            "exempt:\n  - path: tools/x.py\n    ticket: o/r#1\n"
        ),
        ["tests/a.py", ".github/b_rows.py"],
    ),
    ("rows:\n- tests/a.py\nexempt: []\n", ["tests/a.py"]),
    ("guards:\n  - tools/*.py\nrows: []\nexempt: []\n", []),
    ("rows:\n  - tests/a.py", ["tests/a.py"]),
)
REFUSED_READS = (
    ("no rows key", "guards:\n  - tools/*.py\n", "no top-level `rows:`"),
    ("a nested rows key", "x:\n  rows:\n    - a.py\n", "no top-level `rows:`"),
    ("two rows keys", "rows:\n  - a.py\nrows:\n  - b.py\n", "appears 2 times"),
    ("a flow list", "rows: [tests/a.py]\n", "inline value"),
    ("a null", "rows: ~\n", "inline value"),
    ("an alias", "rows: *rows\n", "inline value"),
    ("no value", "rows:\nexempt: []\n", "has no value"),
    ("items after []", "rows: []\n  - a.py\n", "followed by items"),
    ("a mapping item", "rows:\n  - path: tests/a.py\n", "not a plain repo-relative path"),
    ("a glob", "rows:\n  - tests/*_rows.py\n", "not a plain repo-relative path"),
    ("a quoted path", "rows:\n  - 'tests/a.py'\n", "not a plain repo-relative path"),
    ("an absolute path", "rows:\n  - /tests/a.py\n", "not a plain repo-relative path"),
    ("a dot-slash path", "rows:\n  - ./tests/a.py\n", "not a plain repo-relative path"),
    ("a parent path", "rows:\n  - tests/../a.py\n", "not a plain repo-relative path"),
    ("mixed indents", "rows:\n  - a.py\n    - b.py\n", "mixed indents"),
    ("a line that is not an item", "rows:\n  - a.py\n  b.py\n", "is not a `- <path>` item"),
    ("an empty item", "rows:\n  -\n", "is not a `- <path>` item"),
    ("a tab", "rows:\n\t- a.py\n", "a tab"),
    ("a duplicate", "rows:\n  - a.py\n  - a.py\n", "listed twice"),
)


def _reader_findings(module) -> list[str]:
    """``module.read_rows`` against the shapes above, written down by hand."""
    findings = []
    for text, expected in READS:
        try:
            got = module.read_rows(text)
        except module.Refusal as exc:
            findings.append(f"refused {text!r}: {exc}")
            continue
        if got != expected:
            findings.append(f"read {text!r} as {got}, not {expected}")
    for label, text, words in REFUSED_READS:
        try:
            got = module.read_rows(text)
        except module.Refusal as exc:
            if words not in str(exc):
                findings.append(f"refused {label} for another reason")
            continue
        findings.append(f"accepted {label}: {got}")
    return findings


# --- the weekly target and the workflows ---------------------------------------


def _weekly_entry(text: str) -> list[str]:
    """The command the weekly workflow runs, from its ``ENTRY`` literal."""
    found = re.findall(r"^\s+ENTRY:\s*\$\{\{\s*'([^']*)'\s*\}\}\s*$", text, re.MULTILINE)
    assert len(found) == 1, f"expected one ENTRY literal in {WEEKLY.name}, found {found}"
    return shlex.split(found[0])


def _run_target(tmp_path: Path, makefile: Path | None = None):
    """The weekly entry, run against ``makefile`` with a venv whose python records
    each call and runs nothing. Returns the make process and the calls."""
    entry = _weekly_entry(WEEKLY.read_text())
    assert entry[:1] == ["make"], f"the weekly entry is not a make target: {entry}"
    assert shutil.which("make"), "make is not on PATH, so the weekly target cannot be read"
    log = tmp_path / "calls.jsonl"
    recorder = tmp_path / "record.py"
    recorder.write_text(
        "import json, os, sys\n"
        "with open(os.environ['MUTATION_TARGET_LOG'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    shim = venv / "bin" / "python"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{recorder}" "$@"\n')
    shim.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MAKELEVEL", "MFLAGS")}
    env.update(MUTATION_VENV=str(venv), PYTHON=sys.executable, MUTATION_TARGET_LOG=str(log),
               MUTATION_LEG="1", MUTATION_LEGS="1")
    proc = subprocess.run(
        [entry[0], "-f", str(makefile or REPO_ROOT / "Makefile"), *entry[1:]],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300, check=False,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return proc, calls


def _target_findings(proc, calls, rows: list[str]) -> list[str]:
    if proc.returncode != 0:
        return [f"the weekly target exited {proc.returncode}: {proc.stderr.strip()[-300:]}"]
    findings = []
    installs = [c for c in calls if c[:3] == ["-m", "pip", "install"]]
    if not any(TEST_REQUIREMENTS in c for c in installs):
        findings.append(f"the weekly target does not install {TEST_REQUIREMENTS} into its venv")
    runs = [c[2:] for c in calls if c[:2] == ["-m", "pytest"]]
    if len(runs) != 1:
        return findings + [f"the weekly target ran pytest {len(runs)} times"]
    if OPTION not in runs[0]:
        findings.append(f"the weekly target's pytest run does not pass {OPTION}")
    files = [a for a in runs[0] if not a.startswith("-")]
    missing = [r for r in rows if r not in files]
    extra = [f for f in files if f not in rows]
    if missing:
        findings.append(f"the weekly target does not run rows file(s) {missing}")
    if extra:
        findings.append(f"the weekly target runs {extra}, which `rows:` does not list")
    return findings


def _triggers(text: str) -> list[str]:
    """The event names under a workflow's top-level ``on:``."""
    lines = workflow_text.strip_comments(text).splitlines()
    assert lines.count("on:") == 1, "expected one top-level `on:` block"
    events = []
    for line in lines[lines.index("on:") + 1:]:
        if line.strip() and not line.startswith(" "):
            break
        m = re.match(r"^  ([A-Za-z_]+):", line)
        if m:
            events.append(m.group(1))
    assert events, "no events parsed under `on:`"
    return events


def _workflow_findings(texts: dict) -> list[str]:
    """``texts``: every workflow's text by file name."""
    findings = []
    for name, text in sorted(texts.items()):
        for job in workflow_text.jobs(text):
            for body in workflow_text.run_bodies(workflow_text.job_block(text, job)):
                if re.search(rf"(?<![\w-]){re.escape(OPTION)}(?![\w-])", body):
                    findings.append(f"{name}: job {job} passes {OPTION}")
                if re.search(r"\bmake\b[^\n]*\bmutation-nightly\b", body):
                    findings.append(f"{name}: job {job} runs make mutation-nightly")
    for event in _triggers(texts[WEEKLY.name]):
        if event in ("pull_request", "pull_request_target", "push", "merge_group"):
            findings.append(f"{WEEKLY.name} runs on {event}")
    return findings


def _workflows() -> dict:
    texts = {p.name: p.read_text() for p in sorted(WORKFLOWS.glob("*.yml"))}
    assert WEEKLY.name in texts and "ci.yml" in texts, sorted(texts)
    return texts


# --- the tests -----------------------------------------------------------------


def test_the_reader_takes_the_block_list_and_refuses_every_other_shape():
    assert _reader_findings(mutation_selection) == []


def test_the_hooks_drop_exactly_the_rows_on_a_pull_request_and_keep_exactly_them_weekly(
    tmp_path,
):
    assert _selection_findings(_tree(tmp_path)) == []


def test_the_pull_request_run_reports_what_it_deselected(tmp_path):
    code, _, output = _collect(_tree(tmp_path), "tests/")
    assert code == 0, output
    assert "mutation rows: 6 deselected" in output, output


def test_every_weekly_run_that_would_be_narrower_is_refused(tmp_path):
    assert _refusal_findings(tmp_path) == []


def test_the_weekly_entry_runs_every_rows_file_with_the_option(tmp_path):
    rows = mutation_selection.row_files()
    assert rows, "the declaration lists no rows file, so this compares nothing"
    assert _target_findings(*_run_target(tmp_path), rows) == []


def test_no_workflow_runs_the_rows_and_the_weekly_one_runs_on_no_pull_request():
    assert _workflow_findings(_workflows()) == []


def test_every_real_rows_file_gives_the_weekly_run_a_row_and_the_pull_request_none():
    rows = mutation_selection.row_files()
    assert rows, "the declaration lists no rows file"
    code, weekly, output = _collect(REPO_ROOT, OPTION, *rows)
    assert code == 0, output
    code, pull_request, output = _collect(REPO_ROOT, *rows)
    assert code == 0, output
    for path in rows:
        assert any(i.startswith(Path(path).name + "::") for i in weekly), path
    assert not weekly & pull_request, sorted(weekly & pull_request)
