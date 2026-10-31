"""Mutation rows for the weekly selection.

What they break: the rule and its reader in ``tests/mutation_selection.py``, the
hooks ``tests/conftest.py`` composes, the Makefile's ``mutation-nightly`` target,
``ci.yml`` and the weekly workflow. Each row starts from the real file and sees the
judge in ``tests/test_mutation_selection.py`` pass it, plants one fault, asserts the
fault applied, and asserts the exact finding that judge reports.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mutation_selection
import test_mutation_selection as behaviour
from workflow_text import mutate

ROOT = behaviour.REPO_ROOT
SELECTION = "tests/mutation_selection.py"
CONFTEST = "tests/conftest.py"
ALPHA = ["test_alpha_rows.py::test_one", "test_alpha_rows.py::test_two"]
MIXED_PLAIN = ["test_mixed.py::test_plain", "test_mixed.py::test_rowboat"]
OTHER = ["test_other.py::test_other", "test_other.py::test_row_lookalike"]


def _source(rel: str) -> str:
    return (ROOT / rel).read_text()


def _selection_under(tmp_path: Path, rel: str, old: str, new: str) -> list[str]:
    """The selection judge on the real hooks, which must pass, then on ``rel`` mutated."""
    assert behaviour._selection_findings(behaviour._tree(tmp_path / "real")) == []
    mutant = mutate(_source(rel), old, new)
    tree = behaviour._tree(tmp_path / "mutant", sources={rel: mutant})
    return behaviour._selection_findings(tree)


def _refusals_under(tmp_path: Path, old: str, new: str) -> list[str]:
    assert behaviour._refusal_findings(tmp_path / "real") == []
    mutant = mutate(_source(SELECTION), old, new)
    return behaviour._refusal_findings(tmp_path / "mutant", {SELECTION: mutant})


def _reader_under(tmp_path: Path, old: str, new: str) -> list[str]:
    assert behaviour._reader_findings(mutation_selection) == []
    path = tmp_path / "mutant_selection.py"
    path.write_text(mutate(_source(SELECTION), old, new))
    spec = importlib.util.spec_from_file_location("mutant_selection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return behaviour._reader_findings(module)


def _target_under(tmp_path: Path, old: str, new: str) -> tuple[list[str], list[str]]:
    rows = mutation_selection.row_files()
    assert rows
    (tmp_path / "real").mkdir()
    assert behaviour._target_findings(*behaviour._run_target(tmp_path / "real"), rows) == []
    makefile = tmp_path / "Makefile"
    makefile.write_text(mutate(_source("Makefile"), old, new))
    (tmp_path / "mutant").mkdir()
    run = behaviour._run_target(tmp_path / "mutant", makefile)
    return behaviour._target_findings(*run, rows), rows


def _workflows_under(name: str, old: str, new: str) -> list[str]:
    texts = behaviour._workflows()
    assert behaviour._workflow_findings(texts) == []
    texts[name] = mutate(texts[name], old, new)
    return behaviour._workflow_findings(texts)


# --- the hooks -------------------------------------------------------------------


def test_a_pull_request_run_that_keeps_the_rows_is_caught(tmp_path):
    findings = _selection_under(
        tmp_path, SELECTION,
        "(keep if row == weekly else drop)", "(keep if row == weekly or not weekly else drop)",
    )
    assert findings == [f"the pull request run selected {sorted(behaviour.WEEKLY_RUN)}"]


def test_a_weekly_run_that_keeps_the_other_tests_is_caught(tmp_path):
    findings = _selection_under(
        tmp_path, SELECTION,
        "(keep if row == weekly else drop)", "(keep if row == weekly or weekly else drop)",
    )
    assert findings == [
        f"the weekly run of the rows files selected {MIXED_PLAIN}",
        f"the weekly run of tests/ selected {sorted(MIXED_PLAIN + OTHER)}",
    ]


def test_a_rule_that_forgets_the_MUTATION_prefix_is_caught(tmp_path):
    findings = _selection_under(
        tmp_path, SELECTION,
        'ROW_PREFIXES = ("test_row_", "test_MUTATION_")', 'ROW_PREFIXES = ("test_row_",)',
    )
    missed = ["test_mixed.py::test_MUTATION_b"]
    assert findings == [
        f"the pull request run selected {missed}",
        f"the weekly run of the rows files did not select {missed}",
        f"the weekly run of tests/ did not select {missed}",
    ]


def test_a_rule_that_forgets_the_rows_only_file_is_caught(tmp_path):
    findings = _selection_under(
        tmp_path, SELECTION,
        "return path.endswith(ROWS_ONLY_SUFFIX) or name.startswith(ROW_PREFIXES)",
        "return name.startswith(ROW_PREFIXES)",
    )
    assert findings == [
        f"the pull request run selected {ALPHA}",
        "the weekly run of the rows files exited 4",
        "the weekly run of tests/ exited 4",
    ]


def test_a_conftest_that_drops_the_selection_is_caught(tmp_path):
    findings = _selection_under(
        tmp_path, CONFTEST, "    mutation_selection.select(config, items)\n", ""
    )
    assert findings == [
        f"the pull request run selected {sorted(behaviour.WEEKLY_RUN)}",
        f"the weekly run of the rows files selected {MIXED_PLAIN}",
        f"the weekly run of tests/ selected {sorted(MIXED_PLAIN + OTHER)}",
    ]


def test_a_conftest_that_drops_the_producer_marker_is_caught(tmp_path):
    findings = _selection_under(
        tmp_path, CONFTEST,
        "    ingest_producer_marker.pytest_collection_modifyitems(config, items)\n", "",
    )
    assert findings == ["the pull request run selected ['test_other.py::test_needs_producer']"]


# --- the refusals ----------------------------------------------------------------


def test_accepting_a_rows_file_that_gives_no_test_is_caught(tmp_path):
    findings = _refusals_under(tmp_path, "        if silent:\n", "        if silent and False:\n")
    assert findings == ["accepted a rows file that gives the weekly run no test (exit 0)"]


def test_accepting_a_weekly_run_that_leaves_out_a_rows_file_is_caught(tmp_path):
    findings = _refusals_under(tmp_path, "        if absent:\n", "        if absent and False:\n")
    assert findings == ["refused a weekly run that leaves out a rows file for another reason"]


def test_accepting_a_weekly_run_that_selects_nothing_is_caught(tmp_path):
    findings = _refusals_under(tmp_path, "if not any(given.values()):", "if False:")
    assert findings == ["accepted a weekly run when `rows:` is empty (exit 5)"]


def test_accepting_a_rows_entry_that_names_no_file_is_caught(tmp_path):
    findings = _refusals_under(
        tmp_path,
        "missing = [entry for entry in entries if not (root / entry).is_file()]",
        "missing = []",
    )
    assert findings == [
        "accepted a rows entry that names no file, on a pull request (exit 0)",
        "refused a rows entry that names no file, weekly for another reason",
    ]


# --- the reader ------------------------------------------------------------------


def test_a_reader_that_accepts_a_glob_is_caught(tmp_path):
    findings = _reader_under(
        tmp_path,
        '_SEGMENT = re.compile(r"[A-Za-z0-9_.-]+")',
        '_SEGMENT = re.compile(r"[A-Za-z0-9_.*-]+")',
    )
    assert findings == ["accepted a glob: ['tests/*_rows.py']"]


def test_a_reader_that_reads_the_first_of_two_rows_keys_is_caught(tmp_path):
    findings = _reader_under(tmp_path, "if len(keys) > 1:", "if len(keys) > 2:")
    assert findings == ["accepted two rows keys: ['a.py']"]


# --- the weekly target -----------------------------------------------------------


def test_a_target_that_drops_a_rows_file_is_caught(tmp_path):
    findings, rows = _target_under(
        tmp_path,
        'files="$$($(PYTHON) tests/mutation_selection.py)"',
        'files="$$($(PYTHON) tests/mutation_selection.py | sed 1d)"',
    )
    assert findings == [f"the weekly target does not run rows file(s) {rows[:1]}"]


def test_a_target_that_drops_the_option_is_caught(tmp_path):
    findings, _ = _target_under(tmp_path, "-m pytest --mutation-rows -v", "-m pytest -v")
    assert findings == ["the weekly target's pytest run does not pass --mutation-rows"]


def test_a_target_that_does_not_install_the_pinned_pytest_is_caught(tmp_path):
    findings, _ = _target_under(
        tmp_path,
        "--disable-pip-version-check -r .github/requirements/test.txt",
        "--disable-pip-version-check pytest",
    )
    assert findings == [
        "the weekly target does not install .github/requirements/test.txt into its venv"
    ]


# --- the workflows ---------------------------------------------------------------


def test_a_pull_request_job_that_passes_the_option_is_caught():
    findings = _workflows_under(
        "ci.yml",
        "-r .github/requirements/sklearn.txt\n      - run: pytest tests/ -v\n",
        "-r .github/requirements/sklearn.txt\n      - run: pytest tests/ -v --mutation-rows\n",
    )
    assert findings == ["ci.yml: job test-sklearn passes --mutation-rows"]


def test_a_weekly_workflow_that_runs_on_pull_requests_is_caught():
    findings = _workflows_under(
        behaviour.WEEKLY.name, "\non:\n  schedule:\n", "\non:\n  pull_request:\n  schedule:\n"
    )
    assert findings == [f"{behaviour.WEEKLY.name} runs on pull_request"]
