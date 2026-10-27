"""ci.yml's ``ingest-schema-guard``: the vendored category enum held to the
producer's ``develop`` without running the PR's code or publishing the producer.

This repo is public, and so are its run logs and artifacts; the producer is
not. The guard job holds the producer token and the producer's schema, so:

* the comparison it runs is ``tools/`` at the merge base, extracted outside the
  checkout, never the PR's copy -- the token is readable by any code in the job;
* nothing leaves the job but its log, and the log carries one verdict: no
  artifact, no cache, no output, no report;
* ``category-contract``, the check branch protection requires, runs the PR's
  tree tests with no token and is red unless the guard succeeded.

Each property is RUN against a synthetic producer, and each has a mutation row
at the bottom that breaks it and sees it caught.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

import workflow_text as wf

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
TOOL_REL = "tools/ingest_category_contract.py"
VENDORED_REL = "tests/contracts/tracebloc_ingestor/ingest_categories.v1.json"
GUARD_JOB = "ingest-schema-guard"
GUARD_STEP = "Assert the vendored category enum matches the producer's"
GATE_JOB = "category-contract"
GATE_STEP = "The vendored category enum agrees with the producer's develop"


def _tool():
    spec = importlib.util.spec_from_file_location("_ingest_gate_for_guard", REPO_ROOT / TOOL_REL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _tool()


def _text() -> str:
    return WORKFLOW.read_text()


def _guard() -> str:
    return wf.job_block(_text(), GUARD_JOB)


def _guard_shell() -> str:
    return wf.step_shell(_guard(), GUARD_STEP)


def _checkouts(job_text: str) -> list[str]:
    return [s for s in wf.steps(job_text) if "actions/checkout@" in s]


def _producer_dir() -> str:
    """Where the guard job checks the producer out, read from the YAML."""
    producer = [s for s in _checkouts(_guard()) if "token:" in s]
    assert len(producer) == 1, f"expected ONE producer checkout in {GUARD_JOB}: {producer}"
    m = re.search(r"^\s*path:\s*(\S+)\s*$", producer[0], re.MULTILINE)
    assert m, f"the producer checkout declares no path:\n{producer[0]}"
    return m.group(1)


def _sparse_paths() -> list[str]:
    producer = [s for s in _checkouts(_guard()) if "token:" in s][0]
    lines = producer.splitlines()
    at = next(i for i, ln in enumerate(lines) if ln.strip() == "sparse-checkout: |")
    out = []
    for line in lines[at + 1 :]:
        if not line.strip().startswith("/"):
            break
        out.append(line.strip())
    return out


# --------------------------------------------------------------------------
# WIRING
# --------------------------------------------------------------------------
def test_the_guard_checks_out_every_candidate_path_in_order():
    """The sparse checkout is the only list of candidates besides
    ``CANDIDATE_PATHS``. A path added to one and not the other is a candidate
    the comparison says it tries and does not fetch."""
    assert _sparse_paths() == ["/" + p for p in gate.CANDIDATE_PATHS]


def test_the_guard_checkout_is_deep_and_neither_checkout_keeps_a_credential():
    own = [s for s in _checkouts(_guard()) if "repository:" not in s]
    assert len(own) == 1 and "fetch-depth: 0" in own[0], (
        "a shallow checkout cannot resolve `git merge-base origin/<base> HEAD`"
    )
    for step in _checkouts(_guard()):
        assert "persist-credentials: false" in step, step


def test_the_required_check_runs_the_tree_tests_and_waits_on_the_guard():
    job = wf.strip_comments(wf.job_block(_text(), GATE_JOB))
    assert f"needs: {GUARD_JOB}" in job
    assert "    if: ${{ !cancelled() }}" in job, "the job must run when the guard failed"
    assert "pytest tests/test_zoo_category_contract.py" in job
    assert f"needs.{GUARD_JOB}.result" in wf.step_shell(job, GATE_STEP) + job


#: Every value `needs.<job>.result` takes, plus empty for an expression typo.
_RESULTS = ("success", "failure", "cancelled", "skipped", "")


def _gate_shell() -> str:
    return wf.step_shell(wf.job_block(_text(), GATE_JOB), GATE_STEP)


def _gate_findings(tmp_path: Path, shell: str) -> list[str]:
    """Guard results the required check reads the wrong way."""
    return [
        result
        for result in _RESULTS
        if (wf.run_step(tmp_path, shell, {"GUARD_RESULT": result}).returncode == 0)
        != (result == "success")
    ]


def test_the_required_check_is_green_only_on_a_guard_success(tmp_path):
    """A skipped or cancelled guard compared nothing, and a check that read it
    as anything but red would pass a fork PR that could not look."""
    assert _gate_findings(tmp_path, _gate_shell()) == []


# --------------------------------------------------------------------------
# THE STEP, RUN against a synthetic producer
# --------------------------------------------------------------------------
#: Producer-only content: a category no vendored copy carries and prose the
#: vendored copy never takes, so either in the log can only have come from the
#: producer.
_PRIVATE_CATEGORY = "zz_unreleased_category_7f3a"
_PRIVATE_PROSE = "producer-internal description 7f3a"


def _vendored() -> list[str]:
    return json.loads((REPO_ROOT / VENDORED_REL).read_text())["categories"]


def _write_schema(producer: Path, enum, rel: str | None = None) -> None:
    path = producer / (rel or gate.CANDIDATE_PATHS[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "description": _PRIVATE_PROSE,
        "properties": {"category": {"enum": enum}, "internal_only": {"type": "string"}},
    }
    path.write_text(json.dumps(doc))


def _stage_repo(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    for rel in (TOOL_REL, VENDORED_REL):
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((REPO_ROOT / rel).read_bytes())
    wf.commit_base(root)
    return root


#: Every state the guard can meet, the verdict its log must give, and whether
#: the private report must name producer-only content (the negative control).
_SCENARIOS = {
    "aligned": ("OK", False),
    "producer-moved": ("DRIFT", True),
    "vendored-moved": ("DRIFT", False),
    "no-schema": ("NONE", False),
    "malformed": ("NONE", True),
    "crash": ("NONE", False),
}


def _stage_scenario(tmp_path: Path, kind: str) -> tuple[Path, str]:
    """A checkout and a producer in state ``kind``, and the ``base_ref`` to use."""
    root = _stage_repo(tmp_path)
    producer = root / _producer_dir()
    producer.mkdir()
    enum = _vendored()
    if kind == "producer-moved":
        enum = enum + [_PRIVATE_CATEGORY]
    if kind == "malformed":
        enum = {_PRIVATE_CATEGORY: 1}
    if kind != "no-schema":
        _write_schema(producer, enum)
    if kind == "vendored-moved":
        doc = json.loads((root / VENDORED_REL).read_text())
        doc["categories"] = doc["categories"][1:]
        wf.commit_pr_change(root, VENDORED_REL, json.dumps(doc, indent=2) + "\n")
        return root, "develop"
    if kind == "crash":
        wf.commit_pr_change(root, VENDORED_REL, "not json\n")
        return root, "develop"
    return root, ""


def _run(root: Path, base_ref: str, shell: str | None = None):
    return wf.run_step(root, _guard_shell() if shell is None else shell, {"BASE_REF": base_ref})


def _public(proc) -> str:
    return proc.stdout + proc.stderr


def _report(root: Path) -> str:
    return "".join(p.read_text() for p in sorted((root / "_temp").glob("ingest-schema.*/report.log")))


_VERDICT_LINES = {
    "OK": f"ingest schema OK: {VENDORED_REL} matches the producer's category enum",
    "DRIFT": f"INGEST CATEGORY DRIFT — {VENDORED_REL} disagrees with the producer's category enum",
    "NONE": f"INGEST CATEGORY CHECK FAILED — no verdict for {VENDORED_REL}",
}


def _verdict(public: str) -> str:
    found = [kind for kind, line in _VERDICT_LINES.items() if line in public]
    assert len(found) == 1, f"got {found or 'no'} verdict(s):\n{public}"
    return found[0]


_SECRETS = (_PRIVATE_CATEGORY, _PRIVATE_PROSE, "internal_only")


def _leaks(text: str) -> list[str]:
    return [s for s in _SECRETS if s in text]


@pytest.mark.parametrize("kind", sorted(_SCENARIOS))
def test_THE_LOG_GIVES_ONE_VERDICT_AND_NOTHING_FROM_THE_PRODUCER(tmp_path, kind):
    root, base_ref = _stage_scenario(tmp_path, kind)
    proc = _run(root, base_ref)
    public = _public(proc)
    assert _leaks(public) == [], f"the public log carries producer content:\n{public}"
    verdict, names_private = _SCENARIOS[kind]
    if names_private:
        # THE NEGATIVE CONTROL: the comparison did name producer-only content,
        # and the step withheld it.
        assert _leaks(_report(root)), f"the private report holds nothing private:\n{_report(root)}"
    assert _verdict(public) == verdict, public
    assert (proc.returncode == 0) is (verdict == "OK"), (proc.returncode, public)


def test_the_newest_candidate_is_compared_when_both_are_present(tmp_path):
    """The old path agrees and the new one does not: the new one decides."""
    root, _ = _stage_scenario(tmp_path, "aligned")
    producer = root / _producer_dir()
    _write_schema(producer, _vendored(), gate.CANDIDATE_PATHS[1])
    _write_schema(producer, _vendored() + [_PRIVATE_CATEGORY], gate.CANDIDATE_PATHS[0])
    assert _verdict(_public(_run(root, ""))) == "DRIFT"


_IMPOSTOR_MARKER = "THE PR'S COMPARISON RAN"


def _stage_impostor(tmp_path: Path) -> Path:
    """A PR that rewrites the tool to print the producer and agree, against a
    producer that moved, so the trusted comparison must say DRIFT."""
    root, _ = _stage_scenario(tmp_path, "producer-moved")
    impostor = (
        "import pathlib\n"
        f"print({_IMPOSTOR_MARKER!r})\n"
        f"for p in sorted(pathlib.Path({_producer_dir()!r}).rglob('*')):\n"
        "    if p.is_file():\n"
        "        print(p.read_text())\n"
        "class Refusal(SystemExit):\n    pass\n"
        "def select_from_tree(root):\n    return 'x', b''\n"
        "def categories_from(body, source):\n    return ()\n"
        "def load_vendored(path):\n    return ()\n"
        "def compare(*a):\n    return []\n"
    )
    wf.commit_pr_change(root, TOOL_REL, impostor)
    return root


def _isolation_findings(proc, root: Path) -> list[str]:
    findings = []
    if _IMPOSTOR_MARKER in _public(proc) + _report(root):
        findings.append("the PR's comparison ran in the token job")
    if proc.returncode == 0:
        findings.append("a producer that moved went green")
    return findings


def test_THE_TOKEN_JOB_RUNS_THE_MERGE_BASE_COMPARISON_NEVER_THE_PRS(tmp_path):
    root = _stage_impostor(tmp_path)
    proc = _run(root, "develop")
    assert _isolation_findings(proc, root) == [], _public(proc)
    assert _verdict(_public(proc)) == "DRIFT", _public(proc)


# --------------------------------------------------------------------------
# THE BOUNDARY, read from the YAML
# --------------------------------------------------------------------------
def _token_jobs(text: str) -> list[str]:
    """Jobs that mint a token for the producer or read the minted one."""
    found = []
    for job in wf.jobs(text):
        code = wf.strip_comments(wf.job_block(text, job))
        if f"repositories: {gate.PRODUCER_REPO}" in code or "steps.ingestor-token." in code:
            found.append(job)
    return found


def test_THE_PRODUCER_TOKEN_IS_IN_THE_GUARD_JOB_ALONE():
    assert _token_jobs(_text()) == [GUARD_JOB]


def test_NOTHING_LEAVES_THE_TOKEN_JOB():
    assert wf.publishers(_guard()) == []


def test_NO_OTHER_JOB_READS_THE_PRODUCER():
    text = _text()
    producer = re.compile(rf"(?<![\w-]){re.escape(_producer_dir())}(?![\w-])")
    readers = [
        job
        for job in wf.jobs(text)
        if job != GUARD_JOB and producer.search(wf.strip_comments(wf.job_block(text, job)))
    ]
    assert readers == []
    assert not re.search(r"name:\s*ingest-schema\s*$", wf.strip_comments(text), re.MULTILINE)


def test_EVERY_INTERPRETER_RUN_IN_THE_TOKEN_JOB_IS_THE_TRUSTED_COPY():
    assert any('"$trusted/' in body for body in wf.run_bodies(_guard())), (
        "the token job runs no trusted comparison at all, so this test reads nothing"
    )
    assert wf.untrusted_runs(_guard()) == []


# --------------------------------------------------------------------------
# MUTATION ROWS. Each starts from the real workflow and sees it clean, applies
# one change, asserts the change applied, and asserts the detector above that
# owns the property names it.
# --------------------------------------------------------------------------
def test_MUTATION_the_report_printed_to_the_log_is_caught(tmp_path):
    root, base_ref = _stage_scenario(tmp_path, "producer-moved")
    assert _leaks(_public(_run(root, base_ref))) == []
    mutant = wf.mutate(
        _guard_shell(),
        """> "$report" 2>&1 <<'PY' || status=$?""",
        """2>&1 <<'PY' | tee "$report" || status=$?""",
    )
    assert _leaks(_public(_run(root, base_ref, mutant)))


def test_MUTATION_running_the_prs_comparison_is_caught(tmp_path):
    root = _stage_impostor(tmp_path)
    assert _isolation_findings(_run(root, "develop"), root) == []
    old, new = 'python3 -I - "$trusted/tools"', "python3 -I - tools"
    proc = _run(root, "develop", wf.mutate(_guard_shell(), old, new))
    assert "the PR's comparison ran in the token job" in _isolation_findings(proc, root)
    assert wf.untrusted_runs(wf.mutate(_guard(), old, new))


def test_MUTATION_trusting_the_prs_head_is_caught(tmp_path):
    root = _stage_impostor(tmp_path)
    assert _isolation_findings(_run(root, "develop"), root) == []
    mutant = wf.mutate(_guard_shell(), 'git merge-base "origin/$BASE_REF" HEAD', "git rev-parse HEAD")
    proc = _run(root, "develop", mutant)
    assert "the PR's comparison ran in the token job" in _isolation_findings(proc, root)


def test_MUTATION_a_crash_read_as_drift_is_caught(tmp_path):
    root, base_ref = _stage_scenario(tmp_path, "crash")
    assert _verdict(_public(_run(root, base_ref))) == "NONE"
    mutant = wf.mutate(
        _guard_shell(),
        """[ "$status" -eq 1 ] && ! grep -q '^Traceback ' "$report\"""",
        '[ "$status" -eq 1 ]',
    )
    assert _verdict(_public(_run(root, base_ref, mutant))) == "DRIFT"


def test_MUTATION_an_artifact_from_the_token_job_is_caught():
    assert wf.publishers(_guard()) == []
    mutant = wf.mutate(
        _guard(),
        "          sparse-checkout-cone-mode: false\n",
        "          sparse-checkout-cone-mode: false\n"
        "      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4.6.2\n"
        "        with:\n"
        "          name: ingest-schema\n"
        f"          path: {_producer_dir()}\n",
    )
    assert wf.publishers(mutant) == ["upload-artifact"]


def test_MUTATION_the_producer_token_in_a_second_job_is_caught():
    text = _text()
    assert _token_jobs(text) == [GUARD_JOB]
    mint = re.search(r"^      - name: Mint an ingestor-read token\n(?:        .*\n)+", text, re.MULTILINE)
    assert mint, "no producer mint step to copy"
    anchor = "      - name: model_zoo/ agrees with the vendored category enum\n"
    mutant = wf.mutate(text, anchor, mint.group(0) + anchor)
    assert _token_jobs(mutant) == [GUARD_JOB, GATE_JOB]


def test_MUTATION_a_required_check_that_accepts_a_skipped_guard_is_caught(tmp_path):
    shell = _gate_shell()
    assert _gate_findings(tmp_path, shell) == []
    mutant = wf.mutate(shell, '[ "$GUARD_RESULT" != "success" ]', '[ "$GUARD_RESULT" = "failure" ]')
    assert "skipped" in _gate_findings(tmp_path, mutant)
