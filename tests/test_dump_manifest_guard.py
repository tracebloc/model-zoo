"""verify-dumps-engine-pin.yml's ``dump-manifest-guard``: the private dump
manifest checked against the templates without running the PR's code or
publishing the manifest.

This repo is public, and so are its run logs and artifacts; the manifest is
not. The guard job holds the backend token and the manifest, so:

* the checkers it runs are ``tools/`` at the merge base, extracted outside the
  checkout, never the PR's copies -- the token is readable by any code in the
  job -- and they read the PR's templates as data, importing none;
* nothing leaves the job but its log, and the log carries one verdict per
  check: no artifact, no cache, no output, no report.

Each property is RUN against a synthetic manifest derived from the real
templates, and each has a mutation row at the bottom that breaks it and sees
it caught.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

import workflow_text as wf

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "verify-dumps-engine-pin.yml"
COVERAGE_REL = "tools/check_dump_coverage.py"
VERIFIER_REL = "tools/verify_dumps_against_engine_pin.py"
SEED_INDEX_REL = "tools/seed_index.py"
CONTRACT_TEST_REL = "tests/test_model_contract.py"
GUARD_JOB = "dump-manifest-guard"
GUARD_STEP = "Assert the manifest and the templates account for each other"
SWEEP_JOB = "verify-dumps"
MANIFEST_REPO = "backend"
#: The one condition under which a job other than the guard may hold the
#: backend token: it builds templates, so it must never run a pull request's.
DEVELOP_ONLY = "    if: github.event_name != 'pull_request' && github.ref == 'refs/heads/develop'\n"


def _coverage():
    spec = importlib.util.spec_from_file_location("_coverage_for_guard", REPO_ROOT / COVERAGE_REL)
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(REPO_ROOT / "tools"))
    return mod


def _text() -> str:
    return WORKFLOW.read_text()


def _guard() -> str:
    return wf.job_block(_text(), GUARD_JOB)


def _guard_shell() -> str:
    return wf.step_shell(_guard(), GUARD_STEP)


def _checkouts(job_text: str) -> list[str]:
    return [s for s in wf.steps(job_text) if "actions/checkout@" in s]


def _manifest_checkout() -> str:
    found = [s for s in _checkouts(_guard()) if "token:" in s]
    assert len(found) == 1, f"expected ONE manifest checkout in {GUARD_JOB}: {found}"
    return found[0]


def _manifest_dir() -> str:
    """Where the guard job checks the manifest out, read from the YAML."""
    m = re.search(r"^\s*path:\s*(\S+)\s*$", _manifest_checkout(), re.MULTILINE)
    assert m, f"the manifest checkout declares no path:\n{_manifest_checkout()}"
    return m.group(1)


def _manifest_rel() -> str:
    """The manifest's path in the job: the checkout's path plus its one file."""
    m = re.search(r"^\s*sparse-checkout:\s*(\S+)\s*$", _manifest_checkout(), re.MULTILINE)
    assert m, f"the manifest checkout names no single sparse path:\n{_manifest_checkout()}"
    return f"{_manifest_dir()}/{m.group(1)}"


# --------------------------------------------------------------------------
# WIRING
# --------------------------------------------------------------------------
def test_the_step_reads_the_file_the_checkout_produces():
    assert f"manifest={_manifest_rel()}\n" in _guard_shell()


def test_the_guard_checkout_is_deep_and_neither_checkout_keeps_a_credential():
    own = [s for s in _checkouts(_guard()) if "repository:" not in s]
    assert len(own) == 1 and "fetch-depth: 0" in own[0], (
        "a shallow checkout cannot resolve `git merge-base origin/<base> HEAD`"
    )
    for step in _checkouts(_guard()):
        assert "persist-credentials: false" in step, step


# --------------------------------------------------------------------------
# THE STEP, RUN against a manifest derived from the real templates
# --------------------------------------------------------------------------
#: Manifest-only content: names, files and hashes no template and no public
#: file carries, so any of them in the log can only have come from the manifest.
_PRIVATE_RETIRED = "zz_private_retired_dump_9c1e"
_PRIVATE_ORPHAN = "zz_private_orphan_dump_9c1e"
_PRIVATE_PREFIX = "private-prefix-9c1e"
_PRIVATE_VERSION = "9.9.9c1e"


def _live_entries() -> dict[str, dict]:
    """One live entry per seed-expecting template, named the way the coverage
    tool looks for it -- derived from the real survey, never listed here."""
    coverage = _coverage()
    entries = {}
    for key, record in sorted(coverage.survey(REPO_ROOT).items()):
        if record["status"] != coverage.EXPECTS_SEED:
            continue
        name = record["candidates"][0]
        entries[name] = {
            "file": f"{name}_weights.pkl",
            "sha256": hashlib.sha256(f"private-{name}-9c1e".encode()).hexdigest(),
            "size_bytes": 1,
            "category": key.split("/", 1)[0],
        }
    assert entries, "the real survey found no seed-expecting template"
    return entries


def _manifest(kind: str) -> str:
    entries = _live_entries()
    entries[_PRIVATE_RETIRED] = {
        "file": f"{_PRIVATE_RETIRED}_weights.pkl",
        "sha256": "0" * 64,
        "status": "retired",
    }
    if kind == "orphan":
        entries[_PRIVATE_ORPHAN] = {"file": f"{_PRIVATE_ORPHAN}_weights.pkl", "sha256": "1" * 64}
    if kind == "missing":
        entries.pop(sorted(n for n in entries if n != _PRIVATE_RETIRED)[0])
    if kind == "all-retired":
        entries = {n: dict(e, status="retired") for n, e in entries.items()}
    doc = {
        "schema": 2,
        "prefix": _PRIVATE_PREFIX,
        "built_with": {"torch": _PRIVATE_VERSION},
        "entries": entries,
    }
    return json.dumps(doc, indent=1)


def _stage_repo(tmp_path: Path) -> Path:
    """The checkout the step reads: the tools it extracts, the templates it
    reads as data, and the contract test the verifier parses."""
    root = tmp_path / "checkout"
    wanted = [COVERAGE_REL, VERIFIER_REL, SEED_INDEX_REL, CONTRACT_TEST_REL]
    wanted += [p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "model_zoo").rglob("*.py")]
    for rel in wanted:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dst)
    wf.commit_base(root)
    return root


#: Every state the guard can meet, and the verdict its log must give each check.
_SCENARIOS = {
    "aligned": {"coverage": "OK", "resolve": "OK"},
    "orphan": {"coverage": "FAILED", "resolve": "FAILED"},
    "missing": {"coverage": "FAILED", "resolve": "OK"},
    "all-retired": {"coverage": "FAILED", "resolve": "NONE"},
    "not-json": {"coverage": "NONE", "resolve": "NONE"},
}


def _stage_scenario(tmp_path: Path, kind: str) -> Path:
    root = _stage_repo(tmp_path)
    manifest = root / _manifest_rel()
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{not json" if kind == "not-json" else _manifest(kind))
    return root


def _run(root: Path, base_ref: str = "", shell: str | None = None):
    return wf.run_step(root, _guard_shell() if shell is None else shell, {"BASE_REF": base_ref})


def _public(proc) -> str:
    return proc.stdout + proc.stderr


def _reports(root: Path) -> str:
    return "".join(
        p.read_text() for p in sorted((root / "_temp").glob("dump-manifest.*/*")) if p.is_file()
    )


_VERDICT_LINES = {
    "coverage": {
        "OK": "dump coverage OK: every seed-expecting template has a live dump, and every live dump a template",
        "FAILED": "DUMP COVERAGE FAILED — a template and the manifest disagree",
        "NONE": "DUMP COVERAGE CHECK FAILED — no verdict",
    },
    "resolve": {
        "OK": "manifest OK: every live entry resolves to one template",
        "FAILED": "MANIFEST ENTRIES UNRESOLVED — a live entry maps to no template, or is malformed",
        "NONE": "MANIFEST CHECK FAILED — no verdict",
    },
}


def _verdicts(public: str) -> dict[str, str]:
    out = {}
    for check, lines in _VERDICT_LINES.items():
        found = [kind for kind, line in lines.items() if line in public]
        assert len(found) == 1, f"{check} got {found or 'no'} verdict(s):\n{public}"
        out[check] = found[0]
    return out


_SECRETS = (_PRIVATE_RETIRED, _PRIVATE_ORPHAN, _PRIVATE_PREFIX, _PRIVATE_VERSION, "0" * 64, "1" * 64)


def _leaks(text: str) -> list[str]:
    return [s for s in _SECRETS if s in text]


@pytest.mark.parametrize("kind", sorted(_SCENARIOS))
def test_THE_LOG_GIVES_A_VERDICT_PER_CHECK_AND_NOTHING_FROM_THE_MANIFEST(tmp_path, kind):
    root = _stage_scenario(tmp_path, kind)
    proc = _run(root)
    public = _public(proc)
    assert _leaks(public) == [], f"the public log carries manifest content:\n{public}"
    if kind != "not-json":
        # THE NEGATIVE CONTROL: the checkers did name manifest-only content,
        # and the step withheld it.
        assert _PRIVATE_RETIRED in _reports(root), f"the reports hold nothing private:\n{_reports(root)}"
    assert _verdicts(public) == _SCENARIOS[kind], public
    assert (proc.returncode == 0) is (kind == "aligned"), (proc.returncode, public)


def test_THE_RETIRED_AND_NOT_GATED_LINES_STAY_ON_THE_RUNNER(tmp_path):
    """The two lines that used to put manifest entry names in every public log:
    the coverage checker's `RETIRED DUMP <name>` and the verifier's
    `NOT GATED: ... <names>`. Both are still written, to the reports."""
    root = _stage_scenario(tmp_path, "aligned")
    proc = _run(root)
    for prefix, line in (
        ("RETIRED DUMP", f"RETIRED DUMP  {_PRIVATE_RETIRED}"),
        ("NOT GATED", "NOT GATED: 1 retired entr(ies)"),
    ):
        assert line in _reports(root), f"{line!r} not in the reports:\n{_reports(root)}"
        assert prefix not in _public(proc), _public(proc)


def test_an_absent_manifest_is_named_and_red(tmp_path):
    root = _stage_repo(tmp_path)
    proc = _run(root)
    assert proc.returncode != 0
    assert "DUMP MANIFEST MISSING" in _public(proc), _public(proc)


_IMPOSTOR_MARKER = "THE PR'S CHECKER RAN"


def _stage_impostor(tmp_path: Path) -> Path:
    """A PR that rewrites both checkers and their shared index to print the
    manifest and pass, against a manifest with an orphan, so the trusted
    checkers must fail."""
    root = _stage_scenario(tmp_path, "orphan")
    impostor = (
        "import pathlib, sys\n"
        f"print({_IMPOSTOR_MARKER!r})\n"
        f"print(pathlib.Path({_manifest_rel()!r}).read_text())\n"
        "def main(argv=None):\n    return 0\n"
        "if __name__ == '__main__':\n    sys.exit(0)\n"
    )
    for rel in (COVERAGE_REL, VERIFIER_REL, SEED_INDEX_REL):
        wf.commit_pr_change(root, rel, impostor)
    return root


def _isolation_findings(proc, root: Path) -> list[str]:
    findings = []
    if _IMPOSTOR_MARKER in _public(proc) + _reports(root):
        findings.append("the PR's checker ran in the token job")
    if proc.returncode == 0:
        findings.append("a manifest with an orphan went green")
    return findings


def test_THE_TOKEN_JOB_RUNS_THE_MERGE_BASE_CHECKERS_NEVER_THE_PRS(tmp_path):
    root = _stage_impostor(tmp_path)
    proc = _run(root, "develop")
    assert _isolation_findings(proc, root) == [], _public(proc)
    assert _verdicts(_public(proc)) == _SCENARIOS["orphan"], _public(proc)


# --------------------------------------------------------------------------
# THE BOUNDARY, read from the YAML
# --------------------------------------------------------------------------
def _token_jobs(text: str) -> list[str]:
    """Jobs that mint a token for the manifest's repo or read the minted one."""
    found = []
    for job in wf.jobs(text):
        code = wf.strip_comments(wf.job_block(text, job))
        if f"repositories: {MANIFEST_REPO}" in code or "steps.backend-token." in code:
            found.append(job)
    return found


def _token_job_findings(text: str) -> list[str]:
    """Jobs holding the backend token that may run a pull request's code."""
    return [
        job
        for job in _token_jobs(text)
        if job != GUARD_JOB and DEVELOP_ONLY not in wf.job_block(text, job)
    ]


def test_THE_BACKEND_TOKEN_IS_IN_THE_GUARD_AND_THE_DEVELOP_ONLY_SWEEP_ALONE():
    assert _token_jobs(_text()) == [SWEEP_JOB, GUARD_JOB]
    assert _token_job_findings(_text()) == []


def test_NOTHING_LEAVES_A_TOKEN_JOB():
    text = _text()
    assert {job: wf.publishers(wf.job_block(text, job)) for job in _token_jobs(text)} == {
        SWEEP_JOB: [],
        GUARD_JOB: [],
    }


def test_NO_JOB_BUT_A_TOKEN_JOB_READS_THE_MANIFEST():
    text = _text()
    backend = re.compile(rf"(?<![\w-]){re.escape(_manifest_dir())}(?![\w-])")
    readers = [
        job for job in wf.jobs(text) if backend.search(wf.strip_comments(wf.job_block(text, job)))
    ]
    assert readers == _token_jobs(text)
    code = wf.strip_comments(text)
    assert not re.search(r"name:\s*dump-manifest\s*$", code, re.MULTILINE)
    assert "download-artifact" not in code and "upload-artifact" not in code


def test_EVERY_INTERPRETER_RUN_IN_THE_TOKEN_JOB_IS_THE_TRUSTED_COPY():
    assert any('"$trusted/' in body for body in wf.run_bodies(_guard())), (
        "the token job runs no trusted checker at all, so this test reads nothing"
    )
    assert wf.untrusted_runs(_guard()) == []


# --------------------------------------------------------------------------
# MUTATION ROWS. Each starts from the real workflow and sees it clean, applies
# one change, asserts the change applied, and asserts the detector above that
# owns the property names it.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "old, new",
    [
        (
            """> "$work/coverage.log" 2>&1 <<'PY' || status=$?""",
            """2>&1 <<'PY' | tee "$work/coverage.log" || status=$?""",
        ),
        ("""> "$work/resolve.log" 2>&1 || status=$?""", """2>&1 | tee "$work/resolve.log" || status=$?"""),
    ],
    ids=["coverage", "resolve"],
)
def test_MUTATION_a_report_printed_to_the_log_is_caught(tmp_path, old, new):
    root = _stage_scenario(tmp_path, "orphan")
    assert _leaks(_public(_run(root))) == []
    assert _leaks(_public(_run(root, shell=wf.mutate(_guard_shell(), old, new))))


@pytest.mark.parametrize(
    "old, new",
    [
        ('python3 -I - "$trusted/tools"', "python3 -I - tools"),
        ('python3 -I "$trusted/tools/verify_dumps_against_engine_pin.py"', "python3 -I tools/verify_dumps_against_engine_pin.py"),
    ],
    ids=["coverage", "resolve"],
)
def test_MUTATION_running_the_prs_checker_is_caught(tmp_path, old, new):
    root = _stage_impostor(tmp_path)
    assert _isolation_findings(_run(root, "develop"), root) == []
    proc = _run(root, "develop", wf.mutate(_guard_shell(), old, new))
    assert "the PR's checker ran in the token job" in _isolation_findings(proc, root)
    assert wf.untrusted_runs(wf.mutate(_guard(), old, new))


def test_MUTATION_trusting_the_prs_head_is_caught(tmp_path):
    root = _stage_impostor(tmp_path)
    assert _isolation_findings(_run(root, "develop"), root) == []
    mutant = wf.mutate(_guard_shell(), 'git merge-base "origin/$BASE_REF" HEAD', "git rev-parse HEAD")
    assert "the PR's checker ran in the token job" in _isolation_findings(_run(root, "develop", mutant), root)


@pytest.mark.parametrize("check, log", [("coverage", "coverage.log"), ("resolve", "resolve.log")])
def test_MUTATION_a_crash_read_as_a_finding_is_caught(tmp_path, check, log):
    root = _stage_scenario(tmp_path, "not-json")
    assert _verdicts(_public(_run(root)))[check] == "NONE"
    mutant = wf.mutate(
        _guard_shell(),
        f"""[ "$status" -eq 1 ] && ! grep -q '^Traceback ' "$work/{log}\"""",
        '[ "$status" -eq 1 ]',
    )
    assert _verdicts(_public(_run(root, shell=mutant)))[check] == "FAILED"


def test_MUTATION_an_artifact_from_the_token_job_is_caught():
    assert wf.publishers(_guard()) == []
    mutant = wf.mutate(
        _guard(),
        "          sparse-checkout-cone-mode: false\n",
        "          sparse-checkout-cone-mode: false\n"
        "      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4.6.2\n"
        "        with:\n"
        "          name: dump-manifest\n"
        f"          path: {_manifest_rel()}\n",
    )
    assert wf.publishers(mutant) == ["upload-artifact"]


def test_MUTATION_the_backend_token_in_a_job_that_runs_pull_requests_is_caught():
    text = _text()
    assert _token_job_findings(text) == []
    mint = re.search(r"^      - name: Mint a backend-read token\n(?:        .*\n)+", text, re.MULTILINE)
    assert mint, "no backend mint step to copy"
    anchor = "  seed-contract-drift:\n    needs: engine-pin-drift-guard\n"
    mutant = wf.mutate(text, anchor, anchor + "    steps:\n" + mint.group(0))
    assert _token_job_findings(mutant) == ["seed-contract-drift"]


def test_MUTATION_the_sweep_opened_to_pull_requests_is_caught():
    text = _text()
    assert _token_job_findings(text) == []
    mutant = wf.mutate(text, f"  {SWEEP_JOB}:\n    needs: [engine-pin-drift-guard]\n" + DEVELOP_ONLY,
                       f"  {SWEEP_JOB}:\n    needs: [engine-pin-drift-guard]\n")
    assert _token_job_findings(mutant) == [SWEEP_JOB]
