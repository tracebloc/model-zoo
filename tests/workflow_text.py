"""Reading a workflow's jobs and steps as text, and running a step's shell.

Shared by the tests that pin a token job's public boundary
(``tests/test_ingest_schema_guard.py``, ``tests/test_dump_manifest_guard.py``).
No PyYAML: these suites run in the CI framework envs, and none installs it.
Every extraction failure is an assertion, never a skip.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

#: Ways a job hands what it holds to something that outlives it or to another
#: job, every one of them readable from a public repo's run.
PUBLISHERS = ("upload-artifact", "actions/cache", "GITHUB_STEP_SUMMARY", "outputs:")


def jobs(text: str) -> list[str]:
    """The workflow's job ids, in order."""
    assert "\njobs:\n" in text, "no top-level `jobs:` key"
    body = text.split("\njobs:\n", 1)[1]
    found = re.findall(r"^  ([A-Za-z0-9_-]+):\n", body, re.MULTILINE)
    assert found, "no jobs parsed"
    return found


def job_block(text: str, job: str) -> str:
    """One job's YAML, sliced to the next top-level key at its indent."""
    key = f"\n  {job}:\n"
    assert key in text, f"no job named {job!r}"
    rest = text[text.index(key) + len(key) :]
    offset = 0
    for line in rest.splitlines(keepends=True):
        if line.strip() and not line.startswith("   "):
            return rest[:offset]
        offset += len(line)
    return rest


def strip_comments(text: str) -> str:
    """Text without its full-line comments, so prose cannot satisfy a check."""
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))


def steps(job_text: str) -> list[str]:
    """The job's steps, each as comment-free text."""
    return re.split(r"(?m)^      - ", strip_comments(job_text))[1:]


def step_shell(job_text: str, name: str) -> str:
    """The ``run: |`` body of the step called ``name``, dedented and runnable.

    EXTRACTED, never restated: a step edited in the workflow and not in a test
    fails the test rather than drifting from it.
    """
    lines = job_text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip() == f"- name: {name}"), None)
    assert start is not None, f"no step named {name!r}; rename it in the test in the same commit"
    marker = len(lines[start]) - len(lines[start].lstrip())
    run_at = None
    for i in range(start + 1, len(lines)):
        if lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) <= marker:
            break
        if lines[i].strip() in ("run: |", "run: |-"):
            run_at = i
            break
    assert run_at is not None, f"step {name!r} has no block `run: |` body"
    indent = len(lines[run_at]) - len(lines[run_at].lstrip()) + 2
    body = []
    for line in lines[run_at + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) < indent:
            break
        body.append(line[indent:])
    shell = "\n".join(body).rstrip() + "\n"
    assert shell.strip(), f"step {name!r} has an empty run body"
    return shell


def run_bodies(job_text: str) -> list[str]:
    """Every ``run:`` body in a job, comment-free."""
    lines = job_text.splitlines()
    bodies = []
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)(?:- )?run:\s*(.*)$", line)
        if not m:
            continue
        indent, rest = len(m.group(1)), m.group(2).strip()
        if rest not in ("|", "|-", ">", ">-"):
            bodies.append(rest)
            continue
        body = []
        for nxt in lines[i + 1 :]:
            if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= indent:
                break
            body.append(nxt)
        bodies.append(strip_comments("\n".join(body)))
    return bodies


def publishers(job_text: str) -> list[str]:
    code = strip_comments(job_text)
    return [marker for marker in PUBLISHERS if marker in code]


def untrusted_runs(job_text: str) -> list[str]:
    """Interpreter runs and installs in a job that are not the trusted copy.

    Trusted means isolated (``-I``) and reading its code from the extracted
    ``$trusted`` tree: a script under it, or stdin (``-``) with that tree as
    its first argument. Any ``pip``, ``pytest`` or ``make`` runs this
    checkout's code or its dependencies, so it is never trusted.
    """
    found = []
    for body in run_bodies(job_text):
        for m in re.finditer(r"\bpython3?((?:\s+-[A-Za-z]+)*)\s+(\S+)(?:[ \t]+(\S+))?", body):
            flags, arg, nxt = m.group(1).split(), m.group(2), m.group(3) or ""
            trusted = "-I" in flags and (
                arg.startswith('"$trusted/') or (arg == "-" and nxt.startswith('"$trusted/'))
            )
            if not trusted:
                found.append(m.group(0))
        found += [m.group(0) for m in re.finditer(r"\b(?:pip3?|pytest|make)\b[^\n]*", body)]
    return found


def mutate(text: str, old: str, new: str) -> str:
    """``text`` with ``old`` replaced once; asserts the mutation applied."""
    assert text.count(old) == 1, f"mutation anchor is not there exactly once: {old!r}"
    mutant = text.replace(old, new)
    assert mutant != text, "the mutation did not apply, so it proves nothing"
    return mutant


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def commit_base(root: Path, base_branch: str = "develop") -> None:
    """Commit ``root`` as the tip of ``origin/<base_branch>``.

    A git repo because a guard step takes its checker out of git at the merge
    base, never from the working tree. A local ref of that name is enough.
    """
    git(root, "init", "-q", "-b", base_branch)
    git(root, "config", "user.email", "t@t.invalid")
    git(root, "config", "user.name", "t")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    git(root, "update-ref", f"refs/remotes/origin/{base_branch}", "HEAD")


def commit_pr_change(root: Path, rel: str, text: str) -> None:
    """One commit on top of the base: what the PR changed."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    git(root, "add", rel)
    git(root, "commit", "-q", "-m", f"change {rel}")


def run_step(root: Path, shell: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    """``shell`` run the way a job runs it, with ``python3`` this interpreter.

    ``RUNNER_TEMP`` is emptied and set on every run, so what a test reads back
    under it is this run's. ``bash`` is whichever the ambient PATH offers
    (``GUARD_TEST_BASH`` overrides it), so the suite can run under both
    bash 5 and the 3.2 macOS ships.
    """
    script = root / "_step.sh"
    script.write_text(shell)
    shim_dir = root / "_bin"
    shim_dir.mkdir(exist_ok=True)
    shim = shim_dir / "python3"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    shim.chmod(0o755)
    temp = root / "_temp"
    shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir()
    full = dict(os.environ)
    full["PATH"] = f"{shim_dir}{os.pathsep}{full.get('PATH', '')}"
    full["RUNNER_TEMP"] = str(temp)
    full.update(env)
    bash = os.environ.get("GUARD_TEST_BASH", "bash")
    proc = subprocess.run(
        [bash, str(script)], cwd=str(root), capture_output=True, text=True, env=full
    )
    # A step killed by a signal said nothing, and that is not any verdict.
    assert proc.returncode >= 0, f"the step was killed by signal {-proc.returncode}"
    return proc
