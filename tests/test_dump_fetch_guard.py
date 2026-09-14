"""The dump-fetch step in verify-dumps-engine-pin.yml must name the precondition
it is missing, and must invoke the fetch hook with arguments the hook accepts
(the arming overclaim).

Why this is a test and not a comment
------------------------------------
The step used to guard with a single
``[ -f manifest.json ] && [ -f tools/sync_zoo_weights.py ]`` and print one
message for every no-op: *"dumps hosting is the hosting decision."* Two distinct
failures wore that one sentence:

  * the store location was undecided (true, and (internal ref)'s to decide); and
  * ``tools/sync_zoo_weights.py`` was **not in the repo at all** — it existed
    only as an uncommitted file on one laptop, so ``-f`` selected the no-op
    branch on every run and the message blamed hosting.

The second one survived for weeks precisely because nothing watched the branch
being taken. So the branches are asserted here, from the workflow's own shell —
the script under test is EXTRACTED from the YAML rather than restated, so a
guard edited in the workflow and not here fails instead of drifting.

The invocation is asserted against the REAL tool, not a stub, because the
committed call was wrong in a second way nothing could see: it passed
``--manifest/--out``, flags ``sync_zoo_weights.py`` has never had. With the
tool absent that call was unreachable; with it present it would have died on
argparse the first time a manifest landed.

The NEXT hop, and why it is in this file
----------------------------------------
Fixing the workflow-to-hook interface exposed the same defect one step further
along: the fetch hook and the gate it feeds read **the same** ``manifest.json``
under **different** top-level keys — ``entries`` for the hook, ``dumps`` for
``verify_dumps_against_engine_pin.py`` — so no single-key manifest can take the
job green, in either direction. That was recorded as a "known gap" in the
hook's docstring and deferred to the schema reconciliation, but nothing measured which side
reddens first or drove both, and both tools' skip messages still promised a
sweep "the moment a manifest lands". The two steps live in one job, so the
agreement between them is asserted here, end to end, from the workflow's own
shell.

Deliberately torch-free. ``tests/test_verify_dumps_against_engine_pin.py``
opens with ``pytest.importorskip("torch")``, so it is exercised in exactly one
of the three CI framework envs; the branches asserted below all return before
the verifier's first ``import torch``, so they run in all three.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from test_check_dump_coverage import _job_block

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "verify-dumps-engine-pin.yml"
STEP_NAME = "Obtain staged dumps from the model store"
TOOL_REL = Path("tools") / "sync_zoo_weights.py"
VERIFIER_REL = Path("tools") / "verify_dumps_against_engine_pin.py"
STORE_URI = "s3://zoo-weights-test-bucket/zoo-weights"


# --------------------------------------------------------------------------
# Extracting the step's shell out of the workflow
# --------------------------------------------------------------------------
# Hand-rolled rather than PyYAML: none of the three CI framework envs
# (.github/requirements/{pytorch,sklearn,survival}.txt) install PyYAML, and a
# test that imports it would skip in all three — i.e. never run, which is the
# state this file exists to end. Every extraction failure below is an assertion,
# never a skip, so restructuring the YAML breaks this loudly.


def _step_block(text: str) -> list[str]:
    """Return the lines of the ``- name: <STEP_NAME>`` step, marker included."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == f"- name: {STEP_NAME}":
            start = i
            break
    assert start is not None, (
        f"no step named {STEP_NAME!r} in {WORKFLOW}. If it was renamed, rename "
        "STEP_NAME here in the same commit."
    )
    marker_indent = len(lines[start]) - len(lines[start].lstrip())
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if line.strip() and (len(line) - len(line.lstrip())) <= marker_indent:
            break
        block.append(line)
    return block


def _run_script(block: list[str]) -> str:
    """Dedent the step's ``run: |`` block body."""
    run_at = None
    for i, line in enumerate(block):
        if line.strip() in ("run: |", "run: |-"):
            run_at = i
            break
    assert run_at is not None, f"step {STEP_NAME!r} has no literal `run: |` block"
    run_indent = len(block[run_at]) - len(block[run_at].lstrip())
    body = []
    for line in block[run_at + 1 :]:
        if line.strip() and (len(line) - len(line.lstrip())) <= run_indent:
            break
        body.append(line[run_indent + 2 :] if line.strip() else "")
    script = "\n".join(body)
    assert "mkdir -p dist" in script, (
        "extracted the wrong text for the fetch step — expected its shell, got:\n"
        f"{script[:400]}"
    )
    return script


@pytest.fixture(scope="module")
def guard_script() -> str:
    return _run_script(_step_block(WORKFLOW.read_text()))


# --------------------------------------------------------------------------
# Fixture: a checkout-shaped tmp dir + stubbed `aws` and `python3`
# --------------------------------------------------------------------------

_AWS_STUB = """#!/bin/sh
# Minimal `aws s3 cp <src> <dst>` that serves objects out of $FAKE_STORE.
[ "$1" = "s3" ] || { echo "aws stub: unexpected argv: $*" >&2; exit 64; }
[ "$2" = "cp" ] || { echo "aws stub: unexpected argv: $*" >&2; exit 64; }
src="$3"; dst="$4"
rel=`echo "$src" | sed -e 's|^s3://||'`
if [ ! -f "$FAKE_STORE/$rel" ]; then
  echo "aws stub: no such object: $rel" >&2
  exit 1
fi
mkdir -p "`dirname "$dst"`"
cp "$FAKE_STORE/$rel" "$dst"
"""


# A wrapper that makes the interpreter refuse every import the skipped install
# could be providing, then runs the verifier as `__main__`. The verifier
# resolves its repo root from `__file__`, which runpy sets to the path handed
# in, so its default --manifest/--dumps-dir still point at this checkout.
#
# ALLOW-list, not a deny-list. An earlier version named nine packages to block;
# that list was a restatement of tools/requirements-engine-pin.txt, already
# omitted a direct pin (scikit-learn -> `sklearn`) and every transitive the
# install brings (packaging, huggingface_hub, tokenizers, PIL, ...), and so
# would have stayed green while the runner's bare python3 raised
# ModuleNotFoundError. What the skip path actually has is the interpreter's
# own stdlib plus this repo's modules -- so that is what is allowed, derived:
# `sys.stdlib_module_names` from the interpreter, the repo's module names from
# a directory listing (argv[2], see `_repo_modules`). Anything else is exactly
# "something the install provides", whatever its name.
_NO_ENGINE_PIN_RUNNER = """import importlib.abc
import runpy
import sys

ALLOWED = set(sys.stdlib_module_names) | set(sys.argv[2].split(","))


class _NoEnginePin(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] not in ALLOWED:
            raise ImportError(
                f"BLOCKED: {name} (not stdlib and not a module of this repo -- "
                "only the skipped engine-pin install could provide it)"
            )


sys.meta_path.insert(0, _NoEnginePin())
sys.argv = [sys.argv[1]]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def _repo_modules() -> set[str]:
    """Top-level module names a checkout of THIS repo provides, by listing --
    the two packages and every script under tools/ (a sibling script is
    importable from the verifier's own directory)."""
    names = {p.stem for p in (REPO_ROOT / "tools").glob("*.py")}
    names |= {"tools", "model_zoo"}
    assert names > {"tools", "model_zoo"}, f"tools/ listing came back empty: {names}"
    return names


def _write_exec(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class Checkout:
    """A tmp dir shaped like a checkout, plus a fake store to fetch from."""

    def __init__(self, root: Path, script: str):
        self.root = root
        self.script = script
        self.store = root / "_fake_store"
        self.bin = root / "_bin"
        (root / "tools").mkdir()
        self.store.mkdir()
        self.bin.mkdir()
        _write_exec(self.bin / "aws", _AWS_STUB)
        # `python3` in the workflow must be THIS interpreter, not whatever the
        # ambient PATH offers.
        _write_exec(
            self.bin / "python3",
            f'#!/bin/sh\nexec "{sys.executable}" "$@"\n',
        )

    def install_real_tool(self) -> None:
        shutil.copy2(REPO_ROOT / TOOL_REL, self.root / TOOL_REL)

    def install_real_verifier(self) -> None:
        """Copy the gate the fetch step feeds, so the two can be driven in the
        order the job runs them. The verifier resolves its own repo root (and
        therefore its default --manifest and --dumps-dir) from ``__file__``, so
        a copy inside this checkout points at this checkout."""
        shutil.copy2(REPO_ROOT / VERIFIER_REL, self.root / VERIFIER_REL)

    def serve(self, name: str, payload: bytes) -> str:
        """Put one object in the fake store, content-addressed as the hook
        expects, and return its sha256. Does NOT touch manifest.json — the
        callers below write the manifest in whichever shape they are testing."""
        sha = hashlib.sha256(payload).hexdigest()
        fname = f"{name}_weights.pkl"
        obj = self.store / STORE_URI[len("s3://") :] / name / sha[:12] / fname
        obj.parent.mkdir(parents=True, exist_ok=True)
        obj.write_bytes(payload)
        return sha

    def write_manifest(self, manifest: dict) -> None:
        (self.root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )

    def run_verifier(self, *, engine_pin_blocked: bool = False) -> subprocess.CompletedProcess:
        """Run the gate exactly as the next workflow step does: no arguments, so
        every path it resolves is a default.

        ``engine_pin_blocked=True`` runs it in an interpreter that REFUSES to
        import anything the engine-pin install provides -- what the job's
        skip path amounts to now that the install is conditional."""
        if engine_pin_blocked:
            runner = self.root / "_no_engine_pin.py"
            runner.write_text(_NO_ENGINE_PIN_RUNNER)
            argv = [
                sys.executable,
                str(runner),
                str(VERIFIER_REL),
                ",".join(sorted(_repo_modules())),
            ]
        else:
            argv = [sys.executable, str(VERIFIER_REL)]
        return subprocess.run(argv, cwd=self.root, capture_output=True, text=True)

    def stage(self, name: str, payload: bytes, *, serve: bytes | None = None) -> str:
        """Declare one dump in manifest.json and put `serve` in the fake store.

        `payload` is what the manifest's sha256 describes; `serve` is what the
        store actually hands back (defaults to `payload`). Passing a different
        `serve` models a corrupted object.
        """
        sha = hashlib.sha256(payload).hexdigest()
        fname = f"{name}_weights.pkl"
        obj = self.store / STORE_URI[len("s3://") :] / name / sha[:12] / fname
        obj.parent.mkdir(parents=True, exist_ok=True)
        obj.write_bytes(payload if serve is None else serve)
        mpath = self.root / "manifest.json"
        manifest = json.loads(mpath.read_text()) if mpath.exists() else {
            "schema": 2,
            "prefix": "zoo-weights",
            "built_with": {},
            "entries": {},
        }
        manifest["entries"][name] = {
            "file": fname,
            "sha256": sha,
            "size_bytes": len(payload),
        }
        mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return sha

    def run(self, store_uri: str | None = None) -> subprocess.CompletedProcess:
        env = {
            "PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
            "HOME": str(self.root),
            "FAKE_STORE": str(self.store),
            # Mirrors the step's `env:` mapping: an unset repo variable arrives
            # as the empty string, never as a missing name.
            "TRACEBLOC_ZOO_WEIGHTS_URI": store_uri or "",
        }
        return subprocess.run(
            ["bash", "-c", self.script],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
        )


@pytest.fixture
def checkout(tmp_path: Path, guard_script: str) -> Checkout:
    return Checkout(tmp_path, guard_script)


# --------------------------------------------------------------------------
# The three skip branches must be distinguishable
# --------------------------------------------------------------------------


def test_no_manifest_says_no_manifest(checkout: Checkout):
    """Nothing staged: NOT attributed to hosting, NOT attributed to the tool,
    and honest that nothing in this job would ever put a manifest here."""
    checkout.install_real_tool()
    proc = checkout.run(store_uri=STORE_URI)
    assert proc.returncode == 0, proc.stderr
    assert "SKIP (no manifest)" in proc.stdout, proc.stdout
    assert "no sync tool" not in proc.stdout, proc.stdout
    assert "no store URI" not in proc.stdout, proc.stdout
    # The old guard's failure mode: blaming hosting for everything.
    assert "the hosting decision" not in proc.stdout, (
        "an absent manifest is not the hosting decision — that attribution is "
        f"the bug this branch exists to fix:\n{proc.stdout}"
    )
    # …and the SAME failure mode one level up: no step in the verify-dumps job
    # fetches a manifest (it lives in `backend` and reaches CI as an artifact
    # only dump-coverage consumes), so "the gate activates when a manifest
    # lands" would be as reassuring and as untrue as the message this replaced.
    assert "STRUCTURAL" in proc.stdout, (
        "this branch is taken on every run by construction; a message implying "
        f"a manifest might arrive repeats the original bug:\n{proc.stdout}"
    )
    assert "dump-manifest" in proc.stdout, (
        f"the message must name where the manifest actually is:\n{proc.stdout}"
    )


def test_missing_tool_says_missing_tool(checkout: Checkout):
    """A declared manifest with the fetch hook gone: the exact state that hid
    for weeks behind 'dumps hosting is the hosting decision'."""
    checkout.stage("bert_base_uncased", b"dump-bytes")
    assert not (checkout.root / TOOL_REL).exists()
    proc = checkout.run(store_uri=STORE_URI)
    assert proc.returncode == 0, proc.stderr
    assert "SKIP (no sync tool)" in proc.stdout, proc.stdout
    assert "sync_zoo_weights.py" in proc.stdout, proc.stdout
    assert "DEFECT" in proc.stdout, (
        "a missing fetch hook is a repo defect, not a pending decision; the "
        f"message must say so:\n{proc.stdout}"
    )
    assert "no manifest" not in proc.stdout, proc.stdout


def test_no_store_uri_says_no_store_uri(checkout: Checkout):
    """Manifest and hook both present, nowhere to fetch from: THIS is (internal ref)."""
    checkout.stage("bert_base_uncased", b"dump-bytes")
    checkout.install_real_tool()
    proc = checkout.run(store_uri=None)
    assert proc.returncode == 0, proc.stderr
    assert "SKIP (no store URI)" in proc.stdout, proc.stdout
    assert "TRACEBLOC_ZOO_WEIGHTS_URI" in proc.stdout, proc.stdout
    assert "the hosting decision" in proc.stdout, proc.stdout
    assert "no sync tool" not in proc.stdout, proc.stdout
    # It must not have tried to run the tool: the tool exits 1 on an unset URI,
    # which would have made a skip look like a failure.
    assert "fetched + verified" not in proc.stdout, proc.stdout


def test_the_three_branches_print_different_things(checkout: Checkout):
    """No two skip branches may be confusable — the whole point of the change."""
    checkout.install_real_tool()
    no_manifest = checkout.run(store_uri=STORE_URI).stdout
    checkout.stage("bert_base_uncased", b"dump-bytes")
    no_uri = checkout.run(store_uri=None).stdout
    os.remove(checkout.root / TOOL_REL)
    no_tool = checkout.run(store_uri=STORE_URI).stdout
    outputs = [no_manifest, no_uri, no_tool]
    for out in outputs:
        assert out.strip(), "a skip branch printed nothing at all"
    assert len(set(outputs)) == 3, (
        "two skip branches produced identical output, so CI cannot tell them "
        f"apart:\n{outputs}"
    )


# --------------------------------------------------------------------------
# The happy path must actually invoke the committed tool, correctly
# --------------------------------------------------------------------------


def test_happy_path_fetches_every_declared_dump(checkout: Checkout):
    """With all three preconditions met, the step runs the REAL hook and the
    dumps land in dist/ verified. This is the assertion the committed
    `--manifest/--out` invocation could never have passed."""
    checkout.install_real_tool()
    checkout.stage("bert_base_uncased", b"bert-dump-bytes")
    checkout.stage("resnet_50", b"resnet-dump-bytes")
    proc = checkout.run(store_uri=STORE_URI)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert (checkout.root / "dist" / "bert_base_uncased_weights.pkl").read_bytes() == (
        b"bert-dump-bytes"
    )
    assert (checkout.root / "dist" / "resnet_50_weights.pkl").read_bytes() == (
        b"resnet-dump-bytes"
    )
    assert "fetched + verified 2 dump(s)" in proc.stdout, proc.stdout
    for marker in ("SKIP (no manifest)", "SKIP (no sync tool)", "SKIP (no store URI)"):
        assert marker not in proc.stdout, proc.stdout


def test_corrupted_object_fails_the_step_and_leaves_nothing(checkout: Checkout):
    """A store object whose bytes do not match the manifest must redden the step
    and be removed — never left in dist/ for the verifier to strict-load."""
    checkout.install_real_tool()
    checkout.stage("bert_base_uncased", b"bert-dump-bytes", serve=b"tampered")
    proc = checkout.run(store_uri=STORE_URI)
    assert proc.returncode != 0, proc.stdout
    assert "sha256 mismatch" in proc.stderr, f"{proc.stdout}\n{proc.stderr}"
    assert not (checkout.root / "dist" / "bert_base_uncased_weights.pkl").exists()


def test_manifest_with_no_entries_fails_the_step(checkout: Checkout):
    """A manifest that declares nothing protects nothing: fail, do not fetch
    zero dumps and report success (mirrors the verifier's stub-manifest rule)."""
    checkout.install_real_tool()
    (checkout.root / "manifest.json").write_text(
        json.dumps({"schema": 2, "prefix": "zoo-weights", "entries": {}})
    )
    proc = checkout.run(store_uri=STORE_URI)
    assert proc.returncode != 0, proc.stdout
    assert "declares no" in proc.stderr, f"{proc.stdout}\n{proc.stderr}"


# --------------------------------------------------------------------------
# The plumbing the branches depend on
# --------------------------------------------------------------------------


def test_step_plumbs_the_store_uri_variable():
    """The URI must reach the step's shell. Drop this `env:` mapping and the
    third branch is taken forever — a permanent no-op that looks like a
    pending decision, which is the shape of the original bug."""
    block = "\n".join(_step_block(WORKFLOW.read_text()))
    assert "TRACEBLOC_ZOO_WEIGHTS_URI:" in block, block
    assert "vars.TRACEBLOC_ZOO_WEIGHTS_URI" in block, block


def test_fetch_hook_is_committed():
    """(internal ref) in one line: the hook the workflow calls by path must exist at
    that path in the repo, not in someone's working tree."""
    tool = REPO_ROOT / TOOL_REL
    assert tool.is_file(), f"{TOOL_REL} is not in the repo"
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(TOOL_REL)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert tracked.returncode == 0, (
        f"{TOOL_REL} exists on disk but is not tracked by git — an untracked "
        "hook is exactly the failure the arming overclaim recorded"
    )


def test_fetch_hook_embeds_no_developer_path_or_bucket():
    """The hook runs on CI runners and is version-controlled in a public repo:
    no home directories, no real bucket names."""
    src = (REPO_ROOT / TOOL_REL).read_text()
    assert "os.path.expanduser" not in src, (
        "a home-relative default is wrong everywhere this file actually runs"
    )
    for needle in ("/Users/", "/home/", "~/work"):
        assert needle not in src, f"developer-local path {needle!r} in {TOOL_REL}"
    # The placeholder is fine; a resolved bucket is configuration, not source.
    assert "s3://<internal-bucket>" in src, "the store URI placeholder went missing"
    # A real bucket name starts with an alphanumeric; the `<placeholder>` form
    # and the prose "s3://-compatible" do not.
    real_uris = [
        line for line in src.splitlines() if re.search(r"s3://[A-Za-z0-9]", line)
    ]
    assert not real_uris, f"hardcoded store URI in {TOOL_REL}: {real_uris}"


# --------------------------------------------------------------------------
# The next hop: the fetch hook and the gate read the same manifest.json under
# DIFFERENT top-level keys
# --------------------------------------------------------------------------
# `fetch-all` requires `entries`; `verify_dumps_against_engine_pin.py` requires
# `dumps`. Both default to <repo-root>/manifest.json and <repo-root>/dist, so
# the divergence is purely the key — which is why it reads as a stub manifest
# rather than a schema mismatch when it fires. The two are driven in job order
# below so that the pair's behaviour is a measurement rather than an inference
# from two docstrings.

_TEMPLATE_REL = "model_zoo/text_classification/pytorch/bert_base_uncased.py"


def _entries_manifest(name: str, sha: str, size: int) -> dict:
    """The shape tools/sync_zoo_weights.py writes and `fetch-all` consumes."""
    return {
        "schema": 2,
        "prefix": "zoo-weights",
        "built_with": {"torch": "2.11.0"},
        "entries": {
            name: {"file": f"{name}_weights.pkl", "sha256": sha, "size_bytes": size}
        },
    }


def _dumps_manifest(name: str, sha: str) -> dict:
    """The shape verify_dumps_against_engine_pin.py reads."""
    return {
        "schema": 2,
        "prefix": "zoo-weights",
        "built_with": {"torch": "2.11.0"},
        "dumps": [
            {
                "name": name,
                "template": _TEMPLATE_REL,
                "weights": f"{name}_weights.pkl",
                "sha256": sha,
            }
        ],
    }


def test_an_entries_only_manifest_fetches_then_reddens_the_gate(checkout: Checkout):
    """The hook's OWN manifest shape: the fetch step succeeds and the gate then
    fails closed. Not a hypothetical — `entries` is the only shape `fetch-all`
    accepts, so this is what arming the job would actually produce."""
    checkout.install_real_tool()
    checkout.install_real_verifier()
    payload = b"bert-dump-bytes"
    sha = checkout.serve("bert_base_uncased", payload)
    checkout.write_manifest(_entries_manifest("bert_base_uncased", sha, len(payload)))

    fetch = checkout.run(store_uri=STORE_URI)
    assert fetch.returncode == 0, f"{fetch.stdout}\n{fetch.stderr}"
    assert "fetched + verified 1 dump(s)" in fetch.stdout, fetch.stdout
    assert (checkout.root / "dist" / "bert_base_uncased_weights.pkl").exists()

    gate = checkout.run_verifier()
    assert gate.returncode != 0, (
        "the gate accepted a manifest with no 'dumps' list — it would have swept "
        f"nothing and reported success:\n{gate.stdout}"
    )
    # And it must say WHICH of the two causes of that exit code this is. Reading
    # a schema mismatch as a stub manifest makes "populate it" the advice, when
    # the manifest is fully populated under the other key.
    assert "SCHEMA DIVERGENCE" in gate.stderr, (
        "a populated 'entries' manifest is not a stub, and an operator told to "
        f"populate it has nothing to do:\n{gate.stderr}"
    )
    assert "entries" in gate.stderr and "the schema reconciliation" in gate.stderr, gate.stderr
    assert "it is a stub" not in gate.stderr, (
        f"a manifest declaring 1 dump under 'entries' is not a stub:\n{gate.stderr}"
    )


def test_a_dumps_only_manifest_reddens_the_fetch_step_first(checkout: Checkout):
    """The gate's manifest shape never reaches the gate: the fetch step reddens
    on it, so no dump is fetched at all. This is the direction the workflow's
    comment did not name."""
    checkout.install_real_tool()
    checkout.install_real_verifier()
    sha = checkout.serve("bert_base_uncased", b"bert-dump-bytes")
    checkout.write_manifest(_dumps_manifest("bert_base_uncased", sha))

    fetch = checkout.run(store_uri=STORE_URI)
    assert fetch.returncode != 0, (
        "the fetch step accepted a manifest it cannot read, and would have left "
        f"dist/ empty for the gate to call MISSING:\n{fetch.stdout}"
    )
    assert "entries" in fetch.stderr, f"{fetch.stdout}\n{fetch.stderr}"
    assert not (checkout.root / "dist" / "bert_base_uncased_weights.pkl").exists()


def test_no_single_key_manifest_can_arm_this_job(checkout: Checkout):
    """The two directions above, as one claim: whichever key a manifest carries,
    a DIFFERENT side of the job reddens. So "a manifest lands" is not the
    remaining precondition, and no message may imply that it is."""
    checkout.install_real_tool()
    checkout.install_real_verifier()
    payload = b"bert-dump-bytes"
    sha = checkout.serve("bert_base_uncased", payload)

    checkout.write_manifest(_entries_manifest("bert_base_uncased", sha, len(payload)))
    entries_fetch = checkout.run(store_uri=STORE_URI).returncode
    entries_gate = checkout.run_verifier().returncode

    shutil.rmtree(checkout.root / "dist", ignore_errors=True)
    checkout.write_manifest(_dumps_manifest("bert_base_uncased", sha))
    dumps_fetch = checkout.run(store_uri=STORE_URI).returncode
    dumps_gate = checkout.run_verifier().returncode

    # Neither shape gets both sides to 0 — that is the whole finding.
    assert not (entries_fetch == 0 and entries_gate == 0), (
        "an 'entries'-only manifest took the whole job green; if the schemas "
        "were reconciled (the schema reconciliation) this guard and the workflow comment "
        "above the fetch step are both stale and must be revisited"
    )
    assert not (dumps_fetch == 0 and dumps_gate == 0), (
        "a 'dumps'-only manifest took the whole job green; see the schema reconciliation — "
        "revisit this guard and the workflow comment"
    )
    # …and they fail on OPPOSITE sides, which is what makes the divergence
    # invisible: each side's error looks like a local problem.
    assert entries_fetch == 0 and dumps_fetch != 0, (
        f"expected the fetch step to accept only 'entries' "
        f"(entries={entries_fetch}, dumps={dumps_fetch})"
    )
    assert entries_gate != 0, f"expected the gate to reject 'entries' ({entries_gate})"


# --------------------------------------------------------------------------
# Neither tool's no-op message may promise a sweep it cannot deliver
# --------------------------------------------------------------------------
# (internal ref) removed "the verifier runs armed; it activates when a manifest
# lands" from the workflow's shell. The gate carried its own copy of the same
# sentence and kept it, because the shell and the tool each phrase the skip in
# their own words. Both are asserted now, in one place.


def _verifier_module():
    spec = importlib.util.spec_from_file_location(
        "_verifier_for_message_guard", REPO_ROOT / VERIFIER_REL
    )
    assert spec and spec.loader, VERIFIER_REL
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_gates_absent_manifest_message_promises_no_sweep(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """The branch CI takes on every run. It must name the structural cause and
    BOTH outstanding preconditions, not just hosting."""
    mod = _verifier_module()
    rc = mod.run_sweep(
        tmp_path / "manifest.json",
        tmp_path / "dist",
        tmp_path,
        tmp_path / "report.json",
        False,
        True,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "STRUCTURAL" in out, (
        f"nothing in the verify-dumps job puts a manifest here; say so:\n{out}"
    )
    for ticket in ("the hosting decision", "the schema reconciliation", "the arming overclaim"):
        assert ticket in out, f"{ticket} not named in the no-op message:\n{out}"
    assert "dumps" in out, (
        "the message must name the manifest KEY this tool requires — a manifest "
        f"arriving in the other shape is not an arming:\n{out}"
    )
    # The exact overclaim this replaced. It is asserted as an absence because
    # the sentence was true of the author's intent and of nothing else.
    for claim in (
        "will verify every dump the moment",
        "activates when a manifest lands",
        "it activates when a manifest",
    ):
        assert claim not in out, f"the overclaim from the arming overclaim is back:\n{out}"


def test_neither_the_shell_nor_the_gate_claims_arming_on_a_manifest(
    checkout: Checkout,
):
    """Same claim, two sites, and they are what a CI log actually shows. The
    shell's copy was fixed in (internal ref) and the gate's was not — a guard
    reading only one of them would have passed for the four days between. So
    both are asserted against their OUTPUT, in one test, on the branch CI takes.

    Source text is deliberately not what is inspected: the fix's own comments
    quote the removed sentence to explain why it went, so a substring search
    over the file would have to be taught which occurrences are allowed.
    """
    checkout.install_real_tool()
    checkout.install_real_verifier()
    shell = checkout.run(store_uri=STORE_URI)
    gate = checkout.run_verifier()
    assert shell.returncode == 0, shell.stderr
    assert gate.returncode == 0, f"{gate.stdout}\n{gate.stderr}"
    for label, out in (("workflow shell", shell.stdout), ("gate", gate.stdout)):
        assert out.strip(), f"{label} printed nothing on the no-manifest branch"
        for claim in (
            "activates when a manifest lands",
            "will verify every dump the moment",
            "the gate is armed and will verify",
        ):
            assert claim not in out.lower(), (
                f"{label} promises arming on a manifest — the overclaim "
                f"the arming overclaim recorded:\n{out}"
            )
        assert "STRUCTURAL" in out, (
            f"{label} does not say the branch is taken on every run:\n{out}"
        )


def test_the_step_comment_names_both_halves_of_the_divergence():
    """The comment above the fetch step is where the next person decides whether
    arming is a one-line change. It named the gate's side only, which reads as
    "fix the verifier and go"; the fetch step reddens on the other shape, so
    that reading costs a debugging session."""
    text = WORKFLOW.read_text()
    lines = text.splitlines()
    start = next(
        i for i, ln in enumerate(lines) if ln.strip() == f"- name: {STEP_NAME}"
    )
    comment = []
    for line in reversed(lines[:start]):
        if not line.strip().startswith("#"):
            break
        comment.append(line)
    comment_text = "\n".join(reversed(comment))
    assert comment_text.strip(), f"no comment block precedes {STEP_NAME!r}"
    # The needles are the KEYED forms, not the bare words: "dumps" alone is
    # satisfied by the phrase "declares no dumps" and by the verifier's own
    # filename, so a guard spelled that way passes with one direction deleted
    # (measured — it survived mutation M7).
    for needle in (
        "`entries`-keyed",
        "`dumps`-keyed",
        "sync_zoo_weights",
        "verify_dumps_against_engine_pin",
        "the schema reconciliation",
    ):
        assert needle in comment_text, (
            f"the fetch step's comment does not name {needle!r}; a half-stated "
            f"divergence is how the arming overclaim happened:\n{comment_text}"
        )
    # This is a wording-shaped guard and that is its limit: it proves the
    # comment MENTIONS both directions, never that either mention is true.
    # The two tests above are what prove the behaviour.


def test_the_fetch_destination_is_the_directory_the_gate_sweeps():
    """The step writes dumps into `--dest <D>`; the next step runs the gate with
    NO arguments, so it sweeps its own `--dumps-dir` default. Nothing connected
    the two — they agree at `dist` today by coincidence, and moving either alone
    would leave the gate calling every fetched dump MISSING.

    Asserted from the two files rather than by running them: the gate's
    MISSING verdict lives past its first `import torch`, so the behavioural
    version of this test would run in one of the three CI framework envs — and
    a guard exercised in one env is how the arming overclaim's whole class of defect
    survives. Read the real default expression, not a copy of it, so there is
    no transcription to rot.

    What it cannot catch: an ARGUMENT the gate is given in the workflow. It
    reads the gate's *default*, so `--dumps-dir` passed explicitly on the gate's
    own step would make this assertion irrelevant without failing.
    """
    dest = re.findall(
        r"sync_zoo_weights\.py .*fetch-all --dest (\S+)",
        "\n".join(_step_block(WORKFLOW.read_text())),
    )
    assert len(dest) == 1, (
        f"expected exactly one `fetch-all --dest` in the step, found {dest}"
    )
    src = (REPO_ROOT / VERIFIER_REL).read_text()
    m = re.search(
        r'"--dumps-dir",\s*\n\s*default=str\(repo_root_default / "([^"]+)"\)', src
    )
    assert m, (
        f"could not read the --dumps-dir default out of {VERIFIER_REL}. If its "
        "argparse block was restructured, update this pattern in the same "
        "commit — an unreadable default is an assertion failure here, never a "
        "skip."
    )
    assert dest[0] == m.group(1), (
        f"the fetch step writes dumps into {dest[0]!r} but the gate sweeps "
        f"{m.group(1)!r} by default, so every fetched dump would be reported "
        "MISSING. Move both or neither."
    )
    # The other half of the same coincidence: the gate reads its manifest from a
    # default too, and the step's `--staging .` makes ./manifest.json the file
    # the hook reads. Same file, or the two steps are looking at different
    # manifests.
    staging = re.findall(
        r"sync_zoo_weights\.py --staging (\S+) fetch-all",
        "\n".join(_step_block(WORKFLOW.read_text())),
    )
    assert staging == ["."], f"unexpected --staging in the fetch step: {staging}"
    mm = re.search(
        r'"--manifest",\s*\n\s*default=str\(repo_root_default / "([^"]+)"\)', src
    )
    assert mm, f"could not read the --manifest default out of {VERIFIER_REL}"
    assert mm.group(1) == "manifest.json", (
        f"the hook reads ./manifest.json (--staging .) but the gate defaults to "
        f"{mm.group(1)!r} — the two steps would read different manifests"
    )


def test_the_hook_still_fails_closed_on_an_unset_store_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`_store_uri()` exits when TRACEBLOC_ZOO_WEIGHTS_URI is unset. That is the
    right behaviour — an unconfigured store must not be guessed at — and it had
    no guard, because the workflow's third branch short-circuits before the tool
    is ever invoked on that path. So making the hook fail OPEN was invisible:
    every test above still passed with `_store_uri()` returning "" (measured).

    Driven through `cmd_fetch_all`, not just the helper, so a caller that stops
    consulting the helper fails here too.
    """
    spec = importlib.util.spec_from_file_location(
        "_hook_for_failclosed_guard", REPO_ROOT / TOOL_REL
    )
    assert spec and spec.loader, TOOL_REL
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    monkeypatch.delenv("TRACEBLOC_ZOO_WEIGHTS_URI", raising=False)
    (tmp_path / "manifest.json").write_text(
        json.dumps(_entries_manifest("bert_base_uncased", "ab" * 32, 1))
    )
    with pytest.raises(SystemExit) as excinfo:
        hook.cmd_fetch_all(str(tmp_path), str(tmp_path / "dist"))
    message = str(excinfo.value)
    assert "TRACEBLOC_ZOO_WEIGHTS_URI" in message, message
    assert "the hosting decision" in message, (
        f"the exit must name the decision that is missing, not just the var:\n{message}"
    )
    # An empty-string URI is the SAME condition, and the one CI actually
    # produces: an unset repo variable arrives as "" through the step's `env:`
    # mapping, never as an absent name.
    monkeypatch.setenv("TRACEBLOC_ZOO_WEIGHTS_URI", "")
    with pytest.raises(SystemExit):
        hook.cmd_fetch_all(str(tmp_path), str(tmp_path / "dist"))


# --------------------------------------------------------------------------
# The engine-pin install runs only when there is something to verify
# --------------------------------------------------------------------------
# Every run of this job to date took the no-manifest branch -- after ~2 minutes
# of setup-python plus `pip install -r tools/requirements-engine-pin.txt` that
# the branch never used. So the two install steps are conditional on a
# manifest.json being present. That reordering is safe on exactly one premise,
# and it is asserted here rather than stated: the no-manifest path of BOTH tools
# imports nothing the install provides. The shape half is asserted too --
# which steps carry the condition and, as importantly, which must not, because a
# condition that spread to the fetch step or the verifier would turn "skip the
# install" into "skip the verdict".

VERIFY_JOB = "verify-dumps"
ENGINE_PIN_INSTALL = "pip install -r tools/requirements-engine-pin.txt"
# A step-level `if:` key in EITHER of the two places YAML lets it sit: the
# 8-space body form (`        if:`) and the 6-space marker form (`      - if:`),
# which GitHub reads identically. An earlier version matched only the body
# form, so a condition written on the marker line of the verifier step --
# the form the checkout and verifier steps are already written in -- was
# invisible to the "must not carry a condition" half below (measured: the
# verdict step gated, 21 passed). Never a shell `if` inside a `run: |` block
# (deeper indent, and `if [`, not `if:`), and never a comment (dropped before
# a step reaches this).
_STEP_IF = re.compile(r"^(?:      - |        )if:\s*(.+?)\s*$", re.MULTILINE)
_HASHFILES = re.compile(r"^hashFiles\('([^']+)'\)\s*!=\s*''$")


def _verify_job_steps() -> list[str]:
    """The verify-dumps job's steps, each as its own text, in job order.

    Comment lines are dropped: the prose between steps names both tools and
    would otherwise make a step look like the one it merely discusses."""
    steps: list[list[str]] = []
    for line in _job_block(VERIFY_JOB).splitlines():
        if line.lstrip().startswith("#"):
            continue
        if line.startswith("      - "):
            steps.append([line])
        elif steps:
            steps[-1].append(line)
    assert len(steps) >= 4, f"expected the job's steps, sliced:\n{steps}"
    return ["\n".join(step) for step in steps]


def _condition_of(step: str) -> str | None:
    found = _STEP_IF.findall(step)
    assert len(found) <= 1, (
        f"a step carrying more than one `if:` (marker form and body form are the "
        f"SAME key -- a step never has both):\n{step}"
    )
    return found[0] if found else None


def test_condition_of_sees_both_if_forms():
    """The matcher behind the shape test, driven on the two forms GitHub
    accepts -- written out here independently of the regex, so a regex that
    sees one form only reddens THIS test instead of leaving the shape test
    quietly half-blind."""
    body_form = (
        "      - uses: actions/setup-python@sha\n"
        "        if: hashFiles('manifest.json') != ''\n"
        "        with:\n"
        "          python-version: '3.11'"
    )
    marker_form = (
        "      - if: hashFiles('manifest.json') != ''\n"
        "        run: python3 tools/verify_dumps_against_engine_pin.py"
    )
    shell_if_only = (
        "      - run: |\n"
        "          if [ ! -f manifest.json ]; then\n"
        "            echo skip\n"
        "          fi"
    )
    assert _condition_of(body_form) == "hashFiles('manifest.json') != ''"
    assert _condition_of(marker_form) == "hashFiles('manifest.json') != ''"
    assert _condition_of(shell_if_only) is None
    with pytest.raises(AssertionError, match="more than one `if:`"):
        _condition_of(
            "      - if: hashFiles('manifest.json') != ''\n"
            "        if: success()\n"
            "        run: true"
        )


def _install_gate_file(installs: list[str]) -> str:
    """The ONE file both install conditions key on -- read off the workflow,
    so nothing below restates its name."""
    gated_on: set[str] = set()
    for step in installs:
        cond = _condition_of(step)
        assert cond is not None, f"an install step runs unconditionally again:\n{step}"
        m = _HASHFILES.match(cond)
        assert m, f"the install condition is not a hashFiles presence test: {cond!r}"
        gated_on.add(m.group(1))
    # Both installs key on ONE file, and it is the file the fetch step's own
    # shell tests for -- derived from that shell, not restated here.
    assert len(gated_on) == 1, f"the two install steps are gated on different files: {gated_on}"
    (manifest,) = gated_on
    return manifest


def test_the_install_steps_are_gated_on_the_manifest_and_the_verdict_steps_are_not(
    guard_script: str,
):
    steps = _verify_job_steps()
    installs = [s for s in steps if "actions/setup-python@" in s or ENGINE_PIN_INSTALL in s]
    verdicts = [s for s in steps if STEP_NAME in s or VERIFIER_REL.name in s]
    checkouts = [s for s in steps if "actions/checkout@" in s]
    assert len(installs) == 2, f"expected setup-python + the pip install, got:\n{installs}"
    assert len(verdicts) == 2, f"expected the fetch step + the verifier, got:\n{verdicts}"
    assert len(checkouts) == 1, checkouts

    manifest = _install_gate_file(installs)
    assert f"[ ! -f {manifest} ]" in guard_script, (
        f"the install is gated on {manifest!r} but the fetch step tests a "
        f"different path:\n{guard_script[:300]}"
    )
    # The checkout must run for hashFiles to see the file at all, and the two
    # verdict steps must run so the verdict stays the tools' on every path.
    for step in checkouts + verdicts:
        assert _condition_of(step) is None, (
            f"a step that must run on every path carries a condition:\n{step}"
        )


# `hashFiles` sees only what is in the checkout when the condition is evaluated.
# The arming path the workflow's own comments describe is wiring the
# `dump-manifest` artifact into this job, and a download step's natural home is
# beside the fetch step -- AFTER both installs. In that arrangement the installs
# skip, the manifest then exists, and the verifier's first `import torch` raises
# under the runner's bare python3: fail-closed, but as a ModuleNotFoundError
# rather than a dump verdict. So the job must contain no step that could put the
# gated file into the checkout, and arming has to revisit the condition on
# purpose. Anything mentioning the file other than the two known READS
# (the install condition and the fetch step's presence test) -- or messages that
# merely talk about it -- is a finding; an unrecognised form fails rather than
# being guessed benign.
_ARTIFACT_DOWNLOAD = re.compile(r"^\s*(?:- )?uses:\s*\S*download-artifact", re.MULTILINE)


def _manifest_mentions_that_are_not_reads(step: str, manifest: str) -> list[str]:
    reads = (f"hashFiles('{manifest}')", f"[ ! -f {manifest} ]")
    hits: list[str] = []
    for line in step.splitlines():
        if manifest not in line:
            continue
        rest = line.strip()
        for read in reads:
            rest = rest.replace(read, "")
        if manifest not in rest:
            continue
        if rest.startswith("echo ") and ">" not in rest:
            continue  # a message about the file, with nowhere to write it
        hits.append(line)
    return hits


def test_nothing_in_the_job_can_produce_the_file_the_install_is_gated_on():
    steps = _verify_job_steps()
    installs = [s for s in steps if "actions/setup-python@" in s or ENGINE_PIN_INSTALL in s]
    manifest = _install_gate_file(installs)
    for step in steps:
        assert not _ARTIFACT_DOWNLOAD.search(step), (
            f"a step downloads an artifact into the checkout AFTER the install "
            f"condition was evaluated -- if it can carry {manifest!r}, the install "
            f"skips and the verifier runs without its stack. Revisit the "
            f"`hashFiles` condition before arming:\n{step}"
        )
        hits = _manifest_mentions_that_are_not_reads(step, manifest)
        assert not hits, (
            f"a step touches {manifest!r} in a form this test does not know to be "
            f"a read. If it can WRITE the file, the install condition above it is "
            f"already decided:\n" + "\n".join(hits)
        )

    # Anchors -- the same two detectors, on the shapes they exist to catch, so
    # a green run above means "nothing found", not "nothing looked for".
    download = (
        "      - uses: actions/download-artifact@0123456789abcdef # v5\n"
        "        with:\n"
        "          name: dump-manifest"
    )
    assert _ARTIFACT_DOWNLOAD.search(download)
    assert _ARTIFACT_DOWNLOAD.search("      - if: always()\n        uses: actions/download-artifact@v5")
    for write in (
        f"          curl -fsSL $URL -o {manifest}",
        f"          python3 tools/mint.py > {manifest}",
        f"          cp dist/{manifest} {manifest}",
        f'          echo "{{}}" > {manifest}',
    ):
        assert _manifest_mentions_that_are_not_reads("      - run: |\n" + write, manifest), write
    assert not _manifest_mentions_that_are_not_reads(
        f"      - run: pip install\n        if: hashFiles('{manifest}') != ''", manifest
    )


def _module_scope_imports(source: str, filename: str) -> set[str]:
    """Top-level names imported when the module is IMPORTED -- module scope
    plus any if/try/with block at module scope, which execute then too; not
    function bodies, which run only when called."""
    names: set[str] = set()

    def visit(nodes) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                # A relative import can only name a module of this repo.
                names.add((node.module or "").split(".")[0] if not node.level else "tools")
            else:
                for field in ("body", "orelse", "finalbody", "handlers"):
                    visit(getattr(node, field, []))

    visit(ast.parse(source, filename=filename).body)
    assert names, f"parsed no module-scope imports at all from {filename} -- wrong text?"
    return names


def test_the_verifier_imports_nothing_at_module_scope_that_the_install_provides():
    """The static half of the premise, DERIVED: every name the verifier imports
    at module scope must come from the interpreter's own stdlib or from this
    repo -- i.e. the no-manifest verdict needs nothing the skipped install
    would have put on the path. No list of packages is held here; the runtime
    half (`test_the_no_manifest_verdict_needs_nothing_the_install_provides`)
    enforces the same rule in a live interpreter."""
    source = (REPO_ROOT / VERIFIER_REL).read_text()
    imported = _module_scope_imports(source, str(VERIFIER_REL))
    allowed = set(sys.stdlib_module_names) | _repo_modules()
    offenders = sorted(imported - allowed)
    assert not offenders, (
        f"{VERIFIER_REL} imports {offenders} at module scope. On the no-manifest "
        f"path the install is skipped, so the runner's bare python3 has neither "
        f"the engine pin nor its transitives, and the job would redden with a "
        f"ModuleNotFoundError instead of a verdict. Move the import inside the "
        f"function that needs a dump, or revisit the install condition."
    )

    # Anchor: the walker must SEE a third-party import at module scope (both
    # forms, and inside a module-level try:) and must NOT count one inside a
    # function -- else the assertion above is a lint of nothing.
    seen = _module_scope_imports(
        "import sys\n"
        "from packaging.version import Version\n"
        "try:\n    import tokenizers\nexcept ImportError:\n    tokenizers = None\n"
        "def f():\n    import torch\n    return torch\n",
        "<anchor>",
    )
    assert {"packaging", "tokenizers"} <= seen, seen
    assert "torch" not in seen, seen


def test_the_no_manifest_verdict_needs_nothing_the_install_provides(checkout: Checkout):
    """The premise of skipping the install, measured against the REAL verifier
    in an interpreter that refuses every engine-pin import."""
    checkout.install_real_verifier()
    proc = checkout.run_verifier(engine_pin_blocked=True)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "OK (nothing to verify)" in proc.stdout, proc.stdout
    assert "BLOCKED" not in proc.stderr, proc.stderr

    # The blocker has to be shown to bite, or the assertion above is vacuous:
    # the moment a dump IS declared, the same interpreter must fall over on the
    # first engine-pin import -- which is also the proof that the install is
    # needed precisely when the condition lets it run.
    payload = b"bert-dump-bytes"
    (checkout.root / "dist").mkdir()
    (checkout.root / "dist" / "bert_base_uncased_weights.pkl").write_bytes(payload)
    checkout.write_manifest(
        _dumps_manifest("bert_base_uncased", hashlib.sha256(payload).hexdigest())
    )
    proc = checkout.run_verifier(engine_pin_blocked=True)
    assert proc.returncode != 0, (
        f"a declared dump was 'verified' without torch:\n{proc.stdout}"
    )
    assert "BLOCKED: torch" in proc.stderr, proc.stderr


def test_the_no_manifest_fetch_branch_invokes_no_python(checkout: Checkout):
    """The other tool on the skip path: its no-manifest branch is plain shell,
    so it cannot notice that setup-python did not run."""
    checkout.install_real_tool()
    _write_exec(
        checkout.bin / "python3",
        "#!/bin/sh\necho 'python3 invoked on the no-manifest path' >&2\nexit 99\n",
    )
    proc = checkout.run(store_uri=STORE_URI)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "SKIP (no manifest)" in proc.stdout, proc.stdout

    # Anchor: the same stub reddens the step once a manifest sends it to the
    # hook, so the run above passed because no python was needed, not because
    # the stub was inert.
    checkout.stage("bert_base_uncased", b"dump-bytes")
    proc = checkout.run(store_uri=STORE_URI)
    assert proc.returncode == 99, f"{proc.stdout}\n{proc.stderr}"
