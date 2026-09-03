"""The dump-fetch step in verify-dumps-engine-pin.yml must name the precondition
it is missing, and must invoke the fetch hook with arguments the hook accepts
(backend#3060).

Why this is a test and not a comment
------------------------------------
The step used to guard with a single
``[ -f manifest.json ] && [ -f tools/sync_zoo_weights.py ]`` and print one
message for every no-op: *"dumps hosting is backend#2659."* Two distinct
failures wore that one sentence:

  * the store location was undecided (true, and #2659's to decide); and
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
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "verify-dumps-engine-pin.yml"
STEP_NAME = "Obtain staged dumps from the model store"
TOOL_REL = Path("tools") / "sync_zoo_weights.py"
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
    assert "backend#2659" not in proc.stdout, (
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
    for weeks behind 'dumps hosting is backend#2659'."""
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
    """Manifest and hook both present, nowhere to fetch from: THIS is #2659."""
    checkout.stage("bert_base_uncased", b"dump-bytes")
    checkout.install_real_tool()
    proc = checkout.run(store_uri=None)
    assert proc.returncode == 0, proc.stderr
    assert "SKIP (no store URI)" in proc.stdout, proc.stdout
    assert "TRACEBLOC_ZOO_WEIGHTS_URI" in proc.stdout, proc.stdout
    assert "backend#2659" in proc.stdout, proc.stdout
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
    """#3060 in one line: the hook the workflow calls by path must exist at
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
        "hook is exactly the failure backend#3060 recorded"
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
