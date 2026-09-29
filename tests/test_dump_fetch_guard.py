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
import itertools
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
VERIFY_STEP_NAME = "Verify the dumps against the engine pin"
TOOL_REL = Path("tools") / "sync_zoo_weights.py"
VERIFIER_REL = Path("tools") / "verify_dumps_against_engine_pin.py"
# Imported by the verifier once it reads a manifest (never at module scope).
VERIFIER_SIBLINGS = (Path("tools") / "check_dump_coverage.py", Path("tools") / "seed_index.py")
STORE_URI = "s3://zoo-weights-test-bucket/zoo-weights"
ROLE_ARN = "arn:aws:iam::000000000000:role/zoo-weights-read-test"


# --------------------------------------------------------------------------
# Extracting the step's shell out of the workflow
# --------------------------------------------------------------------------
# Hand-rolled rather than PyYAML: none of the three CI framework envs
# (.github/requirements/{pytorch,sklearn,survival}.txt) install PyYAML, and a
# test that imports it would skip in all three — i.e. never run, which is the
# state this file exists to end. Every extraction failure below is an assertion,
# never a skip, so restructuring the YAML breaks this loudly.


def _step_block(text: str, name: str = STEP_NAME) -> list[str]:
    """Return the lines of the ``- name: <name>`` step, marker included."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == f"- name: {name}":
            start = i
            break
    assert start is not None, (
        f"no step named {name!r} in {WORKFLOW}. If it was renamed, rename "
        "it here in the same commit."
    )
    marker_indent = len(lines[start]) - len(lines[start].lstrip())
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if line.strip() and (len(line) - len(line.lstrip())) <= marker_indent:
            break
        block.append(line)
    return block


def _run_script(block: list[str], sentinel: str = "mkdir -p dist") -> str:
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
    assert sentinel in script, (
        f"extracted the wrong text for the step — expected {sentinel!r} in its "
        f"shell, got:\n{script[:400]}"
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
sys.argv = [sys.argv[1], *sys.argv[3:]]
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
        a copy inside this checkout points at this checkout. Its two sibling
        modules come too: it resolves entries through them once a manifest is
        read."""
        for rel in (VERIFIER_REL, *VERIFIER_SIBLINGS):
            shutil.copy2(REPO_ROOT / rel, self.root / rel)

    def install_template(self, category: str, stem: str, source: str) -> str:
        """A template in this checkout's zoo; returns its repo-relative path."""
        rel = Path("model_zoo") / category / "pytorch" / f"{stem}.py"
        (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.root / rel).write_text(source)
        return rel.as_posix()

    def outputs(self) -> dict[str, str]:
        """What the last step wrote to $GITHUB_OUTPUT, as GitHub would read it."""
        path = self.root / "_github_output"
        if not path.exists():
            return {}
        pairs = (ln.split("=", 1) for ln in path.read_text().splitlines() if "=" in ln)
        return dict(pairs)

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

    def run(
        self,
        store_uri: str | None = None,
        *,
        role_arn: str | None = ROLE_ARN,
        trusted: str = "true",
        credentials: str = "success",
    ) -> subprocess.CompletedProcess:
        """Run the step's shell. The keyword defaults are the ARMED state for
        the three credential preconditions (a role is set, the event is
        trusted, the credential step succeeded), so a test that is about the
        store URI or the manifest varies only what it is about."""
        env = {
            "PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
            "HOME": str(self.root),
            "FAKE_STORE": str(self.store),
            # Mirrors the step's `env:` mapping: an unset repo variable arrives
            # as the empty string, never as a missing name.
            "TRACEBLOC_ZOO_WEIGHTS_URI": store_uri or "",
            "ZOO_WEIGHTS_READ_ROLE_ARN": role_arn or "",
            # A GitHub expression renders a boolean as `true` / `false`, and a
            # skipped step's outcome as `skipped`.
            "ZOO_WEIGHTS_TRUSTED_EVENT": trusted,
            "ZOO_WEIGHTS_CREDENTIALS": credentials,
            # A real file, fresh per run, exactly as the runner provides one.
            "GITHUB_OUTPUT": str(self.root / "_github_output"),
        }
        (self.root / "_github_output").write_text("")
        return subprocess.run(
            ["bash", "-c", self.script],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
        )


    def run_verify_step(
        self, not_fetched: str, *, engine_pin_blocked: bool = False
    ) -> subprocess.CompletedProcess:
        """Run the VERIFIER step's own shell, extracted from the workflow, with
        ``not_fetched`` as the fetch step's output would hand it over.

        ``engine_pin_blocked=True`` makes its ``python3`` the import-refusing
        interpreter — what that step has when the install was skipped."""
        script = _run_script(
            _step_block(WORKFLOW.read_text(), VERIFY_STEP_NAME), VERIFIER_REL.as_posix()
        )
        bin_dir = self.root / "_bin_verify"
        bin_dir.mkdir(exist_ok=True)
        if engine_pin_blocked:
            runner = self.root / "_no_engine_pin.py"
            runner.write_text(_NO_ENGINE_PIN_RUNNER)
            mods = ",".join(sorted(_repo_modules()))
            _write_exec(
                bin_dir / "python3",
                f'#!/bin/sh\nscript="$1"; shift\n'
                f'exec "{sys.executable}" "{runner}" "$script" "{mods}" "$@"\n',
            )
        else:
            _write_exec(bin_dir / "python3", f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        env = {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "HOME": str(self.root),
            # An absent step output renders as the empty string.
            "DUMPS_NOT_FETCHED": not_fetched,
        }
        return subprocess.run(
            ["bash", "-c", script], cwd=self.root, env=env, capture_output=True, text=True
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
    # …and the SAME failure mode one level up. This branch used to be taken on
    # every run, because nothing in the job put a manifest here; the job now
    # downloads the dump-manifest artifact first, so reaching it in CI is a
    # FAULT, and the message must say so rather than read as a pending state.
    # (Premise changed by the schema reconciliation: this assertion used to
    # require "STRUCTURAL".)
    assert "FAULT" in proc.stdout, (
        f"an absent manifest is now a fault in CI, not a pending state:\n{proc.stdout}"
    )
    assert "STRUCTURAL" not in proc.stdout, proc.stdout
    assert "dump-manifest" in proc.stdout, (
        f"the message must name where the manifest actually is:\n{proc.stdout}"
    )
    assert "--require-manifest" in proc.stdout, (
        f"the message must say what reddens on it:\n{proc.stdout}"
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
# One schema across both tools: the canonical `entries` manifest, in job order
# --------------------------------------------------------------------------
# The fetch hook and the gate used to read the same manifest.json under
# DIFFERENT top-level keys — `entries` for the hook, `dumps` for the gate — so
# no single-key manifest could take the job green, in either direction. Both
# now read the canonical `entries` dict (the schema reconciliation). The tests
# below drive ONE schema-2 manifest carrying the per-entry shapes the real one
# carries — live entries, a retired entry, and a `category`-carrying entry for
# a stem that ships in two categories — through the fetch step's shell and
# then the verifier step's shell, both extracted from the workflow, in the
# order the job runs them.
#
# PREMISE RETIRED, TESTS REPLACED: `test_an_entries_only_manifest_fetches_
# then_reddens_the_gate` and `test_no_single_key_manifest_can_arm_this_job`
# asserted the divergence itself (the gate MUST redden on an `entries`
# manifest). Their successors below assert the opposite, which is the fix, and
# the `dumps` direction is kept as it was, plus the gate's own refusal.

_TEMPLATE_REL = "model_zoo/text_classification/pytorch/bert_base_uncased.py"


def _entries_manifest(name: str, sha: str, size: int) -> dict:
    """The canonical shape: what tools/sync_zoo_weights.py writes and reads,
    and what the verifier reads."""
    return {
        "schema": 2,
        "prefix": "zoo-weights",
        "built_with": {"torch": "2.11.0"},
        "entries": {
            name: {"file": f"{name}_weights.pkl", "sha256": sha, "size_bytes": size}
        },
    }


def _dumps_manifest(name: str, sha: str) -> dict:
    """The retired `dumps` list shape. Nothing writes it; both tools refuse it."""
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


def _linear_template(out_features: int) -> str:
    return (
        "from torch import nn\n"
        "main_class = 'MyModel'\n"
        "class MyModel(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        f"        self.fc = nn.Linear(4, {out_features})\n"
    )


def _stage_canonical(checkout: Checkout, payloads: dict[str, bytes], built_with: dict) -> dict:
    """The three-shape manifest over this checkout, and its store objects.

    * ``alpha``  — live, one template (image_classification);
    * ``twin``   — live, and its stem ships in TWO categories with different
      heads; the entry records ``category: text_classification``;
    * ``detr``   — retired: no template, and deliberately NOT in the store, so
      a fetch that tried for it would fail on the stub's "no such object".

    ``payloads`` gives the bytes served for alpha and twin.
    """
    checkout.install_template("image_classification", "alpha", _linear_template(3))
    checkout.install_template("image_classification", "twin", _linear_template(3))
    checkout.install_template("text_classification", "twin", _linear_template(5))
    entries = {}
    for name, payload in payloads.items():
        sha = checkout.serve(name, payload)
        entries[name] = {"file": f"{name}_weights.pkl", "sha256": sha, "size_bytes": len(payload)}
    entries["twin"]["category"] = "text_classification"
    entries["detr"] = {
        "file": "detr_weights.pkl",
        "sha256": "d" * 64,
        "size_bytes": 1,
        "status": "retired",
    }
    manifest = {"schema": 2, "prefix": "zoo-weights", "built_with": built_with, "entries": entries}
    checkout.write_manifest(manifest)
    return manifest


def test_fetch_all_fetches_live_entries_and_never_a_retired_one(checkout: Checkout):
    """Torch-free: the fetch half. A retired entry is not requested from the
    store at all (the stub would fail on it), and it is named."""
    checkout.install_real_tool()
    _stage_canonical(checkout, {"alpha": b"alpha-bytes", "twin": b"twin-bytes"}, {})
    proc = checkout.run(store_uri=STORE_URI)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "fetched + verified 2 dump(s)" in proc.stdout, proc.stdout
    assert "NOT FETCHED: 1 retired entr(ies)" in proc.stdout and "detr" in proc.stdout
    assert sorted(p.name for p in (checkout.root / "dist").iterdir()) == [
        "alpha_weights.pkl",
        "twin_weights.pkl",
    ]
    assert checkout.outputs() == {}, "a fetch that RAN must not hand over a skip reason"


def test_a_manifest_of_only_retired_entries_reddens_the_fetch_step(checkout: Checkout):
    checkout.install_real_tool()
    checkout.write_manifest(
        {
            "schema": 2,
            "entries": {"detr": {"file": "detr_weights.pkl", "sha256": "d" * 64, "status": "retired"}},
        }
    )
    proc = checkout.run(store_uri=STORE_URI)
    assert proc.returncode != 0, proc.stdout
    assert "no LIVE entries" in proc.stderr, proc.stderr


def test_an_undefined_status_reddens_the_fetch_step(checkout: Checkout):
    """A typo'd status is refused rather than fetched as live."""
    checkout.install_real_tool()
    sha = checkout.serve("alpha", b"alpha-bytes")
    checkout.write_manifest(
        {
            "schema": 2,
            "entries": {"alpha": {"file": "alpha_weights.pkl", "sha256": sha, "status": "retried"}},
        }
    )
    proc = checkout.run(store_uri=STORE_URI)
    assert proc.returncode != 0, proc.stdout
    assert "'retried'" in proc.stderr, proc.stderr
    assert not (checkout.root / "dist" / "alpha_weights.pkl").exists()


def test_one_canonical_manifest_takes_both_steps_green(checkout: Checkout):
    """THE FIX, measured end to end: the fetch step's shell, then the verifier
    step's shell, on one schema-2 manifest — and green only because each
    per-entry key was honoured (the anchor below removes one and it reddens)."""
    torch = pytest.importorskip("torch")
    import io
    from importlib import metadata

    # The provenance block that matches THIS interpreter, so the only thing
    # under test is the schema, not whichever pins the test env carries.
    installed = {}
    for key in _verifier_module()._PROVENANCE_KEYS:
        try:
            installed[key] = metadata.version(key)
        except metadata.PackageNotFoundError:
            pass

    def fc_state(out_features: int) -> bytes:
        """`fc.` keys: the templates hold their Linear as `self.fc`."""
        buf = io.BytesIO()
        sd = {f"fc.{k}": v for k, v in torch.nn.Linear(4, out_features).state_dict().items()}
        torch.save(sd, buf)
        return buf.getvalue()

    checkout.install_real_tool()
    checkout.install_real_verifier()
    manifest = _stage_canonical(checkout, {"alpha": fc_state(3), "twin": fc_state(5)}, installed)

    fetch = checkout.run(store_uri=STORE_URI)
    assert fetch.returncode == 0, f"{fetch.stdout}\n{fetch.stderr}"
    gate = checkout.run_verify_step(checkout.outputs().get("not_fetched", ""))
    assert gate.returncode == 0, f"{gate.stdout}\n{gate.stderr}"
    assert "All 2 live dump(s) verify against the engine pin" in gate.stdout, gate.stdout
    assert "NOT GATED: 1 retired entr(ies)" in gate.stdout and "detr" in gate.stdout

    # Anchor: drop the recorded category and the same bytes no longer have ONE
    # template — the green above depended on `category` being read.
    del manifest["entries"]["twin"]["category"]
    checkout.write_manifest(manifest)
    gate = checkout.run_verify_step("")
    assert gate.returncode == 1, f"{gate.stdout}\n{gate.stderr}"
    assert "NO_TEMPLATE" in gate.stdout and "twin" in gate.stdout, gate.stdout


def test_a_dumps_only_manifest_is_refused_by_both_steps(checkout: Checkout):
    """The retired `dumps` shape: the fetch step reddens on it, so no dump is
    fetched — and the gate, handed it directly, refuses it BY NAME rather than
    reading it as a stub."""
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

    gate = checkout.run_verifier()
    assert gate.returncode == 2, f"{gate.stdout}\n{gate.stderr}"
    assert "'dumps' list" in gate.stderr and "nothing ever wrote it" in gate.stderr, gate.stderr


def test_a_named_skip_reaches_the_verifier_and_it_loads_nothing(checkout: Checkout):
    """TODAY'S STATE, end to end and torch-free: the store variables are unset,
    so the fetch step names both and hands the reason over; the verifier step
    — its python3 refusing every engine-pin import, because the install was
    skipped — parses the manifest, resolves every live entry, and says SKIP."""
    checkout.install_real_tool()
    checkout.install_real_verifier()
    _stage_canonical(checkout, {"alpha": b"alpha", "twin": b"twin"}, {"torch": "2.11.0"})

    fetch = checkout.run(store_uri=None, role_arn=None, credentials="skipped")
    assert fetch.returncode == 0, f"{fetch.stdout}\n{fetch.stderr}"
    reason = checkout.outputs().get("not_fetched", "")
    assert "no store URI" in reason and "no read role" in reason, checkout.outputs()
    assert "fork" not in reason, reason

    gate = checkout.run_verify_step(reason, engine_pin_blocked=True)
    assert gate.returncode == 0, f"{gate.stdout}\n{gate.stderr}"
    assert "BLOCKED" not in gate.stderr, gate.stderr
    assert f"SKIP (dumps not fetched): {reason}" in gate.stdout, gate.stdout
    assert "all 2 live entr(ies) resolve to a template" in gate.stdout, gate.stdout
    assert "This is not a pass" in gate.stdout, gate.stdout
    assert "NOT GATED: 1 retired" in gate.stdout, gate.stdout

    # Anchor 1: WITHOUT the reason, the same step on the same interpreter tries
    # to sweep — so the green above is the not-fetched mode, not an inert gate.
    swept = checkout.run_verify_step("", engine_pin_blocked=True)
    assert swept.returncode != 0 and "BLOCKED: torch" in swept.stderr, swept.stderr

    # Anchor 2: a structural defect is still RED on the skip path — a live entry
    # that maps to no template is not excused by the store being unconfigured.
    manifest = json.loads((checkout.root / "manifest.json").read_text())
    manifest["entries"]["ghost"] = {"file": "ghost_weights.pkl", "sha256": "e" * 64}
    checkout.write_manifest(manifest)
    ghost = checkout.run_verify_step(reason, engine_pin_blocked=True)
    assert ghost.returncode == 1, f"{ghost.stdout}\n{ghost.stderr}"
    assert "NO_TEMPLATE" in ghost.stdout and "ghost" in ghost.stdout, ghost.stdout


def test_the_verifier_refuses_an_unnamed_skip(checkout: Checkout):
    """`--dumps-not-fetched ''` would be exactly the unnamed skip this gate's
    history is made of. The step never passes it (it tests `-n` first); the
    tool refuses it anyway."""
    checkout.install_real_verifier()
    checkout.write_manifest(_entries_manifest("alpha", "a" * 64, 1))
    proc = subprocess.run(
        [sys.executable, str(VERIFIER_REL), "--dumps-not-fetched", " "],
        cwd=checkout.root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, proc.stdout
    assert "needs a reason" in proc.stderr, proc.stderr


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
    """The branch CI USED to take on every run. Premise changed (the schema
    reconciliation): the job now downloads the canonical manifest first, so
    this message must say the branch is NOT the expected CI path, name where
    the manifest comes from and the key it carries — and still promise no
    sweep. (Its earlier assertions — "STRUCTURAL", the two outstanding
    preconditions, the `dumps` key — described the divergence this PR ends.)"""
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
    assert "not the expected path" in out, (
        f"in CI a manifest is always downloaded here; an absent one is a fault:\n{out}"
    )
    assert "STRUCTURAL" not in out, out
    for needle in ("dump-manifest", "'entries'"):
        assert needle in out, f"{needle} not named in the no-op message:\n{out}"
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
        # Premise changed (the schema reconciliation): this used to require
        # "STRUCTURAL" — the branch was taken on every run. The job now
        # downloads the manifest, so both sites must instead point at the
        # artifact whose absence put them here.
        assert "dump-manifest" in out, (
            f"{label} does not name the artifact that should have supplied it:\n{out}"
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
        f"the exit must cite the decision that says where the store is, not just "
        f"the var:\n{message}"
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


def _folded_condition(step: str) -> str | None:
    """A step's `if:` as one line, whether written inline or folded (`>-`)."""
    lines = step.splitlines()
    at = [i for i, ln in enumerate(lines) if re.match(r"^(?:      - |        )if:", ln)]
    if not at:
        return None
    assert len(at) == 1, f"a step carrying more than one `if:`:\n{step}"
    line = lines[at[0]]
    first = line.split("if:", 1)[1].strip()
    if first not in (">-", ">", "|", "|-"):
        return first
    indent = len(line) - len(line.lstrip())
    body = []
    for ln in lines[at[0] + 1 :]:
        if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent:
            break
        body.append(ln.strip())
    cond = " ".join(b for b in body if b)
    assert cond, f"empty folded `if:`:\n{step}"
    return cond


def _install_gate_file(installs: list[str]) -> str:
    """The ONE file both install conditions key on -- read off the workflow,
    so nothing below restates its name.

    Premise changed (the schema reconciliation): the install used to be gated
    on the manifest ALONE. With the manifest now always downloaded, that would
    install the engine pin on every run for a verifier that loads nothing, so
    the install condition is the CREDENTIAL step's condition, verbatim --
    "install iff the dumps will be fetched". Its FIRST clause is still the
    manifest presence test this function reads the file name from."""
    gated_on: set[str] = set()
    creds = _creds_condition()
    for step in installs:
        cond = _folded_condition(step)
        assert cond is not None, f"an install step runs unconditionally again:\n{step}"
        assert cond == creds, (
            "an install step's condition differs from the credential step's, so "
            "the stack is installed when no fetch will run (wasted minutes) or "
            f"NOT installed when one will (the sweep dies on import torch):\n"
            f"install: {cond!r}\ncreds:   {creds!r}"
        )
        m = _HASHFILES.match(cond.split("&&")[0].strip())
        assert m, f"the install condition does not start with a hashFiles presence test: {cond!r}"
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
    downloads = [s for s in steps if _ARTIFACT_DOWNLOAD.search(s)]
    assert len(installs) == 2, f"expected setup-python + the pip install, got:\n{installs}"
    assert len(verdicts) == 2, f"expected the fetch step + the verifier, got:\n{verdicts}"
    assert len(checkouts) == 1, checkouts
    assert len(downloads) == 1, downloads

    manifest = _install_gate_file(installs)
    assert f"[ ! -f {manifest} ]" in guard_script, (
        f"the install is gated on {manifest!r} but the fetch step tests a "
        f"different path:\n{guard_script[:300]}"
    )
    # The checkout must run for hashFiles to see the file at all, the manifest
    # download must run for there to BE a file, and the two verdict steps must
    # run so the verdict stays the tools' on every path.
    for step in checkouts + downloads + verdicts:
        assert _folded_condition(step) is None, (
            f"a step that must run on every path carries a condition:\n{step}"
        )


# `hashFiles` sees only what is in the checkout when the condition is evaluated.
# That is why this test used to forbid ANY artifact download into this job: a
# download's natural home is beside the fetch step -- AFTER both installs --
# and in that arrangement the installs skip, the manifest then exists, and the
# verifier's first `import torch` raises under the runner's bare python3.
#
# The job now DOES download the dump-manifest artifact (the schema
# reconciliation), so the rule is narrowed to what it was protecting: EXACTLY
# ONE step may produce the gated file -- the dump-manifest download, into the
# checkout root -- and it must come before EVERY step whose condition reads the
# file. Anything else mentioning the file other than the known READS (the
# conditions and the fetch step's presence test) -- or messages that merely
# talk about it -- is a finding; an unrecognised form fails rather than being
# guessed benign.
_ARTIFACT_DOWNLOAD = re.compile(r"^\s*(?:- )?uses:\s*\S*download-artifact", re.MULTILINE)
_MANIFEST_ARTIFACT = "dump-manifest"


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


def _producer_violations(steps: list[str], manifest: str) -> list[str]:
    """Why the job's producers of `manifest` are unsafe, or [] if they are not.

    The one allowed producer is a download of the dump-manifest artifact into
    the checkout root, ahead of every step whose condition reads the file."""
    problems: list[str] = []
    readers = [
        i
        for i, st in enumerate(steps)
        if f"hashFiles('{manifest}')" in (_folded_condition(st) or "")
    ]
    producers = [i for i, st in enumerate(steps) if _ARTIFACT_DOWNLOAD.search(st)]
    if not readers:
        problems.append("no step's condition reads the manifest -- wrong file?")
    if len(producers) != 1:
        problems.append(f"expected exactly one artifact download, found {len(producers)}")
    for i in producers:
        st = steps[i]
        if not re.search(rf"^\s+name: {_MANIFEST_ARTIFACT}\s*$", st, re.MULTILINE):
            problems.append(f"a download that is not the {_MANIFEST_ARTIFACT} artifact:\n{st}")
        if not re.search(r"^\s+path: \.\s*$", st, re.MULTILINE):
            problems.append(f"the manifest download does not land in the checkout root:\n{st}")
        late = [r for r in readers if r < i]
        if late:
            problems.append(
                f"the download runs AFTER {len(late)} step(s) whose condition reads "
                f"{manifest!r} -- they were decided before the file existed:\n{st}"
            )
    for st in steps:
        hits = _manifest_mentions_that_are_not_reads(st, manifest)
        if hits:
            problems.append(
                f"a step touches {manifest!r} in a form this test does not know to "
                "be a read:\n" + "\n".join(hits)
            )
    return problems


def test_the_one_producer_of_the_gated_file_precedes_every_step_that_reads_it():
    steps = _verify_job_steps()
    installs = [s for s in steps if "actions/setup-python@" in s or ENGINE_PIN_INSTALL in s]
    manifest = _install_gate_file(installs)
    assert _producer_violations(steps, manifest) == []

    # Anchors -- the same detector on the shapes it exists to catch, built by
    # rearranging the REAL steps, so a green run above means "nothing found",
    # not "nothing looked for".
    (dl,) = [i for i, st in enumerate(steps) if _ARTIFACT_DOWNLOAD.search(st)]
    moved = steps[:dl] + steps[dl + 1 :]
    fetch_at = next(i for i, st in enumerate(moved) if f"- name: {STEP_NAME}" in st)
    late = moved[:fetch_at] + [steps[dl]] + moved[fetch_at:]
    assert any("AFTER" in p for p in _producer_violations(late, manifest))
    assert any("exactly one" in p for p in _producer_violations(steps + [steps[dl]], manifest))
    other = steps[dl].replace(f"name: {_MANIFEST_ARTIFACT}", "name: engine-pin")
    assert any(
        "not the" in p for p in _producer_violations(steps[:dl] + [other] + steps[dl + 1 :], manifest)
    )
    elsewhere = steps[dl].replace("path: .", "path: _manifest")
    assert any(
        "checkout root" in p
        for p in _producer_violations(steps[:dl] + [elsewhere] + steps[dl + 1 :], manifest)
    )
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


def test_the_verify_job_needs_the_job_that_mints_the_manifest():
    """The download names an artifact another job uploads; without `needs:` it
    races that job and fails, or -- worse -- a re-run reads a stale one."""
    block = _job_block(VERIFY_JOB)
    m = re.search(r"^    needs: \[([^\]]+)\]\s*$", block, re.MULTILINE)
    assert m, f"verify-dumps has no list-form needs:\n{block[:400]}"
    needs = {n.strip() for n in m.group(1).split(",")}
    assert {"engine-pin-drift-guard", "fetch-dump-manifest"} <= needs, needs
    uploader = _job_block("fetch-dump-manifest")
    assert f"name: {_MANIFEST_ARTIFACT}" in uploader and "upload-artifact@" in uploader


def test_no_manifest_json_is_committed_at_the_repo_root():
    """The download lands at ./manifest.json. A committed file there would be
    silently replaced by the canonical one on every run -- and read as the
    manifest by anyone looking at the tree."""
    tracked = subprocess.run(
        ["git", "ls-files", "--", "manifest.json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert tracked.stdout.strip() == "", tracked.stdout


def test_the_verifier_step_hands_over_the_fetch_steps_skip_reason():
    """The fetch step writes `not_fetched`; the verifier step must read THAT
    output, by the fetch step's id, and require a manifest."""
    fetch = _verify_step(f"- name: {STEP_NAME}")
    m = re.search(r"^\s+id: (\S+)\s*$", fetch, re.MULTILINE)
    assert m, f"the fetch step has no id:\n{fetch}"
    assert 'echo "not_fetched=$missing" >> "$GITHUB_OUTPUT"' in fetch, fetch
    verify = _verify_step(f"- name: {VERIFY_STEP_NAME}")
    assert f"DUMPS_NOT_FETCHED: ${{{{ steps.{m.group(1)}.outputs.not_fetched }}}}" in verify, verify
    assert verify.count("--require-manifest") == 2, verify
    assert '--dumps-not-fetched "$DUMPS_NOT_FETCHED"' in verify, verify


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
    #
    # (Adjusted by the schema reconciliation: the declared dump is now in the
    # canonical `entries` shape, with its template in the checkout — a `dumps`
    # manifest is refused before any import, which would make this anchor pass
    # without the blocker ever being reached.)
    payload = b"bert-dump-bytes"
    (checkout.root / "dist").mkdir()
    (checkout.root / "dist" / "bert_base_uncased_weights.pkl").write_bytes(payload)
    checkout.install_template("text_classification", "bert_base_uncased", "")
    checkout.write_manifest(
        _entries_manifest(
            "bert_base_uncased", hashlib.sha256(payload).hexdigest(), len(payload)
        )
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


# --------------------------------------------------------------------------
# The bucket is private: credentials are a named precondition too
# --------------------------------------------------------------------------
# Until the credential step existed, this job had no AWS step, no AWS env, no
# secret and no `id-token` permission, so setting TRACEBLOC_ZOO_WEIGHTS_URI
# would have sent `aws s3 cp` at a private bucket with nothing to sign it --
# the gate could never have armed, and no skip message said so. Each of the
# three credential-side preconditions (a store URI, a read role, a trusted
# event) is now reported by name, every one that is missing and only those,
# and the one disagreement between the credential step's `if:` and this shell
# that could fetch without credentials is a red.

CREDS_STEP_NAME = "Assume the zoo-weights read role"
CREDS_ACTION = "aws-actions/configure-aws-credentials@"
# The marker each precondition's SKIP line carries, keyed by the precondition.
# Written out here, not read from the shell, so a shell that renames or drops
# one reddens instead of agreeing with itself.
_SKIP_MARKER = {
    "uri": "SKIP (no store URI)",
    "role": "SKIP (no read role)",
    "trust": "SKIP (fork pull request)",
}


def _run_with_missing(checkout: Checkout, missing: frozenset[str]):
    return checkout.run(
        store_uri=None if "uri" in missing else STORE_URI,
        role_arn=None if "role" in missing else ROLE_ARN,
        trusted="false" if "trust" in missing else "true",
        # What GitHub would report: the credential step's `if:` fails on any
        # missing precondition, so its outcome is `skipped`.
        credentials="skipped" if missing else "success",
    )


_MISSING_SETS = [
    frozenset(c)
    for n in range(1, len(_SKIP_MARKER) + 1)
    for c in itertools.combinations(sorted(_SKIP_MARKER), n)
]


@pytest.mark.parametrize("missing", _MISSING_SETS, ids=lambda m: "+".join(sorted(m)))
def test_each_missing_credential_precondition_is_named_and_only_it(
    checkout: Checkout, missing: frozenset[str]
):
    """All seven non-empty subsets of {URI, role, trust}: exactly the missing
    ones are named, none of the present ones is, and the hook is never run."""
    checkout.install_real_tool()
    checkout.stage("bert_base_uncased", b"dump-bytes")
    proc = _run_with_missing(checkout, missing)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    for key, marker in _SKIP_MARKER.items():
        if key in missing:
            assert marker in proc.stdout, f"{key} missing but not named:\n{proc.stdout}"
        else:
            assert marker not in proc.stdout, (
                f"{key} is present but its SKIP was printed:\n{proc.stdout}"
            )
    assert "fetched + verified" not in proc.stdout, proc.stdout
    assert not any((checkout.root / "dist").iterdir()), "fetched with a precondition missing"
    assert "SKIP (no manifest)" not in proc.stdout
    assert "SKIP (no sync tool)" not in proc.stdout


def test_no_read_role_names_the_variable_and_read_only_scope(checkout: Checkout):
    checkout.install_real_tool()
    checkout.stage("bert_base_uncased", b"dump-bytes")
    proc = checkout.run(store_uri=STORE_URI, role_arn=None, credentials="skipped")
    assert proc.returncode == 0, proc.stderr
    assert "ZOO_WEIGHTS_READ_ROLE_ARN" in proc.stdout, proc.stdout
    assert "READ-ONLY" in proc.stdout, proc.stdout


def test_no_skip_message_says_the_hosting_decision_is_open(checkout: Checkout):
    """The decision is made. A message calling it open sends an admin to wait
    for a decision instead of setting two variables."""
    checkout.install_real_tool()
    checkout.stage("bert_base_uncased", b"dump-bytes")
    out = checkout.run(store_uri=None, role_arn=None, credentials="skipped").stdout
    assert "SKIP (no store URI)" in out, out
    assert "decision is made" in out, out
    for stale in ("undecided", "decision is open", "open decision", "once the store exists"):
        assert stale not in out.lower(), f"stale hosting text {stale!r}:\n{out}"
    # The same stale claim in the workflow's own prose, which is where the
    # next person reads whether hosting is still pending.
    text = WORKFLOW.read_text().lower()
    for stale in ("hosting decision is open", "open decision", "the store exists"):
        assert stale not in text, f"stale hosting text {stale!r} in {WORKFLOW.name}"


def test_all_preconditions_met_without_credentials_is_red(checkout: Checkout):
    """Every precondition the shell checks holds, but the credential step did
    not succeed: the `if:` and the shell disagree. Fetching anyway is the
    original defect (a private bucket, no credentials); skipping would be a
    green that verified nothing. Red, and the hook is not run."""
    checkout.install_real_tool()
    checkout.stage("bert_base_uncased", b"dump-bytes")
    for outcome in ("skipped", "", "failure"):
        proc = checkout.run(store_uri=STORE_URI, credentials=outcome)
        assert proc.returncode != 0, f"outcome={outcome!r}:\n{proc.stdout}"
        assert CREDS_STEP_NAME in proc.stderr, proc.stderr
        assert "fetched + verified" not in proc.stdout, proc.stdout
    # Anchor: the same checkout with the step's success fetches.
    ok = checkout.run(store_uri=STORE_URI)
    assert ok.returncode == 0 and "fetched + verified 1 dump(s)" in ok.stdout, ok.stdout


def test_an_unreadable_trust_flag_fails_closed(checkout: Checkout):
    """`true` / `false` is all the expression can render. Anything else means
    the mapping was dropped or mangled, and "cannot tell whether this run may
    hold credentials" is a red, never a guess in either direction."""
    checkout.install_real_tool()
    checkout.stage("bert_base_uncased", b"dump-bytes")
    for flag in ("", "True", "yes"):
        proc = checkout.run(store_uri=STORE_URI, trusted=flag)
        assert proc.returncode != 0, f"trusted={flag!r}:\n{proc.stdout}"
        assert "ZOO_WEIGHTS_TRUSTED_EVENT" in proc.stderr, proc.stderr
        assert "fetched + verified" not in proc.stdout, proc.stdout


def _verify_step(needle: str) -> str:
    matches = [s for s in _verify_job_steps() if needle in s]
    assert len(matches) == 1, f"expected one {VERIFY_JOB} step with {needle!r}, got {matches}"
    return matches[0]


def _trust_expression_of_fetch_step() -> str:
    m = re.search(
        r"^\s+ZOO_WEIGHTS_TRUSTED_EVENT: \$\{\{ (.+) \}\}\s*$",
        _verify_step(f"- name: {STEP_NAME}"),
        re.MULTILINE,
    )
    assert m, "the fetch step no longer maps ZOO_WEIGHTS_TRUSTED_EVENT from an expression"
    return m.group(1)


def _creds_condition() -> str:
    """The credential step's `if:`, folded (`>-`) or single-line, as one line."""
    step = _verify_step(f"- name: {CREDS_STEP_NAME}")
    lines = step.splitlines()
    at = next(i for i, ln in enumerate(lines) if ln.strip().startswith("if:"))
    first = lines[at].strip()[len("if:") :].strip()
    if first not in (">-", ">", "|", "|-"):
        return first
    indent = len(lines[at]) - len(lines[at].lstrip())
    body = []
    for ln in lines[at + 1 :]:
        if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent:
            break
        body.append(ln.strip())
    cond = " ".join(b for b in body if b)
    assert cond, f"empty folded `if:` on {CREDS_STEP_NAME!r}"
    return cond


def test_the_credential_step_precedes_the_fetch_and_is_pinned():
    steps = _verify_job_steps()
    creds_at = [i for i, s in enumerate(steps) if f"- name: {CREDS_STEP_NAME}" in s]
    fetch_at = [i for i, s in enumerate(steps) if f"- name: {STEP_NAME}" in s]
    assert len(creds_at) == 1 and len(fetch_at) == 1, (creds_at, fetch_at)
    assert creds_at[0] < fetch_at[0], (
        "the credential step runs AFTER the fetch -- the fetch would hit the "
        "private bucket with no credentials"
    )
    creds = steps[creds_at[0]]
    assert re.search(re.escape(CREDS_ACTION) + r"[0-9a-f]{40} # v\d", creds), (
        f"{CREDS_ACTION} must be pinned by full commit SHA like every action here:\n{creds}"
    )
    assert "role-to-assume: ${{ vars.ZOO_WEIGHTS_READ_ROLE_ARN }}" in creds, creds
    assert "aws-region: eu-central-1" in creds, creds
    # No stored key: OIDC only.
    for key in ("aws-access-key-id", "aws-secret-access-key", "secrets."):
        assert key not in creds, f"{key!r} in the credential step -- OIDC only:\n{creds}"
    # The fetch step reads THIS step's outcome, by its id.
    m = re.search(r"^\s+id: (\S+)\s*$", creds, re.MULTILINE)
    assert m, f"the credential step has no id:\n{creds}"
    fetch = steps[fetch_at[0]]
    assert f"ZOO_WEIGHTS_CREDENTIALS: ${{{{ steps.{m.group(1)}.outcome }}}}" in fetch, fetch
    assert "ZOO_WEIGHTS_READ_ROLE_ARN: ${{ vars.ZOO_WEIGHTS_READ_ROLE_ARN }}" in fetch, fetch


def test_the_credential_step_requires_every_precondition_the_shell_names():
    """The step's `if:` must name the manifest, both variables and the SAME
    trust expression the fetch step renders -- two spellings of one rule are
    held equal here, since GitHub gives no way to share one."""
    cond = _creds_condition()
    trust = _trust_expression_of_fetch_step()
    for needle in (
        "hashFiles('manifest.json') != ''",
        "vars.TRACEBLOC_ZOO_WEIGHTS_URI != ''",
        "vars.ZOO_WEIGHTS_READ_ROLE_ARN != ''",
        f"({trust})",
    ):
        assert needle in cond, f"credential step `if:` lacks {needle!r}: {cond!r}"
    clauses = [c.strip() for c in cond.split("&&")]
    assert len(clauses) == 4, f"expected exactly four ANDed preconditions: {clauses}"
    # And the trust expression is the fork refusal it claims to be: a
    # pull_request is trusted only when its head is this repository.
    assert trust == (
        "github.event_name != 'pull_request' || "
        "github.event.pull_request.head.repo.full_name == github.repository"
    ), trust


def _job_permissions(job: str) -> str:
    block = _job_block(job)
    m = re.search(r"^    permissions:\n((?:      .*\n)+)", block, re.MULTILINE)
    return m.group(1) if m else ""


def test_id_token_write_is_scoped_to_the_verify_job_only():
    text = WORKFLOW.read_text()
    top = re.search(r"^permissions:\n((?:  .*\n)+)", text, re.MULTILINE)
    assert top, "no workflow-level permissions block"
    assert top.group(1).strip() == "contents: read", (
        f"workflow-level permissions widened: {top.group(1)!r}"
    )
    jobs = re.findall(r"^  ([A-Za-z0-9_-]+):\n    ", text.split("\njobs:\n", 1)[1], re.MULTILINE)
    assert VERIFY_JOB in jobs and len(jobs) >= 6, jobs
    with_id_token = [j for j in jobs if "id-token: write" in _job_permissions(j)]
    assert with_id_token == [VERIFY_JOB], (
        f"id-token: write must be granted to {VERIFY_JOB} alone, got {with_id_token}"
    )
    perms = _job_permissions(VERIFY_JOB)
    assert "contents: read" in perms, (
        "a job-level permissions block replaces the workflow's, so the checkout "
        f"needs contents: read restated:\n{perms}"
    )
    assert "write" not in perms.replace("id-token: write", ""), perms


def test_the_permission_matcher_sees_a_job_block():
    """Anchor for the test above: `_job_permissions` finds a planted block and
    reports none where there is none, so `== [VERIFY_JOB]` is a measurement."""
    assert "id-token: write" in _job_permissions(VERIFY_JOB)
    assert _job_permissions("fetch-engine-pin") == ""


# --------------------------------------------------------------------------
# The hook and the gate agree on the manifest's VOCABULARY, not just its key
# --------------------------------------------------------------------------


def _hook_module():
    spec = importlib.util.spec_from_file_location("_hook_for_vocab_guard", REPO_ROOT / TOOL_REL)
    assert spec and spec.loader, TOOL_REL
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    return hook


def test_the_hook_knows_exactly_the_statuses_the_gates_know():
    """The hook holds its own `RETIRED` (it runs by hand outside this checkout's
    tools/), so it is held equal to check_dump_coverage's vocabulary here --
    the one the verifier reads. A status added there and not here would be
    fetched-then-refused, or refused-then-never-fetched."""
    spec = importlib.util.spec_from_file_location(
        "_coverage_for_vocab_guard", REPO_ROOT / VERIFIER_SIBLINGS[0]
    )
    assert spec and spec.loader
    coverage = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        spec.loader.exec_module(coverage)
    finally:
        sys.path.remove(str(REPO_ROOT / "tools"))
    assert (_hook_module().RETIRED,) == tuple(coverage.KNOWN_STATUSES)


def test_the_hook_records_every_provenance_key_the_gate_reconciles():
    """`_build_env` used to record four of the verifier's five keys (no
    torchvision), so a manifest the hook wrote was red-by-omission on the
    gate's partial-block rule. Read from `_build_env`'s source rather than by
    calling it, which would import the whole ML stack to list five names."""
    src = (REPO_ROOT / TOOL_REL).read_text()
    fn = next(
        node
        for node in ast.parse(src).body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_env"
    )
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For) and isinstance(n.iter, ast.Tuple)]
    assert len(loops) == 1, "could not find _build_env's package tuple"
    recorded = tuple(ast.literal_eval(loops[0].iter))
    assert set(recorded) == set(_verifier_module()._PROVENANCE_KEYS), recorded
