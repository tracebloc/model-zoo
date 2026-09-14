"""The engine-pin drift guard must watch BOTH mirrors of the engine's pin (internal ref).

There are two copies of the engine's pin in this repo:

  * ``tools/requirements-engine-pin.txt`` — what prep and ``verify-dumps``
    install; and
  * ``.github/requirements/pytorch.txt`` — what ci.yml's REQUIRED
    ``test-pytorch`` job installs.

The second one's own header states the rule — *"Bump these together with the
engine's pins, never ahead of them"* — and it was the one copy no guard
checked, for two independent reasons, both of which had to be fixed:

  1. ``verify-dumps-engine-pin.yml``'s ``paths:`` filters named the tools/
     mirror but not ``.github/requirements/**``, so a PR touching only
     ``pytorch.txt`` fired **no job at all**; and
  2. the guard step had exactly one ``--mirror``, pointed at the tools/ copy,
     so it would not have looked even if it had run.

(internal ref) is the proof case: as dependabot opened it, it touched only
``pytorch.txt``, sat two minors ahead of the engine, and was **all green**.

Each leg is asserted separately below, because either one alone restores the
blind spot while the other keeps passing — which is exactly how the gap
survived. The trigger half reads the real ``paths:`` lists back out of the
YAML and matches the real filename against them; the invocation half runs the
step's own shell against a synthetic engine pin and requires it to REFUSE.

No PyYAML. This suite runs in all three CI framework envs
(``.github/requirements/{pytorch,sklearn,survival}.txt``) and none installs
it, so a test that imported it would skip in all three — i.e. never run, which
is the state this file exists to end. Every extraction failure is an
assertion, never a skip. Nothing here imports torch either: the checker and
the verifier it borrows ``_PROVENANCE_KEYS`` from are stdlib-only at import
time (the verifier imports torch lazily, inside functions).
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "verify-dumps-engine-pin.yml"
CHECKER_REL = "tools/check_engine_pin_drift.py"
VERIFIER_REL = "tools/verify_dumps_against_engine_pin.py"
TOOLS_MIRROR = "tools/requirements-engine-pin.txt"
CI_MIRROR = ".github/requirements/pytorch.txt"
REQUIREMENTS_DIR = REPO_ROOT / ".github" / "requirements"
GUARD_JOB = "engine-pin-drift-guard"
GUARD_STEP = "Assert every mirror matches the engine's pin"

# The engine's torch/torchvision come from requirements_cuda.txt, everything
# else from requirements.txt — mirrored here so the synthetic engine pin the
# behavioural tests build has the same shape the workflow downloads.
_CUDA_PINS = ("torch", "torchvision")

# THE ENGINE'S REAL CANDIDATE FILENAMES, measured on `tracebloc-engine@develop`
# (`292fe502`) with `git grep -lP '(?:^|\s)torch=='` plus a full exact-pin scan
# over `Dockerfile.base.*` + `use_cases/requirements*.txt` (internal ref):
#
#   10 candidates in the search space
#    7 carry a literal `pkg==ver` -> the derived engine source set
#    3 GPU Dockerfiles carry none  -> they pin torch via a cu129 wheel index
#
# The old workflow fetched two of the seven. `requirements_vision_cv_cuda.txt`
# — the torch version of the `cv:gpu` base image — was not one of them.
_ENGINE_REQUIREMENTS = (
    "use_cases/requirements.txt",
    "use_cases/requirements_cuda.txt",
    "use_cases/requirements_vision_cv.txt",
    "use_cases/requirements_vision_cv_cuda.txt",
)
#: The engine's non-CUDA requirements files carry no torch pin of their own.
_ENGINE_PLAIN_ONLY = (
    "use_cases/requirements.txt",
    "use_cases/requirements_vision_cv.txt",
)
_ENGINE_CPU_DOCKERFILES = (
    "Dockerfile.base.cpu",
    "Dockerfile.base.cv.cpu",
    "Dockerfile.base.text.cpu",
)
#: The GPU bases and the requirements file each one GREPS its torch pin out
#: of, measured on `tracebloc-engine@develop` (292fe502) from their
#: `ARG REQUIREMENTS_FILE=` defaults. `Dockerfile.base.cv.gpu` derives from
#: the very file (internal ref) says was never read, which is what makes that
#: file structurally required rather than merely desirable.
_ENGINE_GPU_DERIVES_FROM = {
    "Dockerfile.base.gpu": "use_cases/requirements_cuda.txt",
    "Dockerfile.base.cv.gpu": "use_cases/requirements_vision_cv_cuda.txt",
    "Dockerfile.base.text.gpu": "use_cases/requirements_cuda.txt",
}
_ENGINE_GPU_DOCKERFILES = tuple(_ENGINE_GPU_DERIVES_FROM)
_ENGINE_CANDIDATES = (
    *_ENGINE_CPU_DOCKERFILES,
    *_ENGINE_GPU_DOCKERFILES,
    *_ENGINE_REQUIREMENTS,
)
#: The CV GPU requirements file, named on its own because it IS the ticket.
CV_GPU_REQUIREMENTS = "use_cases/requirements_vision_cv_cuda.txt"

_CPU_DOCKERFILE = """FROM python:3.11-slim
# A CPU base pins torch inline, so it IS an engine source.
RUN pip install --no-cache-dir \\
    %s
"""

#: The GPU bases' real idiom: no literal pin, a grep into a wheel-index install.
#: A reader that treated this as a source would add a path contributing zero
#: pins, which is a silent no-op rather than a crash -- worse than the gap.
_GPU_DOCKERFILE = """FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04
ARG REQUIREMENTS_FILE=%s
COPY ${REQUIREMENTS_FILE} /tmp/requirements.txt
RUN set -eux; \\
    grep -E '^torch(vision)?==' /tmp/requirements.txt > /tmp/torch-pins.txt; \\
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu129 \\
      -r /tmp/torch-pins.txt
"""

_EXACT = re.compile(r"^([A-Za-z0-9._-]+)==([^\s#]+)")


# --------------------------------------------------------------------------
# Reading the workflow. Hand-rolled slicing, for the reason in the docstring.
# --------------------------------------------------------------------------
def _lines() -> list[str]:
    return WORKFLOW.read_text().splitlines()


def _event_paths(event: str) -> list[str]:
    """The ``paths:`` filter of ``on.<event>``, as written.

    Comment lines are skipped rather than terminating the list: the trigger
    entry added for (internal ref) carries a comment explaining why it is the whole
    directory, and a naive reader would stop at it and report the filter as
    lacking the very entry it documents.
    """
    lines = _lines()
    start = None
    for i, line in enumerate(lines):
        if line == f"  {event}:":
            start = i
            break
    assert start is not None, (
        f"no `on.{event}` trigger in {WORKFLOW}. If the trigger was removed or "
        "renamed, this test's premise changed — update it in the same commit."
    )
    block: list[str] = []
    for line in lines[start + 1 :]:
        # A non-blank line at 2-space indent or less is the next `on:` key.
        if line.strip() and not line.startswith("    "):
            break
        block.append(line)
    p_start = None
    for i, line in enumerate(block):
        if line.strip() == "paths:":
            p_start = i
            break
    assert p_start is not None, f"`on.{event}` declares no `paths:` filter"
    entries: list[str] = []
    for line in block[p_start + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not stripped.startswith("- "):
            break
        entries.append(stripped[2:].strip().strip("\"'"))
    assert entries, f"`on.{event}.paths` is empty"
    return entries


def _glob_matches(pattern: str, path: str) -> bool:
    """GitHub's ``paths:`` glob, enough of it for the patterns this file uses.

    ``**`` crosses ``/``, a single ``*`` does not. Written out rather than
    handed to ``fnmatch``, which treats ``*`` as crossing ``/`` and would call
    ``tools/*.py`` a match for ``tools/sub/x.py`` — i.e. it would report a
    trigger that GitHub does not fire.
    """
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.fullmatch("".join(out), path) is not None


def _triggers(path: str, event: str) -> bool:
    return any(_glob_matches(p, path) for p in _event_paths(event))


def _job_block(job: str = GUARD_JOB) -> str:
    """One job's YAML as text, sliced to the next top-level key at its indent."""
    text = WORKFLOW.read_text()
    key = f"  {job}:\n"
    assert "\n" + key in text, f"no job named {job!r} in {WORKFLOW}"
    start = text.index("\n" + key) + 1
    rest = text[start + len(key) :]
    end = len(rest)
    offset = 0
    for line in rest.splitlines(keepends=True):
        if line.strip() and not line.startswith("   ") and not line.startswith("\t"):
            end = offset
            break
        offset += len(line)
    return rest[:end]


def _guard_step_shell() -> str:
    """The ``run:`` body of the drift-guard step, dedented and runnable.

    EXTRACTED, never restated: a step edited in the workflow and not here must
    fail rather than drift (the convention tests/test_dump_fetch_guard.py set).
    """
    lines = _job_block().splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == f"- name: {GUARD_STEP}":
            start = i
            break
    assert start is not None, (
        f"no step named {GUARD_STEP!r} in job {GUARD_JOB}. If it was renamed, "
        "rename GUARD_STEP here in the same commit."
    )
    marker_indent = len(lines[start]) - len(lines[start].lstrip())
    run_at = None
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.strip() and (len(line) - len(line.lstrip())) <= marker_indent:
            break
        if line.strip() in ("run: |", "run: |-"):
            run_at = i
            break
    assert run_at is not None, f"step {GUARD_STEP!r} has no block `run: |` body"
    body_indent = len(lines[run_at]) - len(lines[run_at].lstrip()) + 2
    body: list[str] = []
    for line in lines[run_at + 1 :]:
        if line.strip() and (len(line) - len(line.lstrip())) < body_indent:
            break
        body.append(line[body_indent:] if len(line) >= body_indent else "")
    shell = "\n".join(body).rstrip() + "\n"
    assert shell.strip(), f"step {GUARD_STEP!r} has an empty run body"
    return shell


def _guard_step_code() -> str:
    """The guard step with its SHELL COMMENTS REMOVED.

    Any assertion about what the step DOES must read this, not the raw text.
    A `needle in raw_text` assertion is satisfiable by a comment quoting the
    needle — it greens with the code gone — and the negative direction is
    worse: a comment mentioning a removed flag reds a healthy tree. Both were
    live in this file's (internal ref) assertions and are proved in
    `test_THE_STEP_ASSERTIONS_CANNOT_BE_SATISFIED_BY_A_COMMENT` with the
    three results that fix needs, including the negative control.

    Comments are stripped from the WHOLE extracted step, which is a complete
    shell script, rather than from a slice of it. `_guard_step_shell()` is
    left returning raw text so its pre-existing callers are untouched.

    A `#` inside a single- or double-quoted string would be dropped by this
    naive line test, so it refuses rather than guessing: a wrong strip is the
    same false verdict one level down. The step contains no such string today
    and the assertion says so out loud.
    """
    raw = _guard_step_shell()
    kept = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "#" not in line or stripped.startswith("#"), (
            "a trailing `#` appeared in the guard step, so a line-based "
            "comment strip can no longer be trusted (it may be inside a "
            f"quoted string): {line!r}. Teach this helper real shell "
            "tokenisation rather than letting it guess."
        )
        kept.append(line)
    return "\n".join(kept)


def _guarded_mirrors() -> list[str]:
    """The mirror list the guard step iterates — the single place both are named."""
    shell = _guard_step_shell()
    m = re.search(r"for\s+mirror\s+in\s+(.+?);\s*do", shell)
    assert m, (
        "the drift-guard step no longer loops over a mirror list; if the shape "
        "changed, teach _guarded_mirrors() the new one rather than dropping the "
        "coverage assertions below"
    )
    mirrors = m.group(1).split()
    assert mirrors, "the guard step's mirror list is empty"
    return mirrors


def _exact_pins(text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _EXACT.match(line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def _provenance_keys() -> tuple[str, ...]:
    """``_PROVENANCE_KEYS`` from the verifier — the pins that invalidate a dump.

    Imported rather than restated so this test cannot claim coverage of a set
    the tools no longer use. Stdlib-only at import time; torch is imported
    lazily inside the verifier's functions.
    """
    path = REPO_ROOT / VERIFIER_REL
    spec = importlib.util.spec_from_file_location("_vdaep_for_pin_guard", path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    keys = mod._PROVENANCE_KEYS
    assert keys, "_PROVENANCE_KEYS is empty"
    return tuple(keys)


# --------------------------------------------------------------------------
# LEG 1 — THE TRIGGER. A pytorch.txt-only PR must fire the workflow.
# --------------------------------------------------------------------------
def test_the_paths_extractor_reads_the_real_filters():
    """The extractor itself, because both trigger tests trust its boundaries."""
    for event in ("pull_request", "push"):
        entries = _event_paths(event)
        assert "model_zoo/**" in entries, f"on.{event}.paths looks mis-sliced: {entries}"
        assert TOOLS_MIRROR in entries, f"on.{event}.paths lost the tools/ mirror"
        assert not any(
            e.endswith(":") for e in entries
        ), f"the slice ran past the paths list into another key: {entries}"


def test_A_PYTORCH_TXT_ONLY_CHANGE_TRIGGERS_THE_WORKFLOW():
    """(internal ref) leg 1. Without this the guard cannot fail because it never runs."""
    for event in ("pull_request", "push"):
        assert _triggers(CI_MIRROR, event), (
            f"on.{event}.paths does not match {CI_MIRROR}: a PR touching only "
            f"the CI pytorch mirror fires no job at all (internal ref). Filters: "
            f"{_event_paths(event)}"
        )


def test_the_tools_mirror_still_triggers_the_workflow():
    """Adding the second mirror must not cost the first one its trigger."""
    for event in ("pull_request", "push"):
        assert _triggers(TOOLS_MIRROR, event)


def test_the_filter_is_not_a_CATCH_ALL():
    """The control. If everything matched, the test above would be vacuous —
    and the workflow would run its 60-minute sweep on every PR ((internal ref)'s
    acceptance: a PR touching neither mirror still skips)."""
    for event in ("pull_request", "push"):
        for untouched in ("README.md", "LICENSE", "CLAUDE.md", "Makefile"):
            assert not _triggers(untouched, event), (
                f"on.{event}.paths matches {untouched}: the filter has become a "
                "catch-all, so the trigger assertions above prove nothing"
            )


def test_the_glob_matcher_does_not_let_a_single_star_cross_a_slash():
    """The matcher is the instrument; a wrong one would report a trigger
    GitHub does not fire (which is the bug, restated)."""
    assert _glob_matches(".github/requirements/**", CI_MIRROR)
    assert _glob_matches("tools/*.py", "tools/x.py")
    assert not _glob_matches("tools/*.py", "tools/sub/x.py")
    assert _glob_matches("tools/**", "tools/sub/x.py")
    assert not _glob_matches("model_zoo/**", "README.md")


# --------------------------------------------------------------------------
# LEG 2, WIRING — the guard must be POINTED at the file.
# --------------------------------------------------------------------------
def test_the_guard_job_invokes_the_drift_checker_at_all():
    assert CHECKER_REL in _job_block(), f"{GUARD_JOB} no longer runs the drift checker"


def test_BOTH_MIRRORS_ARE_CHECKED():
    """(internal ref) leg 2. One `--mirror` pointed at the tools/ copy is what let a
    pytorch.txt bump through even when the job did run."""
    mirrors = _guarded_mirrors()
    assert CI_MIRROR in mirrors, (
        f"the drift guard never checks {CI_MIRROR} — the mirror whose own header "
        f"carries the rule. Guarded: {mirrors}"
    )
    assert TOOLS_MIRROR in mirrors, (
        f"the drift guard stopped checking {TOOLS_MIRROR}; the second mirror is "
        f"an addition, not a replacement. Guarded: {mirrors}"
    )


def test_every_guarded_mirror_actually_exists():
    """A path nothing writes rots silently; here it would make the checker exit
    on a missing file, which reads as a broken gate rather than a moved one."""
    for mirror in _guarded_mirrors():
        assert (REPO_ROOT / mirror).is_file(), f"guarded mirror {mirror} is not in the repo"


def test_EVERY_CI_REQUIREMENT_FILE_THAT_MIRRORS_AN_ENGINE_PIN_IS_GUARDED():
    """The forward guard, and the reason the trigger is the whole directory.

    (internal ref) was not "pytorch.txt was forgotten" so much as "a file could mirror
    the engine's pin and no one would notice it was unguarded". So the
    property is stated over the directory rather than over one filename: any
    requirement set that exact-pins a dump-invalidating package must appear in
    the guard's mirror list.

    Today only pytorch.txt qualifies — sklearn.txt and survival.txt pin their
    own frameworks and none of _PROVENANCE_KEYS, and lint.txt pins ruff. A
    fourth env that pinned `transformers` would fail here instead of shipping
    ungated.
    """
    keys = set(_provenance_keys())
    guarded = set(_guarded_mirrors())
    unguarded = []
    for req in sorted(REQUIREMENTS_DIR.glob("*.txt")):
        rel = req.relative_to(REPO_ROOT).as_posix()
        mirrored = keys & set(_exact_pins(req.read_text()))
        if mirrored and rel not in guarded:
            unguarded.append((rel, sorted(mirrored)))
    assert not unguarded, (
        "these CI requirement sets exact-pin dump-invalidating packages but are "
        f"not in the drift guard's mirror list (internal ref): {unguarded}"
    )


# --------------------------------------------------------------------------
# LEG 2, BEHAVIOUR — the guard must be SEEN TO REFUSE.
#
# The (internal ref) scenario is reconstructed from the REAL pytorch.txt rather than
# from literal version strings: the mirror is left exactly as committed and a
# synthetic engine pin is written two minors BEHIND it. That is "the mirror is
# ahead of the engine", the direction the file's header forbids, and it stays
# true after any legitimate future bump.
# --------------------------------------------------------------------------
def _shift_minor(version: str, delta: int) -> str:
    parts = version.split(".")
    assert len(parts) >= 2 and parts[1].isdigit(), f"cannot shift minor of {version!r}"
    parts[1] = str(int(parts[1]) + delta)
    return ".".join(parts)


def _write_engine_pin(dest: Path, pins: dict[str, str]) -> None:
    """The engine-pin artifact, at the layout the workflow actually downloads.

    NOT A FLAT DIRECTORY OF TWO FILES ANY MORE (internal ref). The artifact
    now carries the whole derivation search space, so `upload-artifact` roots
    it at `_engine/` rather than `_engine/use_cases/` and every requirements
    path keeps its `use_cases/` segment.

    ALL TEN REAL CANDIDATE NAMES, split the way the engine really splits them:
    seven carry a literal `pkg==ver`, and the three GPU Dockerfiles carry none
    because each greps its torch pin out of whichever requirements file its
    `REQUIREMENTS_FILE` build-arg names and installs it from the cu129 wheel
    index. The derivation must therefore SEE ten and SELECT seven — a fixture
    holding only the two files the workflow used to fetch could not tell a
    reader that visits seven from one that visits two, which is the whole
    ticket.

    THE ENGINE IS WRITTEN SELF-CONSISTENT: one version per package across every
    source. A fixture that skewed one engine source against another would be
    refused by `_merge_engine_sources` (exit 2) before any mirror comparison
    ran — correct behaviour, and not what the callers below are measuring.
    """
    plain = [f"{p}=={v}" for p, v in sorted(pins.items()) if p not in _CUDA_PINS]
    cuda = [f"{p}=={v}" for p, v in sorted(pins.items()) if p in _CUDA_PINS]
    # Loud, because a fixture with no torch pin would quietly make the three
    # CPU Dockerfiles derive nothing and move the derived count from 7 to 4.
    assert cuda, (
        f"_write_engine_pin was given no {_CUDA_PINS} pin, so the CPU "
        "Dockerfiles would carry nothing and the derived source count would "
        "silently drop; pass a torch/torchvision pin or teach the count "
        "assertions the new shape"
    )
    (dest / "use_cases").mkdir(parents=True, exist_ok=True)
    for rel in _ENGINE_REQUIREMENTS:
        body = plain if rel in _ENGINE_PLAIN_ONLY else plain + cuda
        (dest / rel).write_text("\n".join(body) + "\n")
    for rel in _ENGINE_CPU_DOCKERFILES:
        (dest / rel).write_text(_CPU_DOCKERFILE % "  \\\n    ".join(cuda))
    for rel, derives_from in _ENGINE_GPU_DERIVES_FROM.items():
        (dest / rel).write_text(_GPU_DOCKERFILE % derives_from)


def _engine_pins_for(mirror: str, shift: int = 0, only: tuple[str, ...] = ()) -> dict[str, str]:
    """A synthetic engine pin derived from a real mirror, optionally skewed."""
    pins = _exact_pins((REPO_ROOT / mirror).read_text())
    assert pins, f"{mirror} has no exact pins to derive an engine pin from"
    if shift:
        targets = only or tuple(pins)
        pins = {
            p: (_shift_minor(v, shift) if p in targets else v) for p, v in pins.items()
        }
    return pins


def _run_checker(
    mirror: str, engine_dir: Path, base: Path | None = None
) -> subprocess.CompletedProcess:
    """The checker, optionally handed the mirror's value at the merge base.

    `base` is the ONLY evidence that settles which side moved
    (internal ref); it is a keyword so every pre-existing caller keeps
    exercising the no-evidence path, which is what push/schedule runs get.

    `--engine-root`, matching what the workflow now passes (internal ref), so
    these tests exercise the derivation rather than a hand-listed pair that CI
    no longer uses. The explicit `--engine` interface is still supported and is
    covered directly in LEG 5.
    """
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / CHECKER_REL),
            "--mirror",
            str(REPO_ROOT / mirror),
            "--engine-root",
            str(engine_dir),
            *(["--mirror-at-base", str(base)] if base is not None else []),
        ],
        capture_output=True,
        text=True,
    )


def test_THE_227_SCENARIO_IS_REFUSED(tmp_path, capsys):
    """pytorch.txt two minors AHEAD of the engine must go red.

    This is the check that could not be seen refusing before this PR, because
    nothing pointed it at this file.
    """
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, _engine_pins_for(CI_MIRROR, shift=-2, only=("torch", "transformers")))
    proc = _run_checker(CI_MIRROR, engine)
    assert proc.returncode == 1, (
        "a mirror two minors ahead of the engine PASSED the drift check:\n"
        f"{proc.stdout}{proc.stderr}"
    )
    assert "ENGINE PIN DRIFT" in proc.stderr
    for pkg in ("torch", "transformers"):
        assert f"{pkg}: mirror pins ==" in proc.stderr, (
            f"the refusal does not name the drifted package {pkg}:\n{proc.stderr}"
        )
    # Shown in the test log, so the refusal this guard exists for is visible in
    # the run rather than only inferable from a green assertion.
    with capsys.disabled():
        print("\n--- (internal ref) scenario, real refusal output ---")
        print(proc.stderr.rstrip())


def test_the_refusal_NAMES_WHICH_MIRROR_DRIFTED(tmp_path):
    """(internal ref)'s acceptance. The header used to be the literal string
    'tools/requirements-engine-pin.txt is stale', so a pytorch.txt drift would
    have sent the fix at the wrong file."""
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, _engine_pins_for(CI_MIRROR, shift=-2, only=("torch",)))
    proc = _run_checker(CI_MIRROR, engine)
    assert proc.returncode == 1
    header = proc.stderr.splitlines()[0]
    assert CI_MIRROR in header, f"the failure header does not name the mirror: {header!r}"
    assert TOOLS_MIRROR not in proc.stderr, (
        "the failure blames the tools/ mirror for a drift in the CI mirror:\n"
        f"{proc.stderr}"
    )


def test_an_engine_AHEAD_of_the_mirror_is_refused_too(tmp_path):
    """Both directions, per (internal ref)'s acceptance. This is the ordinary case: the
    engine bumps, and every mirror is stale until it follows."""
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, _engine_pins_for(CI_MIRROR, shift=+1))
    proc = _run_checker(CI_MIRROR, engine)
    assert proc.returncode == 1, f"an engine ahead of the mirror passed:\n{proc.stdout}"


def test_an_ALIGNED_pytorch_txt_passes(tmp_path):
    """The other half of a working guard: it must go green when the file is
    right, or it is noise people route around."""
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, _engine_pins_for(CI_MIRROR))
    proc = _run_checker(CI_MIRROR, engine)
    assert proc.returncode == 0, f"an aligned mirror was refused:\n{proc.stderr}"
    assert CI_MIRROR in proc.stdout, "the passing line does not say which mirror it checked"


def test_the_tools_mirror_check_can_STILL_fail(tmp_path):
    """Both mirrors must be able to fail. A second check that quietly disarmed
    the first would trade one blind spot for another."""
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, _engine_pins_for(TOOLS_MIRROR, shift=-2, only=("transformers",)))
    proc = _run_checker(TOOLS_MIRROR, engine)
    assert proc.returncode == 1, f"the tools/ mirror check no longer refuses:\n{proc.stdout}"
    assert TOOLS_MIRROR in proc.stderr.splitlines()[0]


# --------------------------------------------------------------------------
# LEG 2, THE STEP ITSELF — run the workflow's own shell.
#
# The tests above prove the checker refuses and that the YAML names both
# mirrors. This one runs the extracted step against a synthetic engine pin, so
# a step that named both files but could not actually fail on the second — a
# `set -e` that returns after the first, a typo in the loop — is caught.
# --------------------------------------------------------------------------
def _stage_repo(tmp_path: Path) -> Path:
    """A minimal checkout the guard step can run in: the two tools it imports,
    plus every mirror, at their real relative paths.

    Both known mirrors are staged even if the step stopped iterating one, so a
    step that dropped a mirror fails on the assertion about its report rather
    than on a FileNotFoundError from the staging helper.
    """
    root = tmp_path / "checkout"
    wanted = dict.fromkeys(
        (CHECKER_REL, VERIFIER_REL, TOOLS_MIRROR, CI_MIRROR, *_guarded_mirrors())
    )
    for rel in wanted:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dst)
    return root


def _run_guard_step(root: Path) -> subprocess.CompletedProcess:
    script = root / "guard_step.sh"
    script.write_text(_guard_step_shell())
    # `python3` in the workflow must be THIS interpreter, not whatever the
    # ambient PATH offers — the shim idiom tests/test_dump_fetch_guard.py uses.
    # PREPENDED to the real PATH rather than replacing it: the step's shell also
    # needs `bash` itself, and hard-coding a minimal PATH assumes where that
    # lives.
    shim_dir = root / "_bin"
    shim_dir.mkdir(exist_ok=True)
    shim = shim_dir / "python3"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    shim.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{shim_dir}{os.pathsep}{env.get('PATH', '')}"
    return subprocess.run(
        ["bash", str(script)],
        cwd=str(root),
        capture_output=True,
        text=True,
        env=env,
    )


def test_the_guard_step_passes_when_every_mirror_is_aligned(tmp_path):
    root = _stage_repo(tmp_path)
    pins: dict[str, str] = {}
    for mirror in _guarded_mirrors():
        pins.update(_engine_pins_for(mirror))
    _write_engine_pin(root / "_engine_pin", pins)
    proc = _run_guard_step(root)
    assert proc.returncode == 0, f"aligned mirrors were refused:\n{proc.stdout}{proc.stderr}"
    for mirror in _guarded_mirrors():
        assert mirror in proc.stdout, f"the step never mentioned {mirror}"


def test_THE_STEP_REPORTS_EVERY_DRIFTED_MIRROR_NOT_JUST_THE_FIRST(tmp_path):
    """An engine bump skews BOTH mirrors at once. Stopping at the first would
    report half the work and hide the rest behind another round-trip."""
    root = _stage_repo(tmp_path)
    pins: dict[str, str] = {}
    for mirror in _guarded_mirrors():
        pins.update(_engine_pins_for(mirror, shift=-2, only=("transformers",)))
    _write_engine_pin(root / "_engine_pin", pins)
    proc = _run_guard_step(root)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"the step went green on drifted mirrors:\n{combined}"
    for mirror in _guarded_mirrors():
        assert f"ENGINE PIN DRIFT — {mirror}" in combined, (
            f"the step did not report drift for {mirror} — it stopped at the "
            f"first failure:\n{combined}"
        )


def test_THE_STEP_GOES_RED_ON_A_PYTORCH_TXT_ONLY_DRIFT(tmp_path):
    """(internal ref) replayed as a tree, which is the whole ticket in one test.

    The staged ``.github/requirements/pytorch.txt`` is bumped two minors — the
    dependabot PR, exactly as it was opened — while the tools/ mirror and the
    engine pin are left in agreement. That tree was ALL GREEN on develop. The
    step must now be red, must name pytorch.txt, and must NOT accuse the
    tools/ mirror, which is correct here.
    """
    root = _stage_repo(tmp_path)
    _write_engine_pin(root / "_engine_pin", _engine_pins_for(TOOLS_MIRROR))

    # The dependabot bump: pytorch.txt alone moves ahead of the engine.
    ci = root / CI_MIRROR
    text = ci.read_text()
    for pkg in ("torch", "transformers"):
        was = _exact_pins(text)[pkg]
        text = text.replace(f"{pkg}=={was}", f"{pkg}=={_shift_minor(was, +2)}")
    ci.write_text(text)
    bumped = _exact_pins(text)

    proc = _run_guard_step(root)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, (
        "a pytorch.txt-only bump two minors ahead of the engine went GREEN — "
        f"(internal ref) is back:\n{combined}"
    )
    assert f"ENGINE PIN DRIFT — {CI_MIRROR}" in combined, (
        f"the step did not name {CI_MIRROR} as drifted:\n{combined}"
    )
    assert f"ENGINE PIN DRIFT — {TOOLS_MIRROR}" not in combined, (
        "the step accused the tools/ mirror, which agrees with the engine "
        f"here:\n{combined}"
    )
    for pkg in ("torch", "transformers"):
        assert f"{pkg}: mirror pins =={bumped[pkg]}" in combined, (
            f"the refusal does not name {pkg}'s bumped version:\n{combined}"
        )


# --------------------------------------------------------------------------
# (internal ref) — THE DIAGNOSIS. The detection above is correct and untouched;
# what follows is about what the red SAYS.
#
# On (internal ref) the guard printed "the engine moved; regenerate the mirror AND the
# dumps built against it" for torch and torchvision. The engine had not moved:
# (internal ref) WAS the mirror bump to 2.13.0, and the engine sat at 2.11.0 until
# (internal ref) merged at 11:44:02Z. Two pins tell you they disagree;
# they carry no history, so they cannot tell you who changed.
#
# The consequence is not cosmetic. Follow that sentence in the mirror-moved
# direction and you regenerate the mirror back DOWN to the engine's older pin,
# which greens the guard and silently reverts the security bump you are
# shipping. And that is the COMMON direction, because the ordering rule
# rule (engine base first, mirrors follow) means the mirror PR is normally
# opened before the engine bump lands.
#
# Every scenario below asserts a COUNT of reported drift lines as well as
# their content: a refusal that named one package while the mirror drifted on
# two would satisfy any "does it mention torch" check.
# --------------------------------------------------------------------------
_ATTRIBUTION = "the engine moved; regenerate the mirror"
_ORDERING_RULE = "the ordering rule"


def _drift_lines(stderr: str) -> list[str]:
    """The per-package disagreement lines, so counts can be asserted."""
    return [ln.strip() for ln in stderr.splitlines() if "mirror pins ==" in ln]


def _write_base(tmp_path: Path, pins: dict[str, str]) -> Path:
    """A merge-base snapshot of a mirror, as the workflow's `git show` writes."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    dest = tmp_path / "mirror-at-base.txt"
    dest.write_text("\n".join(f"{p}=={v}" for p, v in sorted(pins.items())) + "\n")
    return dest


def test_THE_271_MESSAGE_NO_LONGER_ASSERTS_WHICH_SIDE_MOVED(tmp_path, capsys):
    """The ticket. With no base evidence the report must state the
    disagreement, name BOTH remedies and the ordering rule, and attribute
    nothing — because nothing in its inputs can settle it."""
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(
        engine, _engine_pins_for(CI_MIRROR, shift=-2, only=("torch", "transformers"))
    )
    proc = _run_checker(CI_MIRROR, engine)
    assert proc.returncode == 1, "the detection was loosened — (internal ref) must stay red"
    assert _ATTRIBUTION not in proc.stderr, (
        "the report still asserts the engine moved, which is unknowable from "
        f"two pins and wrong half the time:\n{proc.stderr}"
    )
    lines = _drift_lines(proc.stderr)
    assert len(lines) == 2, lines
    for line in lines:
        assert line.endswith("— these must match"), line
    # Both remedies AND the rule that chooses between them.
    assert "If the ENGINE moved" in proc.stderr
    assert "If THIS PR is raising the mirror" in proc.stderr
    assert _ORDERING_RULE in proc.stderr
    assert "would revert your bump" in proc.stderr
    with capsys.disabled():
        print("\n--- (internal ref): the (internal ref) report, no base evidence ---")
        print(proc.stderr.rstrip())


def test_A_MIRROR_MOVED_PR_IS_TOLD_TO_LAND_THE_ENGINE_BUMP_FIRST(tmp_path, capsys):
    """The direction that used to get the reverting advice, derived.

    The base snapshot carries the ENGINE's (older) versions, so the mirror
    changed in this PR — (internal ref) exactly. The remedy must be "land the engine
    bump first", and must NOT be "regenerate the mirror"."""
    engine_pins = _engine_pins_for(CI_MIRROR, shift=-2, only=("torch", "transformers"))
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, engine_pins)
    proc = _run_checker(CI_MIRROR, engine, base=_write_base(tmp_path, engine_pins))
    assert proc.returncode == 1
    assert "The mirror CHANGED in this PR" in proc.stderr, proc.stderr
    assert "Do NOT regenerate the mirror" in proc.stderr
    assert _ORDERING_RULE in proc.stderr
    assert _ATTRIBUTION not in proc.stderr
    lines = _drift_lines(proc.stderr)
    assert len(lines) == 2, lines
    for line in lines:
        assert "this PR moved the mirror" in line, line
    # The engine-moved remedy must NOT also be offered: with evidence in hand
    # the point is to give ONE answer, and listing both here would be the same
    # "regenerate the mirror" instruction hiding in a menu.
    assert "If the ENGINE moved" not in proc.stderr
    with capsys.disabled():
        print("\n--- (internal ref): derived MIRROR-MOVED (the internal ref direction) ---")
        print(proc.stderr.rstrip())


def test_AN_ENGINE_MOVED_PR_IS_STILL_TOLD_TO_REGENERATE(tmp_path, capsys):
    """The other direction must keep the advice it always had, or this fix
    trades one wrong instruction for another. Base == mirror, so the mirror is
    untouched by this PR and the engine is what moved."""
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, _engine_pins_for(CI_MIRROR, shift=+1))
    base = _write_base(tmp_path, _exact_pins((REPO_ROOT / CI_MIRROR).read_text()))
    proc = _run_checker(CI_MIRROR, engine, base=base)
    assert proc.returncode == 1
    assert "the ENGINE moved" in proc.stderr, proc.stderr
    assert "Regenerate the mirror AND the dumps built against it" in proc.stderr
    assert "Do NOT regenerate the mirror" not in proc.stderr
    lines = _drift_lines(proc.stderr)
    assert lines, "no disagreement was reported at all"
    for line in lines:
        assert "mirror unchanged at the merge base" in line, line
    with capsys.disabled():
        print("\n--- (internal ref): derived ENGINE-MOVED ---")
        print(proc.stderr.rstrip())


def test_A_PIN_THIS_PR_ADDED_COUNTS_AS_THE_MIRROR_MOVING(tmp_path):
    """A package pinned now and absent at the base was added BY THIS PR, so
    the PR is the change. An empty base file is the same fact for the whole
    file — which is what the workflow writes for a mirror the PR created."""
    engine_pins = _engine_pins_for(CI_MIRROR, shift=-2, only=("torch",))
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, engine_pins)
    empty = tmp_path / "empty-base.txt"
    empty.write_text("")
    proc = _run_checker(CI_MIRROR, engine, base=empty)
    assert proc.returncode == 1
    assert "The mirror CHANGED in this PR" in proc.stderr
    lines = _drift_lines(proc.stderr)
    assert len(lines) == 1, lines
    assert "absent at the merge base" in lines[0], lines[0]


def test_MIXED_DIRECTIONS_FALL_BACK_TO_BOTH_REMEDIES(tmp_path):
    """A PR bumping one pin while the engine moved another is a real state,
    and no single remedy is right for the whole file. Guessing one there would
    be the original defect with an extra step."""
    mirror_now = _exact_pins((REPO_ROOT / CI_MIRROR).read_text())
    # torch: the engine is BEHIND and the base agrees with the engine  -> this
    #        PR raised it (mirror moved).
    # transformers: the engine is AHEAD and the base agrees with the mirror ->
    #        the engine moved.
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(
        engine,
        {
            **mirror_now,
            "torch": _shift_minor(mirror_now["torch"], -2),
            "transformers": _shift_minor(mirror_now["transformers"], +2),
        },
    )
    base = _write_base(
        tmp_path, {**mirror_now, "torch": _shift_minor(mirror_now["torch"], -2)}
    )
    proc = _run_checker(CI_MIRROR, engine, base=base)
    assert proc.returncode == 1
    lines = _drift_lines(proc.stderr)
    assert len(lines) == 2, lines
    assert "NOT KNOWABLE FROM TWO PINS" in proc.stderr, proc.stderr
    assert "If the ENGINE moved" in proc.stderr
    assert "If THIS PR is raising the mirror" in proc.stderr
    # Both per-package lines still carry their OWN derived note, which is the
    # part that is knowable even when the file-level remedy is not.
    assert any("this PR moved the mirror" in ln for ln in lines), lines
    assert any("mirror unchanged at the merge base" in ln for ln in lines), lines


def test_AN_UNREADABLE_BASE_IS_A_LOUD_FAILURE_NOT_A_SILENT_UNKNOWN(tmp_path):
    """No `except: pass`. A `--mirror-at-base` pointed at nothing would
    otherwise demote every verdict to "I cannot tell" while looking like a
    working run — a checker that quietly stopped using the evidence it was
    handed, which is this ticket's shape one level up."""
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, _engine_pins_for(CI_MIRROR, shift=-2, only=("torch",)))
    proc = _run_checker(CI_MIRROR, engine, base=tmp_path / "does-not-exist.txt")
    assert proc.returncode not in (0, 1), (
        "a missing --mirror-at-base was absorbed instead of raising:\n"
        f"rc={proc.returncode}\n{proc.stdout}{proc.stderr}"
    )
    assert "does-not-exist.txt" in proc.stderr


def test_THE_DETECTION_IS_NOT_LOOSENED_IN_ANY_BASE_CONFIGURATION(tmp_path):
    """(internal ref)'s non-goal, asserted. A mirror ahead of the engine is a stack skew
    whichever side caused it, so every direction stays exit 1 — and an ALIGNED
    mirror stays exit 0, or the guard is noise people route around."""
    mirror_now = _exact_pins((REPO_ROOT / CI_MIRROR).read_text())
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, _engine_pins_for(CI_MIRROR, shift=-2, only=("torch",)))
    bases = {
        "no base": None,
        "mirror moved": _write_base(
            tmp_path / "a", {**mirror_now, "torch": _shift_minor(mirror_now["torch"], -2)}
        ),
        "engine moved": _write_base(tmp_path / "b", mirror_now),
    }
    for label, base in bases.items():
        proc = _run_checker(CI_MIRROR, engine, base=base)
        assert proc.returncode == 1, f"{label}: a real skew went green:\n{proc.stdout}"

    aligned = tmp_path / "_aligned"
    _write_engine_pin(aligned, mirror_now)
    for label, base in bases.items():
        proc = _run_checker(CI_MIRROR, aligned, base=base)
        assert proc.returncode == 0, f"{label}: an aligned mirror was refused:\n{proc.stderr}"


def test_a_MISSING_required_pin_does_not_get_a_which_side_moved_paragraph(tmp_path):
    """The remedy paragraph answers "which side moved". An emptied mirror is
    not a move, so printing it there would be an answer to a question nobody
    asked — and would bury the real finding."""
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, _engine_pins_for(CI_MIRROR))
    empty_mirror = tmp_path / "empty-mirror.txt"
    empty_mirror.write_text("# every pin removed\n")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / CHECKER_REL),
            "--mirror",
            str(empty_mirror),
            # The EXPLICIT interface, deliberately, so it keeps being
            # exercised alongside the derived one the workflow uses — at the
            # real relative layout the artifact now has (internal ref).
            "--engine",
            str(engine / "use_cases" / "requirements.txt"),
            "--engine",
            str(engine / "use_cases" / "requirements_cuda.txt"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "REQUIRED exact pin is missing" in proc.stderr
    assert not _drift_lines(proc.stderr), proc.stderr
    assert "NOT KNOWABLE FROM TWO PINS" not in proc.stderr
    assert "If the ENGINE moved" not in proc.stderr


def test_the_header_no_longer_calls_the_mirror_STALE(tmp_path):
    """"is stale" is itself an attribution: it names the mirror as the wrong
    side, which is the same unfounded claim the per-package line made."""
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, _engine_pins_for(CI_MIRROR, shift=-2, only=("torch",)))
    proc = _run_checker(CI_MIRROR, engine)
    header = proc.stderr.splitlines()[0]
    assert CI_MIRROR in header, header
    assert "is stale" not in header, header


# --------------------------------------------------------------------------
# THE WORKFLOW HALF. The checker can derive a direction; the step has to
# actually hand it the evidence, and the checkout has to be deep enough for
# `git show <merge base>:<mirror>` to resolve. A step that named the flag but
# ran against a shallow tip would be (internal ref)'s shape again: wired but unable to
# look.
# --------------------------------------------------------------------------
def _guard_job_checkout_is_deep() -> bool:
    """`fetch-depth: 0` on the guard job's checkout, read out of the YAML."""
    block = _job_block()
    head = block[: block.index("- name: Download the engine pin")]
    return "fetch-depth: 0" in head


def test_THE_GUARD_STEP_PASSES_THE_MERGE_BASE_EVIDENCE(tmp_path):
    shell = _guard_step_shell()
    assert "--mirror-at-base" in shell, (
        "the guard step never passes --mirror-at-base, so every PR report falls "
        "back to 'both remedies' even though the evidence is one `git show` "
        "away (internal ref)"
    )
    assert "git merge-base" in shell, shell


def test_THE_GUARD_CHECKOUT_IS_DEEP_ENOUGH_TO_RESOLVE_THE_MERGE_BASE():
    assert _guard_job_checkout_is_deep(), (
        "the drift-guard job checks out at the default shallow depth, so "
        "`git merge-base origin/<base> HEAD` cannot resolve and the step's "
        "own `set -e` would red the job for a reason unrelated to any drift"
    )


def _stage_git_repo(tmp_path: Path, base_branch: str = "develop") -> Path:
    """A real git repo the extracted step can run `git merge-base` in.

    The mirrors and tools are committed on `origin/<base_branch>`, then the
    checkout advances one commit. Whatever the caller writes into the mirror
    afterwards is therefore "changed by this PR" and the step must derive it.
    """
    root = _stage_repo(tmp_path)
    git = ["git", "-C", str(root)]
    subprocess.run([*git, "init", "-q", "-b", base_branch], check=True)
    subprocess.run([*git, "config", "user.email", "t@t.invalid"], check=True)
    subprocess.run([*git, "config", "user.name", "t"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run(
        [*git, "commit", "-q", "-m", "base"], check=True, capture_output=True
    )
    # `origin/<base>` is what the step resolves against; a local ref of that
    # name is enough and needs no remote.
    subprocess.run(
        [*git, "update-ref", f"refs/remotes/origin/{base_branch}", "HEAD"], check=True
    )
    return root


def _run_guard_step_in_repo(root: Path, base_ref: str) -> subprocess.CompletedProcess:
    script = root / "guard_step.sh"
    script.write_text(_guard_step_shell())
    shim_dir = root / "_bin"
    shim_dir.mkdir(exist_ok=True)
    shim = shim_dir / "python3"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    shim.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{shim_dir}{os.pathsep}{env.get('PATH', '')}"
    env["BASE_REF"] = base_ref
    env["RUNNER_TEMP"] = str(root / "_temp")
    (root / "_temp").mkdir(exist_ok=True)
    return subprocess.run(
        ["bash", str(script)], cwd=str(root), capture_output=True, text=True, env=env
    )


def test_THE_STEP_DERIVES_MIRROR_MOVED_FROM_A_REAL_MERGE_BASE(tmp_path, capsys):
    """(internal ref) replayed as a git history, which is the whole ticket end to end.

    The engine and both mirrors agree at the merge base; then this "PR" raises
    pytorch.txt two minors, exactly as the mirror bump did. The step must go
    red (the detection is untouched) AND must tell the author to land the
    engine bump first rather than to regenerate the mirror.
    """
    root = _stage_git_repo(tmp_path)
    _write_engine_pin(root / "_engine_pin", _engine_pins_for(TOOLS_MIRROR))

    ci = root / CI_MIRROR
    text = ci.read_text()
    for pkg in ("torch", "transformers"):
        was = _exact_pins(text)[pkg]
        text = text.replace(f"{pkg}=={was}", f"{pkg}=={_shift_minor(was, +2)}")
    ci.write_text(text)
    subprocess.run(["git", "-C", str(root), "add", CI_MIRROR], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-q", "-m", "bump the mirror"],
        check=True,
        capture_output=True,
    )

    proc = _run_guard_step_in_repo(root, "develop")
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"a real skew went green:\n{combined}"
    assert f"ENGINE PIN DRIFT — {CI_MIRROR}" in combined, combined
    assert "The mirror CHANGED in this PR" in combined, (
        "the step did not derive the direction from the merge base, so it fell "
        f"back to both remedies:\n{combined}"
    )
    assert "Do NOT regenerate the mirror" in combined
    assert _ATTRIBUTION not in combined
    # The tools/ mirror is untouched by this PR and agrees with the engine, so
    # it must not be accused — (internal ref)'s property, still holding.
    assert f"ENGINE PIN DRIFT — {TOOLS_MIRROR}" not in combined
    with capsys.disabled():
        print("\n--- (internal ref): the real step, real git history, (internal ref) ---")
        print(combined.rstrip())


def test_THE_STEP_DERIVES_ENGINE_MOVED_WHEN_THE_PR_TOUCHED_NO_MIRROR(tmp_path):
    """The ordinary case: nothing in the PR moved, the engine did. Same
    history, but the skew is put in the ENGINE pin instead of the mirror."""
    root = _stage_git_repo(tmp_path)
    pins: dict[str, str] = {}
    for mirror in _guarded_mirrors():
        pins.update(_engine_pins_for(mirror, shift=+1, only=("transformers",)))
    _write_engine_pin(root / "_engine_pin", pins)
    proc = _run_guard_step_in_repo(root, "develop")
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, combined
    assert "the ENGINE moved" in combined, combined
    assert "Regenerate the mirror AND the dumps built against it" in combined
    assert "Do NOT regenerate the mirror" not in combined


def test_THE_STEP_STILL_WORKS_WITH_NO_BASE_REF(tmp_path):
    """push/schedule runs have no merge base. The step must skip the
    derivation and still check both mirrors, rather than dying on an unset
    variable under `set -u`."""
    root = _stage_git_repo(tmp_path)
    pins: dict[str, str] = {}
    for mirror in _guarded_mirrors():
        pins.update(_engine_pins_for(mirror))
    _write_engine_pin(root / "_engine_pin", pins)
    proc = _run_guard_step_in_repo(root, "")
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"aligned mirrors were refused:\n{combined}"
    for mirror in _guarded_mirrors():
        assert mirror in proc.stdout, f"the step never mentioned {mirror}:\n{combined}"


# --------------------------------------------------------------------------
# LEG 5 — (internal ref). THE ENGINE'S OWN SIDE: a merge that refused a
# disagreement by picking one, and a coverage claim that read two of seven.
#
# Two defects in one reducer:
#
#   engine: dict[str, str] = {}
#   for eng_path in args.engine:
#       engine.update(_exact_pins(Path(eng_path)))
#
#   1. `dict.update` is last-write-wins and has no notion of a conflict, so
#      two engine files pinning one package differently left whichever came
#      LAST as "the engine's pin". The verdict was a function of ARGUMENT
#      ORDER. A disagreement between engine files was invisible — a false
#      clean, which is strictly worse than a false alarm.
#   2. The list handed in WAS the coverage claim, and it named two of the
#      seven engine files that pin something a mirror mirrors. The missing
#      `use_cases/requirements_vision_cv_cuda.txt` is the torch version of the
#      `cv:gpu` base image. An unread source cannot disagree.
#
# EVERY COUNT BELOW IS ASSERTED, not just the absence of findings: a reader
# that visits 2 of 10 candidates and reports agreement is indistinguishable
# from one that visits all 10 and reports agreement, and the count is the only
# thing that separates them.
# --------------------------------------------------------------------------
def _checker_module():
    """`check_engine_pin_drift` as a module, for the pure derivation helpers.

    Loaded the same way `_provenance_keys` loads the verifier, and for the same
    reason: the derivation is asserted against the REAL constant and the REAL
    scanner rather than a restatement of either, so this file cannot certify a
    search space the tool no longer uses.
    """
    path = REPO_ROOT / CHECKER_REL
    spec = importlib.util.spec_from_file_location("_checker_for_pin_guard", path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _engine_like_candidates() -> dict[str, str]:
    """The ten real candidate names with the engine's real two idioms.

    Built through `_write_engine_pin`, so this fixture and every behavioural
    test above describe the same engine.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_engine_pin(root, _engine_pins_for(TOOLS_MIRROR))
        return {rel: (root / rel).read_text() for rel in _ENGINE_CANDIDATES}


def _skew_one_source(engine: Path, rel: str, pkg: str, version: str) -> None:
    """Make ONE engine source disagree with its siblings about `pkg`.

    This is the state the old reducer could not see. It is not hypothetical:
    a torch bump that lands in `requirements_cuda.txt` but not in
    `requirements_vision_cv_cuda.txt` produces exactly it, and the `cv:gpu`
    edge then runs a torch the dumps were never prepped under.
    """
    path = engine / rel
    lines = path.read_text().splitlines()
    out, seen = [], False
    for line in lines:
        if re.match(rf"^{re.escape(pkg)}==", line.strip()):
            out.append(f"{pkg}=={version}")
            seen = True
        else:
            out.append(line)
    assert seen, f"{rel} does not pin {pkg}, so skewing it would prove nothing"
    path.write_text("\n".join(out) + "\n")


def _run_checker_explicit(mirror: str, engine_paths: list[Path]):
    """The checker over an EXPLICIT `--engine` list, in the given order.

    The explicit interface deliberately, and in a caller that controls the
    ORDER: it is the interface that existed before (internal ref), so these
    assertions run against the unfixed checker and are seen to fail there.
    """
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / CHECKER_REL),
            "--mirror",
            str(REPO_ROOT / mirror),
            *[a for p in engine_paths for a in ("--engine", str(p))],
        ],
        capture_output=True,
        text=True,
    )


def _disagreeing_engine(tmp_path: Path) -> tuple[Path, list[Path]]:
    """An engine whose own sources disagree about torch, and nothing else.

    `requirements_cuda.txt` keeps the mirror's torch; the CV GPU file — the one
    the workflow never fetched — is two minors ahead. Every other pin agrees,
    so the ONLY thing that can decide the verdict is how the disagreement is
    handled.
    """
    engine = tmp_path / "_engine_pin"
    pins = _engine_pins_for(TOOLS_MIRROR)
    _write_engine_pin(engine, pins)
    skewed = _shift_minor(pins["torch"], +2)
    _skew_one_source(engine, CV_GPU_REQUIREMENTS, "torch", skewed)
    return engine, [
        engine / "use_cases" / "requirements.txt",
        engine / "use_cases" / "requirements_cuda.txt",
        engine / CV_GPU_REQUIREMENTS,
    ]


def test_THE_VERDICT_DOES_NOT_DEPEND_ON_ARGUMENT_ORDER(tmp_path, capsys):
    """THE TICKET, as one assertion. Same repo state, two argument orders.

    Under the old reducer this pair returned 1 and 0 — a red and a green from
    the same files, decided by which one `dict.update` happened to apply last.
    Now both orders refuse identically, and "identically" is asserted on the
    bytes: a refusal whose text depended on argument order would be the same
    defect moved into the report.
    """
    engine, paths = _disagreeing_engine(tmp_path)
    forward = _run_checker_explicit(TOOLS_MIRROR, paths)
    reversed_ = _run_checker_explicit(TOOLS_MIRROR, list(reversed(paths)))

    with capsys.disabled():
        print("\n--- (internal ref): the disagreement, refused in both orders ---")
        print(forward.stderr)

    assert forward.returncode == reversed_.returncode, (
        "the verdict still depends on the ORDER of --engine arguments: "
        f"{[p.name for p in paths]} exited {forward.returncode} and the same "
        f"files reversed exited {reversed_.returncode} (internal ref)"
    )
    assert forward.returncode == 2, (
        "two engine sources disagreeing about torch must be REFUSED with exit "
        "2 — 'this gate cannot establish what the engine pins' — not answered "
        f"with 0 or 1. Got {forward.returncode}:\n{forward.stderr}"
    )
    assert forward.stderr == reversed_.stderr, (
        "the refusal text differs between argument orders, so the report still "
        "carries the ordering the verdict no longer does"
    )


def test_TWO_DISAGREEING_ENGINE_SOURCES_NAME_BOTH_FILES_AND_BOTH_VERSIONS(tmp_path):
    """A refusal that does not say WHICH sources disagree is unactionable.

    The old reducer named neither: it did not report a conflict at all, so
    there was nothing to act on. `backend`'s `_agreeing_torch_pin` is the
    precedent for naming both.
    """
    engine, paths = _disagreeing_engine(tmp_path)
    proc = _run_checker_explicit(TOOLS_MIRROR, paths)
    assert proc.returncode == 2, proc.stderr
    mirror_torch = _exact_pins((REPO_ROOT / TOOLS_MIRROR).read_text())["torch"]
    skewed = _shift_minor(mirror_torch, +2)
    for expected in (
        "requirements_cuda.txt",
        CV_GPU_REQUIREMENTS.rsplit("/", 1)[-1],
        f"=={mirror_torch}",
        f"=={skewed}",
        "torch",
    ):
        assert expected in proc.stderr, (expected, proc.stderr)


def test_THE_REFUSAL_DOES_NOT_ASSERT_WHICH_SOURCE_MOVED(tmp_path):
    """(internal ref) must not come back through the new message.

    Two pins cannot tell you which side moved, and neither can seven. The
    refusal states that they disagree and stops.
    """
    engine, paths = _disagreeing_engine(tmp_path)
    proc = _run_checker_explicit(TOOLS_MIRROR, paths)
    assert proc.returncode == 2, proc.stderr
    assert _ATTRIBUTION not in proc.stderr, proc.stderr
    assert "is stale" not in proc.stderr, proc.stderr
    for claim in ("the engine moved", "the mirror moved", "regenerate the mirror"):
        assert claim not in proc.stderr, (claim, proc.stderr)


def test_A_DISAGREEMENT_ON_A_PIN_NO_MIRROR_MIRRORS_DOES_NOT_RED_THIS_GATE(tmp_path):
    """The scoping boundary, asserted rather than commented.

    The refusal covers the packages that can change THIS gate's answer — the
    mirror's own pins plus the required ones. Different engine base families
    legitimately differ on packages no mirror mirrors, and refusing on those
    would turn a model-zoo PR red over an engine-internal matter its author
    cannot act on.
    """
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, _engine_pins_for(TOOLS_MIRROR))
    mirror_pins = _exact_pins((REPO_ROOT / TOOLS_MIRROR).read_text())
    unmirrored = "azure-servicebus"
    assert unmirrored not in mirror_pins, (
        f"{unmirrored} is now mirrored, so it can no longer stand in for a pin "
        "outside this gate's scope; pick another"
    )
    for rel, ver in (
        ("use_cases/requirements.txt", "7.12.0"),
        ("use_cases/requirements_cuda.txt", "7.14.0"),
    ):
        with (engine / rel).open("a") as fh:
            fh.write(f"{unmirrored}=={ver}\n")
    proc = _run_checker(TOOLS_MIRROR, engine)
    assert proc.returncode == 0, (
        "a disagreement on a package no mirror pins cannot move this gate's "
        f"verdict and must not red it:\n{proc.stderr}"
    )


def test_A_NAMED_ENGINE_SOURCE_THAT_PINS_NOTHING_IS_A_LOUD_EXIT_2(tmp_path):
    """The GPU-Dockerfile trap, which the old reducer swallowed in silence.

    `use_cases/requirements_vision_cv_cuda.txt` was missing from the scan, so
    the obvious repair is "add the files that pin torch" — and three of the
    engine's eight torch-pinning files pin it via a cu129 wheel index, not a
    literal `pkg==ver`. Handing one of those to `--engine` contributes zero
    pins: under the old reducer a silent no-op that made the invocation LOOK
    wider while reading nothing more. It is now refused by name.
    """
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, _engine_pins_for(TOOLS_MIRROR))
    gpu = engine / _ENGINE_GPU_DOCKERFILES[0]
    assert not _exact_pins(gpu.read_text()), (
        "the GPU Dockerfile fixture now carries a literal pin, so it no longer "
        "reproduces the wheel-index idiom this test is about"
    )
    proc = _run_checker_explicit(
        TOOLS_MIRROR,
        [
            engine / "use_cases" / "requirements.txt",
            engine / "use_cases" / "requirements_cuda.txt",
            gpu,
        ],
    )
    assert proc.returncode == 2, (
        "an engine source contributing NO exact pin must be refused, not "
        f"counted as one that happened to agree:\n{proc.stdout}{proc.stderr}"
    )
    assert _ENGINE_GPU_DOCKERFILES[0] in proc.stderr, proc.stderr


def test_AN_UNREADABLE_ENGINE_SOURCE_IS_A_LOUD_EXIT_2_NOT_A_TRACEBACK(tmp_path):
    """A missing artifact must be exit 2, distinguishable from a drift."""
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, _engine_pins_for(TOOLS_MIRROR))
    proc = _run_checker_explicit(
        TOOLS_MIRROR,
        [engine / "use_cases" / "requirements.txt", engine / "does-not-exist.txt"],
    )
    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr, proc.stderr
    assert "does-not-exist.txt" in proc.stderr, proc.stderr


def test_NO_ENGINE_SOURCE_AT_ALL_IS_REFUSED(tmp_path):
    """An empty engine agrees with every mirror, so it cannot be a pass."""
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / CHECKER_REL),
            "--mirror",
            str(REPO_ROOT / TOOLS_MIRROR),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, f"{proc.returncode}\n{proc.stdout}{proc.stderr}"


def test_AN_ENGINE_ROOT_WHOSE_GLOB_MATCHES_NOTHING_IS_REFUSED(tmp_path):
    """"I read no files and found no disagreement" is the vacuous green."""
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, _engine_pins_for(TOOLS_MIRROR))
    for rel in _ENGINE_CPU_DOCKERFILES + _ENGINE_GPU_DOCKERFILES:
        (engine / rel).unlink()
    proc = _run_checker(TOOLS_MIRROR, engine)
    assert proc.returncode == 2, (
        "an engine-source glob that matched nothing means the artifact did not "
        f"arrive intact; that is not agreement:\n{proc.stdout}{proc.stderr}"
    )
    assert "Dockerfile.base.*" in proc.stderr, proc.stderr


# --------------------------------------------------------------------------
# THE DERIVED COVERAGE CLAIM, and its counts.
# --------------------------------------------------------------------------
def test_THE_CV_GPU_REQUIREMENTS_FILE_IS_IN_THE_DERIVED_SET():
    """The coverage half of (internal ref), as one assertion."""
    mod = _checker_module()
    derived = mod.derive_engine_sources(_engine_like_candidates())
    assert CV_GPU_REQUIREMENTS in derived, (
        f"{CV_GPU_REQUIREMENTS} is still outside the engine-source scan, so a "
        "torch pin living only there — the cv:gpu base image's torch — cannot "
        "disagree with a mirror and reports as agreement (internal ref); "
        f"derived: {derived}"
    )


def test_the_derived_set_is_EXACTLY_the_engine_files_that_pin():
    """COUNTS, because absence alone proves nothing here.

    A checker that reads 2 of 10 candidates and finds no disagreement is
    indistinguishable from one that reads all 10 and finds none. Ten
    candidates, seven sources, three GPU Dockerfiles that pin via a wheel
    index — asserted as numbers AND as names.
    """
    mod = _checker_module()
    candidates = _engine_like_candidates()
    assert len(candidates) == 10, sorted(candidates)
    derived = mod.derive_engine_sources(candidates)
    assert len(derived) == 7, derived
    assert derived == sorted(_ENGINE_REQUIREMENTS + _ENGINE_CPU_DOCKERFILES), derived
    not_derived = sorted(set(candidates) - set(derived))
    assert len(not_derived) == 3, not_derived
    assert not_derived == sorted(_ENGINE_GPU_DOCKERFILES), not_derived


def test_THE_DERIVATION_USES_THE_SAME_SCANNER_AS_THE_READER():
    """A derivation on a different rule than the reader could certify a file
    the reader then finds nothing in — a coverage claim that is true about a
    scanner nobody uses."""
    mod = _checker_module()
    candidates = _engine_like_candidates()
    for rel in mod.derive_engine_sources(candidates):
        assert mod._exact_pins_in(candidates[rel]), (
            f"{rel} was derived as an engine source but the reader finds no "
            "pin in it"
        )


def test_AN_EMPTY_SEARCH_SPACE_DERIVES_A_REFUSAL_NOT_AN_EMPTY_SET():
    mod = _checker_module()
    try:
        mod.derive_engine_sources({})
    except mod.EnginePinError:
        return
    raise AssertionError(
        "nothing looked at derived nothing missing, and that then 'agrees' "
        "vacuously — the exact shape of (internal ref)"
    )


def test_a_TORCH_ADJACENT_NAME_IS_NOT_A_TORCH_PIN():
    """Word boundaries, not substrings. `pytorch-forecasting` contains
    `torch`, and a substring reader would derive a torch source out of a file
    that pins no torch at all — after which the merge would find nothing in it
    and this gate would be one file wider on paper only."""
    mod = _checker_module()
    decoys = "\n".join(
        (
            "pytorch-forecasting==1.4.0",
            "torchmetrics==1.9.0",
            "torchao==0.14.0",
            "my-torch==9.9.9",
        )
    )
    pins = mod._exact_pins_in(decoys)
    assert "torch" not in pins, pins
    assert set(pins) == {"pytorch-forecasting", "torchmetrics", "torchao", "my-torch"}


def test_THE_OK_LINE_STATES_ITS_OWN_COVERAGE(tmp_path):
    """A green that does not say how much it read cannot be told from a green
    that read nothing. So the counts are in the passing output, not only the
    failing one."""
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, _engine_pins_for(TOOLS_MIRROR))
    proc = _run_checker(TOOLS_MIRROR, engine)
    assert proc.returncode == 0, proc.stderr
    assert "10 candidate file(s) read" in proc.stdout, proc.stdout
    assert "7 of 10" in proc.stdout, proc.stdout
    assert CV_GPU_REQUIREMENTS in proc.stdout, proc.stdout
    assert "read from 7 engine source(s)" in proc.stdout, proc.stdout


# --------------------------------------------------------------------------
# THE WORKFLOW HALF. The globs are the only hand-kept claim left, and they are
# written down in THREE places — the sparse-checkout, the artifact path, and
# the module constant. Three copies of one claim is how (internal ref) happened.
# --------------------------------------------------------------------------
def _engine_fetch_with(key: str) -> list[str]:
    """A block-scalar `with:` value from the fetch-engine-pin job, as a list."""
    block = _job_block("fetch-engine-pin")
    marker = f"{key}: |"
    # EXACTLY ONCE. Two occurrences means `.index()` silently picks the first
    # and this helper would report a different key's patterns as this one's.
    assert block.count(marker) == 1, (
        f"{marker!r} appears {block.count(marker)} times in the "
        "fetch-engine-pin job; .index() would pick one arbitrarily"
    )
    body = block[block.index(marker) + len(marker) :]
    out = []
    for raw in body.splitlines()[1:]:
        if not raw.strip() or not raw.startswith(" " * 12):
            break
        out.append(raw.strip())
    assert out, (key, body[:400])
    return out


def test_THE_GUARD_STEP_DERIVES_THE_ENGINE_SET_RATHER_THAN_LISTING_IT():
    # THE COMMENT-STRIPPED VIEW, deliberately. Read raw, the positive
    # assertion below greens on a step whose command was deleted and merely
    # described in a comment, and the negative one reds when a comment
    # explains the flag it forbids. This PR's own workflow comments discuss
    # `--engine` at length, which is exactly the hazard.
    shell = _guard_step_code()
    assert "--engine-root _engine_pin" in shell, (
        "the guard step still hands the checker a LIST of engine files. That "
        "list was the coverage claim and it named two of seven, and the "
        "reducer resolved a disagreement between them by argument order "
        "(internal ref). Pass --engine-root and let the set be derived.\n"
        f"{shell}"
    )
    assert "--engine _engine_pin" not in shell, (
        "a hand-listed --engine path is back alongside --engine-root, which "
        f"reintroduces the list that rots:\n{shell}"
    )


def test_THE_SPARSE_CHECKOUT_THE_ARTIFACT_AND_THE_MODULE_ALL_DECLARE_ONE_SPACE():
    """The three copies of the search space must be the same search space."""
    mod = _checker_module()
    globs = list(mod._ENGINE_SOURCE_GLOBS)
    assert len(globs) == 2, globs

    sparse = _engine_fetch_with("sparse-checkout")
    assert len(sparse) == len(globs), (sparse, globs)
    assert [p.lstrip("/") for p in sparse] == globs, (sparse, globs)
    for pattern in sparse:
        assert pattern.startswith("/"), (
            f"{pattern!r} is not anchored at the engine's root. Non-cone sparse "
            "patterns are gitignore-style, so an unanchored pattern also "
            "matches at depth — it would fetch a vendored "
            "sub/Dockerfile.base.cpu that Path.glob on the consuming side then "
            "does not match"
        )

    artifact = _engine_fetch_with("path")
    assert len(artifact) == len(globs), (artifact, globs)
    assert [p.removeprefix("_engine/") for p in artifact] == globs, (artifact, globs)


def test_THE_ARTIFACT_KEEPS_A_ROOT_LEVEL_PATTERN_SO_THE_LAYOUT_DOES_NOT_FLATTEN():
    """`upload-artifact` roots the archive at the least common ancestor of the
    paths it is given. Drop the root-level Dockerfile pattern and the LCA
    collapses to `_engine/use_cases`, the `use_cases/` segment disappears from
    every downloaded path, and `--engine-root` globs nothing — a coverage
    failure arriving as a layout change."""
    artifact = _engine_fetch_with("path")
    root_level = [p for p in artifact if "/" not in p.removeprefix("_engine/")]
    assert root_level, (
        "no root-level `_engine/<file>` pattern is uploaded any more, so the "
        f"artifact re-roots at _engine/use_cases and flattens: {artifact}"
    )
    assert any(
        p.removeprefix("_engine/").startswith("use_cases/") for p in artifact
    ), artifact


def test_EVERY_DERIVABLE_ENGINE_SOURCE_IS_INSIDE_THE_DECLARED_SEARCH_SPACE():
    """The ten real candidate names must each be matched by a declared glob.

    This is the assertion that would have failed BEFORE the ticket: the
    workflow declared `use_cases/requirements.txt` and
    `use_cases/requirements_cuda.txt`, which match two of the ten.
    """
    globs = [p.lstrip("/") for p in _engine_fetch_with("sparse-checkout")]
    unmatched = [
        rel
        for rel in _ENGINE_CANDIDATES
        if not any(_glob_matches(g, rel) for g in globs)
    ]
    assert not unmatched, (
        f"{unmatched} are engine files in the pin space that no declared glob "
        f"fetches, so nothing here can ever compare against them; globs: {globs}"
    )
    assert len(_ENGINE_CANDIDATES) == 10, _ENGINE_CANDIDATES
    assert any(_glob_matches(g, CV_GPU_REQUIREMENTS) for g in globs), globs


def test_A_SINGLE_STAR_IN_THE_TOOLS_OWN_GLOB_DOES_NOT_CROSS_A_SLASH(tmp_path):
    """The tool leans on `Path.glob` rather than a hand-written matcher, so
    prove the semantics instead of assuming them.

    `(internal ref)` hand-rolled `_matches_glob` for exactly this: `fnmatch`
    would call a vendored `sub/Dockerfile.base.cpu` a match for
    `Dockerfile.base.*`, and a derivation that reached into a subtree would
    claim engine sources the sparse-checkout never fetched. `Path.glob` keeps
    `*` inside one path component — asserted here, and asserted by COUNT so a
    glob that silently stopped matching anything cannot pass as one that
    correctly matched nothing extra.
    """
    mod = _checker_module()
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, _engine_pins_for(TOOLS_MIRROR))
    decoy = engine / "vendor" / "Dockerfile.base.cpu"
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_text("torch==9.9.9\n")

    candidates = mod._candidates_under(engine)
    assert len(candidates) == 10, sorted(candidates)
    assert "vendor/Dockerfile.base.cpu" not in candidates, sorted(candidates)
    # And the decoy's absurd pin cannot reach the verdict either.
    proc = _run_checker(TOOLS_MIRROR, engine)
    assert proc.returncode == 0, proc.stderr
    assert "9.9.9" not in proc.stdout + proc.stderr


def test_THE_TOOLS_GLOBS_AND_THE_TESTS_GLOB_MATCHER_AGREE():
    """`_glob_matches` above is what the workflow-side assertions use; the
    tool uses `Path.glob`. If those two disagreed, the YAML-reading tests
    would certify a search space the tool does not actually walk."""
    mod = _checker_module()
    for rel in _ENGINE_CANDIDATES:
        assert any(_glob_matches(g, rel) for g in mod._ENGINE_SOURCE_GLOBS), rel
    for rel in ("vendor/Dockerfile.base.cpu", "use_cases/sub/requirements.txt"):
        assert not any(
            _glob_matches(g, rel) for g in mod._ENGINE_SOURCE_GLOBS
        ), rel


# --------------------------------------------------------------------------
# PINS vs DERIVES. "It carries no literal pin, so it is not a source" is true
# of the three GPU bases AND of a base family this tool has never heard of.
# One silent bucket for both is a coverage claim that cannot fail — the defect
# being fixed here, one level up. So the two categories are named, counted,
# and an UNCLASSIFIED candidate is an error.
# --------------------------------------------------------------------------
def test_EVERY_CANDIDATE_IS_CLASSIFIED_AND_BOTH_COUNTS_ARE_ASSERTED():
    mod = _checker_module()
    candidates = _engine_like_candidates()
    pinning, deriving, unclassified = mod.classify_engine_candidates(candidates)
    assert len(candidates) == 10, sorted(candidates)
    assert len(pinning) == 7, pinning
    assert len(deriving) == 3, deriving
    assert not unclassified, unclassified
    # 7 + 3 == 10 leaves no residue. A candidate silently in neither bucket is
    # the whole failure mode.
    assert len(pinning) + len(deriving) + len(unclassified) == len(candidates)
    assert sorted(deriving) == sorted(_ENGINE_GPU_DOCKERFILES), deriving
    assert deriving == _ENGINE_GPU_DERIVES_FROM, deriving


def test_A_DERIVING_BASE_MAKES_ITS_SOURCE_FILE_STRUCTURALLY_REQUIRED():
    """The strongest form of the coverage claim.

    `Dockerfile.base.cv.gpu` greps its torch pin out of
    `use_cases/requirements_vision_cv_cuda.txt`. So that file is not merely
    "one we should also read" — the cv:gpu base image's torch version comes
    from it, and a scan that omits it cannot know what that image installs.
    Under the old two-file list it was not even fetched.
    """
    mod = _checker_module()
    candidates = _engine_like_candidates()
    pinning, deriving = mod.assert_engine_pin_coverage(candidates)
    assert deriving["Dockerfile.base.cv.gpu"] == CV_GPU_REQUIREMENTS
    assert CV_GPU_REQUIREMENTS in pinning, pinning
    for base, target in deriving.items():
        assert target in pinning, (base, target, pinning)


def test_AN_UNRECOGNISED_BASE_FAMILY_IS_REFUSED_NOT_QUIETLY_EXCLUDED():
    """A new `Dockerfile.base.*` that neither pins nor greps must be loud."""
    mod = _checker_module()
    candidates = dict(_engine_like_candidates())
    candidates["Dockerfile.base.tpu"] = "FROM python:3.11\nRUN echo no pin here\n"
    _, _, unclassified = mod.classify_engine_candidates(candidates)
    assert unclassified == ["Dockerfile.base.tpu"], unclassified
    try:
        mod.assert_engine_pin_coverage(candidates)
    except mod.EnginePinError as exc:
        assert "Dockerfile.base.tpu" in str(exc), str(exc)
        assert "neither pin" in str(exc), str(exc)
        return
    raise AssertionError(
        "an engine base this tool cannot classify was excluded from the scan "
        "in silence, which is (internal ref)'s disease one level up"
    )


def test_A_GPU_BASE_THAT_STARTS_PINNING_JOINS_THE_SCAN_AUTOMATICALLY():
    """The category change the coordinator asked about, in the safe direction.

    No special case is needed: a bare `torch==` makes the file classify as
    pinning, so it enters the scanned set on its own rather than staying
    excluded because it used to derive.
    """
    mod = _checker_module()
    candidates = dict(_engine_like_candidates())
    # The engine's real pinning idiom: one pin per line, inside a continued
    # `pip install`. That is how all five of its pinning files are written and
    # it is what the line-anchored scanner reads -- see the companion test
    # below, which asserts that assumption instead of leaving it implicit.
    candidates["Dockerfile.base.cv.gpu"] += (
        "RUN pip install --no-cache-dir \\\n    torch==7.7.7\n"
    )
    pinning, deriving, unclassified = mod.classify_engine_candidates(candidates)
    assert "Dockerfile.base.cv.gpu" in pinning, pinning
    assert "Dockerfile.base.cv.gpu" not in deriving, deriving
    assert not unclassified, unclassified
    assert len(pinning) == 8 and len(deriving) == 2


def test_A_DERIVING_BASE_POINTED_OUTSIDE_THE_SCAN_IS_REFUSED():
    mod = _checker_module()
    candidates = dict(_engine_like_candidates())
    candidates["Dockerfile.base.cv.gpu"] = candidates[
        "Dockerfile.base.cv.gpu"
    ].replace(CV_GPU_REQUIREMENTS, "somewhere/else.txt")
    try:
        mod.assert_engine_pin_coverage(candidates)
    except mod.EnginePinError as exc:
        assert "somewhere/else.txt" in str(exc), str(exc)
        return
    raise AssertionError(
        "a GPU base installing torch from a file outside the scanned set was "
        "accepted, so the version that image runs comes from where this gate "
        "cannot see — (internal ref) exactly"
    )


def test_NEITHER_CATEGORY_MAY_QUIETLY_EMPTY_OUT():
    """Both counts are asserted, so a set that stopped being populated is a
    failure rather than a quiet zero."""
    mod = _checker_module()
    base = _engine_like_candidates()

    only_pinning = {
        k: v for k, v in base.items() if k not in _ENGINE_GPU_DOCKERFILES
    }
    try:
        mod.assert_engine_pin_coverage(only_pinning)
        raise AssertionError("an empty DERIVING set passed unnoticed")
    except mod.EnginePinError as exc:
        assert "derives its pin from another" in str(exc), str(exc)

    only_deriving = {k: base[k] for k in _ENGINE_GPU_DOCKERFILES}
    try:
        mod.assert_engine_pin_coverage(only_deriving)
        raise AssertionError("an empty PINNING set passed unnoticed")
    except mod.EnginePinError as exc:
        assert "NO engine file carries an exact pin" in str(exc), str(exc)


def test_THE_OK_LINE_NAMES_THE_READ_AND_EXCLUDED_FILES(tmp_path):
    """A read-and-excluded set that is never printed is an exclusion nobody
    can review."""
    engine = tmp_path / "_engine_pin"
    _write_engine_pin(engine, _engine_pins_for(TOOLS_MIRROR))
    proc = _run_checker(TOOLS_MIRROR, engine)
    assert proc.returncode == 0, proc.stderr
    assert "pins a version: 7 of 10" in proc.stdout, proc.stdout
    assert "derives one   : 3 of 10" in proc.stdout, proc.stdout
    for base, target in _ENGINE_GPU_DERIVES_FROM.items():
        assert f"{base} <- {target}" in proc.stdout, proc.stdout


def test_THE_SCANNER_IS_LINE_ANCHORED_AND_THAT_IS_WHY_A_GPU_BASE_READS_CLEAN():
    """Name the assumption the classification rests on, and assert it.

    `_EXACT` is anchored at the start of a stripped line, so `pkg==ver` must
    stand at the head of its own line to count. That is not incidental — it is
    what lets a GPU Dockerfile be read without its own
    `grep -E '^torch(vision)?=='` pattern being mistaken for a torch pin. A
    scanner widened to match mid-line would classify all three GPU bases as
    PINNING files carrying a garbage version, which is worse than the gap this
    ticket closes.

    The cost of that choice, stated rather than hidden: a pin written inline
    (`RUN pip install torch==X`) in a file that also greps and uses the wheel
    index is not detectable here. The engine writes one pin per line in all
    five of its pinning files, so this matches its actual convention -- and a
    file that stops matching any category lands in `unclassified`, which is an
    error, so a NEW family cannot slip through the same way.
    """
    mod = _checker_module()
    gpu_text = _GPU_DOCKERFILE % "use_cases/requirements_cuda.txt"
    # The GPU base's own grep pattern must not read as a pin.
    assert "torch(vision)?==" in gpu_text
    assert mod._exact_pins_in(gpu_text) == {}, mod._exact_pins_in(gpu_text)
    # A pin at the head of its own line does read as one.
    assert mod._exact_pins_in("    torch==2.13.0 \\\n") == {"torch": "2.13.0"}
    # And mid-line does not, which is the documented cost.
    assert mod._exact_pins_in("RUN pip install torch==2.13.0\n") == {}


def test_THE_STEP_ASSERTIONS_CANNOT_BE_SATISFIED_BY_A_COMMENT():
    """A source-text assertion is satisfiable by a comment. Prove it is not.

    THREE RESULTS, not two — without the third, a strip that rejected
    everything would pass the first two and look like a fix:

      1. raw text on the mutant  -> True   the FALSE PASS being fixed
      2. comments stripped       -> False  the fix catches it
      3. unmutated + stripped    -> True   negative control

    The mutant deletes the flag from the command and quotes it only in a
    comment, which is the realistic regression: someone rewrites the step and
    leaves the prose describing what it used to do. This PR's own workflow
    comments discuss `--engine` and `--engine-root` at length, so the hazard is
    not hypothetical here.
    """
    needle = "--engine-root _engine_pin"
    raw = _guard_step_shell()
    code = _guard_step_code()

    def strip(text: str) -> str:
        return "\n".join(
            ln for ln in text.splitlines() if not ln.strip().startswith("#")
        )

    mutant = raw.replace(f"{needle} \\", "\\")
    assert mutant != raw, "the mutation did not apply, so it proves nothing"
    assert needle not in strip(mutant), (
        "the mutation left the flag in real code, so result 1 below would be "
        "measuring the unmutated step"
    )
    mutant = mutant.replace(
        "# `--engine-root`, NOT A LIST",
        f"# the step used to pass {needle}, NOT A LIST",
    )

    assert needle in mutant, (
        "result 1: raw text no longer greens on a step whose command is gone "
        "and only described in a comment — if this is False the hazard is "
        "gone and this test can go with it"
    )
    assert needle not in strip(mutant), "result 2: the strip failed to catch it"
    assert needle in code, (
        "result 3 (NEGATIVE CONTROL): the comment strip rejected the REAL "
        "step too, so it is not a fix — it is a blanket reject that would "
        "pass results 1 and 2 while asserting nothing"
    )


def test_THE_NEGATIVE_STEP_ASSERTION_IS_NOT_REDDENED_BY_PROSE():
    """The other direction, which is the one people forget.

    `"--engine _engine_pin" not in shell` reds a HEALTHY tree the moment a
    comment explains the hand-listed form it forbids. Written against the raw
    text that is a latent false failure; against the stripped view it is not.
    """
    forbidden = "--engine _engine_pin"
    raw_with_prose = _guard_step_shell().replace(
        "# `--engine-root`, NOT A LIST",
        f"# it no longer passes {forbidden}/requirements.txt, NOT A LIST",
    )

    def strip(text: str) -> str:
        return "\n".join(
            ln for ln in text.splitlines() if not ln.strip().startswith("#")
        )

    assert forbidden in raw_with_prose, "the prose mutation did not apply"
    assert forbidden not in strip(raw_with_prose), (
        "a comment describing the forbidden form still reds the tree, so the "
        "negative assertion is measuring prose rather than code"
    )
    assert forbidden not in _guard_step_code(), (
        "the real step passes a hand-listed --engine path again"
    )
