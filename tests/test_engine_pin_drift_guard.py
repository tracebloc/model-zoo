"""The engine-pin drift guard must watch BOTH mirrors of the engine's pin (#229).

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

#227 is the proof case: as dependabot opened it, it touched only
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

_EXACT = re.compile(r"^([A-Za-z0-9._-]+)==([^\s#]+)")


# --------------------------------------------------------------------------
# Reading the workflow. Hand-rolled slicing, for the reason in the docstring.
# --------------------------------------------------------------------------
def _lines() -> list[str]:
    return WORKFLOW.read_text().splitlines()


def _event_paths(event: str) -> list[str]:
    """The ``paths:`` filter of ``on.<event>``, as written.

    Comment lines are skipped rather than terminating the list: the trigger
    entry added for #229 carries a comment explaining why it is the whole
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
    """#229 leg 1. Without this the guard cannot fail because it never runs."""
    for event in ("pull_request", "push"):
        assert _triggers(CI_MIRROR, event), (
            f"on.{event}.paths does not match {CI_MIRROR}: a PR touching only "
            f"the CI pytorch mirror fires no job at all (#229). Filters: "
            f"{_event_paths(event)}"
        )


def test_the_tools_mirror_still_triggers_the_workflow():
    """Adding the second mirror must not cost the first one its trigger."""
    for event in ("pull_request", "push"):
        assert _triggers(TOOLS_MIRROR, event)


def test_the_filter_is_not_a_CATCH_ALL():
    """The control. If everything matched, the test above would be vacuous —
    and the workflow would run its 60-minute sweep on every PR (#229's
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
    """#229 leg 2. One `--mirror` pointed at the tools/ copy is what let a
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

    #229 was not "pytorch.txt was forgotten" so much as "a file could mirror
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
        f"not in the drift guard's mirror list (#229): {unguarded}"
    )


# --------------------------------------------------------------------------
# LEG 2, BEHAVIOUR — the guard must be SEEN TO REFUSE.
#
# The #227 scenario is reconstructed from the REAL pytorch.txt rather than
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
    dest.mkdir(parents=True, exist_ok=True)
    plain = [f"{p}=={v}" for p, v in sorted(pins.items()) if p not in _CUDA_PINS]
    cuda = [f"{p}=={v}" for p, v in sorted(pins.items()) if p in _CUDA_PINS]
    (dest / "requirements.txt").write_text("\n".join(plain) + "\n")
    (dest / "requirements_cuda.txt").write_text("\n".join(cuda) + "\n")


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


def _run_checker(mirror: str, engine_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / CHECKER_REL),
            "--mirror",
            str(REPO_ROOT / mirror),
            "--engine",
            str(engine_dir / "requirements.txt"),
            "--engine",
            str(engine_dir / "requirements_cuda.txt"),
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
        print("\n--- #227 scenario, real refusal output ---")
        print(proc.stderr.rstrip())


def test_the_refusal_NAMES_WHICH_MIRROR_DRIFTED(tmp_path):
    """#229's acceptance. The header used to be the literal string
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
    """Both directions, per #229's acceptance. This is the ordinary case: the
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
    return subprocess.run(
        ["bash", str(script)],
        cwd=str(root),
        capture_output=True,
        text=True,
        env={"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "HOME": str(root)},
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
    """#227 replayed as a tree, which is the whole ticket in one test.

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
        f"#229 is back:\n{combined}"
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
