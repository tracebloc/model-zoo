#!/usr/bin/env python3
"""Assert one mirror of the engine's pin still equals the engine's live pin.

The engine (tracebloc-engine) is the single source of truth for the
versions a weight dump must be built and verified against. A mirror lets
prep-runners and CI install one file, but a mirror that drifts from the engine
silently reintroduces exactly the skew (internal ref) was about. So the CI gate
runs this check: every exact (``pkg==ver``) line in the mirror must match the
engine's pin, or the job goes red until the mirror (and then the dumps) are
regenerated.

THERE IS MORE THAN ONE MIRROR, and this tool takes whichever one it is handed.
`tools/requirements-engine-pin.txt` is the prep/verify environment;
`.github/requirements/pytorch.txt` is the environment ci.yml's required
`test-pytorch` job installs, and it carries the same rule in its own header
("never ahead of them"). The workflow calls this script once per mirror, so
the failure output below names the file it was given rather than assuming the
tools/ copy — (internal ref), where naming one mirror in the message was part
of what made the other one's absence easy to miss.

Floors (``pkg>=ver``, e.g. safetensors) are intentionally not exact-pinned by
the engine and are skipped here — the engine comment says the resolver governs
the build.

IT DETECTS A DISAGREEMENT; IT DOES NOT KNOW WHICH SIDE MOVED (internal ref).
Two pins tell you they differ. They carry no history, so they cannot tell you
who changed — and this tool used to assert one anyway, printing *"the engine
moved; regenerate the mirror"* in BOTH directions. Half the time that is the
wrong instruction, and not harmlessly wrong: when the MIRROR is the side that
moved — the normal case, since the ordering rule in the ordering rule means the
mirror PR is usually opened before the engine bump lands — following it
regenerates the mirror back DOWN to the engine's older pin, greens the check,
and silently reverts the bump being shipped. `(internal ref)` was exactly that:
the mirror was raising torch to 2.13.0 and the engine sat at 2.11.0 until
`(internal ref)` merged.

So the report states the disagreement, and the direction comes from evidence
or not at all. `--mirror-at-base` supplies the mirror's own value at the PR's
merge base, which DOES settle it: the mirror changed in this PR => the mirror
moved; unchanged => the engine did. Without it the output names both remedies
and the ordering rule that chooses between them.

THE DETECTION IS NOT LOOSENED BY ANY OF THIS. `(internal ref)` should have been red
until `(internal ref)` landed, and it still would be — a mirror ahead of the engine is a
stack skew whichever side caused it. Only the diagnosis changed.

WHICH ENGINE FILES ARE "THE ENGINE" IS DERIVED, NOT DECLARED (internal ref).
This script used to take the engine's side as a hand-listed pair of `--engine`
paths and reduce them with `dict.update`, which produced two separate defects:

  1. `dict.update` has no notion of a conflict. Two `--engine` files pinning
     one package to different versions left whichever came LAST on the command
     line as "the engine's pin" and dropped the other silently, so the verdict
     was a function of argument order rather than of the repo — and a
     disagreement BETWEEN engine files was invisible. A false clean, not a
     false alarm. `_merge_engine_sources` now refuses instead.
  2. The pair itself was the coverage claim, and it was wrong. The engine pins
     torch in FIVE files; `use_cases/requirements_vision_cv_cuda.txt` — what
     `Dockerfile.base.cv.gpu` greps its torch pin out of, i.e. the torch
     version of the `cv:gpu` base image — was not one of the two, so no gate
     in this repo had ever compared a mirror against it. An unread source
     cannot disagree.

The two compound: adding the third file by hand fixes the coverage and hands
the verdict to argument order, so both halves land together. `--engine-root`
derives the set from the engine tree instead (`_ENGINE_SOURCE_GLOBS`), which is
the shape `(internal ref)` used to fix the identical defect in
`backend`'s `tools/offline_weights/engine_pin.py`, where a hand-maintained
`_TORCH_SOURCES` tuple declared itself complete and listed four of five.

Usage:
    # CI: derive the engine's sources from the downloaded pin artifact.
    python tools/check_engine_pin_drift.py \
        --mirror tools/requirements-engine-pin.txt \
        --engine-root _engine_pin \
        [--mirror-at-base /tmp/mirror-at-merge-base.txt]

    # By hand: name them yourself. Still refuses a disagreement between them.
    python tools/check_engine_pin_drift.py \
        --mirror tools/requirements-engine-pin.txt \
        --engine _engine/use_cases/requirements.txt \
        --engine _engine/use_cases/requirements_cuda.txt

Exit codes: 0 the mirror matches; 1 the mirror and the engine disagree; 2 this
gate could not establish what the engine pins at all.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Single source of truth for which pins are dump-invalidating, shared with the
# verifier so the two tools cannot disagree. tools/ is on sys.path when this is
# run as a script (python tools/check_engine_pin_drift.py); make the import work
# regardless of the invoking cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_dumps_against_engine_pin import (  # noqa: E402
    _PROVENANCE_KEYS as _REQUIRED_PINS,
)

_EXACT = re.compile(r"^([A-Za-z0-9._-]+)==([^\s#]+)")


def _exact_pins_in(text: str) -> dict[str, str]:
    """THE one scanner. `derive_engine_sources` calls it too, deliberately.

    A derivation that decided which files are engine sources by any other rule
    than the one the reader uses could certify coverage of a file the reader
    then finds nothing in — a coverage claim that is true about a different
    scanner than the one doing the work (internal ref).
    """
    pins: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _EXACT.match(line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def _exact_pins(path: Path) -> dict[str, str]:
    return _exact_pins_in(path.read_text())


class EnginePinError(Exception):
    """This gate could not establish what the engine pins. Exit 2, never 1.

    EXIT 2 IS NOT EXIT 1 AND MUST NOT BE FOLDED INTO EITHER BRANCH. 1 is an
    answer: the mirror and the engine disagree. 2 is the absence of one, and
    "could not tell" collapsed into the benign branch is how a gate reports
    agreement it never established.
    """


#: The search space the engine's source set is DERIVED from, anchored at the
#: engine's root — and the only hand-kept claim left in this file.
#:
#: WHY A DERIVATION RATHER THAN FILENAMES (internal ref). The workflow used to
#: name two engine files, and that pair WAS the coverage claim; it listed two
#: of the seven files that pin something a mirror mirrors, and two of the five
#: that pin torch. `(internal ref)` is the same defect in `backend`'s
#: copy of this idea and `(internal ref)` the same remedy. A hand-kept list of
#: ANOTHER repo's filenames rots on that repo's next base family and nothing
#: here notices; a claim as weak as "the engine pins things in a base
#: Dockerfile or a use-case requirements file" survives it, and what it selects
#: is then measured rather than asserted.
_ENGINE_SOURCE_GLOBS = ("Dockerfile.base.*", "use_cases/requirements*.txt")


def _candidates_under(root: Path) -> dict[str, str]:
    """Every file in the search space, read WITHOUT filtering: path -> text.

    Unfiltered on purpose. The derivation below has to be able to observe that
    a candidate pins nothing, and a candidate that was never read is
    indistinguishable from one that pins nothing. `Path.glob` does not let a
    single `*` cross a `/`, so `Dockerfile.base.*` cannot reach a vendored
    `sub/Dockerfile.base.cpu`.

    A GLOB THAT MATCHES NOTHING IS AN ERROR, not an empty result. It means the
    pin artifact did not arrive intact or the engine renamed that family, and
    "I read no files and found no disagreement" is exactly the vacuous green
    this ticket is about.
    """
    found: dict[str, str] = {}
    for pattern in _ENGINE_SOURCE_GLOBS:
        matched = sorted(p for p in root.glob(pattern) if p.is_file())
        if not matched:
            raise EnginePinError(
                f"the engine-source glob {pattern!r} matched NO file under "
                f"{root}, so the search space is incomplete. Either the "
                "engine-pin artifact did not arrive intact or the engine "
                "renamed that family. Refusing: a search space this gate "
                "could not read is not one it agrees with."
            )
        for p in matched:
            found[p.relative_to(root).as_posix()] = p.read_text()
    return found


#: A candidate that carries NO literal pin but DERIVES one from another
#: candidate: the three GPU bases `grep` a torch pin out of whichever
#: requirements file their `REQUIREMENTS_FILE` build-arg names and install it
#: from the cu129 wheel index. Two independent signals are required so a
#: stray `grep` somewhere else cannot classify a file as deriving.
_WHEEL_INDEX = re.compile(r"--index-url\s+https://download\.pytorch\.org/whl/")
_GREPS_A_TORCH_PIN = re.compile(r"grep\b[^\n]*\btorch[^\n]*==")
_DERIVES_FROM = re.compile(r"^ARG\s+REQUIREMENTS_FILE=(\S+)", re.MULTILINE)


def classify_engine_candidates(
    candidates: dict[str, str],
) -> tuple[list[str], dict[str, str], list[str]]:
    """Sort the search space into files that PIN a version and files that
    DERIVE one — and return whatever fits neither.

    WHY THE DISTINCTION IS EXPLICIT RATHER THAN IMPLIED. "It pins nothing, so
    it is not a source" is true of the three GPU Dockerfiles and it is also
    true of a brand-new base family that this tool has never heard of. Folding
    both into one silent "not a source" bucket is a coverage claim that cannot
    fail — the same defect as the one this file is fixing, one level up, which
    is why an UNCLASSIFIED candidate is an error rather than an omission.

    Returns `(pinning, deriving, unclassified)` where `deriving` maps each
    deriving file to the candidate it takes its pin FROM.
    """
    pinning: list[str] = []
    deriving: dict[str, str] = {}
    unclassified: list[str] = []
    for rel in sorted(candidates):
        text = candidates[rel]
        if _exact_pins_in(text):
            pinning.append(rel)
        elif _WHEEL_INDEX.search(text) and _GREPS_A_TORCH_PIN.search(text):
            m = _DERIVES_FROM.search(text)
            deriving[rel] = m.group(1) if m else ""
        else:
            unclassified.append(rel)
    return pinning, deriving, unclassified


def assert_engine_pin_coverage(
    candidates: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """Every candidate must be a pinning source or a deriving one. Or exit 2.

    FOUR WAYS THE CLAIM CAN BE FALSE, and each is raised rather than absorbed:

      * a candidate that neither pins nor derives — an unrecognised base
        family, silently outside the scan under the old reducer;
      * NO pinning source at all, which agrees with every mirror vacuously;
      * NO deriving source at all, which means the wheel-index idiom stopped
        being recognised — the set quietly emptying out is indistinguishable
        from an engine that stopped using it, and only one of those is fine;
      * a deriving source whose `REQUIREMENTS_FILE` target is not itself a
        pinning candidate. That is the strongest of the four: it makes the
        file each GPU base greps STRUCTURALLY REQUIRED to be in the scan, and
        `Dockerfile.base.cv.gpu` derives from
        `use_cases/requirements_vision_cv_cuda.txt` — precisely the file
        (internal ref) is about. Under the old two-file list that target was
        not even fetched.

    A GPU Dockerfile that STARTS carrying a bare `torch==` needs no special
    case: it classifies as pinning and joins the scanned set automatically.
    """
    pinning, deriving, unclassified = classify_engine_candidates(candidates)
    problems: list[str] = []
    if unclassified:
        problems.append(
            "these engine files neither pin a version nor derive one, so this "
            "gate cannot say whether they belong in the scan: "
            + ", ".join(unclassified)
            + ". An unrecognised base family must be classified deliberately, "
            "not left out quietly — that is a coverage claim that cannot fail."
        )
    if not pinning:
        problems.append(
            "NO engine file carries an exact pin, so the engine's side would "
            "be empty and agree with every mirror vacuously."
        )
    if not deriving:
        problems.append(
            "NO engine file derives its pin from another any more. Either the "
            "engine dropped the wheel-index bases — update the expectation "
            "deliberately — or the idiom changed and this tool stopped "
            "recognising it. Those are different facts and this is not the "
            "place to guess between them."
        )
    for rel, target in sorted(deriving.items()):
        if not target:
            problems.append(
                f"{rel} derives its pin but names no REQUIREMENTS_FILE, so the "
                "file it actually installs from cannot be identified."
            )
        elif target not in pinning:
            problems.append(
                f"{rel} derives its torch pin from {target}, which is NOT a "
                "pinning file in the scanned set. The version that base image "
                "actually installs therefore comes from outside this gate's "
                "scan, which is (internal ref) exactly."
            )
    if problems:
        raise EnginePinError(
            "ENGINE PIN COVERAGE IS WRONG — the search space does not account "
            "for itself:\n"
            + "\n".join(f"  {p}" for p in problems)
            + f"\n\nRead {len(candidates)} candidate(s): {len(pinning)} pin a "
            f"version, {len(deriving)} derive one, {len(unclassified)} "
            "unaccounted for."
        )
    return pinning, deriving


def derive_engine_sources(candidates: dict[str, str]) -> list[str]:
    """Which candidates actually carry an exact pin — the derived source set.

    THE THREE GPU DOCKERFILES ARE READ AND DERIVE NOTHING, correctly. Each
    pins torch by `grep -E '^torch(vision)?==' /tmp/requirements.txt` piped
    into a `--index-url .../cu129` install, so it carries no literal pin of its
    own: its torch version IS whichever requirements file its
    `REQUIREMENTS_FILE` build-arg names, and that file is a candidate in its
    own right. Handing one to `--engine` would name a source contributing zero
    pins — a silent no-op here rather than a crash, and so WORSE than the gap
    it looks like it closes, because the invocation would then claim a source
    it reads nothing from.
    """
    if not candidates:
        raise EnginePinError(
            "the engine-source derivation was handed no candidates at all. "
            "Nothing looked at derives nothing missing, and that then 'agrees' "
            "vacuously (internal ref)."
        )
    return sorted(rel for rel, text in candidates.items() if _exact_pins_in(text))


def _conflict_report(conflicts: dict[str, dict[str, str]]) -> str:
    """The refusal, worded so it is byte-identical under any argument order.

    Packages sorted, and each package's sources sorted by path — because the
    whole defect was a verdict that depended on argument order, and a REFUSAL
    whose text depended on argument order would be the same bug in the report.
    """
    lines = [
        "CANNOT ESTABLISH THE ENGINE'S PIN — the engine's own sources disagree "
        "with each other:"
    ]
    for pkg in sorted(conflicts):
        srcs = conflicts[pkg]
        shown = ", ".join(f"{path} pins =={srcs[path]}" for path in sorted(srcs))
        lines.append(f"  {pkg}: {shown}")
    lines.append(
        "\nThere is no honest single version to gate a mirror against while "
        "these disagree, so this check REFUSES rather than picking one. It used "
        "to pick: `dict.update` is last-write-wins, so whichever source came "
        "LAST on the command line became 'the engine's pin' and the others were "
        "discarded without a word — the same files in a different order gave a "
        "red and a green (internal ref)."
    )
    lines.append(
        "WHICH SOURCE IS RIGHT IS NOT KNOWABLE FROM THE PINS (internal ref) — "
        "two pins cannot tell you which side moved, and neither can seven. This "
        "gate READS the engine's sources; it does not choose between them. "
        "Reconcile them in tracebloc-engine, then re-run."
    )
    return "\n".join(lines)


def _merge_engine_sources(paths: list[str], relevant: set[str]) -> dict[str, str]:
    """Merge the engine's sources, refusing a disagreement between them.

    SCOPED TO THE PACKAGES THAT CAN CHANGE THIS GATE'S ANSWER — the mirror's
    own exact pins plus the required ones. That scoping is deliberate and not
    a loosening: a disagreement outside it cannot move any verdict here, and
    refusing on it would turn a model-zoo PR red over an engine-internal
    matter its author cannot act on. Different base families legitimately
    differ on packages no mirror mirrors (measured on `tracebloc-engine@develop`:
    32 distinct packages across the seven sources, zero disagreements, so this
    scoping changes no verdict today either way).
    """
    seen: dict[str, dict[str, str]] = {}
    for path in paths:
        # AN UNREADABLE ENGINE SOURCE IS EXIT 2, NOT A TRACEBACK. It is the
        # same fact as the empty-contribution case below — this gate does not
        # know what the engine pins — and a traceback reports it with no exit
        # status a caller can distinguish from a crash in the compare.
        try:
            pins = _exact_pins(Path(path))
        except OSError as exc:
            raise EnginePinError(
                f"{path} was named as an engine source but could not be read "
                f"({exc.strerror}). Refusing: an engine source this gate never "
                "read cannot disagree with the mirror, so continuing without "
                "it would report agreement it never established (internal ref)."
            ) from exc
        # A NAMED SOURCE THAT CONTRIBUTES NOTHING IS A BROKEN INVOCATION, not
        # a benign no-op. It is what a sparse-checkout typo, a missing
        # artifact, or a GPU Dockerfile handed to `--engine` looks like, and
        # every one of those reads 6 of 7 files and reports agreement.
        if not pins:
            raise EnginePinError(
                f"{path} was named as an engine source but carries NO exact "
                "pin at all, so it cannot contribute to the engine's side. "
                "That is a broken invocation — a missing artifact, a "
                "sparse-checkout that matched nothing, or a file that pins via "
                "a wheel index rather than `pkg==ver` — and it must not pass "
                "as a source that simply happened to agree (internal ref)."
            )
        for pkg, ver in pins.items():
            seen.setdefault(pkg, {})[path] = ver
    conflicts = {
        pkg: srcs
        for pkg, srcs in seen.items()
        if pkg in relevant and len(set(srcs.values())) > 1
    }
    if conflicts:
        raise EnginePinError(_conflict_report(conflicts))
    # Deterministic pick: every remaining package agrees across its sources,
    # so sorting by path only makes the choice reproducible, never material.
    return {pkg: srcs[sorted(srcs)[0]] for pkg, srcs in seen.items()}


#: WHAT TWO PINS CAN AND CANNOT TELL YOU (internal ref). They can tell you
#: the mirror and the engine disagree. They cannot tell you WHICH SIDE MOVED —
#: that is a fact about history, and neither file carries any. So the
#: disagreement is stated without a cause and both remedies are offered,
#: unless `--mirror-at-base` hands over the one piece of evidence that does
#: settle it.
_MIRROR_MOVED = "MIRROR-MOVED"
_ENGINE_MOVED = "ENGINE-MOVED"
_UNKNOWN = "UNKNOWN"

_REMEDIES = {
    _ENGINE_MOVED: (
        "The mirror is UNCHANGED since the merge base, so the ENGINE moved. "
        "Regenerate the mirror AND the dumps built against it."
    ),
    _MIRROR_MOVED: (
        "The mirror CHANGED in this PR, so this PR is the side that moved. Do "
        "NOT regenerate the mirror down to the engine's pin — that would "
        "revert the bump you are shipping. The engine base moves FIRST and "
        "mirrors follow (the ordering rule): land the engine bump, then re-run this."
    ),
    _UNKNOWN: (
        "WHICH SIDE MOVED IS NOT KNOWABLE FROM TWO PINS, so this check does "
        "not guess (internal ref). Both remedies, and the ordering rule that "
        "chooses between them:\n"
        "  * If the ENGINE moved, the mirror is stale: regenerate the mirror "
        "AND the dumps built against it.\n"
        "  * If THIS PR is raising the mirror, the mirror is correct and the "
        "engine is behind: the engine base moves FIRST and mirrors follow "
        "(the ordering rule), so the engine bump must merge before this can go "
        "green. Regenerating the mirror here would revert your bump.\n"
        "  Pass --mirror-at-base <the mirror's content at the merge base> and "
        "this check will derive the direction instead of listing both."
    ),
}


def _direction(pkg: str, mirror_ver: str, base: dict[str, str] | None) -> str:
    """Which side moved, derived from the mirror's own value at the merge base.

    THE ONLY HONEST SOURCE OF A DIRECTION HERE. Mirror changed in this PR =>
    the mirror moved; unchanged => the engine did. A package that is pinned
    now and was NOT pinned at the base also counts as the mirror moving: the
    PR added the pin, so the PR is the change.

    Returns `_UNKNOWN` when no base was supplied, which is the honest answer
    on a schedule/push run where there is no PR and so no merge base.
    """
    if base is None:
        return _UNKNOWN
    return _ENGINE_MOVED if base.get(pkg) == mirror_ver else _MIRROR_MOVED


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mirror", required=True)
    ap.add_argument(
        "--engine",
        action="append",
        default=[],
        help="repeatable. An engine pin file, named explicitly. A disagreement "
        "between two of them is REFUSED, not resolved by argument order "
        "(internal ref).",
    )
    ap.add_argument(
        "--engine-root",
        default=None,
        help="the root of a checked-out/downloaded engine pin. The engine's "
        "source set is DERIVED from it via _ENGINE_SOURCE_GLOBS rather than "
        "hand-listed, which is what keeps a new engine base family from "
        "landing outside this gate's scan (internal ref). Combinable with "
        "--engine; at least one of the two is required.",
    )
    ap.add_argument(
        "--mirror-at-base",
        default=None,
        help="the SAME mirror's content at this PR's merge base (e.g. "
        "`git show origin/develop:<mirror>`). Optional, and it buys exactly "
        "one thing: a derived direction instead of both remedies. Omit it on "
        "a push/schedule run, where there is no merge base to derive from.",
    )
    args = ap.parse_args()

    mirror = _exact_pins(Path(args.mirror))

    # THE ENGINE'S SIDE, AND EVERY WAY IT CAN FAIL TO EXIST, IS EXIT 2. Not a
    # traceback and not exit 1: "the engine pins X and the mirror pins Y" and
    # "this gate never worked out what the engine pins" are different facts and
    # a reader of the exit status has to be able to tell them apart.
    #
    # `except EnginePinError` and nothing wider — a bare `except` here would
    # swallow the very unreadable-file cases the message claims to report.
    try:
        engine_paths = list(args.engine)
        coverage: list[str] = []
        if args.engine_root:
            root = Path(args.engine_root)
            candidates = _candidates_under(root)
            # THE CLAIM IS CHECKED BEFORE IT IS USED. Deriving the set and
            # then gating on it without asking whether the set accounts for
            # the whole search space is how a two-of-seven list passed for
            # years (internal ref).
            pinning, deriving = assert_engine_pin_coverage(candidates)
            derived = derive_engine_sources(candidates)
            assert derived == pinning, (derived, pinning)
            engine_paths.extend(str(root / rel) for rel in derived)
            # A GREEN THAT DOES NOT SAY HOW MUCH IT READ cannot be told from a
            # green that read nothing, which is the whole ticket. So the counts
            # go in the output, on success as well as failure.
            coverage = [
                "  search space : "
                + ", ".join(_ENGINE_SOURCE_GLOBS)
                + f" ({len(candidates)} candidate file(s) read)",
                f"  pins a version: {len(pinning)} of {len(candidates)} — "
                + ", ".join(pinning),
                # NAMED, not merely counted. A read-and-excluded set that is
                # never printed is an exclusion nobody can review.
                f"  derives one   : {len(deriving)} of {len(candidates)} — "
                + ", ".join(
                    f"{rel} <- {target}" for rel, target in sorted(deriving.items())
                ),
            ]
        if not engine_paths:
            raise EnginePinError(
                "no engine source was given: pass --engine-root (derived, what "
                "CI uses) or at least one --engine. Refusing rather than "
                "comparing the mirror against an empty engine, which agrees "
                "with everything."
            )
        engine = _merge_engine_sources(engine_paths, set(mirror) | set(_REQUIRED_PINS))
    except EnginePinError as exc:
        print(exc, file=sys.stderr)
        # The counts go out on the FAILURE path too: a refusal that does not
        # say how many sources it read cannot be told from one that read two.
        for line in coverage:
            print(line, file=sys.stderr)
        return 2
    # NOT a try/except-pass. A `--mirror-at-base` that was handed a path this
    # tool cannot read is a broken invocation, and swallowing it would silently
    # demote every verdict to UNKNOWN — a checker that quietly stopped using
    # the evidence it was given, which is (internal ref)'s shape one level up.
    #
    # EXIT 2, not 1, and not a traceback: 1 means "the two sides disagree" and
    # a broken invocation must not be mistaken for one. An EMPTY file is a
    # different thing entirely and is fine — it is what the workflow writes
    # for a mirror this PR created, and it derives "the mirror moved".
    base = None
    if args.mirror_at_base:
        base_path = Path(args.mirror_at_base)
        if not base_path.is_file():
            print(
                f"--mirror-at-base {base_path} is not a readable file, so the "
                "which-side-moved derivation cannot run. Refusing rather than "
                "falling back to an undiagnosed report: a checker that quietly "
                "stops using the evidence it was handed is (internal ref) again.",
                file=sys.stderr,
            )
            return 2
        base = _exact_pins(base_path)

    problems: list[str] = []
    directions: set[str] = set()
    # Fail closed on an emptied, comment-only, or partial mirror: the loop below
    # only walks pins that ARE present, so a mirror that drops a load-bearing pin
    # (or has none at all) would otherwise pass vacuously and stop turning the
    # schedule red on an engine bump — the only alarm while dumps are unhosted.
    for pkg in _REQUIRED_PINS:
        if pkg not in mirror:
            problems.append(
                f"{pkg}: REQUIRED exact pin is missing from the mirror — an "
                "emptied/comment-only/partial mirror must not pass the drift check"
            )
    for pkg, ver in sorted(mirror.items()):
        eng_ver = engine.get(pkg)
        if eng_ver is None:
            problems.append(
                f"{pkg}=={ver}: pinned in the mirror but NOT found in the engine's "
                "requirements — remove it or add it to the engine"
            )
        elif eng_ver != ver:
            # STATE THE DISAGREEMENT, ATTRIBUTE NOTHING (internal ref). This
            # line used to end "the engine moved; regenerate the mirror" in
            # BOTH directions. When the mirror is the side that moved — the
            # normal case, because someone opens the mirror PR before the
            # engine bump lands — following that sentence regenerates the
            # mirror back DOWN to the engine's older pin, greens the check, and
            # silently reverts the security bump being shipped. A red that
            # tells you to undo your change is worse than one that says "these
            # disagree and I cannot tell which of you is wrong".
            direction = _direction(pkg, ver, base)
            directions.add(direction)
            note = ""
            if direction == _MIRROR_MOVED:
                was = base.get(pkg) if base else None
                note = (
                    f" [this PR moved the mirror: =={was} at the merge base]"
                    if was is not None
                    else " [this PR added this pin; it is absent at the merge base]"
                )
            elif direction == _ENGINE_MOVED:
                note = " [mirror unchanged at the merge base, so the engine moved]"
            problems.append(
                f"{pkg}: mirror pins =={ver}, engine pins =={eng_ver} — these "
                f"must match{note}"
            )

    if problems:
        # Name the mirror we were HANDED, never a hard-coded path: with two
        # mirrors checked in the same job, a fixed string would attribute every
        # drift to the tools/ copy and send the fix at the wrong file (internal ref).
        #
        # AND "is stale" IS ITSELF AN ATTRIBUTION (internal ref) — it names
        # the mirror as the wrong side, which is the same unfounded claim the
        # per-package line used to make. The header states the disagreement.
        print(
            f"ENGINE PIN DRIFT — {args.mirror} disagrees with the engine:",
            file=sys.stderr,
        )
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        # One remedy when the evidence settles the direction, both when it does
        # not. Mixed directions across packages is a real state (a PR bumping
        # one pin while the engine moved another) and gets both, since no
        # single remedy is right for the whole file.
        #
        # `directions` is empty when every problem is a MISSING or
        # NOT-IN-THE-ENGINE pin rather than a version disagreement. Nothing
        # moved in that case, so the which-side-moved paragraph would be an
        # answer to a question nobody asked and is not printed.
        if directions:
            verdict = (
                directions.pop() if len(directions) == 1 else _UNKNOWN
            )
            print(f"\n{_REMEDIES[verdict]}", file=sys.stderr)
        print(
            "\nEither way this is the (internal ref) failure mode while it stands: "
            "prep/verify would run against a different stack than the edge. The "
            "detection is correct — do not loosen it; reconcile the two sides.",
            file=sys.stderr,
        )
        return 1

    print(
        f"engine pin OK: {args.mirror} — {len(mirror)} mirrored pin(s) match the "
        f"engine's requirements, read from {len(engine_paths)} engine source(s)"
    )
    for line in coverage:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
