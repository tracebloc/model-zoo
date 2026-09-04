#!/usr/bin/env python3
"""Assert one mirror of the engine's pin still equals the engine's live pin.

The engine (tracebloc-engine `use_cases/requirements.txt` +
`use_cases/requirements_cuda.txt`) is the single source of truth for the
versions a weight dump must be built and verified against. A mirror lets
prep-runners and CI install one file, but a mirror that drifts from the engine
silently reintroduces exactly the skew backend#2641 was about. So the CI gate
runs this check: every exact (``pkg==ver``) line in the mirror must match the
engine's pin, or the job goes red until the mirror (and then the dumps) are
regenerated.

THERE IS MORE THAN ONE MIRROR, and this tool takes whichever one it is handed.
`tools/requirements-engine-pin.txt` is the prep/verify environment;
`.github/requirements/pytorch.txt` is the environment ci.yml's required
`test-pytorch` job installs, and it carries the same rule in its own header
("never ahead of them"). The workflow calls this script once per mirror, so
the failure output below names the file it was given rather than assuming the
tools/ copy — model-zoo#229, where naming one mirror in the message was part
of what made the other one's absence easy to miss.

Floors (``pkg>=ver``, e.g. safetensors) are intentionally not exact-pinned by
the engine and are skipped here — the engine comment says the resolver governs
the build.

Usage:
    python tools/check_engine_pin_drift.py \
        --mirror tools/requirements-engine-pin.txt \
        --engine _engine/use_cases/requirements.txt \
        --engine _engine/use_cases/requirements_cuda.txt
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


def _exact_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _EXACT.match(line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--engine", action="append", required=True, help="repeatable")
    args = ap.parse_args()

    mirror = _exact_pins(Path(args.mirror))
    engine: dict[str, str] = {}
    for eng_path in args.engine:
        engine.update(_exact_pins(Path(eng_path)))

    problems: list[str] = []
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
            problems.append(
                f"{pkg}: mirror pins =={ver}, engine pins =={eng_ver} — the engine "
                "moved; regenerate the mirror AND the dumps built against it"
            )

    if problems:
        # Name the mirror we were HANDED, never a hard-coded path: with two
        # mirrors checked in the same job, a fixed string would attribute every
        # drift to the tools/ copy and send the fix at the wrong file (#229).
        print(f"ENGINE PIN DRIFT — {args.mirror} is stale:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print(
            "\nThis is the backend#2641 failure mode: prep/verify would run against "
            "a different stack than the edge. Fix the mirror and re-prep affected dumps.",
            file=sys.stderr,
        )
        return 1

    print(
        f"engine pin OK: {args.mirror} — {len(mirror)} mirrored pin(s) match the "
        "engine's requirements"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
