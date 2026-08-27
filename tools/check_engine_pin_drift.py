#!/usr/bin/env python3
"""Assert tools/requirements-engine-pin.txt still equals the engine's live pin.

The engine (tracebloc-engine `use_cases/requirements.txt` +
`use_cases/requirements_cuda.txt`) is the single source of truth for the
versions a weight dump must be built and verified against. This mirror lets
prep-runners and CI install one file, but a mirror that drifts from the engine
silently reintroduces exactly the skew backend#2641 was about. So the CI gate
runs this check: every exact (``pkg==ver``) line in the mirror must match the
engine's pin, or the job goes red until the mirror (and then the dumps) are
regenerated.

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
        print("ENGINE PIN DRIFT — tools/requirements-engine-pin.txt is stale:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print(
            "\nThis is the backend#2641 failure mode: prep/verify would run against "
            "a different stack than the edge. Fix the mirror and re-prep affected dumps.",
            file=sys.stderr,
        )
        return 1

    print(f"engine pin OK: {len(mirror)} mirrored pin(s) match the engine's requirements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
