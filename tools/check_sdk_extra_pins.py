#!/usr/bin/env python3
"""Hold the engine-pin mirror inside the pinned SDK's ``[pytorch]`` extra.

``.github/requirements/sdk.txt`` installs ``tracebloc[pytorch]==X`` next to the
engine-pin mirror ``.github/requirements/pytorch.txt``. The SDK caps torch and
torchvision (and floors the other libraries) in its own metadata; if a mirror
pin leaves that range, pip fails to resolve inside the job. This check reads the
SDK's wheel METADATA and names the pin and the range before the install.

Every exact ``pkg==ver`` line in the mirror is compared with every
``Requires-Dist`` of the ``pytorch`` extra that names the same package. Exit
codes: 0 all inside; 1 a pin outside a range; 2 it could not establish either
side (no SDK pin, no wheel, no pytorch-extra requirement, an operator it does
not know). Stdlib only.

Usage:
    pip download --no-deps "tracebloc==<pin>" -d "$dir"
    python tools/check_sdk_extra_pins.py --wheel "$dir"/tracebloc-*.whl
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_MIRROR = _ROOT / ".github" / "requirements" / "pytorch.txt"
_SDK = _ROOT / ".github" / "requirements" / "sdk.txt"

_EXACT = re.compile(r"^([A-Za-z0-9._-]+)==([^\s#;]+)")
_SDK_PIN = re.compile(r"^tracebloc\[[^\]]*\bpytorch\b[^\]]*\]==([^\s#;]+)")
_REQ = re.compile(r"^([A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*\(?([^;()]*)\)?\s*(?:;(.*))?$")
_EXTRA = re.compile(r"""extra\s*==\s*["']pytorch["']""")
_SPEC = re.compile(r"^(~=|==|!=|<=|>=|<|>)\s*([0-9][0-9A-Za-z.]*)$")


class Unreadable(Exception):
    """This gate cannot tell what one side says: a finding, never a pass."""


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _ver(v: str) -> tuple[int, ...]:
    parts = v.split(".")
    if not all(p.isdigit() for p in parts):
        raise Unreadable(f"version {v!r} is not a plain release version")
    out = [int(p) for p in parts]
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return tuple(out)


def _holds(op: str, have: tuple[int, ...], want_s: str) -> bool:
    want = _ver(want_s)
    if op == "==":
        return have == want
    if op == "!=":
        return have != want
    if op == ">=":
        return have >= want
    if op == "<=":
        return have <= want
    if op == ">":
        return have > want
    if op == "<":
        return have < want
    # ~=X.Y.Z: >= X.Y.Z and == X.Y.*
    base = [int(p) for p in want_s.split(".")]
    if len(base) < 2:
        raise Unreadable(f"~={want_s} needs at least two parts")
    return have >= want and tuple(have[: len(base) - 1]) == tuple(base[:-1])


def sdk_pin(text: str) -> str:
    pins = [m.group(1) for ln in text.splitlines() if (m := _SDK_PIN.match(ln.strip()))]
    if len(pins) != 1:
        raise Unreadable(f"expected one tracebloc[pytorch]==X line, found {len(pins)}")
    return pins[0]


def mirror_pins(text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for ln in text.splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and (m := _EXACT.match(ln)):
            pins[_norm(m.group(1))] = m.group(2)
    if not pins:
        raise Unreadable("the mirror carries no exact pin")
    return pins


def extra_requirements(metadata: str) -> list[tuple[str, str]]:
    """(package, specifier text) of every Requires-Dist gated on extra pytorch."""
    out = []
    for ln in metadata.splitlines():
        if not ln.startswith("Requires-Dist:"):
            continue
        m = _REQ.match(ln.split(":", 1)[1].strip())
        if not m:
            raise Unreadable(f"cannot parse {ln!r}")
        if m.group(3) and _EXTRA.search(m.group(3)):
            out.append((_norm(m.group(1)), m.group(2).strip()))
    if not out:
        raise Unreadable("the SDK metadata has no requirement for extra 'pytorch'")
    return out


def wheel_metadata(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if re.fullmatch(r"[^/]+\.dist-info/METADATA", n)]
        if len(names) != 1:
            raise Unreadable(f"{path} holds {len(names)} METADATA files")
        text = z.read(names[0]).decode("utf-8")
    ver = re.search(r"^Version:\s*(\S+)", text, re.MULTILINE)
    if not ver:
        raise Unreadable("the wheel METADATA names no Version")
    return text, ver.group(1)


def violations(pins: dict[str, str], reqs: list[tuple[str, str]]) -> list[str]:
    bad = []
    for pkg, spec in reqs:
        if pkg not in pins:
            continue
        have = _ver(pins[pkg])
        for part in (s.strip() for s in spec.split(",") if s.strip()):
            m = _SPEC.match(part)
            if not m:
                raise Unreadable(f"cannot read specifier {part!r} for {pkg}")
            if not _holds(m.group(1), have, m.group(2)):
                bad.append(f"{pkg}=={pins[pkg]} is outside the SDK's range '{spec}' (fails {part})")
                break
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wheel", required=True, type=Path)
    ap.add_argument("--mirror", type=Path, default=_MIRROR)
    ap.add_argument("--sdk", type=Path, default=_SDK)
    a = ap.parse_args(argv)
    try:
        want = sdk_pin(a.sdk.read_text(encoding="utf-8"))
        meta, got = wheel_metadata(a.wheel)
        if got != want:
            raise Unreadable(f"{a.sdk} pins tracebloc=={want} but the wheel is {got}")
        pins = mirror_pins(a.mirror.read_text(encoding="utf-8"))
        reqs = extra_requirements(meta)
        shared = sorted({p for p, _ in reqs} & set(pins))
        if not shared:
            raise Unreadable("no package is named by both the mirror and the SDK extra")
        bad = violations(pins, reqs)
    except (Unreadable, OSError, zipfile.BadZipFile) as e:
        print(f"check_sdk_extra_pins: CANNOT TELL: {e}", file=sys.stderr)
        return 2
    if bad:
        print(f"check_sdk_extra_pins: {a.mirror} leaves tracebloc[pytorch]=={want}:", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        print("Move the SDK pin in sdk.txt or the mirror so they agree; the engine decides the mirror.", file=sys.stderr)
        return 1
    print(f"ok: {', '.join(shared)} inside tracebloc[pytorch]=={want}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
