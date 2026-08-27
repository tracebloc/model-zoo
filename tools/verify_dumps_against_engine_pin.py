#!/usr/bin/env python3
"""Sweep every staged weight dump against the engine's pinned dependency set
(backend#2641 prevention half, backend#2658).

Why this exists
---------------
A prepped ``<base>_weights.pkl`` state_dict's key layout is decided by the
*transformers version that built the module tree*, and the engine loads seed
weights with ``load_state_dict(strict=True)``. So a dump produced under a
different version than the engine pins is a hard training abort, discoverable
only on the edge. ``prep_offline_weights.py`` cannot catch this: it builds the
prep and ship modules in one interpreter, so its strict-load check certifies
internal consistency, never agreement with the engine (backend#2641).

This tool closes that gap. Run it in an interpreter pinned to the engine's
stack (``tools/requirements-engine-pin.txt``, itself a guarded mirror of the
engine's ``use_cases/requirements.txt`` — see the CI workflow
``verify-dumps-engine-pin.yml``). For each dump it:

  * builds the *shipped* offline template in THIS interpreter — i.e. what the
    edge builds — and strict-loads the staged dump into it, categorising the
    result OK / KEY_MISMATCH / BUILD_FAIL exactly as the edge would experience
    it; and
  * checks provenance: ``manifest.json``'s schema-2 ``built_with`` block must
    match the versions actually installed here (the engine's pin). A drift —
    e.g. the engine bumps ``transformers`` — turns the gate red loudly instead
    of silently stranding every hosted seed.

Fail-closed contract
---------------------
The process exits non-zero on ANY of: a dump that does not strict-load, a
``built_with`` value that disagrees with the installed engine pin, a declared
dump whose bytes are absent (with ``--require-manifest``/manifest present), or a
sha256 that does not match. A dependency/build error is a hard error, never a
swallowed green.

The one green-with-no-work case is deliberate and loud: on a checkout with no
``manifest.json`` at all, there are no staged dumps to protect yet (hosting is
backend#2659), so the gate reports that it is armed and exits 0. Pass
``--require-manifest`` to make even that red.

Dumps are NOT committed to this repo (see ``prep_offline_weights.py``: they are
served from the tracebloc model store). CI obtains them into ``--dumps-dir``
before invoking this tool; see the workflow for the fetch integration point.

Manifest schema (v2)
--------------------
    {
      "schema": 2,
      "built_with": {"torch": "2.11.0", "transformers": "5.8.0",
                     "timm": "1.0.26", "peft": "0.19.1"},
      "dumps": [
        {"name": "bert_base_uncased",
         "template": "model_zoo/text_classification/pytorch/bert_base_uncased.py",
         "weights": "bert_base_uncased_weights.pkl",
         "sha256": "…"}
      ]
    }
"""
from __future__ import annotations

# Offline flags must be in force before torch/transformers import — several
# frameworks latch them once, at first import (see prep_offline_weights.py).
# The shipped templates are offline-migrated; a hub lookup here is a defect we
# want surfaced, not silently satisfied from a warm cache.
import os as _os

_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

# Package versions to reconcile against manifest["built_with"]. The engine pins
# every one of these; transformers is the load-bearing one (it restructures HF
# module trees between releases), but a drift in any is a dump-invalidating
# change, so all are checked.
_PROVENANCE_KEYS = ("torch", "transformers", "timm", "peft")

OK = "OK"
KEY_MISMATCH = "KEY_MISMATCH"
BUILD_FAIL = "BUILD_FAIL"
MISSING = "MISSING"
SHA_MISMATCH = "SHA_MISMATCH"
# Not a failure: the template's random-init fp32 construction exceeds a standard
# ubuntu-latest runner's RAM, so we skip the BUILD (not the dump's existence/sha)
# rather than let one oversized template OOM and take the whole sweep — and every
# dump that WOULD have verified — down with it. Reported loudly, never folded into
# OK, so the coverage gap is visible.
SKIPPED_RAM = "SKIPPED_RAM"

# Templates too large to construct in CI RAM. Kept in lockstep with
# tests/test_model_contract.py:_TOO_LARGE_FOR_CI_RAM (the source of truth for the
# instantiation suite) — see the verify-tool test that pins them equal. Keyed on
# the path relative to model_zoo/ (directory-scoped, never a bare basename: 19
# basenames are duplicated across task dirs, so a basename key would skip the
# wrong files).
_TOO_LARGE_FOR_CI_RAM = {
    "text_classification/pytorch/gemma_2.py",
}


def _ci_ram_skip_key(template: str) -> str | None:
    """Return the matched _TOO_LARGE_FOR_CI_RAM entry for a manifest template
    path (which may or may not carry a leading model_zoo/), else None. Matches on
    the directory-scoped suffix so it is robust to the prefix yet immune to the
    duplicated-basename trap."""
    posix = Path(template).as_posix()
    for entry in _TOO_LARGE_FOR_CI_RAM:
        if posix == entry or posix.endswith("/" + entry):
            return entry
    return None


def _installed_version(pkg: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_ship(template_path: str):
    """Build the shipped template's model exactly as the edge does — via its
    ``main_class``/``main_method`` entry point (both conventions accepted, as
    prep_offline_weights.py does)."""
    mod = _load_module(template_path, "ship_template")
    entry_name = getattr(mod, "main_class", None) or getattr(mod, "main_method", None)
    if not entry_name or not hasattr(mod, entry_name):
        raise RuntimeError(
            f"{template_path}: no main_class/main_method entry point found"
        )
    return getattr(mod, entry_name)()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_provenance(built_with: dict, installed: dict) -> list[str]:
    """Return a list of human-readable drift messages (empty == clean)."""
    problems: list[str] = []
    for key in _PROVENANCE_KEYS:
        declared = built_with.get(key)
        have = installed.get(key)
        if declared is None and have is None:
            # The engine pin carries no such package and the manifest declares
            # none — consistent, nothing to reconcile.
            continue
        if declared is None:
            # Installed in the engine pin but OMITTED from built_with: we cannot
            # confirm the dump was prepped against this version. Fail closed — a
            # partial built_with block (e.g. only `torch`) must not let a
            # transformers/timm/peft drift pass unseen.
            problems.append(
                f"{key}: engine pin installs {have}, but it is absent from the "
                f"manifest built_with (a partial block cannot hide a drift)"
            )
            continue
        if have is None:
            problems.append(f"{key}: manifest built_with={declared}, but not installed")
        elif have != declared:
            problems.append(
                f"{key}: manifest built_with={declared}, engine pin installs {have}"
            )
    return problems


def _verify_one(entry: dict, dumps_dir: Path, repo_root: Path) -> dict:
    """Verify a single dump. Returns a result dict with a ``category``."""
    import torch

    name = entry.get("name") or entry.get("weights") or "<unnamed>"
    template = entry.get("template")
    weights_name = entry.get("weights")
    result = {"name": name, "template": template, "weights": weights_name}

    if not template or not weights_name:
        result["category"] = BUILD_FAIL
        result["detail"] = "manifest entry missing 'template' or 'weights'"
        return result

    weights_path = dumps_dir / weights_name
    if not weights_path.exists():
        result["category"] = MISSING
        result["detail"] = f"dump bytes not found at {weights_path}"
        return result

    if entry.get("sha256"):
        actual = _sha256(weights_path)
        if actual != entry["sha256"]:
            result["category"] = SHA_MISMATCH
            result["detail"] = f"sha256 {actual} != manifest {entry['sha256']}"
            return result

    if _ci_ram_skip_key(template):
        # Skip the BUILD only — existence + sha above already ran. Constructing
        # this template's fp32 params would exceed CI RAM and OOM the whole job.
        result["category"] = SKIPPED_RAM
        result["detail"] = (
            "random-init construction exceeds CI runner RAM; build not attempted "
            "(see _TOO_LARGE_FOR_CI_RAM)"
        )
        return result

    template_path = repo_root / template
    try:
        model = _build_ship(str(template_path))
    except Exception as exc:  # noqa: BLE001 — any build failure is BUILD_FAIL
        result["category"] = BUILD_FAIL
        result["detail"] = f"{type(exc).__name__}: {exc}"
        return result

    try:
        state_dict = torch.load(weights_path, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:  # noqa: BLE001 — strict-load failure == edge abort
        result["category"] = KEY_MISMATCH
        result["detail"] = f"{type(exc).__name__}: {exc}"
        return result

    result["category"] = OK
    result["tensors"] = len(state_dict)
    return result


def _summary(results: list[dict], transformers_version: str | None) -> str:
    counts: dict[str, int] = {}
    for r in results:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    lines = [f"=== SUMMARY (transformers {transformers_version}) ==="]
    for cat in (OK, KEY_MISMATCH, BUILD_FAIL, MISSING, SHA_MISMATCH, SKIPPED_RAM):
        if counts.get(cat):
            lines.append(f"  {cat:<14} {counts[cat]:>3}")
    return "\n".join(lines)


def run_sweep(
    manifest_path: Path,
    dumps_dir: Path,
    repo_root: Path,
    report_path: Path,
    require_manifest: bool,
    check_provenance: bool,
) -> int:
    installed = {k: _installed_version(k) for k in _PROVENANCE_KEYS}

    if not manifest_path.exists():
        msg = (
            f"no manifest at {manifest_path}: no weight dumps are staged for "
            "hosting yet (backend#2659). The gate is ARMED and will verify every "
            "dump the moment a manifest.json lands."
        )
        if require_manifest:
            print(f"FAIL (fail-closed): {msg}", file=sys.stderr)
            return 2
        print(f"OK (nothing to verify): {msg}")
        return 0

    manifest = json.loads(manifest_path.read_text())
    built_with = manifest.get("built_with", {})
    dumps = manifest.get("dumps", [])

    provenance_problems: list[str] = []
    if check_provenance:
        if not built_with:
            provenance_problems.append(
                "manifest has no 'built_with' block (schema 2 required) — cannot "
                "prove dumps were built against the engine's pin"
            )
        else:
            provenance_problems = _check_provenance(built_with, installed)

    results = [_verify_one(e, dumps_dir, repo_root) for e in dumps]

    report = {
        "engine_pin_installed": installed,
        "manifest_built_with": built_with,
        "provenance_problems": provenance_problems,
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(_summary(results, installed.get("transformers")))
    if provenance_problems:
        print("\n=== PROVENANCE DRIFT (built_with vs engine pin) ===")
        for p in provenance_problems:
            print(f"  {p}")
    for r in results:
        if r["category"] != OK:
            print(f"  {r['category']:<14} {r['name']}: {r.get('detail', '')}")

    # SKIPPED_RAM is a reported coverage gap, not a failure — it must not redden
    # the gate (an OOM would have verified nothing at all).
    failed = [r for r in results if r["category"] not in (OK, SKIPPED_RAM)]
    if failed or provenance_problems:
        print(
            f"\nFAIL: {len(failed)} dump(s) not OK, "
            f"{len(provenance_problems)} provenance drift(s). "
            f"Report: {report_path}",
            file=sys.stderr,
        )
        return 1
    print(f"\nAll {len(results)} dump(s) verify against the engine pin. Report: {report_path}")
    return 0


def _selftest() -> int:
    """Self-contained proof the categorisation + provenance logic works, using
    a synthetic torch template and dumps — no transformers/timm/peft needed.
    Exercised by tests/test_verify_dumps_against_engine_pin.py and runnable as
    ``verify_dumps_against_engine_pin.py --selftest``."""
    import tempfile

    import torch
    from torch import nn

    template_src = (
        "from torch import nn\n"
        "main_class = 'MyModel'\n"
        "class MyModel(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(4, 3)\n"
    )
    broken_src = (
        "main_class = 'MyModel'\n"
        "class MyModel:\n"
        "    def __init__(self):\n"
        "        raise RuntimeError('cannot build on this pin')\n"
    )

    class _Ref(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dumps = root / "dist"
        dumps.mkdir()
        (root / "good.py").write_text(template_src)
        (root / "broken.py").write_text(broken_src)

        # OK dump: exact state_dict of the template.
        torch.save(_Ref().state_dict(), dumps / "good_weights.pkl")
        # KEY_MISMATCH dump: drop a key.
        bad = _Ref().state_dict()
        del bad["fc.bias"]
        torch.save(bad, dumps / "mismatch_weights.pkl")

        manifest = {
            "schema": 2,
            "built_with": {"torch": _installed_version("torch")},
            "dumps": [
                {"name": "good", "template": "good.py", "weights": "good_weights.pkl"},
                {"name": "mismatch", "template": "good.py", "weights": "mismatch_weights.pkl"},
                {"name": "broken", "template": "broken.py", "weights": "good_weights.pkl"},
                {"name": "gone", "template": "good.py", "weights": "absent_weights.pkl"},
            ],
        }
        mpath = root / "manifest.json"
        mpath.write_text(json.dumps(manifest))

        rc = run_sweep(mpath, dumps, root, root / "report.json", False, True)
        report = json.loads((root / "report.json").read_text())
        cats = {r["name"]: r["category"] for r in report["results"]}
        assert cats["good"] == OK, cats
        assert cats["mismatch"] == KEY_MISMATCH, cats
        assert cats["broken"] == BUILD_FAIL, cats
        assert cats["gone"] == MISSING, cats
        assert rc == 1, "sweep with failures must exit non-zero"

        # Provenance drift must fail closed.
        drift = dict(manifest, built_with={"transformers": "9.9.9"})
        dpath = root / "manifest_drift.json"
        dpath.write_text(json.dumps(drift))
        rc_drift = run_sweep(dpath, dumps, root, root / "r2.json", False, True)
        assert rc_drift == 1, "provenance drift must exit non-zero"

        # No manifest, not required → armed-green.
        rc_none = run_sweep(root / "nope.json", dumps, root, root / "r3.json", False, True)
        assert rc_none == 0, "absent manifest (not required) must be green"
        # …but red when required.
        rc_req = run_sweep(root / "nope.json", dumps, root, root / "r4.json", True, True)
        assert rc_req == 2, "absent manifest with --require-manifest must be red"

    print("selftest OK: OK/KEY_MISMATCH/BUILD_FAIL/MISSING + provenance + fail-closed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root_default = Path(__file__).resolve().parent.parent
    ap.add_argument(
        "--manifest",
        default=str(repo_root_default / "manifest.json"),
        help="path to the schema-2 manifest.json (default: repo-root/manifest.json)",
    )
    ap.add_argument(
        "--dumps-dir",
        default=str(repo_root_default / "dist"),
        help="directory holding the <base>_weights.pkl dumps (default: repo-root/dist). "
        "CI fetches these from the tracebloc model store; they are not committed.",
    )
    ap.add_argument("--repo-root", default=str(repo_root_default))
    ap.add_argument("--report", default=str(repo_root_default / "dump_verification.json"))
    ap.add_argument(
        "--require-manifest",
        action="store_true",
        help="fail (red) if manifest.json is absent, instead of the armed-green no-op",
    )
    ap.add_argument("--no-check-provenance", dest="check_provenance", action="store_false")
    ap.add_argument("--selftest", action="store_true", help="run built-in synthetic tests and exit")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    return run_sweep(
        Path(args.manifest),
        Path(args.dumps_dir),
        Path(args.repo_root),
        Path(args.report),
        args.require_manifest,
        args.check_provenance,
    )


if __name__ == "__main__":
    raise SystemExit(main())
