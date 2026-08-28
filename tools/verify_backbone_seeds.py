#!/usr/bin/env python3
"""Prove the backbone-only seeds load into a head the dump never saw.

backend#2642's acceptance, and the thing the old verifier could not ask. That one
built each template at its DEFAULT class count and strict-loaded the matching
dump — so it passed on exactly the configuration the dump was made for, which is
the one configuration that was never in doubt. Every failure in the 2026-08-28
acceptance run (backend#2660) happened at some OTHER class count.

So this builds each template at a class count chosen to differ from the dump's,
and asserts the contract the engine is to enforce:

* ``unexpected`` keys  ⇒ FAIL. A key the model does not have means the wrong dump.
* ``missing`` keys     ⇒ allowed ONLY where the template declared them via
                         ``SEED_EXCLUDED_PREFIXES``. Anything else missing is the
                         averaging-service#94 silent-drop, and it fails here.
* a shape mismatch     ⇒ FAIL (torch raises regardless of ``strict``).

THIS DOUBLES AS THE REFERENCE IMPLEMENTATION for the engine side: `check()` below
is the whole contract, and it is deliberately small enough to port verbatim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

#: Deliberately not a round number and not any template's default: the point is a
#: head shape no staged dump was ever built with.
PROBE_CLASSES = 7
CLASS_CONSTANTS = ("output_classes", "num_feature_points")
CONSTANT = "SEED_EXCLUDED_PREFIXES"


def read_prefixes(path: Path) -> Tuple[str, ...]:
    import ast

    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == CONSTANT:
                    return tuple(ast.literal_eval(node.value))
    return ()


def build_at(path: Path, value: int, tmp: Path):
    src = path.read_text(encoding="utf-8")
    for name in CLASS_CONSTANTS:
        src = re.sub(rf"(?m)^{name}\s*=\s*\S+", f"{name} = {value}", src, count=1)
    target = tmp / f"probe_{path.stem}.py"
    target.write_text(src, encoding="utf-8")
    for sibling in path.parent.glob(f"{path.stem}_*.json"):
        shutil.copy(sibling, tmp / sibling.name)

    spec = importlib.util.spec_from_file_location(f"probe_{path.stem}", target)
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        spec.loader.exec_module(module)
        if hasattr(module, "MyModel"):
            return module.MyModel()
        main_class = getattr(module, "main_class")
        return getattr(module, main_class)()
    finally:
        os.chdir(cwd)


def check(model, state: Dict, prefixes: Tuple[str, ...]) -> Tuple[bool, str]:
    """THE CONTRACT. Port this to the engine's cycle-0 seed load verbatim."""
    result = model.load_state_dict(state, strict=False)  # raises on shape mismatch
    if result.unexpected_keys:
        return False, (
            f"{len(result.unexpected_keys)} unexpected key(s), e.g. "
            f"{result.unexpected_keys[:3]} — this dump does not belong to this model"
        )
    undeclared = [
        key for key in result.missing_keys if not key.startswith(tuple(prefixes))
    ]
    if undeclared:
        return False, (
            f"{len(undeclared)} key(s) missing that the template did not declare, "
            f"e.g. {undeclared[:3]} — these would silently keep their fresh init"
        )
    return True, f"{len(result.missing_keys)} declared head key(s) left fresh"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zoo", required=True)
    parser.add_argument("--weights", required=True, help="backbone-only seeds")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    import torch

    zoo, weights = Path(args.zoo).expanduser(), Path(args.weights).expanduser()
    root = zoo / "model_zoo" if (zoo / "model_zoo").is_dir() else zoo
    templates: Dict[str, Path] = {}
    for category in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for path in (root / category / "pytorch").glob("*.py"):
            templates.setdefault(path.stem, path)

    results: Dict[str, Dict] = {}
    for directory in sorted(p.name for p in weights.iterdir() if p.is_dir()):
        stem = directory
        for prefix in ("sentence_pair_",):
            if stem.startswith(prefix):
                stem = stem[len(prefix) :]
        dump = weights / directory / f"{directory}_weights.pkl"
        path = templates.get(stem)
        if path is None or not dump.is_file():
            results[directory] = {"status": "NO_TEMPLATE_OR_DUMP"}
        else:
            prefixes = read_prefixes(path)
            with tempfile.TemporaryDirectory(prefix="tb-bbverify-") as raw:
                try:
                    model = build_at(path, PROBE_CLASSES, Path(raw))
                    state = torch.load(dump, weights_only=True, map_location="cpu")
                    ok, detail = check(model, state, prefixes)
                    results[directory] = {
                        "status": "OK" if ok else "CONTRACT_FAIL",
                        "detail": detail,
                        "prefixes": list(prefixes),
                    }
                    del model
                except Exception as exc:  # noqa: BLE001
                    results[directory] = {
                        "status": "LOAD_FAIL",
                        "detail": f"{type(exc).__name__}: {exc}"[:300],
                    }
        r = results[directory]
        print(f"{directory:34s} {r['status']:14s} {r.get('detail', '')}"[:150], flush=True)

    summary: Dict[str, int] = {}
    for record in results.values():
        summary[record["status"]] = summary.get(record["status"], 0) + 1
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\n=== at output_classes={PROBE_CLASSES} — " +
          ", ".join(f"{k}={v}" for k, v in sorted(summary.items())) + " ===")
    return 0 if summary.get("OK") == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
