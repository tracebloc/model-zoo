#!/usr/bin/env python3
"""Derive each template's head keys — the ones a hosted seed must NOT carry.

backend#2642. The seed contract is "backbone only": the hosted dump holds every
parameter except the task head, and the head initialises fresh from the
template's ``output_classes`` (which is where the dataset's class count lands).

WHY DERIVE RATHER THAN HAND-LIST. 52 templates across 5 families name their head
differently — ``fc``, ``classifier``, ``decode_head.classifier``, a detection
head buried under ``head.cls_logits``. A hand-written list is 52 chances to be
subtly wrong, and wrong in the direction that matters: a head key left IN the
seed re-introduces the exact shape mismatch this contract exists to remove, and
it fails on the edge rather than here.

THE DERIVATION IS THE DEFINITION. A head key is precisely a key whose SHAPE is a
function of ``output_classes``. So build the template twice, with two different
class counts, and diff the state_dict shapes. That is family-agnostic, it cannot
disagree with the template it just read, and re-running it is how you check a
template has not drifted.

WHAT IT DELIBERATELY DOES NOT DO: guess. A template whose two builds differ in
their KEY SETS (not just shapes) is reported as INCONCLUSIVE rather than
resolved, because that means something other than the head moved and a
prefix list would be papering over it.
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
from typing import Dict, List, Optional, Tuple

# The seed verifier's isolation, for the same reason it has it: a warm hub cache
# makes an offline build succeed on a template that would abort on a closed edge.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

#: The two class counts to build with. Both are deliberately unusual: 1 and 2 are
#: special-cased by several heads (binary vs multiclass), and a value that
#: happens to equal the template's default would diff to nothing and read as
#: "no head".
PROBE_A, PROBE_B = 7, 13

#: Module-level constants that carry a class count. `num_feature_points` is the
#: keypoint family's, and it multiplies into the head width — so a template can
#: be head-coupled through either.
CLASS_CONSTANTS = ("output_classes", "num_feature_points")


def _rewrite_constants(src: str, value: int) -> Tuple[str, List[str]]:
    """Set every class-count constant in `src` to `value`.

    Word-boundary anchored at line start: a `.replace()` here would corrupt any
    identifier that merely CONTAINS the token, which is the model checker's
    `main_class` substring bug repeated.
    """
    touched = []
    for name in CLASS_CONSTANTS:
        pattern = rf"(?m)^{name}\s*=\s*\S+"
        if re.search(pattern, src):
            src = re.sub(pattern, f"{name} = {value}", src, count=1)
            touched.append(name)
    return src, touched


def _build_state_shapes(path: Path, value: int, tmp: Path) -> Optional[Dict[str, tuple]]:
    """`{key: shape}` for the template built with the class constants at `value`."""
    src = path.read_text(encoding="utf-8")
    rewritten, touched = _rewrite_constants(src, value)
    if not touched:
        return None  # no class constant at all -> cannot be head-coupled

    target = tmp / f"probe_{value}_{path.stem}.py"
    target.write_text(rewritten, encoding="utf-8")
    # Siblings the template may load relative to itself (tokenizer.json etc).
    for sibling in path.parent.glob(f"{path.stem}_*.json"):
        shutil.copy(sibling, tmp / sibling.name)

    spec = importlib.util.spec_from_file_location(f"probe_{value}_{path.stem}", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {target}")
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        os.chdir(tmp)  # templates resolve siblings relative to the process cwd
        spec.loader.exec_module(module)
        if hasattr(module, "MyModel"):
            model = module.MyModel()
        else:
            main_class = getattr(module, "main_class", None)
            if not (main_class and hasattr(module, main_class)):
                raise RuntimeError("no MyModel / main_class entry point")
            model = getattr(module, main_class)()
        return {k: tuple(v.shape) for k, v in model.state_dict().items()}
    finally:
        os.chdir(cwd)


def _collapse(keys: List[str]) -> List[str]:
    """The shortest prefixes covering exactly `keys`.

    `fc.weight` + `fc.bias` -> `fc.`; the engine and the prep tool both match on
    prefixes, so emitting leaves would work but would rot the moment a head grows
    a buffer.
    """
    prefixes = sorted({key.rsplit(".", 1)[0] + "." for key in keys})
    minimal: List[str] = []
    for prefix in prefixes:
        if not any(prefix != other and prefix.startswith(other) for other in prefixes):
            minimal.append(prefix)
    return minimal


def derive(path: Path) -> Dict:
    """`SEED_EXCLUDED_PREFIXES` for one template, or a reason it could not be found."""
    with tempfile.TemporaryDirectory(prefix="tb-derive-") as raw:
        tmp = Path(raw)
        try:
            a = _build_state_shapes(path, PROBE_A, tmp)
            if a is None:
                return {"status": "NO_CLASS_CONSTANT"}
            b = _build_state_shapes(path, PROBE_B, tmp)
        except Exception as exc:  # noqa: BLE001 — any build failure is a finding
            return {"status": "BUILD_FAIL", "error": f"{type(exc).__name__}: {exc}"[:300]}

    if b is None:
        return {"status": "NO_CLASS_CONSTANT"}
    if set(a) != set(b):
        # The key SETS moved, not just shapes: something structural depends on the
        # class count. A prefix list would hide that, so refuse to emit one.
        only_a = sorted(set(a) - set(b))[:5]
        only_b = sorted(set(b) - set(a))[:5]
        return {"status": "INCONCLUSIVE", "only_a": only_a, "only_b": only_b}

    varying = sorted(k for k in a if a[k] != b[k])
    if not varying:
        return {"status": "NO_HEAD", "prefixes": []}
    return {
        "status": "OK",
        "prefixes": _collapse(varying),
        "keys": varying,
        "tensors": len(a),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zoo", required=True, help="a model-zoo checkout")
    parser.add_argument("--only", nargs="*", help="template stems; default: all")
    parser.add_argument("--out", default=None, help="write the JSON verdict here")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip templates already present in --out (each entry is a model build)",
    )
    args = parser.parse_args(argv)

    zoo = Path(args.zoo).expanduser()
    root = zoo / "model_zoo" if (zoo / "model_zoo").is_dir() else zoo

    templates: List[Tuple[str, Path]] = []
    for category in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for path in sorted((root / category / "pytorch").glob("*.py")):
            if path.stem.startswith("_") or path.stem in {"test", "benchmark"}:
                continue
            if args.only and path.stem not in args.only:
                continue
            templates.append((category, path))

    if not templates:
        print("no templates matched — nothing was derived", file=sys.stderr)
        return 2

    # WRITTEN AFTER EVERY TEMPLATE, NOT AT THE END. Each entry costs a model
    # build, the full sweep is ~100 of them, and a run that is interrupted at 90%
    # and writes nothing is a run that has to start over -- which is exactly what
    # happened the first time this was used. `--resume` then makes a re-run cost
    # only what is actually missing.
    out_path = Path(args.out) if args.out else None
    results: Dict[str, Dict] = {}
    if out_path and args.resume and out_path.is_file():
        results = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"resuming: {len(results)} template(s) already derived", flush=True)

    for category, path in templates:
        # KEYED BY category/stem, NOT stem. `bert_base_uncased` ships in BOTH
        # text_classification and sentence_pair_classification; keying on the stem
        # silently kept one and dropped the other.
        key = f"{category}/{path.stem}"
        if key in results:
            continue
        record = derive(path)
        record["category"] = category
        results[key] = record
        if out_path:
            out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        detail = ",".join(record.get("prefixes", [])) or record["status"]
        print(f"{key:52s} {record['status']:18s} {detail}", flush=True)

    summary: Dict[str, int] = {}
    for record in results.values():
        summary[record["status"]] = summary.get(record["status"], 0) + 1
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\n=== " + ", ".join(f"{k}={v}" for k, v in sorted(summary.items())) + " ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
