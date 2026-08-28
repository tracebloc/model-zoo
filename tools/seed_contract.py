#!/usr/bin/env python3
"""Apply backend#2642's backbone-only seed contract to the zoo and the dumps.

Two subcommands, in the order they must run:

  ``apply``  — write ``SEED_EXCLUDED_PREFIXES`` into each template that has a
               head, from ``derive_seed_excluded.py``'s verdict. The constant is
               the SSoT the contract names: the prep side strips exactly these
               keys, the engine allows exactly these to be missing, and neither
               restates the other.

  ``strip``  — turn the staged full dumps into backbone-only seeds by removing
               the keys each template declares. Derives from the EXISTING dumps
               rather than re-downloading from the hub: the staged 52 are already
               wrapper-shaped and 52/52 strict-load under the engine's pinned
               stack (backend#2659), so stripping is a pure subtraction on bytes
               that are already trusted.

WHY THE CONSTANT IS READ WITH ``ast`` AND NOT BY IMPORTING. Reading a module-level
tuple does not need a model, and every template here builds a multi-hundred-MB one
on import. Parsing also cannot execute a template as a side effect of asking it a
question, which matters when the thing being asked about is a security migration.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

from seed_index import (  # noqa: E402 — sibling tool module, same directory
    AmbiguousTemplate,
    CONSTANT,
    build_index,
    read_prefixes,
    resolve,
    zoo_root,
)

_DOC = f"""# backend#2642 — the task head is NOT carried by the hosted seed.
# The seed holds the backbone; the head initialises fresh from output_classes,
# which is where the dataset's class count lands. Derived mechanically by
# tools/derive_seed_excluded.py (build twice, diff the shapes) — regenerate it
# rather than editing by hand if this model's head changes.
"""


def _render(prefixes: List[str]) -> str:
    items = ", ".join(f'"{p}"' for p in prefixes)
    trailing = "," if len(prefixes) == 1 else ""
    return f"{_DOC}{CONSTANT} = ({items}{trailing})\n"


def _insert(src: str, block: str) -> str:
    """Place the constant after the imports, before the first other statement.

    Module-level constants in these templates sit above the model class, and the
    SDK's rewriter reads them from the module namespace — so anywhere top-level
    works mechanically. After the imports is where a reader looks.
    """
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    anchor = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            anchor = node.end_lineno or anchor
        elif anchor:
            break
    return "".join(lines[:anchor]) + "\n" + block + "".join(lines[anchor:])


def cmd_check(zoo: Path, verdict: Path) -> int:
    """Fail if any template's DECLARED head disagrees with the DERIVED one.

    THE DECLARATION IS A LITERAL, AND A LITERAL ROTS. ``apply`` runs once, at
    authoring time, and from then on 46 templates restate a derivation nobody
    re-runs. Rename ``fc.`` to ``head.``, add a layer, and the constant is
    quietly wrong -- after which a seed either carries head weights it should
    not or excludes keys that no longer exist. That is the failure this whole
    contract exists to remove, coming back through the declaration instead of
    the load (Lukas, model-zoo#217).

    So the derivation is the CHECK, not the authoring step. Two ways to be
    wrong, and both are failures rather than one being a warning:

    * declared != derived  -- the head moved and the constant did not.
    * has a head, declares nothing -- a new template that never got a constant
      would otherwise ship a head-carrying seed and fail on the first dataset
      whose class count differs, which is exactly backend#2660's five failures.
    """
    results = json.loads(verdict.read_text(encoding="utf-8"))
    root = zoo_root(zoo)
    checked = 0
    drifted: List[str] = []

    for key, record in sorted(results.items()):
        category, stem = key.split("/", 1)
        path = root / category / "pytorch" / f"{stem}.py"
        if not path.is_file():
            drifted.append(f"{key}: template not found at {path}")
            continue

        status = record.get("status")
        declared = read_prefixes(path)

        if status in {"NO_HEAD", "NO_CLASS_CONSTANT"}:
            # No head: declaring one would exclude keys the seed must carry.
            if declared:
                drifted.append(
                    f"{key}: declares {declared} but the derivation finds no "
                    f"class-dependent parameter ({status})"
                )
            checked += 1
            continue
        if status != "OK":
            drifted.append(f"{key}: derivation returned {status} — cannot be checked")
            continue

        derived = tuple(record["prefixes"])
        if declared is None:
            drifted.append(
                f"{key}: has a head {derived} but declares no {CONSTANT} — its "
                f"seed would carry head weights"
            )
        elif declared != derived:
            drifted.append(f"{key}: declares {declared}, derivation says {derived}")
        checked += 1

    print(f"{checked} template(s) checked, {len(drifted)} drifted")
    if drifted:
        print(
            f"\n{CONSTANT} disagrees with what the template actually builds:",
            file=sys.stderr,
        )
        for line in drifted:
            print(f"  {line}", file=sys.stderr)
        print(
            "\nRe-derive and re-apply rather than editing the constant by hand:\n"
            "  python3 tools/derive_seed_excluded.py --zoo . --out /tmp/seed.json\n"
            "  python3 tools/seed_contract.py --zoo . apply --verdict /tmp/seed.json",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_apply(zoo: Path, verdict: Path, dry_run: bool) -> int:
    results = json.loads(verdict.read_text(encoding="utf-8"))
    root = zoo_root(zoo)
    written = skipped = 0
    blocked: List[str] = []

    for key, record in sorted(results.items()):
        category, stem = key.split("/", 1)
        path = root / category / "pytorch" / f"{stem}.py"
        status = record.get("status")

        if status in {"NO_HEAD", "NO_CLASS_CONSTANT"}:
            skipped += 1
            continue
        if status != "OK":
            # INCONCLUSIVE / BUILD_FAIL are findings, not templates to patch. A
            # guessed prefix list on one of these is worse than no contract: it
            # would strip keys nobody proved were the head.
            blocked.append(f"{key}: {status}")
            continue
        if not path.is_file():
            blocked.append(f"{key}: template not found at {path}")
            continue

        existing = read_prefixes(path)
        wanted = tuple(record["prefixes"])
        if existing == wanted:
            skipped += 1
            continue
        if existing is not None:
            blocked.append(f"{key}: declares {existing}, derivation says {wanted}")
            continue

        src = path.read_text(encoding="utf-8")
        patched = _insert(src, _render(list(wanted)))
        # Parse what will be written, not what was intended.
        ast.parse(patched, filename=str(path))
        if not dry_run:
            path.write_text(patched, encoding="utf-8")
        written += 1
        print(f"{'would write' if dry_run else 'wrote'} {key:52s} {', '.join(wanted)}")

    print(f"\n{written} written, {skipped} skipped (no head / already correct)")
    if blocked:
        print(f"\n{len(blocked)} NOT patched — each needs a look:", file=sys.stderr)
        for line in blocked:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_strip(zoo: Path, weights: Path, dest: Path, dry_run: bool) -> int:
    import torch

    # ONE resolver, shared with verify_backbone_seeds — a stem is not unique
    # (`bert_base_uncased` ships in two categories) and `.setdefault()` used to
    # let one silently win (Bugbot, model-zoo#217).
    index = build_index(zoo)

    entries: Dict[str, Dict] = {}
    problems: List[str] = []
    dest.mkdir(parents=True, exist_ok=True)

    for directory in sorted(p.name for p in weights.iterdir() if p.is_dir()):
        dump = weights / directory / f"{directory}_weights.pkl"
        try:
            _, template = resolve(index, directory)
        except AmbiguousTemplate as exc:
            problems.append(str(exc))
            continue
        if not dump.is_file():
            problems.append(f"{directory}: no dump at {dump}")
            continue

        prefixes = read_prefixes(template)
        out_dir = dest / directory
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{directory}_weights.pkl"

        if not prefixes:
            # No head: the backbone-only seed IS the full dump. Copied rather
            # than symlinked so the store holds real bytes at a content address.
            if not dry_run:
                shutil.copy(dump, out)
            print(f"{directory:34s} no head — carried whole")
        else:
            state = torch.load(dump, weights_only=True, map_location="cpu")
            dropped = [k for k in state if k.startswith(tuple(prefixes))]
            if not dropped:
                # The template declares a head the dump does not contain. That is
                # a real disagreement, not a no-op: either the dump predates the
                # declaration or the prefixes are wrong.
                problems.append(
                    f"{directory}: declares {prefixes} but no dump key matches"
                )
                continue
            kept = {k: v for k, v in state.items() if k not in set(dropped)}
            if not dry_run:
                torch.save(kept, out)
            print(
                f"{directory:34s} dropped {len(dropped):3d} of {len(state):4d} "
                f"keys  ({', '.join(prefixes)})"
            )

        if not dry_run:
            entries[directory] = {
                "file": out.name,
                "sha256": _sha256(out),
                "size_bytes": out.stat().st_size,
                "seed_excluded_prefixes": list(prefixes or ()),
            }

    if not dry_run:
        (dest / "manifest_entries.json").write_text(
            json.dumps(entries, indent=1, sort_keys=True), encoding="utf-8"
        )
    print(f"\n{len(entries)} backbone-only seed(s) written to {dest}")
    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zoo", required=True, help="a model-zoo checkout")
    parser.add_argument("--dry-run", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("apply", help="write SEED_EXCLUDED_PREFIXES into templates")
    a.add_argument("--verdict", required=True, help="derive_seed_excluded.py --out")

    c = sub.add_parser("check", help="fail if a declared head != the derived one")
    c.add_argument("--verdict", required=True, help="derive_seed_excluded.py --out")

    s = sub.add_parser("strip", help="derive backbone-only dumps from staged ones")
    s.add_argument("--weights", required=True, help="staged full dumps")
    s.add_argument("--dest", required=True, help="where backbone-only seeds go")

    args = parser.parse_args(argv)
    zoo = Path(args.zoo).expanduser()
    if args.cmd == "check":
        return cmd_check(zoo, Path(args.verdict).expanduser())
    if args.cmd == "apply":
        return cmd_apply(zoo, Path(args.verdict).expanduser(), args.dry_run)
    return cmd_strip(
        zoo,
        Path(args.weights).expanduser(),
        Path(args.dest).expanduser(),
        args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
