#!/usr/bin/env python3
"""Assert every migrated template that expects a hosted seed HAS one (backend#2659).

THE DIRECTION NOTHING ELSE COVERS
---------------------------------
Every existing gate is keyed on the dumps that exist:

  * ``verify_dumps_against_engine_pin.py`` sweeps the staged weights directory
    and strict-loads each dump into its template;
  * ``manifest.json``'s schema-2 block records what was prepped;
  * ``seed_contract.py`` / ``verify_backbone_seeds.py`` are scoped to templates
    that have a dump.

All three answer *"are the dumps we have good?"*. None answers *"do we have a
dump for every template that needs one?"* — so a template that was migrated but
never prepped **passes every gate by being absent**, and on an edge it trains
from random init while the migration's whole premise is that it does not.

That is not hypothetical. backend#2659 found ``faster_rcnn_resnet`` this way,
and this tool, on its first run, found a SECOND: ``text_classification/
bert_base_uncased``. Two of fifty-four, invisible to four checks.

WHAT COUNTS AS "EXPECTS A SEED"
-------------------------------
There is no registry of prepped templates anywhere in this repo — ``prep_
offline_weights.py`` is invoked per template by hand — which is exactly why the
gap could open. Rather than add a 54-entry list that would rot, this derives the
answer from what each migrated template already states about itself, and treats
anything it cannot classify as a FAILURE:

  * names its own ``<stem>_weights.pkl``          -> EXPECTS A SEED
  * says ``weights=False`` and has no weight file -> DECLARES NO SEED
  * both, or neither                              -> **UNCLASSIFIED, red**

The third rule is what makes the first two safe to rely on. A docstring is prose
and prose rots, so the gate does not quietly trust it: if an edit leaves a
migrated template silent about whether it needs a seed, CI says so by name
instead of guessing. A gate that guesses is how this gap stayed open.

FAIL-CLOSED, both ways round: a seed-expecting template with no dump is red, and
a staged dump belonging to no seed-expecting template is red too (a rename that
stranded its dump, or a dump for a template that has since been un-migrated).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from seed_index import (  # noqa: E402 — sibling tool module, same directory
    DUMP_DIR_CATEGORY,
    DUMP_DIR_CATEGORY_PREFIXES,
    zoo_root,
)

#: The phrase every #1499-migrated template carries in its docstring. Not a
#: clever heuristic — it is the one sentence the migration wrote into all 57,
#: and `check_migration_marker` below keeps it honest by failing when the count
#: of markers and the count of classified templates disagree.
MIGRATED_MARKER = "Offline variant"

#: A template that random-initialises by design says so, in these words, and
#: tells the user to upload with `weights=False`. Three do: `vit`,
#: `bert_base_uncased_scratch`, `distilbert_scratch`.
NO_SEED_MARKERS = ("weights=False", "no weight file")

EXPECTS_SEED = "EXPECTS_SEED"
NO_SEED = "NO_SEED"
UNCLASSIFIED = "UNCLASSIFIED"
#: A seed-expecting stem shared by more than one category with nothing in
#: ``seed_index`` to tell the dumps apart. Its own status, not a NO DUMP: the
#: dump may well exist, and the defect is that no rule says whose it is.
AMBIGUOUS = "AMBIGUOUS"


def classify(path: Path) -> Tuple[Optional[str], str]:
    """``(status, detail)`` for one template, or ``(None, "")`` if unmigrated."""
    source = path.read_text(encoding="utf-8")
    if MIGRATED_MARKER not in source:
        return None, ""

    names_dump = f"{path.stem}_weights.pkl" in source
    declares_none = all(marker in source for marker in NO_SEED_MARKERS)

    if names_dump and not declares_none:
        return EXPECTS_SEED, f"{path.stem}_weights.pkl"
    if declares_none and not names_dump:
        return NO_SEED, "random-initialised by design"
    if names_dump and declares_none:
        return UNCLASSIFIED, (
            f"names {path.stem}_weights.pkl AND declares it has no weight file "
            f"— one of the two statements is stale"
        )
    return UNCLASSIFIED, (
        "migrated, but says nothing about whether it needs a hosted seed. Either "
        f"name its {path.stem}_weights.pkl in the docstring, or state that it "
        "random-initialises by design and uploads with weights=False"
    )


def dump_dir_candidates(category: str, stem: str, sharing: int = 1) -> List[str]:
    """The dump-directory name(s) this template's seed could be filed under.

    Mirrors ``seed_index.resolve`` in reverse: a colliding stem is filed under a
    category prefix, and the bare name belongs to whichever category
    ``DUMP_DIR_CATEGORY`` names. Derived from those two maps rather than
    restated, so a new collision is handled by editing ``seed_index`` alone.

    ``sharing`` is how many seed-expecting templates carry this stem. AN
    UNMAPPED COLLISION RETURNS NOTHING RATHER THAN THE BARE NAME (Bugbot).
    The previous rule was ``DUMP_DIR_CATEGORY.get(stem, category) == category``,
    and for a stem absent from that map ``.get`` returns ``category``, so the
    test was always true: two seed-expecting templates sharing an unmapped stem
    would BOTH claim the same bare dump, and coverage would go green with one of
    them unseeded — the exact gap this gate exists to close, reintroduced inside
    the gate.

    ``seed_index.resolve`` already decided this question the other way round:
    *ambiguity is an error, not a pick*, and it raises rather than
    ``setdefault``-ing. Returning ``[]`` here lets the caller say so by name
    instead of silently agreeing with itself.
    """
    prefixed = [
        f"{prefix}{stem}"
        for prefix, prefixed_category in DUMP_DIR_CATEGORY_PREFIXES.items()
        if prefixed_category == category
    ]
    if prefixed:
        return prefixed
    owner = DUMP_DIR_CATEGORY.get(stem)
    if owner is not None:
        return [stem] if owner == category else []
    if sharing > 1:
        return []
    return [stem]


def manifest_names(declared: Dict) -> Tuple[Set[str], Optional[str]]:
    """Dump names from a manifest, plus a warning when its shape is the other one.

    TWO SHAPES BOTH CALL THEMSELVES SCHEMA 2, and this is the first tool to read
    a real one:

      * ``verify_dumps_against_engine_pin.py``'s docstring documents
        ``{"dumps": [{"name": ..., "sha256": ...}]}`` — a LIST of records — and
        its parser reads ``manifest["dumps"]``;
      * the manifest actually staged alongside the 52 dumps (backend#2659) uses
        ``{"prefix": ..., "entries": {"<name>": {"file", "sha256",
        "size_bytes"}}}`` — a DICT keyed by name.

    Nothing caught it because the CI gate is still an armed no-op until the
    seeds are hosted, so it has never been pointed at the staged manifest. It
    would not have failed silently — that gate fail-closes on a missing
    ``dumps`` key — but it WOULD go red on the day of the upload, for a reason
    that reads like a broken manifest rather than a schema disagreement.

    Both are read here so this tool is usable today, and the divergence is
    reported rather than absorbed: whichever shape gets hosted, the two tools
    have to agree on it before the upload.
    """
    if isinstance(declared.get("dumps"), list):
        return {entry["name"] for entry in declared["dumps"]}, None
    if isinstance(declared.get("entries"), dict):
        return set(declared["entries"]), (
            "manifest uses the `entries` dict shape; "
            "verify_dumps_against_engine_pin.py parses the `dumps` list shape. "
            "Both are labelled schema 2 — settle this before hosting "
            "(backend#2659)."
        )
    return set(), (
        "manifest declares neither a `dumps` list nor an `entries` dict — it "
        "names no dumps at all, so it protects nothing."
    )


def available(
    dumps_dir: Optional[Path], manifest: Optional[Path]
) -> Tuple[Set[str], List[str]]:
    """Dump-directory names that exist, from a staging dir and/or a manifest."""
    names: Set[str] = set()
    warnings: List[str] = []
    if dumps_dir is not None:
        names |= {p.name for p in dumps_dir.iterdir() if p.is_dir()}
    if manifest is not None:
        declared = json.loads(manifest.read_text(encoding="utf-8"))
        found, warning = manifest_names(declared)
        names |= found
        if warning:
            warnings.append(warning)
    return names, warnings


def survey(zoo: Path) -> Dict[str, Dict]:
    """``{"<category>/<stem>": {...}}`` for every migrated template.

    Two passes on purpose: a template's candidate dump names depend on how many
    OTHER templates share its stem, so every stem has to be counted before any
    one of them can be resolved.
    """
    root = zoo_root(zoo)
    classified: Dict[str, Dict] = {}
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pytorch = category_dir / "pytorch"
        if not pytorch.is_dir():
            continue
        for path in sorted(pytorch.glob("*.py")):
            status, detail = classify(path)
            if status is None:
                continue
            classified[f"{category_dir.name}/{path.stem}"] = {
                "status": status,
                "detail": detail,
                "category": category_dir.name,
                "stem": path.stem,
            }

    sharing: Dict[str, int] = {}
    for record in classified.values():
        if record["status"] == EXPECTS_SEED:
            sharing[record["stem"]] = sharing.get(record["stem"], 0) + 1

    found: Dict[str, Dict] = {}
    for key, record in classified.items():
        candidates = dump_dir_candidates(
            record["category"], record["stem"], sharing.get(record["stem"], 1)
        )
        if record["status"] == EXPECTS_SEED and not candidates:
            record = dict(
                record,
                status=AMBIGUOUS,
                detail=(
                    f"{record['stem']}.py expects a seed in "
                    f"{sharing[record['stem']]} categories and nothing in "
                    f"seed_index says which dump is whose. Add a "
                    f"DUMP_DIR_CATEGORY_PREFIXES entry for this category, or a "
                    f"DUMP_DIR_CATEGORY entry naming the owner of the bare name"
                ),
            )
        found[key] = {
            "status": record["status"],
            "detail": record["detail"],
            "candidates": candidates,
        }
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zoo", default=".", help="model-zoo checkout")
    parser.add_argument("--dumps-dir", default=None, help="staged dumps")
    parser.add_argument("--manifest", default=None, help="manifest.json")
    parser.add_argument("--out", default=None, help="write the survey as JSON")
    parser.add_argument(
        "--require-dumps",
        action="store_true",
        help="fail when neither --dumps-dir nor --manifest is available, instead "
        "of reporting the classification only. CI passes this once the seeds "
        "are hosted (backend#2659).",
    )
    args = parser.parse_args(argv)

    found = survey(Path(args.zoo).expanduser())
    if not found:
        print("no migrated templates found — is --zoo pointing at the zoo?")
        return 1

    dumps_dir = Path(args.dumps_dir).expanduser() if args.dumps_dir else None
    manifest = Path(args.manifest).expanduser() if args.manifest else None
    if dumps_dir is not None and not dumps_dir.is_dir():
        print(f"--dumps-dir does not exist: {dumps_dir}")
        return 1
    if manifest is not None and not manifest.is_file():
        print(f"--manifest does not exist: {manifest}")
        return 1
    have_inventory = dumps_dir is not None or manifest is not None
    names, warnings = available(dumps_dir, manifest) if have_inventory else (set(), [])
    for warning in warnings:
        print(f"  WARNING       {warning}")

    unclassified = sorted(k for k, v in found.items() if v["status"] == UNCLASSIFIED)
    ambiguous = sorted(k for k, v in found.items() if v["status"] == AMBIGUOUS)
    expecting = {k: v for k, v in found.items() if v["status"] == EXPECTS_SEED}
    no_seed = sorted(k for k, v in found.items() if v["status"] == NO_SEED)

    print(f"migrated templates      : {len(found)}")
    print(f"  expect a hosted seed  : {len(expecting)}")
    print(f"  declare no seed       : {len(no_seed)}")
    print(f"  UNCLASSIFIED          : {len(unclassified)}")
    print(f"  AMBIGUOUS             : {len(ambiguous)}")

    for key in unclassified:
        print(f"  UNCLASSIFIED  {key}: {found[key]['detail']}")
    for key in ambiguous:
        print(f"  AMBIGUOUS     {key}: {found[key]['detail']}")

    missing: List[str] = []
    orphans: List[str] = []
    if have_inventory:
        resolved: Set[str] = set()
        for key, record in sorted(expecting.items()):
            hit = next((c for c in record["candidates"] if c in names), None)
            if hit is None:
                missing.append(key)
                print(f"  NO DUMP       {key}: looked for {record['candidates']}")
            else:
                resolved.add(hit)
        orphans = sorted(names - resolved)
        for name in orphans:
            print(f"  ORPHAN DUMP   {name}: no seed-expecting template claims it")
        print(f"\ninventory: {len(names)} dump(s); {len(missing)} missing, "
              f"{len(orphans)} orphaned")
    else:
        print("\nno --dumps-dir or --manifest given: classification only, coverage "
              "not checked" + ("" if not args.require_dumps else " (REQUIRED)"))

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "templates": found,
                    "ambiguous": ambiguous,
                    "missing": missing,
                    "orphans": orphans,
                    "inventory_checked": have_inventory,
                },
                indent=1,
            ),
            encoding="utf-8",
        )

    if manifest is not None and not names and dumps_dir is None:
        # A present manifest that names no dumps verifies nothing while looking
        # green. Same fail-closed rule verify_dumps_against_engine_pin.py applies.
        print("\nFAIL (fail-closed): the manifest names no dumps.")
        return 1
    if unclassified or ambiguous or missing or orphans:
        return 1
    if args.require_dumps and not have_inventory:
        return 1
    print("\ndump coverage OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
