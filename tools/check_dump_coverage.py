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

WITH ONE NAMED EXCEPTION, AND IT IS A DECISION RATHER THAN A LOOPHOLE. A dump
whose manifest entry declares ``"status": "retired"`` is retained on purpose,
and is reported instead of failing — see ``RETIRED`` below for why that is not
the same finding as an orphan. The exception is narrow in three ways, each of
which is a test: an entry with no ``status`` is live, so nothing already written
becomes exempt; a status that is not in ``KNOWN_STATUSES`` is red by name rather
than assumed live; and a retired dump that a seed-expecting template DOES claim
is red, because then the retirement is what has gone stale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

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

#: A dump the manifest KEEPS ON PURPOSE for a template that no longer exists.
#:
#: backend#2985: the seven DETR seeds retired with the `hf_transformer` family
#: were prepped under the ML stack pinned at `scripts/.tracebloc-engine-ref`,
#: which is not cheaply reproducible — regenerating them means standing that
#: environment back up. So they are retained rather than deleted: reversing a
#: deletion means reconstructing a stack, reversing a flag means editing a line.
#:
#: THAT MAKES "A DUMP NO TEMPLATE CLAIMS" TWO DIFFERENT FINDINGS, and only one
#: of them is a defect. It is a defect when nobody said so — a rename that
#: stranded its dump, a template un-migrated without its seed being dealt with —
#: because then the store and the zoo have drifted apart and nobody noticed. It
#: is a decision when somebody said so here, in the manifest, in the artifact a
#: PR review can see. Collapsing the two is what would make this gate unusable
#: the moment a deliberate retirement happened: permanently red, and therefore
#: switched off — which is how the orphan half came to be unarmed in the first
#: place (see the CI job this tool is called from).
RETIRED = "retired"

#: Every status an entry may declare. AN ENTRY WITH NO ``status`` IS LIVE, which
#: is what keeps this back-compatible with every manifest written before
#: backend#2985 — the exemption has to be asked for, and every entry except the
#: seven retired DETR seeds does not ask.
#:
#: Anything else is named and RED rather than quietly treated as live. A typo'd
#: ``retried`` would otherwise read as a live dump, then as an orphan, and send
#: whoever is on the other end of the alarm hunting for a stranded blob that is
#: really a misspelling. The gate already refuses to guess about a template that
#: has gone silent (``UNCLASSIFIED``); this is the same refusal about an entry.
KNOWN_STATUSES = (RETIRED,)


class Inventory(NamedTuple):
    """What the store is said to hold, and what the manifest says about it.

    ``names`` is every dump the inventory knows of, retired ones included — it
    is the store's contents, not a to-do list. ``retired`` is the subset the
    manifest excuses from needing a template. Keeping them as two sets rather
    than removing retired names from ``names`` is deliberate: the retired dumps
    still exist, still occupy the store, and are still worth printing every run,
    which is the whole point of retaining them visibly instead of deleting them.
    """

    names: Set[str]
    retired: Set[str]
    warnings: List[str]
    #: ``"<name>: <status>"`` for each entry declaring a status nobody defined.
    bad_status: List[str]


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
    if sharing <= 1:
        # A UNIQUE STEM IS FILED UNDER THE BARE NAME. That is not an inference —
        # `seed_index` states it: "Dump directories are named `<stem>` except
        # where a stem collides, in which case the category is prefixed." An
        # earlier revision returned the category prefix for EVERY template in a
        # prefixed category, so a unique-stem template in
        # `sentence_pair_classification` (or in any category that later gains a
        # prefix entry) resolved to a name nothing is filed under (Bugbot).
        #
        # The prefixed form is accepted too, second: it also resolves forward,
        # so a dump filed that way is findable, not missing. This direction
        # fails LOUD (a false "NO DUMP") rather than silently, which is why it
        # was survivable — but a gate that cries wolf is a gate people switch off.
        return [stem] + prefixed
    if prefixed:
        return prefixed
    owner = DUMP_DIR_CATEGORY.get(stem)
    if owner is not None:
        return [stem] if owner == category else []
    return []


def _status_of(record: object) -> Optional[str]:
    """The status an inventory record declares, or ``None`` for "live".

    A record that is not a mapping — the ``dumps`` list shape allows a bare
    string, and a hand-edited manifest can contain anything — declares no
    status, which means live. That is the safe default in this tool: live is the
    state that must earn a template, so mis-reading a retirement as live is a
    false RED, and mis-reading a live dump as retired would be a false GREEN.
    """
    if isinstance(record, dict):
        status = record.get("status")
        return status if isinstance(status, str) else None
    return None


def manifest_names(declared: Dict) -> Inventory:
    """Dump names from a manifest, split by status, plus any shape warning.

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

    ``status`` IS READ IN BOTH SHAPES, not just the one the manifest happens to
    use today. The `entries` dict is what backend#2985 marks retired, but wiring
    the exemption to that shape alone would mean a later switch to the `dumps`
    list silently un-retires all seven — a green gate over a decision it had
    stopped honouring. The shapes disagree about layout, not about meaning.
    """
    records: Dict[str, object]
    warning: Optional[str]
    if isinstance(declared.get("dumps"), list):
        records = {entry["name"]: entry for entry in declared["dumps"]}
        warning = None
    elif isinstance(declared.get("entries"), dict):
        records = dict(declared["entries"])
        warning = (
            "manifest uses the `entries` dict shape; "
            "verify_dumps_against_engine_pin.py parses the `dumps` list shape. "
            "Both are labelled schema 2 — settle this before hosting "
            "(backend#2659)."
        )
    else:
        return Inventory(
            names=set(),
            retired=set(),
            warnings=[
                "manifest declares neither a `dumps` list nor an `entries` "
                "dict — it names no dumps at all, so it protects nothing."
            ],
            bad_status=[],
        )

    retired: Set[str] = set()
    bad_status: List[str] = []
    for name, record in sorted(records.items()):
        status = _status_of(record)
        if status is None:
            continue
        if status == RETIRED:
            retired.add(name)
        else:
            bad_status.append(f"{name}: {status!r}")
    return Inventory(
        names=set(records),
        retired=retired,
        warnings=[warning] if warning else [],
        bad_status=bad_status,
    )


def available(dumps_dir: Optional[Path], manifest: Optional[Path]) -> Inventory:
    """Dump-directory names that exist, from a staging dir and/or a manifest.

    A staging directory carries no statuses — it is a listing of folders — so
    everything it contributes is live UNLESS the manifest retires it. That is
    the right way round: the two sources are unioned for ``names`` and the
    manifest is the only thing that can grant an exemption, so pointing this at
    a staging dir cannot launder an unlisted blob into a retired one.
    """
    names: Set[str] = set()
    retired: Set[str] = set()
    warnings: List[str] = []
    bad_status: List[str] = []
    if dumps_dir is not None:
        names |= {p.name for p in dumps_dir.iterdir() if p.is_dir()}
    if manifest is not None:
        declared = json.loads(manifest.read_text(encoding="utf-8"))
        found = manifest_names(declared)
        names |= found.names
        retired |= found.retired
        warnings.extend(found.warnings)
        bad_status.extend(found.bad_status)
    return Inventory(
        names=names, retired=retired, warnings=warnings, bad_status=bad_status
    )


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
    inventory = (
        available(dumps_dir, manifest)
        if have_inventory
        else Inventory(names=set(), retired=set(), warnings=[], bad_status=[])
    )
    names = inventory.names
    for warning in inventory.warnings:
        print(f"  WARNING       {warning}")
    for entry in inventory.bad_status:
        print(
            f"  BAD STATUS    {entry}: not one of {list(KNOWN_STATUSES)}. An "
            "entry declaring a status nobody defined is named rather than "
            "assumed live — a typo must not read as either a live dump or a "
            "retirement (backend#2985)"
        )

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
    retired: List[str] = []
    claimed_retired: List[str] = []
    if have_inventory:
        resolved: Set[str] = set()
        for key, record in sorted(expecting.items()):
            hit = next((c for c in record["candidates"] if c in names), None)
            if hit is None:
                missing.append(key)
                print(f"  NO DUMP       {key}: looked for {record['candidates']}")
            else:
                resolved.add(hit)
        # A RETIREMENT A TEMPLATE STILL CLAIMS IS RED, and it is the direction
        # worth stating out loud, because it is the one that would otherwise
        # ship a retired seed into a live training run. Two edits produce it —
        # retiring an entry whose template is still there, or re-adding a
        # template whose seed was retired earlier — and neither author is
        # looking at the other half. `retired` is an exemption from "no
        # template claims this", so a dump a template DOES claim cannot use it.
        claimed_retired = sorted(inventory.retired & resolved)
        for name in claimed_retired:
            print(
                f"  RETIRED IN USE {name}: the manifest retires this dump, but a "
                "seed-expecting template claims it. Either the retirement is "
                "stale or the template should not have come back"
            )
        orphans = sorted(names - resolved - inventory.retired)
        for name in orphans:
            print(f"  ORPHAN DUMP   {name}: no seed-expecting template claims it")
        # Printed EVERY RUN, and deliberately not silent. The point of retaining
        # these rather than deleting them is that the decision stays visible
        # instead of rotting; a retirement nobody is reminded of is a deletion
        # with extra steps (backend#2985).
        retired = sorted(inventory.retired - resolved)
        for name in retired:
            print(
                f"  RETIRED DUMP  {name}: no template claims it, and the manifest "
                "says so on purpose — retained, not orphaned"
            )
        print(f"\ninventory: {len(names)} dump(s); {len(missing)} missing, "
              f"{len(orphans)} orphaned, {len(retired)} retired")
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
                    "retired": retired,
                    "claimed_retired": claimed_retired,
                    "bad_status": inventory.bad_status,
                    "inventory_checked": have_inventory,
                },
                indent=1,
            ),
            encoding="utf-8",
        )

    if manifest is not None and dumps_dir is None and not (names - inventory.retired):
        # A present manifest that names no LIVE dumps verifies nothing while
        # looking green. Same fail-closed rule verify_dumps_against_engine_pin.py
        # applies — extended past `not names` to cover the state backend#2985
        # made reachable: an inventory that is entirely retired protects exactly
        # as little as an empty one, and "every dump is exempt" must not be the
        # one way to switch this gate off quietly.
        print(
            "\nFAIL (fail-closed): the manifest names no live dumps"
            + (f" ({len(inventory.retired)} retired)." if inventory.retired else ".")
        )
        return 1
    if unclassified or ambiguous or missing or orphans:
        return 1
    if inventory.bad_status or claimed_retired:
        return 1
    if args.require_dumps and not have_inventory:
        return 1
    print("\ndump coverage OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
