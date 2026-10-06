#!/usr/bin/env python3
"""Sweep every staged weight dump against the engine's pinned dependency set
(an internal ticket prevention half).

Why this exists
---------------
A prepped ``<base>_weights.pkl`` state_dict's key layout is decided by the
*transformers version that built the module tree*, and the engine loads seed
weights with ``load_state_dict(strict=True)``. So a dump produced under a
different version than the engine pins is a hard training abort, discoverable
only on the edge. ``prep_offline_weights.py`` cannot catch this: it builds the
prep and ship modules in one interpreter, so its strict-load check certifies
internal consistency, never agreement with the engine.

This tool closes that gap. Run it in an interpreter pinned to the engine's
stack (``tools/requirements-engine-pin.txt``, itself a guarded mirror of the
engine's ``use_cases/requirements.txt`` — see the CI workflow
``verify-dumps-engine-pin.yml``). For each LIVE manifest entry it:

  * resolves the entry to the ONE shipped template it belongs to (the entry's
    name is its dump-directory name, resolved by ``seed_index.resolve`` — the
    same resolution every other seed tool uses — honouring the entry's recorded
    ``category`` when it carries one);
  * builds that template in THIS interpreter — i.e. what the edge builds — and
    loads the staged dump into it under the backbone-seed contract the edge's
    cycle-0 seed load applies (strict, except that keys under the template's
    ``SEED_EXCLUDED_PREFIXES`` may be missing), categorising the result OK /
    KEY_MISMATCH / BUILD_FAIL exactly as the edge would experience it;
  * refuses a dump that still CARRIES a declared head key (HEAD_PRESENT,
    model-zoo#135): such a seed loads only at the class count it was built
    at, so every other ``output_classes`` is refused on a size mismatch. A
    template with no ``SEED_EXCLUDED_PREFIXES`` is checked strict, as before;
    and
  * checks provenance: the ``built_with`` block describing the entry must match
    the versions actually installed here (the engine's pin). A drift — e.g. the
    engine bumps ``transformers`` — turns the gate red loudly instead of
    silently stranding every hosted seed.

ONE SCHEMA: the canonical manifest's ``entries``
------------------------------------------------
The manifest lives in ``backend`` under ``tools/offline_weights`` and reaches
CI as the ``dump-manifest`` artifact. It is the shape
``tools/sync_zoo_weights.py`` writes and its ``fetch-all`` reads:

    {
      "schema": 2,
      "prefix": "zoo-weights",
      "built_with": {"torch": "...", "torchvision": "...",
                     "transformers": "...", "timm": "...", "peft": "..."},
      "entries": {
        "<dump-dir name>": {
          "file": "<name>_weights.pkl",   # fetched FLAT into --dumps-dir
          "sha256": "...",
          "size_bytes": 123,
          "built_with": {...},            # optional: this entry's OWN stack
          "status": "retired",            # optional: absent means live
          "category": "<zoo category>"    # optional: disambiguates a stem
        }
      }
    }

This tool used to read a different shape — a ``dumps`` LIST of ``{name,
template, weights, sha256}`` records — that nothing ever wrote: the fetch hook
writes and reads ``entries``, and so does backend. So no manifest could take the
job green (the schema reconciliation, (internal ref)). The ``dumps`` shape is
now refused BY NAME rather than read, so a manifest in it cannot be mistaken
for an empty or stub one; one schema, not two.

Per-entry semantics, each one the backend gates' (``check_provenance``,
``verify_dumps``) so the two repos cannot disagree about one manifest:

  * ``built_with`` on an entry REPLACES the shared block for that entry (never
    merges). Present-but-empty is refused (exit 2): it declines the shared
    block and records nothing, so nothing describes that dump.
  * ``"status": "retired"`` entries are NOT fetched, NOT built and NOT counted
    as failures; they are listed on a ``NOT GATED`` line every run. A status
    nobody defined is red by name (``check_dump_coverage.KNOWN_STATUSES``). A
    manifest whose every entry is retired verifies nothing and exits 2 — it
    must never read as a green sweep.
  * ``category``, when present, selects between templates that share a stem
    across categories; a recorded category the dump's own name contradicts is
    refused.

Fail-closed contract
---------------------
The process exits non-zero on ANY of: a live dump that does not load under the
seed contract, a live dump that still carries its declared head, a
``built_with`` value that disagrees with the installed engine pin, a live entry
whose bytes are absent, a sha256 that does not match, a live entry that maps to
no template (or to more than one), or a manifest that is not the canonical
shape. A dependency/build error is a hard error, never a swallowed green.

Two green-without-a-sweep cases, both deliberate and both loud:

  * No ``manifest.json`` at all: nothing declares a dump. Exit 0, or red with
    ``--require-manifest``. In CI that is now the exception — the workflow
    puts the canonical manifest here.
  * ``--dumps-not-fetched REASON``: the workflow could not fetch the dumps for a
    NAMED precondition (no store URI, no read role, a fork). The manifest is
    still parsed and every live entry is still resolved to its template — a
    structural defect is red here too — but nothing is strict-loaded, provenance
    is not compared (the engine pin is not installed on that path), and the
    output says ``SKIP (dumps not fetched)`` with the reason. It is not a pass.

Dumps are NOT committed to this repo (see ``prep_offline_weights.py``: they are
served from the tracebloc model store). CI fetches them into ``--dumps-dir``
before invoking this tool; see the workflow's fetch step.
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


def _siblings():
    """``(check_dump_coverage, seed_index)`` — imported when a manifest is read.

    NOT at module scope, on purpose: ``check_engine_pin_drift.py`` imports this
    module for ``_PROVENANCE_KEYS`` alone, and is staged with only this file
    beside it (tests/test_engine_pin_drift_guard.py). Both siblings are
    stdlib-only, so the not-fetched path still runs under the runner's bare
    python3.
    """
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import check_dump_coverage
    import seed_index

    return check_dump_coverage, seed_index


# Package versions to reconcile against manifest["built_with"]. The engine pins
# every one of these; transformers is the load-bearing one (it restructures HF
# module trees between releases), but a drift in any is a dump-invalidating
# change, so all are checked. torchvision is included because it likewise
# determines the key layout of torchvision-detector/keypoint templates and the
# engine mirror pins it.
_PROVENANCE_KEYS = ("torch", "torchvision", "transformers", "timm", "peft")

OK = "OK"
KEY_MISMATCH = "KEY_MISMATCH"
BUILD_FAIL = "BUILD_FAIL"
MISSING = "MISSING"
SHA_MISMATCH = "SHA_MISMATCH"
# A live entry that maps to no template, or to more than one. Its own verdict,
# not a BUILD_FAIL: nothing was built, and the fix is in the manifest or the
# zoo's naming, not in the template.
NO_TEMPLATE = "NO_TEMPLATE"
# A live entry missing `file`/`sha256`, or naming a file that is not a bare
# filename (the fetch hook lays dumps out FLAT in --dumps-dir, so a path here
# could only point outside it).
MALFORMED = "MALFORMED"
# A `status` nobody defined. Red by name, never assumed live or retired.
BAD_STATUS = "BAD_STATUS"
# Not a failure: `"status": "retired"`. Not fetched, not built, not verified —
# reported on a NOT GATED line so the exemption stays visible.
RETIRED_ENTRY = "RETIRED"
# Not a failure: the template's random-init fp32 construction exceeds a standard
# ubuntu-latest runner's RAM, so we skip the BUILD (not the dump's existence/sha)
# rather than let one oversized template OOM and take the whole sweep — and every
# dump that WOULD have verified — down with it. Reported loudly, never folded into
# OK, so the coverage gap is visible.
SKIPPED_RAM = "SKIPPED_RAM"
# The dump still carries keys its template declares in SEED_EXCLUDED_PREFIXES
# (model-zoo#135). The template promises a backbone-only seed, so the head must
# have been stripped (`seed_contract.py strip`) before publishing; a dump that
# kept it fits ONE class count, and any other `output_classes` is refused with a
# size mismatch. It loads cleanly at the default count, which is why the old
# strict-load gate certified it, so it is its own verdict rather than OK.
HEAD_PRESENT = "HEAD_PRESENT"
# Not a failure, and not a pass: --dumps-not-fetched. The entry resolved to its
# template; nothing was loaded.
NOT_FETCHED = "NOT_FETCHED"

#: Verdicts that do not redden the gate. Everything else does.
_NOT_FAILURES = (OK, SKIPPED_RAM, RETIRED_ENTRY, NOT_FETCHED)

# Templates too large to construct in CI RAM. Kept in lockstep with
# tests/test_model_contract.py:_TOO_LARGE_FOR_CI_RAM (the source of truth for the
# instantiation suite) — see the verify-tool test that pins them equal. Keyed on
# the path relative to model_zoo/ (directory-scoped, never a bare basename: 19
# basenames are duplicated across task dirs, so a basename key would skip the
# wrong files).
_TOO_LARGE_FOR_CI_RAM = {
    "text_classification/pytorch/gemma_2.py",
}


class ManifestError(ValueError):
    """The manifest cannot be read as the canonical schema — exit 2."""


def _ci_ram_skip_key(template: str) -> str | None:
    """Return the matched _TOO_LARGE_FOR_CI_RAM entry for a template path
    (which may or may not carry a leading model_zoo/), else None. Matches on
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


def load_entries(manifest: dict) -> dict:
    """The manifest's ``entries`` dict, or ``ManifestError`` naming why not.

    Every refusal here is a manifest that cannot be read as the canonical
    schema, and each says which — a ``dumps`` list is not a stub, and a stub is
    not a schema mismatch, so the advice differs.
    """
    if "dumps" in manifest:
        raise ManifestError(
            "it carries a 'dumps' list. That shape was this tool's own and "
            "nothing ever wrote it; the canonical manifest (backend "
            "tools/offline_weights/manifest.json, and what "
            "tools/sync_zoo_weights.py writes and fetches from) is keyed "
            "'entries': {name: {file, sha256, size_bytes, ...}}. Rewrite it in "
            "that shape — this tool reads one schema, not two"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ManifestError(
            "its 'entries' dict is missing or empty — it is a stub, and it "
            "protects nothing"
        )
    for name, meta in entries.items():
        if isinstance(meta, dict) and "built_with" in meta:
            own = meta["built_with"]
            if not isinstance(own, dict) or not own:
                # Same refusal as backend check_provenance: an entry that
                # declines the shared block and records nothing is described by
                # nothing at all.
                raise ManifestError(
                    f"entry {name!r} carries a built_with that is {own!r} — "
                    "present but empty. It is not covered by the shared block "
                    "and not by itself either. Remove the key to inherit the "
                    "shared block, or record the stack"
                )
    return entries


def resolve_built_with(manifest: dict, meta: dict) -> tuple[dict, bool]:
    """``(the provenance describing this entry, is it the entry's own?)``.

    An entry's own block REPLACES the shared one — the backend's
    ``check_provenance.resolve_built_with`` rule, deliberately not a merge: a
    dump built on a different day was built by a whole different stack, and a
    merge would describe a stack nobody ran.
    """
    own = meta.get("built_with")
    if own:
        return dict(own), True
    return dict(manifest.get("built_with") or {}), False


def plan(manifest: dict, repo_root: Path) -> list[dict]:
    """One planned result per manifest entry, before anything is loaded.

    Retired, malformed, unknown-status and unresolvable entries are FINAL here
    (they carry a verdict already); a live, well-formed, resolved entry carries
    ``template`` and ``weights`` and no verdict yet. Needs only the stdlib and
    a directory listing, so it runs on the not-fetched path too.
    """
    coverage, seeds = _siblings()
    entries = load_entries(manifest)
    index = seeds.build_index(repo_root)
    planned: list[dict] = []
    for name, meta in sorted(entries.items()):
        result: dict = {"name": name}
        planned.append(result)
        if not isinstance(meta, dict):
            result["category"] = MALFORMED
            result["detail"] = f"entry is {type(meta).__name__}, not a mapping"
            continue
        status = coverage._status_of(meta)
        if status == coverage.RETIRED:
            result["category"] = RETIRED_ENTRY
            result["detail"] = "status: retired — not fetched, not verified"
            continue
        if status is not None:
            result["category"] = BAD_STATUS
            result["detail"] = (
                f"status {status!r} is not one of {list(coverage.KNOWN_STATUSES)}; a typo "
                "must not read as either a live dump or a retirement"
            )
            continue
        weights, sha = meta.get("file"), meta.get("sha256")
        if not weights or not sha:
            result["category"] = MALFORMED
            result["detail"] = "entry is missing 'file' or 'sha256'"
            continue
        if Path(weights).name != weights:
            result["category"] = MALFORMED
            result["detail"] = (
                f"file {weights!r} is not a bare filename; dumps are fetched "
                "flat into the dumps directory"
            )
            continue
        result["weights"] = weights
        result["sha256"] = sha
        try:
            zoo_category, path = seeds.resolve(index, name, meta.get("category"))
        except seeds.AmbiguousTemplate as exc:
            result["category"] = NO_TEMPLATE
            result["detail"] = str(exc)
            continue
        result["zoo_category"] = zoo_category
        result["template"] = path.relative_to(repo_root).as_posix()
        built_with, is_own = resolve_built_with(manifest, meta)
        result["built_with"] = built_with
        result["built_with_is_own"] = is_own
    return planned


def declared_head(template_path: Path) -> tuple[str, ...]:
    """The template's ``SEED_EXCLUDED_PREFIXES``, read with ``ast``.

    Normalised the way the engine's cycle-0 seed load normalises
    it: an empty prefix matches every key with ``startswith``, so it is dropped
    rather than honoured — otherwise ``("",)`` would excuse every missing key.
    """
    _, seeds = _siblings()
    return tuple(p for p in (seeds.read_prefixes(Path(template_path)) or ()) if p)


def seed_contract_problem(model, state_dict: dict, head: tuple[str, ...]) -> str | None:
    """The engine's cycle-0 seed load, or why it would refuse (None == loads).

    The same three rules as ``verify_backbone_seeds.check`` and the engine:
    an unexpected key is fatal, a missing key is fatal unless it is under the
    declared head, and a shape mismatch raises. With no declared head this is
    exactly ``strict=True``.
    """
    loaded = model.load_state_dict(state_dict, strict=False)  # raises on shape
    if loaded.unexpected_keys:
        return (
            f"{len(loaded.unexpected_keys)} unexpected key(s), e.g. "
            f"{loaded.unexpected_keys[:3]} — this dump does not belong to this model"
        )
    undeclared = [k for k in loaded.missing_keys if not k.startswith(head)]
    if undeclared:
        return (
            f"{len(undeclared)} missing key(s) the template does not declare in "
            f"SEED_EXCLUDED_PREFIXES, e.g. {undeclared[:3]} — they would keep "
            "their random init"
        )
    return None


def _verify_one(entry: dict, dumps_dir: Path, repo_root: Path) -> dict:
    """Load-verify one PLANNED live entry. Returns it with a ``category``."""
    import torch

    result = dict(entry)
    weights_path = dumps_dir / entry["weights"]
    if not weights_path.exists():
        result["category"] = MISSING
        result["detail"] = f"dump bytes not found at {weights_path}"
        return result

    actual = _sha256(weights_path)
    if actual != entry["sha256"]:
        result["category"] = SHA_MISMATCH
        result["detail"] = f"sha256 {actual} != manifest {entry['sha256']}"
        return result

    if _ci_ram_skip_key(entry["template"]):
        # Skip the BUILD only — existence + sha above already ran. Constructing
        # this template's fp32 params would exceed CI RAM and OOM the whole job.
        result["category"] = SKIPPED_RAM
        result["detail"] = (
            "random-init construction exceeds CI runner RAM; build not attempted "
            "(see _TOO_LARGE_FOR_CI_RAM)"
        )
        return result

    try:
        model = _build_ship(str(repo_root / entry["template"]))
    except Exception as exc:  # noqa: BLE001 — any build failure is BUILD_FAIL
        result["category"] = BUILD_FAIL
        result["detail"] = f"{type(exc).__name__}: {exc}"
        return result

    try:
        state_dict = torch.load(weights_path, weights_only=True)
    except Exception as exc:  # noqa: BLE001 — an unreadable dump == edge abort
        result["category"] = KEY_MISMATCH
        result["detail"] = f"{type(exc).__name__}: {exc}"
        return result

    head = declared_head(repo_root / entry["template"])
    carried = sorted(k for k in state_dict if head and k.startswith(head))
    if carried:
        result["category"] = HEAD_PRESENT
        result["detail"] = (
            f"{len(carried)} key(s) under the template's SEED_EXCLUDED_PREFIXES "
            f"are still in the dump, e.g. {carried[:4]}. The seed fits only the "
            "class count it was built at; publish the `seed_contract.py strip` "
            "output instead"
        )
        return result

    try:
        problem = seed_contract_problem(model, state_dict, head)
    except Exception as exc:  # noqa: BLE001 — shape mismatch == edge abort
        result["category"] = KEY_MISMATCH
        result["detail"] = f"{type(exc).__name__}: {exc}"
        return result
    if problem:
        result["category"] = KEY_MISMATCH
        result["detail"] = problem
        return result

    result["category"] = OK
    result["tensors"] = len(state_dict)
    return result


def _provenance(manifest: dict, live: list[dict], installed: dict) -> list[str]:
    """Drift for the LIVE entries only (retired ones are not gated).

    The shared block is compared ONCE, and only if some live entry inherits it
    — printing the same rows per inheriting entry would bury the per-entry rows
    that matter. Entries with their own block are compared individually.
    """
    problems: list[str] = []
    inheriting = [r for r in live if not r["built_with_is_own"]]
    if inheriting:
        shared = manifest.get("built_with") or {}
        if not shared:
            problems.append(
                f"manifest has no shared 'built_with' block (schema 2 required), "
                f"and {len(inheriting)} live entr(ies) carry no block of their "
                "own — cannot prove those dumps were built against the engine's pin"
            )
        else:
            problems.extend(
                f"<shared built_with, {len(inheriting)} entr(ies)> {p}"
                for p in _check_provenance(shared, installed)
            )
    for r in live:
        if r["built_with_is_own"]:
            problems.extend(
                f"{r['name']} (own built_with) {p}"
                for p in _check_provenance(r["built_with"], installed)
            )
    return problems


def _summary(results: list[dict], transformers_version: str | None) -> str:
    counts: dict[str, int] = {}
    for r in results:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    lines = [f"=== SUMMARY (transformers {transformers_version}) ==="]
    for cat in (
        OK,
        KEY_MISMATCH,
        HEAD_PRESENT,
        BUILD_FAIL,
        MISSING,
        SHA_MISMATCH,
        NO_TEMPLATE,
        MALFORMED,
        BAD_STATUS,
        SKIPPED_RAM,
        NOT_FETCHED,
        RETIRED_ENTRY,
    ):
        if counts.get(cat):
            lines.append(f"  {cat:<14} {counts[cat]:>3}")
    return "\n".join(lines)


def _not_gated_line(results: list[dict]) -> str | None:
    retired = [r["name"] for r in results if r["category"] == RETIRED_ENTRY]
    if not retired:
        return None
    return (
        f"NOT GATED: {len(retired)} retired entr(ies), not fetched and not "
        f"verified: {', '.join(retired)}"
    )


def run_sweep(
    manifest_path: Path,
    dumps_dir: Path,
    repo_root: Path,
    report_path: Path,
    require_manifest: bool,
    check_provenance: bool,
    dumps_not_fetched: str | None = None,
) -> int:
    installed = {k: _installed_version(k) for k in _PROVENANCE_KEYS}

    if not manifest_path.exists():
        # The branch a checkout with no manifest takes. It used to be the branch
        # CI took on EVERY run, because nothing in the verify-dumps job put a
        # manifest here (the arming overclaim); the workflow now downloads the
        # canonical one, so in CI this is the exception rather than the rule.
        msg = (
            f"no manifest at {manifest_path}: no weight dumps are declared here, "
            "so there is nothing to verify. In CI this is not the expected path: "
            "the verify-dumps job downloads the canonical manifest (keyed "
            "'entries', from backend tools/offline_weights, via the dump-manifest "
            "artifact) to this path before this step runs — so reaching this "
            "branch there means that download did not land where this tool reads."
        )
        if require_manifest:
            print(f"FAIL (fail-closed): {msg}", file=sys.stderr)
            return 2
        print(f"OK (nothing to verify): {msg}")
        return 0

    manifest = json.loads(manifest_path.read_text())
    try:
        planned = plan(manifest, repo_root)
    except ManifestError as exc:
        print(
            f"FAIL (fail-closed): manifest {manifest_path} cannot be read: {exc}.",
            file=sys.stderr,
        )
        return 2

    pending = [r for r in planned if "category" not in r]
    if not pending and not any(r["category"] not in _NOT_FAILURES for r in planned):
        # Every entry is retired. A manifest like that verifies NOTHING, and a
        # sweep over zero dumps must not read as green (same rule as backend
        # verify_dumps: "a store of only retired dumps still exits 2").
        print(
            f"FAIL (fail-closed): manifest {manifest_path} declares no LIVE "
            f"entries. {_not_gated_line(planned)}. Nothing would be verified, "
            "and that is not a pass.",
            file=sys.stderr,
        )
        return 2

    provenance_problems: list[str] = []
    if dumps_not_fetched is not None:
        results = [dict(r, category=NOT_FETCHED) if "category" not in r else r for r in planned]
    else:
        if check_provenance:
            provenance_problems = _provenance(manifest, pending, installed)
        results = [
            _verify_one(r, dumps_dir, repo_root) if "category" not in r else r
            for r in planned
        ]

    report = {
        "engine_pin_installed": installed,
        "manifest_built_with": manifest.get("built_with", {}),
        "dumps_not_fetched": dumps_not_fetched,
        "provenance_problems": provenance_problems,
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(_summary(results, installed.get("transformers")))
    not_gated = _not_gated_line(results)
    if not_gated:
        print(not_gated)
    if provenance_problems:
        print("\n=== PROVENANCE DRIFT (built_with vs engine pin) ===")
        for p in provenance_problems:
            print(f"  {p}")
    for r in results:
        if r["category"] not in (OK, RETIRED_ENTRY, NOT_FETCHED):
            print(f"  {r['category']:<14} {r['name']}: {r.get('detail', '')}")

    # SKIPPED_RAM is a reported coverage gap, not a failure — it must not redden
    # the gate (an OOM would have verified nothing at all). RETIRED is a
    # decision recorded in the manifest, reported on the NOT GATED line.
    failed = [r for r in results if r["category"] not in _NOT_FAILURES]
    if failed or provenance_problems:
        print(
            f"\nFAIL: {len(failed)} entr(ies) not OK, "
            f"{len(provenance_problems)} provenance drift(s). "
            f"Report: {report_path}",
            file=sys.stderr,
        )
        return 1
    if dumps_not_fetched is not None:
        live = sum(1 for r in results if r["category"] == NOT_FETCHED)
        print(
            f"\nSKIP (dumps not fetched): {dumps_not_fetched}. The manifest is "
            f"well-formed and all {live} live entr(ies) resolve to a template, "
            "but NO dump was strict-loaded and provenance was not compared. "
            f"This is not a pass. Report: {report_path}"
        )
        return 0
    print(
        f"\nAll {len(pending)} live dump(s) verify against the engine pin. "
        f"Report: {report_path}"
    )
    return 0


_SELFTEST_TEMPLATE = (
    "from torch import nn\n"
    "main_class = 'MyModel'\n"
    "class MyModel(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc = nn.Linear(4, 3)\n"
)
_SELFTEST_BROKEN = (
    "main_class = 'MyModel'\n"
    "class MyModel:\n"
    "    def __init__(self):\n"
    "        raise RuntimeError('cannot build on this pin')\n"
)


def _selftest() -> int:
    """Self-contained proof the categorisation + provenance logic works, using
    a synthetic zoo, torch templates and an ``entries`` manifest — no
    transformers/timm/peft needed. Exercised by
    tests/test_verify_dumps_against_engine_pin.py and runnable as
    ``verify_dumps_against_engine_pin.py --selftest``."""
    import tempfile

    import torch
    from torch import nn

    class _Ref(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        zoo = root / "model_zoo" / "image_classification" / "pytorch"
        zoo.mkdir(parents=True)
        dumps = root / "dist"
        dumps.mkdir()
        for stem in ("good", "mismatch", "gone"):
            (zoo / f"{stem}.py").write_text(_SELFTEST_TEMPLATE)
        (zoo / "broken.py").write_text(_SELFTEST_BROKEN)
        headed = _SELFTEST_TEMPLATE.replace(
            "main_class", 'SEED_EXCLUDED_PREFIXES = ("fc.",)\nmain_class', 1
        )
        (zoo / "headed.py").write_text(headed)
        (zoo / "stripped.py").write_text(headed)

        def dump(name: str, state: dict) -> dict:
            path = dumps / f"{name}_weights.pkl"
            torch.save(state, path)
            return {"file": path.name, "sha256": _sha256(path), "size_bytes": 1}

        bad = _Ref().state_dict()
        del bad["fc.bias"]
        entries = {
            "good": dump("good", _Ref().state_dict()),
            "mismatch": dump("mismatch", bad),
            "broken": dump("broken", _Ref().state_dict()),
            "headed": dump("headed", _Ref().state_dict()),
            "stripped": dump("stripped", {}),
            "gone": {"file": "gone_weights.pkl", "sha256": "0" * 64, "size_bytes": 1},
            "old": {"file": "old_weights.pkl", "sha256": "0" * 64, "status": "retired"},
        }
        manifest = {
            "schema": 2,
            "built_with": {"torch": _installed_version("torch")},
            "entries": entries,
        }
        mpath = root / "manifest.json"
        mpath.write_text(json.dumps(manifest))

        rc = run_sweep(mpath, dumps, root, root / "report.json", False, False)
        report = json.loads((root / "report.json").read_text())
        cats = {r["name"]: r["category"] for r in report["results"]}
        assert cats == {
            "good": OK,
            "mismatch": KEY_MISMATCH,
            "broken": BUILD_FAIL,
            "headed": HEAD_PRESENT,
            "stripped": OK,
            "gone": MISSING,
            "old": RETIRED_ENTRY,
        }, cats
        assert rc == 1, "sweep with failures must exit non-zero"

        # Provenance drift must fail closed.
        only_good = {"good": entries["good"]}
        drift = dict(manifest, entries=only_good, built_with={"transformers": "9.9.9"})
        dpath = root / "manifest_drift.json"
        dpath.write_text(json.dumps(drift))
        rc_drift = run_sweep(dpath, dumps, root, root / "r2.json", False, True)
        assert rc_drift == 1, "provenance drift must exit non-zero"

        # Only retired entries: nothing verified, never green.
        rpath = root / "manifest_retired.json"
        rpath.write_text(json.dumps(dict(manifest, entries={"old": entries["old"]})))
        assert run_sweep(rpath, dumps, root, root / "r5.json", False, True) == 2

        # No manifest, not required → green-with-no-work.
        rc_none = run_sweep(root / "nope.json", dumps, root, root / "r3.json", False, True)
        assert rc_none == 0, "absent manifest (not required) must be green"
        # …but red when required.
        rc_req = run_sweep(root / "nope.json", dumps, root, root / "r4.json", True, True)
        assert rc_req == 2, "absent manifest with --require-manifest must be red"

    print(
        "selftest OK: OK/KEY_MISMATCH/HEAD_PRESENT/BUILD_FAIL/MISSING/RETIRED + provenance + "
        "fail-closed"
    )
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
        help="fail (red) if manifest.json is absent, instead of the green no-op",
    )
    ap.add_argument(
        "--dumps-not-fetched",
        metavar="REASON",
        default=None,
        help="the dumps could not be fetched, for this NAMED reason: parse and "
        "resolve the manifest (red on a structural defect) but load nothing, and "
        "say SKIP with the reason instead of claiming a sweep",
    )
    ap.add_argument("--no-check-provenance", dest="check_provenance", action="store_false")
    ap.add_argument("--selftest", action="store_true", help="run built-in synthetic tests and exit")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    if args.dumps_not_fetched is not None and not args.dumps_not_fetched.strip():
        print(
            "FAIL (fail-closed): --dumps-not-fetched needs a reason; an unnamed "
            "skip is the defect this flag exists to prevent.",
            file=sys.stderr,
        )
        return 2

    return run_sweep(
        Path(args.manifest),
        Path(args.dumps_dir),
        Path(args.repo_root),
        Path(args.report),
        args.require_manifest,
        args.check_provenance,
        args.dumps_not_fetched,
    )


if __name__ == "__main__":
    raise SystemExit(main())
