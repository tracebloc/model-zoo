#!/usr/bin/env python3
"""The zoo's category list, held to the ingestor's published category enum.

WHY THIS EXISTS
---------------
The platform publishes ONE list of task categories -- the ``category`` enum of
the ingestor's ingest schema, which is what an ingest is validated against --
and several repos keep their own copy of it. Nothing compared them. A sibling
repo once carried two categories no other part of the platform implemented, for
months, with every suite green: the list-level question "does anything here
disagree with the producer?" was not being asked anywhere.

This repo's copy is the set of directories under ``model_zoo/``. It is DERIVED
from the tree here, never restated -- adding a directory is what registers a
category, so there is no second list to forget.

THREE LAYERS, each a separate assertion so a failure names which one broke:

1. tree == vendored enum (``tests/test_zoo_category_contract.py``, offline, in
   every test job). Both directions reported separately: a published category
   with no zoo directory is a category shipped with no models; a zoo directory
   the producer does not publish is a zombie.
2. every model module's declared ``category = "..."`` equals the directory it
   sits in -- asserted INTO the directory set, read from the AST so a module
   that cannot be imported in this job is still checked.
3. vendored enum == the producer's enum on its ``develop`` (the marked test,
   run by ci.yml's ``category-contract`` job). A check that compares the tree
   against a stale vendored file agrees with itself; this layer is what stops
   the drift from moving one level up.

This module is the shared logic for all three plus the refresh that writes the
vendored file. ``tools/`` is not a package; tests load it by path, as they do
``prep_offline_weights.py``. Stdlib only.

THE COST, stated rather than discovered: adding a category becomes a change in
more than one repo that fails loudly until every repo agrees.

Refresh the vendored copy (needs ``gh`` logged in with read access to the
producer)::

    python3 tools/ingest_category_contract.py --write
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "model_zoo"
VENDORED = ROOT / "tests" / "contracts" / "tracebloc_ingestor" / "ingest_categories.v1.json"

#: The producer's repository, owner-qualified at run time. Named once.
PRODUCER_REPO = "data-ingestors"

#: The branch the vendored copy is held to.
PRODUCER_REF = "develop"

#: Where the ingest schema is published in the producer, NEWEST FIRST. The
#: producer moved its schema directory into a ``contracts/`` package; ``develop``
#: carries the new path while ``main`` still carries the old one. A single
#: hard-coded path would 404 the day a branch moves over, against a zoo tree
#: that did not change. Delete the old entry once every producer branch carries
#: the new layout.
CANDIDATE_PATHS = (
    "tracebloc_ingestor/contracts/schema/ingest.v1.json",
    "tracebloc_ingestor/schema/ingest.v1.json",
)

#: Where the category enum lives inside that document.
ENUM_POINTER = ("properties", "category", "enum")

#: HTTP statuses that mean "not at this candidate path" -- and ONLY these. Any
#: other failure is fatal from the candidate that raised it: a 401/403/5xx is
#: "could not look", which must never read as "absent, try the next one".
ABSENT_STATUSES = frozenset({404, 410})


class Refusal(SystemExit):
    """Could not establish the comparison. Never a pass, never a skip."""


# ── the zoo's side ─────────────────────────────────────────────────────────


def zoo_directories(model_root: Path = MODEL_ROOT) -> frozenset[str]:
    """Every category directory under ``model_zoo/``, derived by listing.

    A directory whose name starts with ``_`` or ``.`` (``__pycache__``, a hidden
    tool directory) is not a category; everything else is. Files beside the
    directories (``TOKENIZERS.md``) are not categories either.
    """
    return frozenset(
        p.name
        for p in model_root.iterdir()
        if p.is_dir() and not p.name.startswith(("_", "."))
    )


def _top_level_str(tree: ast.Module, name: str):
    """``(found, value)`` for a module-level ``name = <...>``.

    ``value`` is the string if the right-hand side is a string literal, else
    the AST node's type name -- so a computed ``category`` is a finding with a
    reason, not a silent miss.
    """
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        for t in targets:
            if isinstance(t, ast.Name) and t.id == name:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return True, value.value
                return True, f"<not a string literal: {type(value).__name__}>"
    return False, None


def declared_categories(model_root: Path = MODEL_ROOT) -> dict[str, tuple[str, str | None]]:
    """``{relative path: (directory, declared category or None)}`` for every
    MODEL module under ``model_root``.

    A model module is one that declares ``framework`` or ``category`` at module
    level -- the same line ``tests/test_model_contract.py`` draws between a
    template and a support file (``loss.py``, ``utils.py``). Read from the AST,
    not by import: the contract test skips a module whose framework is not
    installed in the current job, and a skip is not a check.
    """
    out: dict[str, tuple[str, str | None]] = {}
    for path in sorted(model_root.rglob("*.py")):
        rel = path.relative_to(model_root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        has_framework, _ = _top_level_str(tree, "framework")
        has_category, category = _top_level_str(tree, "category")
        if has_framework or has_category:
            out[rel.as_posix()] = (rel.parts[0], category)
    return out


# ── the producer's side ────────────────────────────────────────────────────


def categories_from(document: bytes, source: str) -> tuple[str, ...]:
    """The enum at ``ENUM_POINTER``, or a REFUSAL naming what was missing.

    An absent, empty, non-list or non-string enum is not an empty contract: it
    is a document that is not the one we think we are reading, and reporting
    agreement against it would pass for a reason unrelated to the lists
    agreeing. ``bool({})`` is False -- a malformed payload is exactly the shape
    that takes a permissive branch.
    """
    try:
        node = json.loads(document)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise Refusal(f"refused: {source} is not valid JSON: {e}")
    for i, key in enumerate(ENUM_POINTER):
        if not isinstance(node, dict) or key not in node:
            where = "/".join(ENUM_POINTER[: i + 1])
            raise Refusal(f"refused: {source} has no `{where}` -- not the ingest schema this gate reads")
        node = node[key]
    if not isinstance(node, list) or not node:
        raise Refusal(f"refused: {source} has an EMPTY or non-list category enum ({node!r})")
    bad = [c for c in node if not isinstance(c, str) or not c]
    if bad:
        raise Refusal(f"refused: {source} category enum has entries that are not non-empty strings: {bad!r}")
    return tuple(node)


def duplicates(values) -> list[str]:
    """Entries listed more than once. Set equality hides these."""
    return sorted(v for v, n in Counter(values).items() if n > 1)


def select_from_tree(root: Path) -> tuple[str, bytes]:
    """The newest candidate present under a checkout of the producer.

    ``root`` is a (sparse) checkout of the producer: a candidate path that is
    absent there is the checkout's 404 and selects the next one. A candidate
    that is present but is not a readable file is fatal FROM THAT CANDIDATE.
    All absent is a refusal naming every path tried.
    """
    if not root.is_dir():
        raise Refusal(f"refused: producer checkout {root} is not a directory -- nothing was fetched")
    for rel in CANDIDATE_PATHS:
        path = root / rel
        if not path.exists():
            continue
        if not path.is_file():
            raise Refusal(f"refused: candidate {rel} exists under {root} but is not a file")
        try:
            return rel, path.read_bytes()
        except OSError as e:
            raise Refusal(f"refused: candidate {rel} under {root} could not be read: {e}")
    raise Refusal(
        f"refused: none of the candidate schema paths exist under {root}: "
        + ", ".join(CANDIDATE_PATHS)
    )


def contents_url(owner: str, path: str, ref: str) -> str:
    return f"https://api.github.com/repos/{owner}/{PRODUCER_REPO}/contents/{path}?ref={ref}"


def fetch_http(owner: str, ref: str, token: str | None, opener=urllib.request.urlopen):
    """``(path, bytes)`` of the newest candidate the producer serves at ``ref``.

    A 404/410 selects the next candidate. Any other HTTP status, or a transport
    failure, is fatal naming the candidate that raised it. All absent is a
    refusal naming every path.
    """
    headers = {"Accept": "application/vnd.github.raw", "User-Agent": "model-zoo-category-gate"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for rel in CANDIDATE_PATHS:
        url = contents_url(owner, rel, ref)
        try:
            with opener(urllib.request.Request(url, headers=headers), timeout=30) as resp:
                return rel, resp.read()
        except urllib.error.HTTPError as e:
            if e.code in ABSENT_STATUSES:
                continue
            raise Refusal(f"refused: candidate {rel} at {ref}: HTTP {e.code} -- could not look")
        except (urllib.error.URLError, OSError) as e:
            raise Refusal(f"refused: candidate {rel} at {ref}: {e} -- could not look")
    raise Refusal(
        f"refused: the producer serves none of the candidate schema paths at {ref}: "
        + ", ".join(CANDIDATE_PATHS)
    )


# ── the vendored copy ──────────────────────────────────────────────────────


def load_vendored(path: Path = VENDORED) -> tuple[str, ...]:
    """The vendored enum, through the SAME refusals as the producer's."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    categories = doc.get("categories")
    wrapped = json.dumps({"properties": {"category": {"enum": categories}}}).encode()
    return categories_from(wrapped, str(path))


def compare(published, ours, published_name: str, ours_name: str) -> list[str]:
    """Findings, one line per direction, each named. Empty means agreement."""
    findings = []
    dup = duplicates(published)
    if dup:
        findings.append(f"{published_name} lists {dup} more than once")
    missing = sorted(set(published) - set(ours))
    extra = sorted(set(ours) - set(published))
    if missing:
        findings.append(f"only {published_name} has: {missing}")
    if extra:
        findings.append(f"only {ours_name} has: {extra}")
    return findings


# ── refresh ────────────────────────────────────────────────────────────────


def _gh_token() -> str | None:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolve_sha(owner: str, ref: str, token: str | None, opener=urllib.request.urlopen) -> str:
    url = f"https://api.github.com/repos/{owner}/{PRODUCER_REPO}/commits/{ref}"
    headers = {"Accept": "application/vnd.github.sha", "User-Agent": "model-zoo-category-gate"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with opener(urllib.request.Request(url, headers=headers), timeout=30) as resp:
            return resp.read().decode().strip()
    except (urllib.error.URLError, OSError) as e:
        raise Refusal(f"refused: could not resolve {ref} to a commit: {e}")


def render_vendored(categories, path: str, sha: str, date: str) -> str:
    doc = {
        "description": (
            "The category enum of the ingestor's published ingest schema, the list an ingest "
            "is validated against. DERIVED by tools/ingest_category_contract.py --write, never "
            "hand-edited. The zoo's category directories are asserted equal to it, and it is "
            "asserted equal to the producer's develop in CI, so a stale copy cannot pass."
        ),
        "version": 1,
        "generated_from": {
            "path": path,
            "pointer": "/" + "/".join(ENUM_POINTER),
            "ref": sha,
            "ref_branch": PRODUCER_REF,
            "ref_date": date,
        },
        "categories": list(categories),
    }
    return json.dumps(doc, indent=2) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--owner", default="tracebloc")
    ap.add_argument("--ref", default=PRODUCER_REF)
    ap.add_argument("--write", action="store_true", help="rewrite the vendored file from the producer")
    args = ap.parse_args(argv)

    token = _gh_token()
    sha = _resolve_sha(args.owner, args.ref, token)
    path, body = fetch_http(args.owner, sha, token)
    categories = categories_from(body, f"{path}@{sha[:12]}")
    if duplicates(categories):
        raise Refusal(f"refused: the producer lists {duplicates(categories)} more than once")
    if path != CANDIDATE_PATHS[0]:
        print(f"note: the producer still serves the OLD path {path} at {args.ref}", file=sys.stderr)

    findings = compare(categories, load_vendored(), f"the producer ({args.ref})", "the vendored copy")
    if not args.write:
        for f in findings:
            print(f"DRIFT: {f}", file=sys.stderr)
        print("vendored copy agrees with the producer" if not findings else "run with --write to refresh")
        return 1 if findings else 0

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    VENDORED.write_text(render_vendored(categories, path, sha, date), encoding="utf-8")
    print(f"wrote {VENDORED.relative_to(ROOT)} from {path}@{sha[:12]}")
    ours = zoo_directories()
    tree = compare(categories, ours, "the producer", "model_zoo/")
    for f in tree:
        print(f"now reconcile the tree: {f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
