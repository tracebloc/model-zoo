#!/usr/bin/env python3
"""Vendored contract copies, each held to the producer commit it pins.

Every file under ``tests/contracts/`` is a copy of something another repository
owns. Each copy is DERIVED from its producer by this module (``--write``),
never hand-edited, and is held two ways:

1. Offline, on every pull request (``tests/test_contract_copies_held.py``, in
   ci.yml's required ``category-contract``): the copy names its producer, the
   40-hex commit it was derived from, every source file's git blob sha there,
   and ``fields_sha256``, the digest of the contract fields as written. A hand
   edit of a contract field disagrees with that digest. No network.
2. Against the producer's same-named branch (``.github/workflows/
   contract-drift.yml``, weekly, required nowhere): the copies on this repo's
   develop, staging and main are re-derived against the producer's develop,
   staging and main, and a disagreement files one issue.

The ingestor's category copy is held by ``tools/ingest_category_contract.py``
and is registered here as such, so "every copy is held" is one listing.

Only the fields a consumer test reads are compared; the prose around them is
curated here, because this repository is public and the producers are not.
A verdict names files and fields, never a producer value, unless ``--show``.

Exit status: 0 agree, 1 drift, 2 refused (could not tell; never a pass).

    python3 tools/contract_copies.py                 # each copy vs its pinned commit
    python3 tools/contract_copies.py --ref develop   # each copy vs the producer's develop
    python3 tools/contract_copies.py --write --ref develop   # re-derive and re-pin

Stdlib only; needs ``gh`` logged in (or ``GH_TOKEN``) with read access to the
producers for every mode but ``--tree``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "tests" / "contracts"
OWNER = "tracebloc"

#: The producer branches a copy may be derived from; the drift run pairs each
#: with this repository's branch of the same name.
BRANCHES = ("develop", "staging", "main")

SHA40 = re.compile(r"[0-9a-f]{40}")

#: What the public copy carries in place of a reference into a private tracker.
#: tests/test_vendored_contract_carries_no_internal_refs.py's ISSUE_REF; held equal to it
#: by tests/test_contract_copies_held.py.
ISSUE_REF = re.compile(r"(?:[A-Za-z0-9._-]+/)?[A-Za-z0-9._-]*#\d+\b")
PLACEHOLDER = "(internal ref)"


class Refusal(Exception):
    """Could not establish the comparison (exit 2). Never a pass, never a skip."""


# ── derivations: producer source bytes -> contract fields ──────────────────


def _module(source: bytes, name: str) -> ast.Module:
    try:
        return ast.parse(source.decode("utf-8"), filename=name)
    except (SyntaxError, UnicodeDecodeError) as e:
        raise Refusal(f"refused: {name} does not parse: {e}")


def _str_list(values, what: str) -> list[str]:
    if not isinstance(values, list) or not values or not all(isinstance(v, str) and v for v in values):
        raise Refusal(f"refused: {what} is not a non-empty list of non-empty strings")
    if len(set(values)) != len(values):
        raise Refusal(f"refused: {what} lists a value more than once")
    return values


def derive_model_types(sources: dict[str, bytes]) -> dict:
    """``Experiment.MODEL_TYPE_CHOICES``: each ``(MODEL_TYPE_X, _("Label"))``
    with ``MODEL_TYPE_X`` resolved to the string the class assigns it."""
    ((name, body),) = sources.items()
    tree = _module(body, name)
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Experiment"):
        consts, choices = {}, None
        for node in cls.body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
                continue
            target, value = node.targets[0].id, node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                consts[target] = value.value
            elif target == "MODEL_TYPE_CHOICES":
                choices = value
        if not isinstance(choices, (ast.Tuple, ast.List)):
            raise Refusal(f"refused: {name} has no literal Experiment.MODEL_TYPE_CHOICES")
        out = []
        for item in choices.elts:
            try:
                const, label_call = item.elts
                label = label_call.args[0].value if isinstance(label_call, ast.Call) else label_call.value
                out.append({"constant": const.id, "value": consts[const.id], "label": label})
            except (AttributeError, ValueError, KeyError, IndexError) as e:
                raise Refusal(f"refused: a MODEL_TYPE_CHOICES entry in {name} is not (CONSTANT, _('label')): {e!r}")
        values = _str_list([c["value"] for c in out], f"{name} MODEL_TYPE_CHOICES")
        return {"choices": out, "accepted_model_type_values": sorted(values)}
    raise Refusal(f"refused: {name} defines no class Experiment")


def derive_upload_imports(sources: dict[str, bytes]) -> dict:
    """The literal tuple ``ALLOWED_PYTHON_PACKAGE_IMPORTS``, sorted."""
    ((name, body),) = sources.items()
    for node in _module(body, name).body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "ALLOWED_PYTHON_PACKAGE_IMPORTS" for t in node.targets)
        ):
            try:
                values = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError) as e:
                raise Refusal(f"refused: {name} ALLOWED_PYTHON_PACKAGE_IMPORTS is not a literal: {e!r}")
            if not isinstance(values, (tuple, list, set, frozenset)):
                raise Refusal(f"refused: {name} ALLOWED_PYTHON_PACKAGE_IMPORTS is not a tuple, list or set")
            values = list(values)
            return {"allowed_top_level_packages": sorted(_str_list(values, f"{name} ALLOWED_PYTHON_PACKAGE_IMPORTS"))}
    raise Refusal(f"refused: {name} assigns no ALLOWED_PYTHON_PACKAGE_IMPORTS")


#: The keys of an OD family the engine routes on. ``notes`` is prose.
OD_FAMILY_KEYS = ("family", "aliases", "training_supported", "handler")


def derive_od_families(sources: dict[str, bytes]) -> dict:
    """The routing fields of the engine's published OD families schema."""
    ((name, body),) = sources.items()
    try:
        doc = json.loads(body)
        families = [{k: f[k] for k in OD_FAMILY_KEYS} for f in doc["families"]]
        out = {
            "version": doc["version"],
            "families": families,
            "accepted_model_type_values": doc["accepted_model_type_values"],
            "not_accepted": {"examples": doc["not_accepted"]["examples"]},
        }
    except (ValueError, KeyError, TypeError) as e:
        raise Refusal(f"refused: {name} is not the OD families schema: {e!r}")
    _str_list(out["accepted_model_type_values"], f"{name} accepted_model_type_values")
    if not families:
        raise Refusal(f"refused: {name} publishes no families")
    return out


SKL_FAMILIES = ("classifier", "regression")
SKL_STRATEGIES = ("average_estimators", "_average_family_specific")


def _is_model_type(node: ast.AST) -> bool:
    """``model_type`` or ``<anything>.model_type``, on either side of a compare."""
    return getattr(node, "id", None) == "model_type" or getattr(node, "attr", None) == "model_type"


def derive_sklearn_types(sources: dict[str, bytes]) -> dict:
    """``model_type`` values the SDK's sklearn upload path routes, per family.

    The keys of each ``_ensemble_prefixes`` dict, with every string literal
    compared against ``model_type`` inside the averaging strategies. A module
    named ``*classifier.py`` / ``*regression.py`` feeds that family; any other
    (the shared base) feeds both.
    """
    acc: dict[str, set[str]] = defaultdict(set)
    for name, body in sorted(sources.items()):
        fams = [f for f in SKL_FAMILIES if name.endswith(f"{f}.py")] or list(SKL_FAMILIES)
        found = set()
        for node in ast.walk(_module(body, name)):
            if (
                isinstance(node, ast.Assign)
                and any(getattr(t, "attr", getattr(t, "id", None)) == "_ensemble_prefixes" for t in node.targets)
                and isinstance(node.value, ast.Dict)
            ):
                found |= {k.value for k in node.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if isinstance(node, ast.FunctionDef) and node.name in SKL_STRATEGIES:
                for cmp in (n for n in ast.walk(node) if isinstance(n, ast.Compare)):
                    sides = [cmp.left, *cmp.comparators]
                    if not any(_is_model_type(s) for s in sides):
                        continue
                    for c in sides:
                        for v in c.elts if isinstance(c, (ast.Tuple, ast.List, ast.Set)) else [c]:
                            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                                found.add(v.value)
        for f in fams:
            acc[f] |= found
    out = {f: sorted(acc[f]) for f in SKL_FAMILIES}
    for f, values in out.items():
        if not values:
            raise Refusal(f"refused: no sklearn {f} model_type found in {sorted(sources)}")
    return {"accepted_by_family": out}


# ── the registry ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Copy:
    file: str  # relative to CONTRACTS
    repo: str  # the producer, under OWNER
    paths: tuple[str, ...]  # the producer files the copy is derived from
    derive: Callable[[dict[str, bytes]], dict]
    method: str

    @property
    def fields(self) -> tuple[str, ...]:
        return FIELDS[self.derive]


FIELDS = {
    derive_model_types: ("choices", "accepted_model_type_values"),
    derive_upload_imports: ("allowed_top_level_packages",),
    derive_od_families: ("version", "families", "accepted_model_type_values", "not_accepted"),
    derive_sklearn_types: ("accepted_by_family",),
}

COPIES = (
    Copy(
        "tracebloc_backend/model_type_choices.v1.json",
        "backend",
        ("experiments/models/experiment.py",),
        derive_model_types,
        "ast.parse of Experiment.MODEL_TYPE_CHOICES, each MODEL_TYPE_* constant resolved to its literal",
    ),
    Copy(
        "tracebloc_backend/upload_import_allowlist.v1.json",
        "backend",
        ("bandit_tracebloc_model_validation/bandit_plugins/import_names.py",),
        derive_upload_imports,
        "ast.literal_eval of ALLOWED_PYTHON_PACKAGE_IMPORTS, sorted",
    ),
    Copy(
        "tracebloc_engine/object_detection_families.v2.json",
        "tracebloc-engine",
        ("core/schema/object_detection_families.v2.json",),
        derive_od_families,
        "the schema's version, families (routing keys), accepted_model_type_values and not_accepted.examples",
    ),
    Copy(
        "tracebloc_sdk/sklearn_model_types.v1.json",
        "tracebloc-py-package",
        tuple(f"tracebloc/upload/skl_tabular_{m}.py" for m in ("base", "classifier", "regression")),
        derive_sklearn_types,
        "ast.parse: _ensemble_prefixes keys plus literals compared with model_type in the averaging strategies",
    ),
)

#: Copies another tool derives and holds, by the tool that does.
HELD_ELSEWHERE = {
    "tracebloc_ingestor/ingest_categories.v1.json": "tools/ingest_category_contract.py",
}


def copy_named(file: str) -> Copy:
    for c in COPIES:
        if c.file == file:
            return c
    raise Refusal(f"refused: {file} is not a registered copy")


# ── the copy's side ────────────────────────────────────────────────────────


def blob_sha(data: bytes) -> str:
    """The git blob sha of ``data``, as ``git hash-object`` computes it."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fields_of(doc: dict, copy: Copy) -> dict:
    """The contract fields of a copy, in the shape its derivation produces."""
    try:
        out = {k: doc[k] for k in copy.fields}
    except (KeyError, TypeError) as e:
        raise Refusal(f"refused: {copy.file} has no contract field {e}")
    if copy.derive is derive_od_families:
        try:
            out["families"] = [{k: f[k] for k in OD_FAMILY_KEYS} for f in out["families"]]
            out["not_accepted"] = {"examples": out["not_accepted"]["examples"]}
        except (KeyError, TypeError) as e:
            raise Refusal(f"refused: {copy.file} families/not_accepted are malformed: {e!r}")
    return out


def fields_digest(fields: dict) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_doc(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refusal(f"refused: {path.name} is absent or not JSON: {e}")
    if not isinstance(doc, dict):
        raise Refusal(f"refused: {path.name} is not a JSON object")
    return doc


def pin_findings(doc: dict, copy: Copy) -> list[str]:
    """What is wrong with a copy's pin, offline. Empty means it is held."""
    g = doc.get("generated_from")
    if not isinstance(g, dict):
        return ["generated_from is missing"]
    out = []
    if g.get("repo") != f"{OWNER}/{copy.repo}":
        out.append(f"generated_from.repo is not {OWNER}/{copy.repo}")
    if g.get("paths") != list(copy.paths):
        out.append(f"generated_from.paths is not {list(copy.paths)}")
    if not (isinstance(g.get("ref"), str) and SHA40.fullmatch(g["ref"])):
        out.append("generated_from.ref is not a 40-hex commit")
    if g.get("ref_branch") not in BRANCHES:
        out.append(f"generated_from.ref_branch is not one of {list(BRANCHES)}")
    blobs = g.get("blobs")
    if not (isinstance(blobs, dict) and set(blobs) == set(copy.paths)
            and all(isinstance(v, str) and SHA40.fullmatch(v) for v in blobs.values())):
        out.append("generated_from.blobs does not give a 40-hex blob sha for every path")
    if g.get("fields_sha256") != fields_digest(fields_of(doc, copy)):
        out.append(f"the contract fields {list(copy.fields)} are not the ones --write derived (fields_sha256)")
    return out


# ── the producer's side ────────────────────────────────────────────────────


def _token() -> str | None:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _get(url: str, accept: str, token: str | None, opener=urllib.request.urlopen) -> bytes:
    headers = {"Accept": accept, "User-Agent": "model-zoo-contract-copies"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with opener(urllib.request.Request(url, headers=headers), timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        raise Refusal(f"refused: HTTP {e.code} reading {url.split('?')[0]} -- could not look")
    except (urllib.error.URLError, OSError) as e:
        raise Refusal(f"refused: {e} reading {url.split('?')[0]} -- could not look")


def resolve(repo: str, ref: str, token: str | None) -> str:
    sha = _get(f"https://api.github.com/repos/{OWNER}/{repo}/commits/{urllib.parse.quote(ref)}",
               "application/vnd.github.sha", token).decode().strip()
    if not SHA40.fullmatch(sha):
        raise Refusal(f"refused: {repo}@{ref} did not resolve to a commit")
    return sha


def fetch_http(copy: Copy, sha: str, token: str | None) -> dict[str, bytes]:
    return {
        p: _get(f"https://api.github.com/repos/{OWNER}/{copy.repo}/contents/{p}?ref={sha}",
                "application/vnd.github.raw", token)
        for p in copy.paths
    }


def fetch_tree(copy: Copy, root: Path) -> dict[str, bytes]:
    """The copy's sources from a (sparse) checkout of its producer."""
    out = {}
    for p in copy.paths:
        path = root / p
        if path.is_symlink() or not path.is_file():
            raise Refusal(f"refused: {copy.repo}:{p} is not a file in the checkout -- nothing to compare")
        out[p] = path.read_bytes()
    return out


# ── compare ────────────────────────────────────────────────────────────────


def compare(derived: dict, ours: dict) -> list[str]:
    """The contract fields that differ, by name. Empty means agreement."""
    return [k for k in derived if derived[k] != ours.get(k)]


def check(copy: Copy, doc: dict, sources: dict[str, bytes], show: bool = False) -> list[str]:
    derived = copy.derive(sources)
    differ = compare(derived, fields_of(doc, copy))
    findings = [f"{copy.file}: {k} differs from the producer" for k in differ]
    if show:
        findings += [f"  producer {k} = {json.dumps(derived[k])}" for k in differ]
    return findings


# ── write ──────────────────────────────────────────────────────────────────


def scrub(value):
    """Every string with a private tracker reference replaced by PLACEHOLDER."""
    if isinstance(value, str):
        return ISSUE_REF.sub(PLACEHOLDER, value)
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    return value


def render(copy: Copy, doc: dict, sources: dict[str, bytes], sha: str, branch: str, date: str) -> str:
    """``doc`` with its contract fields and pin re-derived; its prose kept."""
    derived = copy.derive(sources)
    doc = dict(doc)
    for k, v in derived.items():
        if copy.derive is derive_od_families and k == "families":
            notes = {f.get("family"): f.get("notes") for f in doc.get("families", []) if isinstance(f, dict)}
            upstream = {f["family"]: f.get("notes", "") for f in json.loads(sources[copy.paths[0]])["families"]}
            v = [{**f, "notes": notes.get(f["family"]) or scrub(upstream[f["family"]])} for f in v]
        elif copy.derive is derive_od_families and k == "not_accepted":
            v = {**doc.get("not_accepted", {}), **v}
        doc[k] = v
    doc["generated_from"] = {
        "repo": f"{OWNER}/{copy.repo}",
        "paths": list(copy.paths),
        "ref": sha,
        "ref_branch": branch,
        "ref_date": date,
        "blobs": {p: blob_sha(sources[p]) for p in copy.paths},
        "fields_sha256": fields_digest(fields_of(doc, copy)),
        "method": copy.method + ". Written by tools/contract_copies.py --write; never hand-edit.",
    }
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


# ── CLI ────────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", help="the producer branch to read (default: the commit each copy pins)")
    ap.add_argument("--copies", type=Path, default=CONTRACTS, help="the contracts directory to check")
    ap.add_argument("--tree", action="append", default=[], metavar="REPO=DIR",
                    help="read REPO from a checkout at DIR instead of the network (repeatable)")
    ap.add_argument("--only", action="append", default=[], metavar="FILE", help="check only this copy")
    ap.add_argument("--write", action="store_true", help="re-derive and re-pin from --ref (a branch)")
    ap.add_argument("--show", action="store_true", help="print the producer's value of a drifted field")
    args = ap.parse_args(argv)

    if args.write and (args.ref not in BRANCHES or args.tree):
        raise Refusal(f"refused: --write derives over the network from a producer branch {list(BRANCHES)}")
    trees = {}
    for t in args.tree:
        repo, sep, d = t.partition("=")
        if not sep:
            raise Refusal(f"refused: --tree takes REPO=DIR, not {t!r}")
        trees[repo] = Path(d)
    copies = [copy_named(f) for f in args.only] if args.only else list(COPIES)
    token = None if all(c.repo in trees for c in copies) else _token()

    findings: list[str] = []
    for copy in copies:
        path = args.copies / copy.file
        doc = load_doc(path)
        if copy.repo in trees:
            sources, sha = fetch_tree(copy, trees[copy.repo]), None
        else:
            ref = args.ref or (doc.get("generated_from") or {}).get("ref")
            if not (isinstance(ref, str) and ref):
                raise Refusal(f"refused: {copy.file} pins no producer commit; pass --ref")
            sha = resolve(copy.repo, ref, token)
            sources = fetch_http(copy, sha, token)
            if not args.ref:
                blobs = (doc.get("generated_from") or {}).get("blobs") or {}
                findings += [f"{copy.file}: {p} at the pinned commit is not the blob it pins"
                             for p in copy.paths if blobs.get(p) != blob_sha(sources[p])]
        if args.write:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path.write_text(render(copy, doc, sources, sha, args.ref, date), encoding="utf-8")
            print(f"wrote {copy.file} from {OWNER}/{copy.repo}@{sha[:12]}")
            continue
        findings += check(copy, doc, sources, show=args.show)

    for f in findings:
        print(f"DRIFT: {f}" if not f.startswith("  ") else f, file=sys.stderr)
    if not args.write:
        print("every copy agrees with its producer" if not findings else f"{len(findings)} finding(s); see above")
    return 1 if findings else 0


def run(argv=None) -> int:
    try:
        return main(argv)
    except Refusal as refusal:
        print(refusal, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(run())
