#!/usr/bin/env python3
"""Sync zoo weight dumps between a local staging dir and the tracebloc model
store — the fetch hook `.github/workflows/verify-dumps-engine-pin.yml` calls
(tracebloc/backend#2658, hosting is tracebloc/backend#2659).

The zoo never ships weight files in git (see CLAUDE.md "Weight file
convention"); the SDK uploads whatever ``<base>_weights.pkl`` sibling sits next
to the template ``.py`` at ``upload_model(..., weights=True)`` time. This tool
is the staging step:

  * ``manifest``    — (re)build ``manifest.json`` over the staging dir: one
                      entry per template with sha256 + size. The manifest is
                      the source of truth for what the store should hold.
  * ``upload``      — push every dump + the manifest to the store.
                      Content-addressed layout so re-uploads are idempotent and
                      a corrupted object can never silently shadow a good one:
                        <STORE>/<template>/<sha256[:12]>/<template>_weights.pkl
                        <STORE>/manifest.json
  * ``fetch T``     — pull template T's dump from the store to --dest,
                      verifying sha256 against the manifest before it is
                      trusted.
  * ``fetch-all``   — the same for every template the manifest declares. This
                      is what CI runs to populate ``dist/`` before
                      ``verify_dumps_against_engine_pin.py`` sweeps it.
  * ``verify``      — sha256-check every staged dump against the manifest.

Store location
--------------
Deliberately NOT hardcoded — where the dumps live is the open decision on
tracebloc/backend#2659. Set it via:

    export TRACEBLOC_ZOO_WEIGHTS_URI="s3://<internal-bucket>/zoo-weights"

Transport is the ``aws`` CLI (no extra Python deps); any s3://-compatible URI
the CLI can reach works. Requires an already-authenticated session — this tool
neither reads nor writes credentials.

Staging location
----------------
Defaults to ``<repo>/dist`` (the same directory CI fetches into and the
verifier sweeps). Override per-run with ``--staging``, or per-environment with
``TRACEBLOC_ZOO_WEIGHTS_STAGING``. Nothing here assumes a path outside the
checkout.

Known gap, not addressed here
-----------------------------
The manifest this tool WRITES (``{"entries": {name: {file, sha256,
size_bytes}}}``) is not the shape ``verify_dumps_against_engine_pin.py`` READS
(``{"dumps": [{name, template, weights, sha256}]}``) — and ``_build_env``
records four of the five provenance keys that verifier reconciles (no
``torchvision``). Both are properties of the manifest's schema and provenance
block, which is the subject of tracebloc/backend#3059; reconciling them is that
ticket's call, so this file records the divergence rather than picking a side.
Until it is reconciled, ``manifest`` is a staging aid — CI's fetch path
(``fetch-all``) only needs ``entries``.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Repo-relative, never a developer's home: this file is version-controlled and
# runs on CI runners, so a default pointing at one machine's working tree would
# be wrong everywhere it actually executes.
STAGING_DEFAULT = os.environ.get("TRACEBLOC_ZOO_WEIGHTS_STAGING") or os.path.join(
    _REPO_ROOT, "dist"
)


def _store_uri() -> str:
    uri = os.environ.get("TRACEBLOC_ZOO_WEIGHTS_URI", "").rstrip("/")
    if not uri:
        sys.exit(
            "TRACEBLOC_ZOO_WEIGHTS_URI is not set. The store location is the "
            "open decision on tracebloc/backend#2659 — set it once confirmed, "
            'e.g.\n  export TRACEBLOC_ZOO_WEIGHTS_URI="s3://<internal-bucket>/zoo-weights"'
        )
    return uri


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(staging: str) -> dict:
    mpath = os.path.join(staging, "manifest.json")
    if not os.path.exists(mpath):
        sys.exit(f"no manifest at {mpath} — run `manifest` first")
    with open(mpath) as fh:
        return json.load(fh)


def _aws(*args: str) -> None:
    cmd = ["aws", *args]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.exit(f"aws command failed ({proc.returncode}): {' '.join(cmd)}")


def _build_env() -> dict:
    """Record the library versions the dumps were produced under.

    A dump's key layout is decided by the transformers version that built the
    module tree, and the engine loads it with ``strict=True`` — so a dump
    produced under a different version than the engine pins is a hard training
    abort (backend#2641). Stamping the version next to the sha256 is what makes
    that checkable later instead of discoverable only on the edge.
    """
    env = {}
    for name in ("torch", "transformers", "timm", "peft"):
        try:
            env[name] = __import__(name).__version__
        except Exception:
            env[name] = None
    return env


def cmd_manifest(staging: str) -> None:
    entries = {}
    for tpl in sorted(os.listdir(staging)):
        d = os.path.join(staging, tpl)
        if not os.path.isdir(d):
            continue
        dumps = [f for f in os.listdir(d) if f.endswith("_weights.pkl")]
        if not dumps:
            continue
        f = os.path.join(d, dumps[0])
        entries[tpl] = {
            "file": dumps[0],
            "sha256": _sha256(f),
            "size_bytes": os.path.getsize(f),
        }
        print(f"{tpl:35s} {entries[tpl]['size_bytes']/1e6:8.1f}MB")
    with open(os.path.join(staging, "manifest.json"), "w") as out:
        json.dump(
            {
                "schema": 2,
                "prefix": "zoo-weights",
                "built_with": _build_env(),
                "entries": entries,
            },
            out,
            indent=2,
            sort_keys=True,
        )
    print(f"manifest.json: {len(entries)} entries")


def cmd_verify(staging: str) -> None:
    manifest = _load_manifest(staging)
    bad = 0
    for tpl, meta in sorted(manifest["entries"].items()):
        f = os.path.join(staging, tpl, meta["file"])
        if not os.path.exists(f):
            print(f"MISSING  {tpl}")
            bad += 1
            continue
        ok = _sha256(f) == meta["sha256"]
        print(f"{'OK      ' if ok else 'MISMATCH'} {tpl}")
        bad += 0 if ok else 1
    sys.exit(1 if bad else 0)


def cmd_upload(staging: str) -> None:
    store = _store_uri()
    manifest = _load_manifest(staging)
    for tpl, meta in sorted(manifest["entries"].items()):
        src = os.path.join(staging, tpl, meta["file"])
        dst = f"{store}/{tpl}/{meta['sha256'][:12]}/{meta['file']}"
        print(f"upload {tpl} -> {dst}")
        _aws("s3", "cp", src, dst)
    _aws("s3", "cp", os.path.join(staging, "manifest.json"), f"{store}/manifest.json")
    print(f"uploaded {len(manifest['entries'])} dumps + manifest to {store}")


def _fetch_one(store: str, template: str, meta: dict, dest: str) -> None:
    """Download one dump and verify its sha256 before it is trusted.

    A dump that arrives corrupted is REMOVED, not left on disk: the verifier
    reports an absent dump as MISSING (fail-closed) but a present-and-wrong one
    could otherwise be picked up by anything that only checks existence.
    """
    src = f"{store}/{template}/{meta['sha256'][:12]}/{meta['file']}"
    out = os.path.join(dest, meta["file"])
    _aws("s3", "cp", src, out)
    got = _sha256(out)
    if got != meta["sha256"]:
        os.remove(out)
        sys.exit(f"sha256 mismatch for {template}: got {got[:16]}…, removed {out}")
    print(f"fetched + verified {out}")


def cmd_fetch(staging: str, template: str, dest: str) -> None:
    store = _store_uri()
    manifest = _load_manifest(staging)
    meta = manifest["entries"].get(template)
    if meta is None:
        sys.exit(f"unknown template {template!r} — not in manifest")
    os.makedirs(dest, exist_ok=True)
    _fetch_one(store, template, meta, dest)


def cmd_fetch_all(staging: str, dest: str) -> None:
    """Fetch every dump the manifest declares — CI's entry point.

    Fails on the FIRST bad object rather than collecting failures: a partial
    ``dist/`` is exactly what the verifier is built to catch (MISSING per dump),
    so continuing past a mismatch would trade one loud error for 50 quiet ones.
    """
    store = _store_uri()
    manifest = _load_manifest(staging)
    entries = manifest.get("entries") or {}
    if not entries:
        sys.exit(
            f"manifest at {os.path.join(staging, 'manifest.json')} declares no "
            "'entries' — nothing to fetch (a manifest that names no dumps "
            "protects nothing)"
        )
    os.makedirs(dest, exist_ok=True)
    for tpl, meta in sorted(entries.items()):
        _fetch_one(store, tpl, meta, dest)
    print(f"fetched + verified {len(entries)} dump(s) from {store} into {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--staging", default=STAGING_DEFAULT)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    sub.add_parser("verify")
    sub.add_parser("upload")
    f = sub.add_parser("fetch")
    f.add_argument("template")
    f.add_argument("--dest", default=".")
    fa = sub.add_parser("fetch-all")
    fa.add_argument("--dest", default=".")
    a = p.parse_args()
    if a.cmd == "manifest":
        cmd_manifest(a.staging)
    elif a.cmd == "verify":
        cmd_verify(a.staging)
    elif a.cmd == "upload":
        cmd_upload(a.staging)
    elif a.cmd == "fetch":
        cmd_fetch(a.staging, a.template, a.dest)
    elif a.cmd == "fetch-all":
        cmd_fetch_all(a.staging, a.dest)


if __name__ == "__main__":
    main()
