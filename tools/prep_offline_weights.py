#!/usr/bin/env python3
"""Offline pretrained-weight prep + strict-load verifier (model-zoo #156).

Purpose
-------
Turn a hub-fetching model-zoo template into the two artifacts the offline
pattern needs:

  1. a wrapper-matched ``<base>_weights.pkl`` state_dict, produced ONCE by
     whoever runs this tool (with network access and any model license
     accepted), by instantiating the ORIGINAL (pre-migration) ``MyModel``
     that still downloads pretrained weights; and
  2. a proof that this dump loads ``strict=True`` into the REWRITTEN
     (offline) ``MyModel`` that the zoo actually ships.

The platform loads seed weights with ``load_state_dict(strict=True)``, and
the SDK's upload-time weight check does the same. So the ONLY thing that
makes a dump valid is that its keys and shapes match the rewritten
``MyModel()`` exactly. This tool builds both modules, dumps from the "prep"
(pretrained) one, and asserts a clean strict load into the "ship" (offline)
one before writing anything final.

Because the pretrained builder and the offline builder must produce the SAME
wrapper module (same attribute names, same head replacement, same
``num_classes``), the safest "prep" builder is literally the pre-migration
template: check it out from git history, point ``--prep`` at it, point
``--ship`` at the rewritten file, and let this tool prove they line up.

Class-count caveat
------------------
The dump contains the FULL state_dict — including the task head, sized to
the template's declared ``output_classes`` (those head tensors are freshly
initialized, not pretrained; the pretrained value lives in the backbone/
encoder). The platform's strict seed load requires them to be present and
shape-matched, so a prepped dump pairs with ONE class count: the one the
template declared when the tool ran. An experiment that overrides
``output_classes`` needs a dump prepped at that count — set the value in
both template copies and re-run this tool (seconds, no re-download once
cached). Head-tolerant seed loading, which would let one dump serve any
class count, is a platform-side change tracked with the rest of the
migration in #156.

How the offline check is enforced
---------------------------------
The two builds have contradictory requirements — the prep build needs
network, the ship build must prove it needs none — and several frameworks
latch their offline switches at import time (huggingface_hub reads
``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE`` once, when first imported), so
the two builds cannot share a process. The ship build therefore runs in a
fresh subprocess (the ``--verify-ship`` mode below) where, before any
framework import:

  * ``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1`` are set in the
    subprocess environment, so they are already in force when the
    interpreter starts;
  * ``HF_HOME`` and ``TORCH_HOME`` point at an empty temp directory, so a
    warm local cache cannot quietly satisfy a lookup the template should
    not be making; and
  * all socket connections are blocked in-process (``_block_network``),
    which also covers download paths that consult none of those variables
    — ``torch.hub`` checkpoint URLs in particular.

Any download attempt left in the rewritten template therefore fails the
verification, whatever library it goes through. ``tests/test_prep_offline_weights.py``
keeps this honest with a mutation test: a deliberately network-fetching
template must make this tool go red.

Usage
-----
    # torchvision example (no config to inline):
    python tools/prep_offline_weights.py \\
        --prep  old/faster_rcnn_resnet.py \\
        --ship  model_zoo/object_detection/pytorch/faster_rcnn_resnet.py \\
        --out   dist/faster_rcnn_resnet_weights.pkl

    # HuggingFace example (offline config inlined in the ship file):
    python tools/prep_offline_weights.py \\
        --prep  old/bert_base_uncased.py \\
        --ship  model_zoo/text_classification/pytorch/bert_base_uncased.py \\
        --out   dist/bert_base_uncased_weights.pkl

Run the prep build in an environment that allows the one-time download
(network on, hub token exported for license-gated models) AND is pinned to the
engine's dependency set — install ``tools/requirements-engine-pin.txt`` first.
That pin mirrors the engine's ``use_cases/requirements.txt`` (transformers /
timm / torch / torchvision / peft), which is the ONLY thing that makes a dump's
key layout match what the edge builds. Record those versions in
``manifest.json``'s schema-2 ``built_with`` block so
``verify_dumps_against_engine_pin.py`` can gate them in CI (backend#2641,
backend#2658). Prep and verify therefore share one environment definition; a
drift between it and the engine is caught by the CI gate.

The produced weight file is NOT committed to this repo — it is uploaded next
to the template via ``upload_model(..., weights=True)`` and served from the
tracebloc model store.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build(path: str, name: str):
    mod = _load_module(path, name)
    entry_name = getattr(mod, "main_class", None) or getattr(mod, "main_method", None)
    if not entry_name or not hasattr(mod, entry_name):
        raise RuntimeError(f"{path}: no main_class/main_method entry point found")
    model = getattr(mod, entry_name)()
    return mod, model


def _offline_env(cache_dir: str) -> dict[str, str]:
    """Environment for the ship-build subprocess: offline flags set before
    the interpreter starts (import-time latches see them), caches pointed at
    an empty directory (a warm cache cannot mask a lookup).

    Every cache variable that can override ``HF_HOME`` is pinned too —
    ``HF_HUB_CACHE``/``HUGGINGFACE_HUB_CACHE`` (and the legacy
    ``TRANSFORMERS_CACHE``) take precedence when set in the parent
    environment, and offline mode happily serves from a warm cache without
    opening a socket, which would keep the verification green for a template
    that still performs hub lookups.
    """
    empty_hub_cache = os.path.join(cache_dir, "hub")
    return {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": cache_dir,
        "HF_HUB_CACHE": empty_hub_cache,
        "HUGGINGFACE_HUB_CACHE": empty_hub_cache,
        "TRANSFORMERS_CACHE": empty_hub_cache,
        "TORCH_HOME": cache_dir,
    }


def _block_network() -> None:
    """Refuse every socket connection in this process.

    Belt-and-braces behind the env flags: this also stops download paths
    that consult no offline variable at all (torch.hub checkpoint URLs).
    Called before any framework import in --verify-ship mode.
    """
    import socket

    def _deny(*_args, **_kwargs):
        raise RuntimeError(
            "offline verification: network access blocked — the ship template "
            "attempted a connection while building"
        )

    socket.socket.connect = _deny
    socket.socket.connect_ex = _deny
    socket.create_connection = _deny
    socket.getaddrinfo = _deny


def _verify_ship(ship_path: str, state_path: str) -> int:
    """Subprocess entry: build the offline template with the network blocked,
    then prove the prep dump strict-loads into it."""
    _block_network()

    import torch

    _, ship_model = _build(ship_path, "ship_template")
    state_dict = torch.load(state_path, weights_only=True)

    # The decisive check: strict load. Keys AND shapes must match exactly,
    # or the platform's strict=True seed load (and the SDK's upload-time
    # weight check) would fail. load_state_dict raises on any mismatch.
    ship_model.load_state_dict(state_dict, strict=True)
    print(
        f"[verify-ship] {ship_path}: built with network blocked; "
        f"strict load of {len(state_dict)} tensors OK"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--prep",
        help="pre-migration template (downloads pretrained weights ONCE)",
    )
    ap.add_argument("--ship", help="rewritten offline template the zoo ships")
    ap.add_argument("--out", help="output weights path (<base>_weights.pkl)")
    ap.add_argument(
        "--verify-ship",
        metavar="SHIP_PY",
        help="verification mode (run internally in a fresh offline subprocess): "
        "build SHIP_PY with the network blocked and strict-load --state into it",
    )
    ap.add_argument(
        "--state", help="state_dict file to strict-load in --verify-ship mode"
    )
    args = ap.parse_args()

    if args.verify_ship:
        if not args.state:
            ap.error("--verify-ship requires --state")
        return _verify_ship(args.verify_ship, args.state)

    if not (args.prep and args.ship and args.out):
        ap.error("--prep, --ship and --out are all required (or use --verify-ship)")

    import torch

    # 1) Build the pretrained ("prep") wrapper. Network is expected here —
    #    this is the single download, done once by the person running this
    #    tool. We do NOT force offline for this build.
    print(f"[prep] building pretrained wrapper from {args.prep} ...")
    _, prep_model = _build(args.prep, "prep_template")
    prep_model.eval()
    state_dict = prep_model.state_dict()
    print(f"[prep] captured state_dict with {len(state_dict)} tensors")

    # 2) + 3) Build the offline ("ship") wrapper and strict-load the dump —
    #    in a FRESH subprocess with the offline environment set before the
    #    interpreter starts and every socket connection blocked (see module
    #    docstring). This process has already imported the frameworks with
    #    network on, so an in-process env flip would verify nothing.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_state = os.path.join(tmp, "prep_state.pt")
        torch.save(state_dict, tmp_state)
        cache_dir = os.path.join(tmp, "empty-cache")
        os.makedirs(cache_dir)
        print(f"[ship] verifying offline build of {args.ship} in a subprocess ...")
        subprocess.run(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--verify-ship",
                args.ship,
                "--state",
                tmp_state,
            ],
            env=_offline_env(cache_dir),
            check=True,
        )

    # 4) Persist. torch.save is fine for the upload path, but tied/shared
    #    tensors (e.g. an MLM decoder tied to the input embeddings) need
    #    care downstream — warn if detected.
    _warn_on_shared_storage(state_dict)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, out)
    print(
        f"[done] wrote {out} ({out.stat().st_size / 1e6:.1f} MB); strict-load verified."
    )
    return 0


def _warn_on_shared_storage(state_dict) -> None:
    seen: dict[int, str] = {}
    for key, tensor in state_dict.items():
        try:
            ptr = tensor.untyped_storage().data_ptr()
        except Exception:
            continue
        if ptr in seen:
            print(
                f"[warn] tied/shared storage: '{key}' shares memory with "
                f"'{seen[ptr]}'. Plain safetensors.save_file rejects shared "
                "tensors; emit via safetensors save_model / clone-before-save "
                "if converting this dump.",
                file=sys.stderr,
            )
        else:
            seen[ptr] = key


if __name__ == "__main__":
    raise SystemExit(main())
