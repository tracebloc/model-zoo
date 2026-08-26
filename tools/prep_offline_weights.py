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
(network on, hub token exported for license-gated models). The --ship build
is verified under ``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1`` so a
lingering download call in the rewritten template is caught here, not later.

The produced weight file is NOT committed to this repo — it is uploaded next
to the template via ``upload_model(..., weights=True)`` and served from the
tracebloc model store.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--prep",
        required=True,
        help="pre-migration template (downloads pretrained weights ONCE)",
    )
    ap.add_argument(
        "--ship", required=True, help="rewritten offline template the zoo ships"
    )
    ap.add_argument(
        "--out", required=True, help="output weights path (<base>_weights.pkl)"
    )
    args = ap.parse_args()

    import torch

    # 1) Build the pretrained ("prep") wrapper. Network is expected here —
    #    this is the single download, done once by the person running this
    #    tool. We do NOT force offline for this build.
    print(f"[prep] building pretrained wrapper from {args.prep} ...")
    _, prep_model = _build(args.prep, "prep_template")
    prep_model.eval()
    state_dict = prep_model.state_dict()
    print(f"[prep] captured state_dict with {len(state_dict)} tensors")

    # 2) Build the offline ("ship") wrapper under a hard offline env — any
    #    download call left in the rewritten template blows up right here.
    print(
        f"[ship] building offline wrapper from {args.ship} under HF_HUB_OFFLINE=1 ..."
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _, ship_model = _build(args.ship, "ship_template")

    # 3) The decisive check: strict load. Keys AND shapes must match exactly,
    #    or the platform's strict=True seed load (and the SDK's upload-time
    #    weight check) would fail.
    print("[verify] loading prep state_dict into offline wrapper with strict=True ...")
    ship_model.load_state_dict(state_dict, strict=True)
    # load_state_dict(strict=True) raises on any mismatch; reaching here is success.

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
