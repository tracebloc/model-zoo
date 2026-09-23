"""Pytest session config for model-zoo.

HuggingFace hub is a closed door (design note D6): every template
must build from local library code or an inlined config, never a runtime hub
fetch. The offline-weights migration ((internal ref)-(internal ref)) already removed every fetch
site; the client spawns training pods with these three vars set
(client-runtime jobs_manager._add_environment_variables), and the engine
tokenizer loader hard-errors under them.

We mirror that here at COLLECTION time — before any template module imports
transformers / timm / torchvision — so the whole suite (contract +
instantiation) exercises the closed door. A template that regressed to a
runtime fetch then fails offline in CI instead of silently downloading. Every
model cache is pointed at a throwaway tmp dir so a model already cached on a
developer's machine cannot mask a fetch that would fail in CI -- TAKEN from
tools/prep_offline_weights.py's `_offline_env`, not mirrored: this file used to
restate it and the restatement had already drifted.

Torch's half of the door is shut below in two layers: a name-rebind that carries
the message a template author needs to read, and `_block_network()` at the
socket, which is the layer that holds regardless of who imported torchvision
first.
"""

import importlib.util
import os
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TOOL = _ROOT / "tools" / "prep_offline_weights.py"


def _prep_offline_weights():
    """`tools/prep_offline_weights.py`, loaded by path.

    Same idiom as `test_prep_offline_weights.py` (`tools/` is not a package), so this
    file and that suite hold ONE `_offline_env` / `_block_network` rather than two
    copies that drift. Import failure is deliberately fatal: the socket door this file
    promises to shut lives in there, and a conftest that quietly fell back to its own
    copy would be a door with no lock and no red test.
    """
    spec = importlib.util.spec_from_file_location("prep_offline_weights", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PREP = _prep_offline_weights()

# The `ingest_producer` marker: deselected unless `-m` names it. The hooks live in
# their own file so the category-gate tests can drive the SAME hooks in a throwaway
# session; see tests/ingest_producer_marker.py for why deselected, not skipped.
from ingest_producer_marker import pytest_collection_modifyitems, pytest_configure  # noqa: E402,F401

# Force the closed door for the whole test session (not setdefault — the point
# is that the suite proves templates build with the hub shut, regardless of the
# ambient environment).
# TAKEN from the ship-build's own environment, not restated (@LukasWodka). The
# header below claimed this "mirrors" `_offline_env` while omitting
# `HUGGINGFACE_HUB_CACHE`, which takes precedence over `HF_HOME` when it is set in
# the parent environment -- and offline mode serves a warm cache happily, without
# opening a socket, so a developer with it exported kept exactly the masking this
# block exists to prevent. Deriving means the two cannot disagree again.
_ISOLATED_CACHE = tempfile.mkdtemp(prefix="model-zoo-offline-cache-")
os.environ.update(_PREP._offline_env(_ISOLATED_CACHE))
# `_offline_env` is written for a SUBPROCESS and carries no datasets flag; templates
# are collected in THIS process, and `datasets.load_dataset` is one of the patterns
# test_model_contract.py forbids.
os.environ["HF_DATASETS_OFFLINE"] = "1"


# The door above is HuggingFace-only, and torch has its own. `TORCH_HOME` only
# RELOCATES torchvision's cache — it does not close it — so `download.pytorch.org` stayed
# wide open and `test_model_instantiates` downloaded silently on a networked runner and
# passed. That is how `deeplab.py` shipped calling `deeplabv3_resnet50(pretrained=False)`
# while its `weights_backbone` default fetched a 97.8 MB ResNet-50 checkpoint on every
# construction (internal ref).
#
# No template may fetch: architectures build from local code or an inlined config, and
# pretrained tensors arrive as a sibling weights file the platform loads AFTER MyModel().
# So torch's URL loaders raise here, for the same reason the HF vars are set and not
# setdefault — the suite proves the door is shut regardless of the ambient environment.
def _shut_the_torch_door() -> None:
    try:
        import torch.hub
    except ImportError:
        # The sklearn and survival matrices do not install torch, and a template that
        # cannot import it cannot fetch through it. Guarded rather than assumed: without
        # this, conftest raised at COLLECTION time and took both matrices to exit 4 —
        # every test in them, not just the ones that touch torch.
        return

    def _refuse(url, *_args, **_kwargs):
        raise AssertionError(
            "A template fetched pretrained weights at build time: "
            f"load_state_dict_from_url({url!r}). Architectures must build offline — pass "
            "`weights=None` AND `weights_backbone=None` (the torchvision detection and "
            "segmentation builders default `weights_backbone` to an ImageNet enum, and "
            "`pretrained=False` does not disable it), and ship the tensors as a sibling "
            "weights file instead."
        )

    # Patch the REAL loader first, and unconditionally. Everything torchvision reaches
    # for lands here.
    torch.hub.load_state_dict_from_url = _refuse

    # `torch.utils.model_zoo.load_url` IS the same function object; rebind the alias too
    # so a template reaching it by that name is caught rather than slipping past. Its own
    # `try`, deliberately: `torch.utils.model_zoo` is a DEPRECATED shim, and while one
    # `try` covered both imports, the day torch drops it the `except` fired before
    # `torch.hub` was patched at all — the door hung open with every test still green
    # (@saadqbal ask 2). The alias is a nice-to-have; the loader above is not.
    try:
        import torch.utils.model_zoo
    except ImportError:
        return
    torch.utils.model_zoo.load_url = _refuse


_shut_the_torch_door()

# AND THE SOCKET, ONCE, REGARDLESS OF IMPORT ORDER (@LukasWodka ask 3).
# The rebind above is the MESSAGE-BEARING layer -- it names `weights_backbone=None`,
# which is what a template author has to read. What it cannot do is survive import
# order. torchvision captures the loader BY VALUE at import
# (`_internally_replaced_utils.py`: `from torch.hub import load_state_dict_from_url`,
# re-exported into `models/_api.py`), so anything that pulls torchvision in before
# this file runs keeps a live copy of the real downloader: measured, `torch.hub.
# load_state_dict_from_url is _refuse` is then True while `torchvision.models._api.
# load_state_dict_from_url is _refuse` is False. Today one conftest and no plugin
# config keep the ordering in our favour, which is a property of the layout, not of
# the door.
#
# `_block_network` refuses at the SOCKET, so it holds whoever imported what, and it
# covers the paths a name-rebind never sees: `torch.hub.download_url_to_file` (timm's
# fetch path), `urllib`, `requests`. It is the same function
# test_prep_offline_weights.py::test_network_fetching_template_goes_red already
# mutation-proves -- one mechanism, two callers, rather than a third narrower one.
#
# Session-wide is safe: no test in this suite needs the network -- nothing under
# tests/ imports requests, urllib or boto3, and the dump and engine-pin tests do not
# reach out either (@saadqbal). Verified by running it: all 186 template
# instantiations pass with the socket shut.
#
# The six per-test `guard_constructs_with_no_network` socket copies in the YOLO and
# object-detection test files exist only because collection never closed this. They
# are now redundant rather than load-bearing; removing them is a separate change.
_PREP._block_network()
