"""Pytest session config for model-zoo.

HuggingFace hub is a closed door (RFC-0003 D6 / backend#1501): every template
must build from local library code or an inlined config, never a runtime hub
fetch. The offline-weights migration (#182-#193) already removed every fetch
site; the client spawns training pods with these three vars set
(client-runtime jobs_manager._add_environment_variables), and the engine
tokenizer loader hard-errors under them.

We mirror that here at COLLECTION time — before any template module imports
transformers / timm / torchvision — so the whole suite (contract +
instantiation) exercises the closed door. A template that regressed to a
runtime fetch then fails offline in CI instead of silently downloading. Every
model cache is pointed at a throwaway tmp dir so a model already cached on a
developer's machine cannot mask a fetch that would fail in CI (mirrors
tools/prep_offline_weights.py's _offline_env).
"""

import os
import tempfile

# Force the closed door for the whole test session (not setdefault — the point
# is that the suite proves templates build with the hub shut, regardless of the
# ambient environment).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Isolate every model/dataset cache so a locally-cached checkpoint cannot mask a
# fetch that would fail on a clean CI runner.
_ISOLATED_CACHE = tempfile.mkdtemp(prefix="model-zoo-offline-cache-")
for _var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME"):
    os.environ[_var] = _ISOLATED_CACHE
