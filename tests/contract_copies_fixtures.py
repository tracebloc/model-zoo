"""Synthetic producer sources for ``tools/contract_copies.py``'s tests.

Written here by hand, in the shapes the producers use, because the real sources
are private and this repository is public. ``producer_tree`` lays them out as
the drift run's sparse checkouts do.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_tool():
    spec = importlib.util.spec_from_file_location("contract_copies", ROOT / "tools" / "contract_copies.py")
    mod = importlib.util.module_from_spec(spec)
    # Registered first: @dataclass resolves the module by name while it loads.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


EXPERIMENT = '''\
from django.utils.translation import gettext_lazy as _


class Experiment:
    MODEL_TYPE_ALPHA = "alpha"
    MODEL_TYPE_BETA = "beta"
    MODEL_TYPE_CHOICES = (
        (MODEL_TYPE_ALPHA, _("Alpha Model")),
        (MODEL_TYPE_BETA, _("Beta Model")),
    )
'''

IMPORT_NAMES = 'ALLOWED_PYTHON_PACKAGE_IMPORTS = (\n    "torch",\n    "math",\n)\n'

OD_FAMILIES = json.dumps({
    "version": 2,
    "families": [
        {"family": "alpha", "aliases": ["a"], "training_supported": True, "handler": "h.alpha",
         "notes": "upstream note citing other-repo#12"},
    ],
    "accepted_model_type_values": ["alpha"],
    "not_accepted": {"description": "prose", "examples": ["gamma"]},
})

SKL = {
    "base": '''\
class SklTabularBase:
    _ensemble_prefixes = {}

    def average_estimators(self, models, model_type, ensemble_cls):
        if model_type == "linear":
            return 1
''',
    "classifier": '''\
class SklTabularClassifier:
    _ensemble_prefixes = {"tree": "t"}

    def _average_family_specific(self, models, model_type, ensemble_cls):
        if model_type in ("naive",):
            return 1
''',
    "regression": '''\
class SklTabularRegression:
    _ensemble_prefixes = {"svm": "s"}
''',
}


def sources() -> dict[str, dict[str, str]]:
    """``{repo: {path: text}}`` for every registered copy's sources."""
    return {
        "backend": {
            "experiments/models/experiment.py": EXPERIMENT,
            "bandit_tracebloc_model_validation/bandit_plugins/import_names.py": IMPORT_NAMES,
        },
        "tracebloc-engine": {"core/schema/object_detection_families.v2.json": OD_FAMILIES},
        "tracebloc-py-package": {f"tracebloc/upload/skl_tabular_{k}.py": v for k, v in SKL.items()},
    }


def producer_tree(root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Write every source under ``root/<repo>/``; ``{repo: checkout dir}``."""
    out = {}
    for repo, files in sources().items():
        for rel, text in files.items():
            path = root / repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        out[repo] = root / repo
    return out


def rendered_copies(tool, trees: dict[str, pathlib.Path], dest: pathlib.Path) -> pathlib.Path:
    """A contracts directory each copy of which ``--write`` derived from ``trees``."""
    for copy in tool.COPIES:
        src = tool.fetch_tree(copy, trees[copy.repo])
        path = dest / copy.file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tool.render(copy, {}, src, "0" * 40, "develop", "2026-01-01"), encoding="utf-8")
    return dest


def tree_args(trees: dict[str, pathlib.Path]) -> list[str]:
    return [a for repo, d in trees.items() for a in ("--tree", f"{repo}={d}")]
