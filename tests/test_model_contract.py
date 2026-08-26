"""Smoke tests: every model file imports, declares the expected metadata, and
— for templates that build locally — actually constructs.

Parametrized over every `.py` under `model_zoo/`. Files that do not declare
a `framework = "..."` module attribute are treated as support files (e.g.
loss.py, utils.py inside a packaged model folder) and skipped. Files whose
framework is not installed in the current environment are also skipped —
this lets the CI matrix run per-framework without re-installing everything.

Two tests, deliberately separate:

`test_model_contract` imports the module and checks module-level metadata.

`test_model_instantiates` calls the declared entry point. Importing is not
enough: a model's architecture is built inside `__init__`, so a template can
import cleanly and still be impossible to construct. That is not
hypothetical — `mambavision.py` asked timm for `mambavision_tiny.fb_in1k`,
an architecture timm has never shipped, and raised
`RuntimeError: Unknown model (mambavision_tiny)` on every construction while
the import-only test stayed green (backend#1859).
"""

import importlib
import importlib.util
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parent.parent
MODEL_ROOT = ROOT / "model_zoo"

FRAMEWORK_IMPORT_NAME = {
    "pytorch": "torch",
    "sklearn": "sklearn",
    "lifelines": "lifelines",
    "scikit_survival": "sksurv",
}
KNOWN_FRAMEWORKS = set(FRAMEWORK_IMPORT_NAME)

OPTIONAL_THIRD_PARTY = {
    "xgboost",
    "lightgbm",
    "catboost",
    "interpret",
    "peft",
    "timm",
}

KNOWN_CATEGORIES = {
    "image_classification",
    "object_detection",
    "text_classification",
    "semantic_segmentation",
    "keypoint_detection",
    "tabular_classification",
    "tabular_regression",
    "time_series_forecasting",
    "time_series_classification",
    "time_to_event_prediction",
    "masked_language_modeling",
    "causal_language_modeling",
    "token_classification",
    "seq2seq",
    "embeddings",
    "sentence_pair_classification",
}


def _read_framework(path: pathlib.Path) -> str | None:
    """Extract `framework = "..."` without importing the file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(r'^\s*framework\s*=\s*["\'](\w+)["\']', text, re.MULTILINE)
    return match.group(1) if match else None


def _is_installed(framework: str) -> bool:
    import_name = FRAMEWORK_IMPORT_NAME[framework]
    try:
        importlib.import_module(import_name)
    except ImportError:
        return False
    return True


def _missing_optional_deps(path: pathlib.Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    found = set(re.findall(r"^\s*(?:from|import)\s+(\w+)", text, re.MULTILINE))
    missing = []
    for mod in sorted(found & OPTIONAL_THIRD_PARTY):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    return missing


# A template "fetches from the hub" — and so cannot be constructed in CI
# without pulling weights over the network — if it does any of:
#   - HuggingFace `from_pretrained(...)`                 -> transformers / timm hub
#   - torchvision `weights="DEFAULT"` (any "UPPER" id)   -> download.pytorch.org
#   - a torchvision `<Arch>_Weights.<MEMBER>` enum value -> download.pytorch.org
# The last two also download ImageNet/COCO checkpoints, so matching only
# `from_pretrained` let them slip through and made CI fetch them anyway.
# Local builds (`pretrained=False`, `weights=None`, timm `create_model(...,
# pretrained=False)`) match none of these and stay covered by the test.
_HUB_FETCH = re.compile(
    r"from_pretrained"
    r"|weights\s*=\s*[\"'][A-Z]"
    r"|_Weights\."
)


def _fetches_from_hub(path: pathlib.Path) -> bool:
    """Does this template pull a model/config from an external hub to build?

    Those templates cannot be constructed in a test without network:
    `from_pretrained` downloads from the HuggingFace hub (some are
    multi-gigabyte — gemma_2, sam2), and torchvision builders asked
    for pretrained checkpoints — `weights="DEFAULT"` or a `<Arch>_Weights.*`
    enum — pull ImageNet/COCO weights from download.pytorch.org. They are
    excluded from the instantiation test rather than making CI download the
    world. Templates that build their architecture from local library code —
    the ones a `create_model("name")` typo can break silently, plus torchvision
    builders called with `pretrained=False`/`weights=None` — are all still
    covered.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return bool(_HUB_FETCH.search(text))


def _model_files() -> list[pathlib.Path]:
    return sorted(MODEL_ROOT.rglob("*.py"))


def _load_or_skip(path: pathlib.Path):
    """Shared preamble: resolve the framework, skip what this job cannot run,
    and exec the module. Returns (framework, module)."""
    framework = _read_framework(path)
    if framework is None:
        pytest.skip("support file (no `framework` declaration)")

    assert framework in KNOWN_FRAMEWORKS, (
        f"{path}: declared framework {framework!r} is not in {KNOWN_FRAMEWORKS}"
    )

    if not _is_installed(framework):
        pytest.skip(f"{framework} not installed in this CI job")

    missing = _missing_optional_deps(path)
    if missing:
        pytest.skip(f"optional dep(s) not installed in this CI job: {', '.join(missing)}")

    module_name = re.sub(r"\W", "_", path.stem)
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader, f"{path}: importlib could not build a spec"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return framework, module


@pytest.mark.parametrize("path", _model_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_model_contract(path: pathlib.Path) -> None:
    framework, module = _load_or_skip(path)

    assert getattr(module, "framework", None) == framework

    category = getattr(module, "category", None)
    assert category in KNOWN_CATEGORIES, (
        f"{path}: declared category {category!r} is not in {KNOWN_CATEGORIES}"
    )

    parts = path.parts
    mz_idx = parts.index("model_zoo")
    task_from_path = parts[mz_idx + 1]
    assert category == task_from_path, (
        f"{path}: category {category!r} does not match task directory {task_from_path!r}"
    )

    entry = getattr(module, "main_class", None) or getattr(module, "main_method", None)
    assert entry, f"{path}: neither main_class nor main_method is defined"
    assert hasattr(module, entry), f"{path}: entry symbol {entry!r} not found in module"


@pytest.mark.parametrize("path", _model_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_model_instantiates(path: pathlib.Path) -> None:
    """The declared entry point must construct with its default arguments.

    Every template is meant to be copied and run as-is, so "constructs with no
    arguments" is the weakest promise it can make. Architecture names are
    looked up in a third-party registry (timm, torchvision) at construction
    time, and nothing else in this repo checks that those strings resolve.
    """
    if _read_framework(path) is not None and _fetches_from_hub(path):
        pytest.skip("builds from an external hub — needs network, see _fetches_from_hub")

    _, module = _load_or_skip(path)

    entry_name = getattr(module, "main_class", None) or getattr(
        module, "main_method", None
    )
    assert entry_name, f"{path}: neither main_class nor main_method is defined"
    entry = getattr(module, entry_name)

    model = entry()
    assert model is not None, f"{path}: {entry_name}() returned None"
