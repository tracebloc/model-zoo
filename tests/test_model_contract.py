"""Smoke test: every model file imports and declares the expected metadata.

Parametrized over every `.py` under `model_zoo/`. Files that do not declare
a `framework = "..."` module attribute are treated as support files (e.g.
loss.py, utils.py inside a packaged model folder) and skipped. Files whose
framework is not installed in the current environment are also skipped —
this lets the CI matrix run per-framework without re-installing everything.
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
    "tensorflow": "tensorflow",
    "sklearn": "sklearn",
    "lifelines": "lifelines",
    "scikit_survival": "sksurv",
}
KNOWN_FRAMEWORKS = set(FRAMEWORK_IMPORT_NAME)

KNOWN_CATEGORIES = {
    "image_classification",
    "object_detection",
    "text_classification",
    "semantic_segmentation",
    "keypoint_detection",
    "tabular_classification",
    "tabular_regression",
    "time_series_forecasting",
    "time_to_event_prediction",
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


def _model_files() -> list[pathlib.Path]:
    return sorted(MODEL_ROOT.rglob("*.py"))


@pytest.mark.parametrize("path", _model_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_model_contract(path: pathlib.Path) -> None:
    framework = _read_framework(path)
    if framework is None:
        pytest.skip("support file (no `framework` declaration)")

    assert framework in KNOWN_FRAMEWORKS, (
        f"{path}: declared framework {framework!r} is not in {KNOWN_FRAMEWORKS}"
    )

    if not _is_installed(framework):
        pytest.skip(f"{framework} not installed in this CI job")

    module_name = re.sub(r"\W", "_", path.stem)
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader, f"{path}: importlib could not build a spec"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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
