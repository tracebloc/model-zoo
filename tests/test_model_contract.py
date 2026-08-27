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


# Runtime hub-fetch patterns forbidden under model_zoo/ (RFC-0003 D6 /
# backend#1501). The HuggingFace hub is a closed door: a template must build
# from local library code or an inlined config, never pull weights, configs, or
# datasets over the network at construction time. Forbidden:
#   - HuggingFace `from_pretrained(...)` / `hf_hub_download` / `snapshot_download`
#   - `datasets.load_dataset(...)`                       -> HuggingFace hub
#   - torchvision `weights="DEFAULT"` (any "UPPER" id)   -> download.pytorch.org
#   - a torchvision `<Arch>_Weights.<MEMBER>` enum value -> download.pytorch.org
#   - `pretrained=True` (timm / torchvision legacy)      -> checkpoint download
#   - `torch.hub.load(...)` / `load_state_dict_from_url` -> arbitrary URL fetch
# Local builds (`pretrained=False`, `weights=None`, timm `create_model(...,
# pretrained=False)`, `AutoConfig.for_model(...)` + `from_config`) match none of
# these and are the required offline pattern.
_RUNTIME_HUB_FETCH = re.compile(
    r"from_pretrained"
    r"|hf_hub_download"
    r"|snapshot_download"
    r"|load_dataset"
    r"|weights\s*=\s*[\"'][A-Z]"
    r"|_Weights\."
    r"|pretrained\s*=\s*True"
    r"|torch\.hub\.load"
    r"|load_state_dict_from_url"
)


# Offline-migrated templates (#156) build from an inlined config with no hub
# fetch, so they are constructible in tests — which is the point. But
# construction materializes the full fp32 random-init parameter set in RAM, and
# for multi-billion-parameter templates that exceeds the ~16GB of a standard
# ubuntu-latest runner (gemma_2: ~2.6B params -> ~10.5GB for the tensors alone,
# before torch/test overhead). This is the ONLY reason a template is excluded
# from the instantiation test — a RAM ceiling, never a network dependency (the
# hub is closed; see test_no_runtime_hub_fetch_patterns).
# Keyed on the path relative to MODEL_ROOT (posix), not the basename: 19
# basenames are duplicated across task directories (e.g. bert_base_uncased.py
# in both text_classification and sentence_pair_classification), so a
# basename key would silently skip every file sharing the name.
_TOO_LARGE_FOR_CI_RAM = {
    "text_classification/pytorch/gemma_2.py",
}


def _ci_ram_skip_key(path: pathlib.Path) -> str:
    return path.relative_to(MODEL_ROOT).as_posix()


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

    The whole session runs with the HuggingFace hub closed (tests/conftest.py:
    HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE), so this also
    proves every template builds offline. There is no "fetches from the hub"
    skip any more — a template that needs a runtime fetch is a bug caught here
    (offline construction error) and at the source by
    test_no_runtime_hub_fetch_patterns.
    """
    if _ci_ram_skip_key(path) in _TOO_LARGE_FOR_CI_RAM:
        pytest.skip(
            "random-init construction exceeds CI runner RAM, see _TOO_LARGE_FOR_CI_RAM"
        )

    _, module = _load_or_skip(path)

    entry_name = getattr(module, "main_class", None) or getattr(
        module, "main_method", None
    )
    assert entry_name, f"{path}: neither main_class nor main_method is defined"
    entry = getattr(module, entry_name)

    model = entry()
    assert model is not None, f"{path}: {entry_name}() returned None"


def test_no_runtime_hub_fetch_patterns() -> None:
    """No template may fetch from a remote hub at construction time.

    The HuggingFace hub is a closed door (RFC-0003 D6 / backend#1501): the
    offline-weights migration (#182-#193) removed every runtime fetch, and this
    guard keeps the door shut. It fails at the SOURCE — a reintroduced
    `from_pretrained` / `pretrained=True` / `load_dataset` / torchvision
    pretrained checkpoint is caught here even for a template that a given CI
    job's framework matrix does not instantiate. Comment text is ignored so
    prose that merely names a pattern does not trip the guard.
    """
    offenders = []
    for path in _model_files():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(lines, start=1):
            code = line.split("#", 1)[0]
            if _RUNTIME_HUB_FETCH.search(code):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "runtime hub-fetch pattern(s) found under model_zoo/ — the HuggingFace "
        "hub is a closed door, build from an inlined config / uploaded weights "
        "(RFC-0003 D6 / backend#1501):\n" + "\n".join(offenders)
    )


def test_ci_ram_skip_entries_exist() -> None:
    """Every skip entry must name a real file, or it silently skips nothing."""
    for entry in _TOO_LARGE_FOR_CI_RAM:
        assert (MODEL_ROOT / entry).is_file(), (
            f"_TOO_LARGE_FOR_CI_RAM entry {entry!r} does not exist under model_zoo/"
        )


def test_ci_ram_skip_key_is_directory_scoped() -> None:
    """A skip entry for one directory must not match a same-named file in
    another. The tree has duplicated basenames (bert_base_uncased.py lives in
    both text_classification and sentence_pair_classification), which is why
    the set is keyed on MODEL_ROOT-relative paths, not basenames."""
    tc = MODEL_ROOT / "text_classification" / "pytorch" / "bert_base_uncased.py"
    spc = (
        MODEL_ROOT / "sentence_pair_classification" / "pytorch" / "bert_base_uncased.py"
    )
    assert tc.is_file() and spc.is_file(), (
        "expected duplicated basename fixture missing — update this test"
    )
    assert _ci_ram_skip_key(spc) not in {_ci_ram_skip_key(tc)}
