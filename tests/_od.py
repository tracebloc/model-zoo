"""Shared object-detection template helpers for the ``tests/`` suite.

Three OD test files had grown byte-identical copies of the same three
helpers — the ``framework`` reader, the template scan, and the
import-and-construct — and had already started drifting: the build-module
name prefix differed, and so did whether ``output_classes`` was consulted.
Three copies of a scan that decides WHICH templates get checked is the
worst place for drift, because a copy that silently narrows makes its file
pass on fewer templates without failing anything.

Not a general "test utils" dumping ground: this module holds only the
helpers that decide *which* OD templates exist and *how* to construct one.

What deliberately stays per-file
--------------------------------
``_read_model_type`` and the file-local ``framework`` regexes in
``test_od_declared_resolution.py`` and
``test_od_torchvision_family_train_step.py``. Those files compare TWO
readers' verdicts against each other to prove their roster partition is
not vacuous, and that argument needs two independent implementations
*within one file*. It was never an argument for re-typing
``declares_framework`` in three files — which is what the review on
model-zoo#251 pointed out.

Migration status (model-zoo#251)
--------------------------------
``test_od_norm_layers_normalise.py`` imports from here.
``test_od_declared_resolution.py`` and
``test_od_torchvision_family_train_step.py`` still hold their own copies:
the first is being rewritten on another branch right now (model-zoo#252,
+571 lines in that one file), so moving its helpers here would conflict
for no benefit. Their imports are a follow-up, to be done when nothing is
in flight on them — the point of landing this module is that the count of
copies stops growing.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re

ROOT = pathlib.Path(__file__).parent.parent
OD_ROOT = ROOT / "model_zoo" / "object_detection"


def declares_framework(path: pathlib.Path) -> str | None:
    """The module-level ``framework``, or ``None`` for a support module.

    This is what separates a TEMPLATE ENTRY POINT from a helper: the
    metadata contract in CLAUDE.md requires it of every model file, and
    the ``yolo_*/loss.py`` helpers declare none. Read statically so file
    SELECTION costs nothing in the CI jobs that cannot import torch.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(r'^\s*framework\s*=\s*["\'](\w*)["\']', text, re.MULTILINE)
    return match.group(1) if match else None


def od_templates() -> list[pathlib.Path]:
    """Every OD template entry point, sorted."""
    return [p for p in sorted(OD_ROOT.rglob("*.py")) if declares_framework(p)]


def template_key(path: pathlib.Path) -> str:
    """Row key for a template.

    The yolo templates are all named ``model.py`` under a versioned
    directory, so a bare ``path.stem`` would collide three ways. Use the
    directory name for those, the file stem otherwise.
    """
    return path.parent.name if path.stem == "model" else path.stem


def build_template(path: pathlib.Path, num_classes: int | None = None, prefix: str = "od"):
    """Import a template and construct its model.

    Returns ``(module, model)``.

    ``num_classes=None`` means "use the template's own declared
    ``output_classes``", which is what a test checking a template on its
    own terms wants; pass an explicit count to construct at a different
    one. That reconciles the two behaviours the copies had drifted into
    rather than picking one and silently changing a caller.

    ``prefix`` namespaces the synthetic module name so two test files
    importing the same template file do not collide in
    ``spec_from_file_location``.
    """
    module_name = re.sub(r"\W", "_", f"{prefix}_{template_key(path)}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader, f"{path}: importlib could not build a spec"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    entry = getattr(module, "main_class", None) or getattr(module, "main_method", None)
    assert entry, f"{path}: neither main_class nor main_method is defined"
    if num_classes is None:
        num_classes = int(getattr(module, "output_classes", 3))
    return module, getattr(module, entry)(num_classes)
