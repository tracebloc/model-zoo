"""Every torchvision builder with a separate BACKBONE must be called with both
weight arguments pinned off (internal ref).

The defect this pins
--------------------
``torchvision.models.segmentation.deeplabv3_resnet50`` and its 17 siblings take
*two* weight arguments, not one::

    deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=3)
                       ^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^
                       the head      the backbone -- DEFAULTS TO
                                     ResNet50_Weights.IMAGENET1K_V1

``weights_backbone`` defaults to an ImageNet enum, so a call that omits it
downloads a 97.8 MB ResNet-50 checkpoint from ``download.pytorch.org`` on every
construction and unpickles what it fetches. ``deeplab.py`` shipped as
``deeplabv3_resnet50(pretrained=False, progress=True, num_classes=...)`` and did
exactly that: measured, it wrote ``resnet50-0676ba61.pth`` (97.8 MB) into
``TORCH_HOME``; corrected, it writes nothing.

``pretrained=False`` is what makes this a trap rather than an oversight. It is
the spelling that *looks* like it settles the question. On torchvision 0.28 -- the
pin in ``.github/requirements/pytorch.txt``, so the version this guard actually
runs against -- it is not even a parameter of these builders any more: the legacy
shim maps it onto ``weights`` and never touches ``weights_backbone``.

WHY THIS FILE EXISTS -- two guards, one blind spot each
-------------------------------------------------------
Neither existing half sees this shape, and this is @LukasWodka's observation on
(internal ref) that neither alone is sufficient:

* ``test_model_contract.py::test_no_runtime_hub_fetch_patterns`` is the zoo-wide
  STATIC guard for runtime hub fetches. It was blind here by construction: its
  patterns look for ``pretrained=True``, an ``UPPER`` weights id or a
  ``_Weights.`` enum, and the shipped call had ``pretrained=False`` and no
  ``weights`` argument at all. There was nothing textual to match -- the fetch
  lived in a DEFAULT, in torchvision's signature, not in our source.
* the runtime door (``conftest._shut_the_torch_door`` +
  ``test_torch_fetch_door.py``) catches it when the model is actually BUILT --
  but only for templates the pytorch matrix instantiates in-process, and
  ``_TOO_LARGE_FOR_CI_RAM`` exempts some. ``test_model_instantiates`` also calls
  each entry point with DEFAULTS, so ``DeepLabV3(backbone="resnet101")`` is never
  constructed at all; that branch was correct by symmetry only (@saadqbal).

So the class needs a static rule with whole-zoo scope, which is this file.

DERIVED FROM SIGNATURES, NOT FROM A LIST
----------------------------------------
The set of affected builders is read out of the installed torchvision with
``inspect.signature`` -- every public callable under ``torchvision.models`` and
its task sub-modules that HAS a ``weights_backbone`` parameter. 18 on the
pinned torchvision 0.28.

That is the point. A hand-maintained list of builder names would be correct on
the day it was written and would silently miss the next builder torchvision
adds, which is the same shape of gap that let this defect ship. Deriving it
means a new sibling is covered the moment it exists, and a builder that loses
the parameter drops out on its own.

``importorskip("torchvision")`` so it runs in the pytorch matrix, where
torchvision is installed and its real signatures are available -- the sklearn and
survival matrices skip it, exactly as they skip the runtime door.

WHAT THIS FILE DOES NOT CLAIM
-----------------------------
It matches ``ast.Call`` nodes by the callee's identifier, so it is a check on how
the zoo SPELLS these calls. Indirection defeats it -- a builder fetched through
``getattr(models.segmentation, name)`` or stored in a dict and called later is
invisible here, and so is a wrapper that forwards ``**kwargs``. The runtime door
is what covers those, for the templates CI builds. Neither half subsumes the
other; that is why both exist.

A syntax error propagates: an unparseable template is a loud failure, because
this file cannot say what it calls and "cannot say" must never read as "clean".
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

pytest.importorskip("torchvision")

import torchvision.models  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
MODEL_ROOT = ROOT / "model_zoo"

#: The task sub-modules that carry backbone-taking builders, plus the root.
#: Named rather than walked: ``torchvision.models`` re-exports a great deal, and
#: an exhaustive walk would pull in ``models.detection.backbone_utils`` helpers
#: whose ``weights_backbone`` is a plumbing argument rather than a builder knob.
_MODEL_SUBMODULES = (
    "",
    "detection",
    "segmentation",
    "video",
    "optical_flow",
    "quantization",
)

#: Both arguments must be present AND pinned off. ``weights`` alone leaves the
#: backbone downloading -- that IS (internal ref) -- and ``weights_backbone`` alone leaves
#: the head able to default in a builder that gains one.
_REQUIRED_OFF = ("weights", "weights_backbone")

#: A COLLAPSE DETECTOR for the derivation, not a census. If a torchvision
#: refactor moves these builders and the signature scan silently finds nothing,
#: every assertion below passes vacuously. 18 on the pinned 0.28; the floor catches a
#: derivation that has stopped deriving, without pinning an exact count that a
#: legitimate torchvision release would break.
_MIN_BACKBONE_BUILDERS = 12


def _backbone_builders() -> dict[str, str]:
    """Public torchvision callables that take ``weights_backbone``.

    Returns ``{callable name: dotted module path}``. Read from the INSTALLED
    library, so the roster tracks whatever torchvision the pytorch matrix pins.
    """
    found: dict[str, str] = {}
    for sub in _MODEL_SUBMODULES:
        module = torchvision.models
        dotted = "torchvision.models"
        if sub:
            module = getattr(module, sub, None)
            dotted = f"{dotted}.{sub}"
            if module is None:
                continue
        for name, obj in vars(module).items():
            if name.startswith("_") or not callable(obj):
                continue
            try:
                signature = inspect.signature(obj)
            except (TypeError, ValueError):
                # Not introspectable (C extension, some descriptors). It cannot be
                # one of our builders, which are plain Python functions.
                continue
            if "weights_backbone" in signature.parameters:
                found[name] = dotted
    return found


def _model_files() -> list[pathlib.Path]:
    return sorted(MODEL_ROOT.rglob("*.py"))


def _callee_name(node: ast.Call) -> str | None:
    """The identifier a call is spelled with -- ``f(...)`` or ``a.b.f(...)``."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _pinned_off(node: ast.Call, key: str) -> bool:
    """Is ``key=None`` / ``key=False`` passed explicitly?

    Keyword only. These builders take ``weights`` and ``weights_backbone`` as
    keyword-or-positional, but no template in this repo passes them positionally
    and the offline pattern we are pinning is the named one -- a positional
    weights argument reads as a checkpoint, not as an opt-out, and should be
    rewritten rather than accepted here.
    """
    for keyword in node.keywords:
        if keyword.arg != key:
            continue
        value = keyword.value
        if isinstance(value, ast.Constant) and (
            value.value is None or value.value is False
        ):
            return True
        return False
    return False


def test_the_builder_roster_was_derived() -> None:
    """The signature scan must actually find the builders.

    Guards against a vacuous pass: with an empty roster no template can offend
    and the real assertion below would be green on nothing.
    """
    builders = _backbone_builders()
    assert len(builders) >= _MIN_BACKBONE_BUILDERS, (
        "the torchvision signature scan found only "
        f"{len(builders)} callable(s) taking `weights_backbone` "
        f"(expected at least {_MIN_BACKBONE_BUILDERS}) — the derivation has "
        "stopped deriving, so every other assertion in this file is passing "
        f"vacuously. Found: {sorted(builders)}"
    )
    # The builder this whole class was filed for.
    assert "deeplabv3_resnet50" in builders


def test_backbone_builders_are_called_with_both_weights_pinned_off() -> None:
    """No template may leave ``weights_backbone`` at its ImageNet default."""
    builders = _backbone_builders()
    offenders: list[str] = []
    call_sites = 0

    for path in _model_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _callee_name(node)
            if name not in builders:
                continue
            call_sites += 1
            missing = [key for key in _REQUIRED_OFF if not _pinned_off(node, key)]
            if missing:
                offenders.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: {name}(...) is missing "
                    + " and ".join(f"`{key}=None`" for key in missing)
                )

    assert not offenders, (
        "torchvision builder call(s) that download a pretrained BACKBONE (internal ref).\n"
        "These builders default `weights_backbone` to an ImageNet enum, so omitting "
        "it fetches a checkpoint from download.pytorch.org at construction time and "
        "unpickles it. `pretrained=False` does NOT disable it — on torchvision 0.28 "
        "it is not a parameter of these builders at all, and the legacy shim maps it "
        "onto `weights` only.\n"
        "Pass BOTH `weights=None` and `weights_backbone=None`, and ship any real "
        "tensors as a sibling weights file:\n" + "\n".join(offenders)
    )

    # Not an assertion about the zoo's contents, a check that the scan LOOKED. If a
    # refactor renames every call site, "no offenders" would otherwise be vacuous.
    assert call_sites > 0, (
        "no call to any backbone-taking torchvision builder was found anywhere under "
        "model_zoo/ — the scan matched nothing, which is not the same as clean"
    )


def test_the_detector_discriminates() -> None:
    """Both directions, on the exact shapes (internal ref) turned on.

    A guard nobody has watched fail is a guard that can quietly stop working.
    `pretrained=False` is the shape that shipped the bug and MUST read as an
    offence; the corrected call MUST read as clean.
    """
    shipped = ast.parse(
        "deeplabv3_resnet50(pretrained=False, progress=True, num_classes=3)"
    ).body[0].value
    assert not _pinned_off(shipped, "weights_backbone")
    assert not _pinned_off(shipped, "weights")

    head_only = ast.parse("deeplabv3_resnet50(weights=None, num_classes=3)").body[0].value
    assert _pinned_off(head_only, "weights")
    assert not _pinned_off(head_only, "weights_backbone")

    fixed = ast.parse(
        "deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=3)"
    ).body[0].value
    assert _pinned_off(fixed, "weights")
    assert _pinned_off(fixed, "weights_backbone")

    # An ImageNet enum is not an opt-out, however it is spelled.
    enum_call = ast.parse(
        "deeplabv3_resnet50(weights=None, "
        "weights_backbone=ResNet50_Weights.IMAGENET1K_V1)"
    ).body[0].value
    assert not _pinned_off(enum_call, "weights_backbone")
