"""Contract test: a ``torchvision_detection`` template's declared ``image_size``
is the resolution it actually runs at.

Background (backend#3058)
-------------------------
``image_size`` is not decoration. The SDK turns it into the experiment's
``data_shape`` parameter, the engine's ``TorchvisionDetectionHandler.data_shape``
reads that value back, and the dataset resizes every image to a
``data_shape x data_shape`` square before the model sees it. So the declared
number decides how much resolution the training data keeps.

Three templates on ``develop`` declare ``image_size = 448`` while their builders
resize to ``min_size=800`` internally: ``faster_rcnn_resnet``, ``fcos`` and
``retinanet``. The dataset therefore downsamples to 448, the model upsamples
straight back to 800, and the run pays two resizes to train on 448-px detail
while reporting 800. Nothing caught it because OD deliberately ships no SDK
shape-probe (#270) — the header is taken at face value everywhere.

What this pins, and how
-----------------------
Not the source text — the **built model**. Every detector in this family owns a
``transform`` submodule (torchvision's ``GeneralizedRCNNTransform``, reused by
the hand-written templates for exactly this reason), and that transform is the
single point where the declared edge either takes effect or does not. A forward
hook on it reports the spatial size of the tensor the backbone is about to
receive, for a square input at the declared ``image_size``. Reading it off the
first convolution would be wrong: YOLOX's ``Focus`` stem slices before it
convolves, so its first conv legitimately sees ``image_size / 2``.

The three known-dishonest templates are listed in
``_DECLARED_SIZE_IS_DECORATIVE`` and pinned from BOTH sides — they are exempted
from the equality assertion, and separately asserted to still mismatch. So the
list cannot rot: fixing one of them turns this file red with "remove it from the
exemption list", which is the failure worth having.
"""

import importlib.util
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parent.parent
OD_ROOT = ROOT / "model_zoo" / "object_detection"
CONTRACTS = pathlib.Path(__file__).parent / "contracts" / "tracebloc_engine"

FAMILY = "torchvision_detection"

#: Templates whose declared ``image_size`` is NOT the resolution they run at.
#: Each is a backend#3058 debt, not a licence: a new template must not join this
#: set, and ``test_every_exempt_template_still_mismatches`` fails if one is fixed
#: and left listed. Keyed by module stem.
_DECLARED_SIZE_IS_DECORATIVE = {
    "faster_rcnn_resnet": 800,
    "fcos": 800,
    "retinanet": 800,
}


def _schema_path() -> pathlib.Path:
    """Newest vendored ``object_detection_families.v<N>.json``, by version number
    rather than by name — the file is version-bumped in place when the engine's
    vocabulary narrows, and pinning the filename breaks on the rename."""
    paths = sorted(
        CONTRACTS.glob("object_detection_families.v*.json"),
        key=lambda p: int(re.search(r"\.v(\d+)\.json$", p.name).group(1)),
    )
    assert paths, f"no vendored OD families schema under {CONTRACTS}"
    return paths[-1]


def _family_values() -> frozenset[str]:
    """Declaration values routing to this family — the family name plus every
    alias, normalized the way the engine's ``resolve_family`` does.

    Selecting on the literal ``"torchvision_detection"`` would silently skip
    ``faster_rcnn_resnet``, which declares the legacy ``rcnn`` alias — and that
    is one of the templates this file exists to pin.
    """
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    entries = [f for f in schema["families"] if f["family"] == FAMILY]
    assert entries, f"{_schema_path().name}: no {FAMILY!r} family entry"
    return frozenset(
        v.strip().lower() for v in {FAMILY, *entries[0].get("aliases", [])}
    )


FAMILY_VALUES = _family_values()


def _read_model_type(path: pathlib.Path) -> str | None:
    """Declared module-level ``model_type``, read statically so file SELECTION
    costs nothing in CI jobs that cannot run torch."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(r'^\s*model_type\s*=\s*["\'](\w*)["\']', text, re.MULTILINE)
    return match.group(1) if match else None


def _family_templates() -> list[pathlib.Path]:
    return [
        p
        for p in sorted(OD_ROOT.rglob("*.py"))
        if (_read_model_type(p) or "").strip().lower() in FAMILY_VALUES
    ]


FAMILY_TEMPLATES = _family_templates()


def _load(path: pathlib.Path):
    module_name = re.sub(r"\W", "_", f"declared_size_{path.stem}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader, f"{path}: importlib could not build a spec"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measured_edge(torch, module, num_classes: int = 3) -> tuple[int, int]:
    """The (H, W) the backbone receives for a square input at the declared size.

    Both tests below share this one measurement rather than each rolling its
    own: the exemption list is only meaningful if it is checked against exactly
    the number the equality assertion would have used.
    """
    declared = getattr(module, "image_size", None)
    assert isinstance(declared, int) and declared > 0, (
        f"{module.__name__}: image_size must be a positive int, got {declared!r}"
    )

    entry_name = getattr(module, "main_class", None) or getattr(
        module, "main_method", None
    )
    model = getattr(module, entry_name)(num_classes)

    transform = getattr(model, "transform", None)
    assert transform is not None, (
        f"{module.__name__}: no `transform` submodule. Every detector in this "
        f"family owns a GeneralizedRCNNTransform — it is the one place the "
        f"declared edge takes effect, and without it this check cannot measure "
        f"anything. A template that resizes elsewhere must expose that seam."
    )

    seen: list[tuple[int, int]] = []

    def record(_module, _inputs, output):
        image_list = output[0]
        seen.append((int(image_list.tensors.shape[-2]), int(image_list.tensors.shape[-1])))

    handle = transform.register_forward_hook(record)
    try:
        model.eval()
        with torch.no_grad():
            model([torch.rand(3, declared, declared)])
    finally:
        handle.remove()

    assert seen, f"{module.__name__}: the transform hook never fired"
    return seen[0]


def test_family_templates_were_found() -> None:
    """Guard the guard: this file is driven by a file scan, so an empty scan
    would pass by checking nothing (the backend#1859 silent-green shape)."""
    assert "rcnn" in FAMILY_VALUES, (
        f"{_schema_path().name}: {FAMILY!r} lost its legacy 'rcnn' alias — the "
        f"scan just stopped covering every template declaring it"
    )
    # A RUNNING TOTAL, like the census in test_check_dump_coverage.py: a
    # rebase re-counts the tree rather than keeping this branch's literal.
    assert len(FAMILY_TEMPLATES) >= 15, (
        f"expected the {FAMILY} roster under {OD_ROOT}, found "
        f"{[p.name for p in FAMILY_TEMPLATES]} — did the tree move? The floor "
        f"tracks the live roster; raise it as the roster grows."
    )


@pytest.mark.parametrize(
    "path", FAMILY_TEMPLATES, ids=lambda p: str(p.relative_to(ROOT))
)
def test_declared_image_size_is_the_measured_edge(path: pathlib.Path) -> None:
    torch = pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")

    if path.stem in _DECLARED_SIZE_IS_DECORATIVE:
        pytest.skip(
            f"{path.stem}: known backend#3058 debt, pinned by "
            f"test_every_exempt_template_still_mismatches instead"
        )

    module = _load(path)
    declared = module.image_size
    height, width = measured_edge(torch, module)
    assert (height, width) == (declared, declared), (
        f"{path.relative_to(ROOT)}: declares image_size={declared} but the "
        f"backbone receives {height}x{width}. The declared edge decides how "
        f"much resolution the dataset keeps, so a template must declare the "
        f"resolution it runs at (backend#3058). Either set image_size to "
        f"{height} or build the transform with min_size == max_size == "
        f"{declared}."
    )


@pytest.mark.parametrize("stem", sorted(_DECLARED_SIZE_IS_DECORATIVE))
def test_every_exempt_template_still_mismatches(stem: str) -> None:
    """The exemption list, pinned from the other side.

    An entry that has been fixed must be removed, or the next template to
    regress under that name inherits a silent pass. This is the same
    fail-closed shape ``test_ci_ram_skip_entries_exist`` uses for its skip set.
    """
    torch = pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")

    matches = [p for p in FAMILY_TEMPLATES if p.stem == stem]
    assert matches, (
        f"_DECLARED_SIZE_IS_DECORATIVE names {stem!r}, which is no longer a "
        f"{FAMILY} template — delete the entry"
    )

    module = _load(matches[0])
    height, width = measured_edge(torch, module)
    expected = _DECLARED_SIZE_IS_DECORATIVE[stem]
    assert (height, width) == (expected, expected), (
        f"{stem}: expected the known backend#3058 mismatch (declares "
        f"{module.image_size}, runs at {expected}) but it now runs at "
        f"{height}x{width}. If it was fixed, remove it from "
        f"_DECLARED_SIZE_IS_DECORATIVE so the equality assertion covers it."
    )
