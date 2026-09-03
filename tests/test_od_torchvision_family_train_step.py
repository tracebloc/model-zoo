"""Contract test: every ``torchvision_detection``-family OD template survives one
train step and one eval step against the targets the platform actually supplies.

Background (backend#2988)
-------------------------
``object_detection/pytorch/mask_rcnn.py`` shipped on ``develop`` and could not
complete a single training step. Mask R-CNN carries a mask head, and
torchvision's ``RoIHeads.forward`` unconditionally reads a ``masks`` key from
every target when one is present::

    gt_masks = [t["masks"] for t in targets]   # torchvision/models/detection/roi_heads.py

Nothing in the object-detection path supplies that key — an OD manifest is
Pascal-VOC XML, which carries boxes, not segmentation masks:

- engine  — ``core/datasets/image_detection_dataset_pytorch.py`` emits
  ``{boxes, labels, area, iscrowd}`` (+ optional ``image_id``)
- SDK      — ``tracebloc/utils/detection_utils.py`` dummy RCNN dataset emits
  ``{boxes, labels}``

so the template raised ``ValueError: Every element of targets should have a
masks key`` on the first batch, in both places. It was deleted rather than
patched: instance segmentation is what Mask R-CNN is *for*, and the box-only
path it was standing in for is already covered by ``faster_rcnn_resnet``.

Why nothing caught it
---------------------
The two checks that would have are excluded from coverage in the SDK
(``general_utils.py`` torchvision branch and ``torch_object_detector.py``
``_rcnn_training`` are both ``# pragma: no cover``), and this repo asserted only
that a template *constructs* (``test_model_contract.py::test_model_instantiates``)
and that its ``model_type`` *routes* (``test_od_model_type_contract.py``).
Constructing and routing are not training. This file closes that gap: it runs the
handler's actual contract, so a template that cannot train fails here rather than
on a customer's edge.

What this pins — the engine's ``TorchvisionDetectionHandler`` contract
----------------------------------------------------------------------
- ``model(images, targets)`` in train mode returns a non-empty dict of finite
  scalar losses
- ``model(images)`` in eval mode returns ``List[Dict]`` with ``boxes`` (pixel
  xyxy), ``scores``, ``labels``
- both hold for **``{boxes, labels}``-only targets** — the intersection of what
  the two producers emit, i.e. the weakest guarantee a template may rely on. A
  template needing more than that is broken by definition.
- both hold when an image in the batch has **zero objects**, which the engine's
  dataset emits explicitly for an unannotated image.

Family, not literal string — the trap
-------------------------------------
The templates are selected by FAMILY, derived from the vendored engine schema:
``torchvision_detection`` **plus its aliases**, and ``rcnn`` is one. Selecting on
the literal string ``"torchvision_detection"`` would have skipped exactly the two
files that declare the legacy alias — ``faster_rcnn_resnet`` and the very
``mask_rcnn`` this test exists because of. A guard that misses the bug that
motivated it is worse than no guard.
"""

import importlib.util
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parent.parent
OD_ROOT = ROOT / "model_zoo" / "object_detection"
CONTRACTS = pathlib.Path(__file__).parent / "contracts" / "tracebloc_engine"

#: The family whose contract this file exercises. ``yolo`` speaks a different
#: contract entirely (grid tensor + external ``loss.py``), so it is out of scope.
FAMILY = "torchvision_detection"


def _schema_path() -> pathlib.Path:
    """Newest vendored ``object_detection_families.v<N>.json``.

    Resolved by glob rather than pinned by name: the vendored schema is version-
    bumped in place when the engine's vocabulary narrows (v1 -> v2 dropped
    ``hf_transformer`` with backend#2973), and a test pinning a filename breaks
    on the rename instead of adopting it. Sorted numerically, so v10 does not
    sort before v2.
    """
    paths = sorted(
        CONTRACTS.glob("object_detection_families.v*.json"),
        key=lambda p: int(re.search(r"\.v(\d+)\.json$", p.name).group(1)),
    )
    assert paths, f"no vendored OD families schema under {CONTRACTS}"
    return paths[-1]


def _family_values() -> frozenset[str]:
    """The declaration values that route to this family — the family name plus
    every alias — normalized the way the engine's ``resolve_family`` does."""
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    entries = [f for f in schema["families"] if f["family"] == FAMILY]
    assert entries, f"{_schema_path().name}: no {FAMILY!r} family entry"
    values = {FAMILY, *entries[0].get("aliases", [])}
    return frozenset(v.strip().lower() for v in values)


FAMILY_VALUES = _family_values()


def _read_model_type(path: pathlib.Path) -> str | None:
    """Declared module-level ``model_type``, read statically — no import — so
    file SELECTION costs nothing in the CI jobs that cannot run torch.

    Deliberately local, not shared with ``test_od_model_type_contract.py``'s
    identical reader (nor ``test_model_contract.py``'s ``_read_framework``):
    every test file in this suite is self-contained, and the one file worth
    extracting a helper into is being rewritten by backend#2973 right now.
    Worth folding the three readers into one ``tests/`` helper once that lands.
    """
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


def _build(path: pathlib.Path, num_classes: int):
    """Import the template and construct its model at ``num_classes``."""
    module_name = re.sub(r"\W", "_", f"od_family_{path.stem}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader, f"{path}: importlib could not build a spec"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    entry = getattr(module, "main_class", None) or getattr(module, "main_method", None)
    assert entry, f"{path}: neither main_class nor main_method is defined"
    return module, getattr(module, entry)(num_classes)


def _targets(torch, num_classes: int) -> list[dict]:
    """Targets carrying ONLY ``boxes`` and ``labels``.

    That is the intersection of the two producers (see the module docstring), so
    a template that trains against these trains against either. Pixel xyxy with
    positive area — torchvision rejects degenerate boxes outright, which would
    mask a real failure behind an input error. The second entry is the engine's
    zero-object target for an unannotated image, shapes and dtypes matched to
    ``image_detection_dataset_pytorch``.

    **Labels span the full model-space range, ``1`` to ``num_classes``.** The
    engine's ``TorchvisionDetectionHandler`` shifts a dataset-space label
    ``[0, C-1]`` up past torchvision's background index before the model sees it
    (backend#3062), so a template in this family is handed ``[1, C]`` and must
    allocate ``C + 1`` head channels — which every one of them does, as
    ``num_classes = num_classes + 1  # 1 for background``.

    These labels used to be ``[1, max(1, num_classes - 1)]``: already
    background-avoiding, so this file would have been red on the pre-#3062
    contract too, but spanning only ``[1, C-1]`` and never the top of the range.
    A template that allocated ``num_classes`` channels instead of
    ``num_classes + 1`` therefore passed here and then raised on the LAST class
    at training time — the one class the old range could not reach. Using
    ``num_classes`` itself makes the ``+ 1`` head width a required property
    rather than an accident, which is what closes that hole for templates not
    written yet and for customer-supplied models.
    """
    return [
        {
            "boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0], [70.0, 70.0, 110.0, 120.0]]),
            "labels": torch.tensor([1, num_classes], dtype=torch.int64),
        },
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        },
    ]


def _assert_speaks_handler_contract(torch, model, name: str, num_classes: int) -> dict:
    """Run one train step and one eval step; return the loss dict.

    Factored out so the guard can be pointed at a model KNOWN to violate the
    contract — see ``test_guard_rejects_a_mask_headed_model``. A gate whose whole
    value is going red has to be shown going red (the convention in
    ``test_check_dump_coverage.py``); without that, this file could assert
    nothing and still pass on every template forever.
    """
    images = [torch.rand(3, 128, 160), torch.rand(3, 144, 128)]

    model.train()
    losses = model(images, _targets(torch, num_classes))
    assert isinstance(losses, dict) and losses, (
        f"{name}: train mode returned {type(losses).__name__}, not a non-empty "
        f"loss dict — the handler calls sum(losses.values()) on this"
    )
    for key, value in losses.items():
        assert torch.is_tensor(value) and value.ndim == 0, (
            f"{name}: loss {key!r} is {value!r}, not a scalar tensor"
        )
        assert torch.isfinite(value).all(), f"{name}: loss {key!r} is not finite"

    model.eval()
    with torch.no_grad():
        preds = model(images)
    assert isinstance(preds, list) and len(preds) == len(images), (
        f"{name}: eval mode returned {type(preds).__name__} of "
        f"{len(preds) if isinstance(preds, list) else '?'}, expected a list of "
        f"{len(images)} dicts"
    )
    for i, pred in enumerate(preds):
        missing = {"boxes", "scores", "labels"} - set(pred)
        assert not missing, f"{name}: eval prediction {i} is missing {sorted(missing)}"
        boxes = pred["boxes"]
        assert boxes.ndim == 2 and boxes.shape[-1] == 4, (
            f"{name}: eval prediction {i} boxes have shape {tuple(boxes.shape)}, "
            f"expected (N, 4)"
        )
        if boxes.numel():
            assert bool((boxes[:, 2] >= boxes[:, 0]).all() and (boxes[:, 3] >= boxes[:, 1]).all()), (
                f"{name}: eval prediction {i} boxes are not xyxy (x2<x1 or y2<y1); "
                f"the metrics read them as xyxy pixels"
            )
    return losses


def test_family_templates_were_found() -> None:
    """Guard the guard: the parametrized test below is driven by a file scan, so
    an empty scan would make the whole file pass by checking nothing — the
    silent-green shape of backend#1859. Also asserts the alias is in play, since
    dropping it is what would silently narrow the scan to nothing useful."""
    assert "rcnn" in FAMILY_VALUES, (
        f"{_schema_path().name}: {FAMILY!r} lost its legacy 'rcnn' alias — if the "
        f"engine really dropped it, update FAMILY_VALUES' users; if not, this "
        f"scan just stopped covering every template declaring it"
    )
    assert len(FAMILY_TEMPLATES) >= 3, (
        f"expected the {FAMILY} roster under {OD_ROOT}, found "
        f"{[p.name for p in FAMILY_TEMPLATES]} — did the tree move? The floor "
        f"tracks the live roster: raise it with backend#2982's Tier 0."
    )


@pytest.mark.parametrize(
    "path", FAMILY_TEMPLATES, ids=lambda p: str(p.relative_to(ROOT))
)
def test_family_template_trains_and_evals(path: pathlib.Path) -> None:
    torch = pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")

    num_classes = 3
    _, model = _build(path, num_classes)
    _assert_speaks_handler_contract(torch, model, path.name, num_classes)


def test_guard_rejects_a_mask_headed_model() -> None:
    """The guard, pointed at the defect it was written for.

    This is torchvision's Mask R-CNN — the architecture the deleted
    ``mask_rcnn.py`` built — constructed directly here rather than kept as a
    template. It must FAIL the contract above, on the missing ``masks`` key.

    Two things this pins that the parametrized test cannot: the check is
    genuinely able to go red, and the reason ``mask_rcnn.py`` was deleted is
    still true of the library. If torchvision ever makes ``masks`` optional this
    test fails, and re-homing Mask R-CNN becomes worth revisiting (backend#795).
    """
    torch = pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")

    from torchvision.models import resnet18
    from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
    from torchvision.models.detection.mask_rcnn import MaskRCNN

    # resnet18, not the template's resnet50: this asserts a LIBRARY behaviour,
    # so the smallest backbone that exercises it is the right one for CI.
    backbone = _resnet_fpn_extractor(resnet18(weights=None), trainable_layers=1)
    model = MaskRCNN(backbone, num_classes=4)

    with pytest.raises((ValueError, KeyError)) as excinfo:
        _assert_speaks_handler_contract(torch, model, "MaskRCNN(probe)", 3)

    assert "mask" in str(excinfo.value).lower(), (
        f"a mask-headed detector failed the contract, but not on the masks key: "
        f"{excinfo.value!r} — if torchvision changed its message, update this "
        f"assertion; if it stopped requiring masks, see backend#795"
    )


def test_guard_rejects_a_head_that_omits_the_background_channel() -> None:
    """The label-range half of the guard, shown going red (backend#3062).

    Same convention as the mask-headed probe above: a check whose whole value is
    going red has to be demonstrated going red, or widening ``_targets`` to
    ``[1, num_classes]`` would be an untested edit that looks like a guard.

    This builds the mistake the widened range exists to catch — a detector in
    this family whose classification head has exactly ``num_classes`` channels
    instead of ``num_classes + 1``, i.e. a template that forgot the
    ``# 1 for background`` line. It must FAIL, and specifically on the top of
    the label range: under the old ``[1, num_classes - 1]`` labels this same
    model passed, which is exactly why the hole was invisible.

    RetinaNet rather than Faster R-CNN on purpose: the sigmoid focal head
    one-hots the label directly into its channel dimension, so the out-of-range
    class raises in the loss itself rather than inside cross-entropy, which
    keeps the failure legible.
    """
    torch = pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")

    from torchvision.models import resnet18
    from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
    from torchvision.models.detection.retinanet import RetinaNet

    num_classes = 3

    # resnet18, not the templates' resnet50: this asserts a LIBRARY behaviour,
    # so the smallest backbone that exercises it is the right one for CI.
    def _under_allocated():
        backbone = _resnet_fpn_extractor(resnet18(weights=None), trainable_layers=1)
        # The bug: num_classes, NOT num_classes + 1.
        return RetinaNet(backbone, num_classes=num_classes)

    with pytest.raises((IndexError, RuntimeError)):
        _assert_speaks_handler_contract(
            torch, _under_allocated(), "RetinaNet(no-background-channel)", num_classes
        )

    # Positive control: the SAME model, the SAME assertions, with the label
    # range this file used before backend#3062. It passes — so the red above is
    # produced by the widened range and by nothing else.
    _assert_speaks_handler_contract(
        torch,
        _under_allocated(),
        "RetinaNet(no-background-channel, old label range)",
        num_classes - 1,
    )
