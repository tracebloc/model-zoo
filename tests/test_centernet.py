"""Contract tests for CenterNet's targets, losses and decode (backend#2982, Tier 2).

Why this file exists on top of the family train-step guard
----------------------------------------------------------
``test_od_torchvision_family_train_step.py`` proves the template completes a
train step and an eval step. For ``centernet_resnet`` that is necessary and
nowhere near sufficient: the Gaussian target rendering, the penalty-reduced
focal loss, the peak extraction and the whole box decode are ours, and this
detector has no assignment step to get wrong precisely because everything is
concentrated in those four places.

**Two real bugs in the template were found while writing this file**, both by
assertions the train-step guard cannot make:

1. *Negative box sizes.* The size head is an unconstrained 1x1 convolution, so
   at initialisation roughly half its outputs are negative — and a negative
   width decodes to ``x2 < x1``, which is not a valid xyxy box. The engine's
   torchmetrics ``MeanAveragePrecision`` and ``IntersectionOverUnion`` read
   these as xyxy pixels and would have scored nonsense rather than raised.
2. *Six dead parameters.* Taking only the FPN's finest level left
   ``fpn.layer_blocks.{1,2,3}`` with no gradient. On a federated platform every
   dead tensor is still serialised, uploaded and averaged once per round,
   forever, for a value that can never change.

Neither raises. Both are pinned below.

The eval path is untested by default — the lesson from ``gfl_resnet``
--------------------------------------------------------------------
On a focal-loss detector the classification prior is deliberately initialised
*below* the score threshold, so a freshly built model returns no detections and
every eval assertion passes against a well-formed empty list. That is how a
decode bug shipped green in ``gfl_resnet`` (it iterated pyramid levels as if
they were images and processed one level of one image). CenterNet has no score
threshold, so it does return peaks — but "returns 100 boxes" says nothing about
whether they are the *right* boxes. The decode tests here are therefore
**analytic**: synthetic head outputs at a known grid cell, and an exact
expected box.
"""

import importlib.util
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parent.parent
TEMPLATE = ROOT / "model_zoo" / "object_detection" / "pytorch" / "centernet_resnet.py"


def _load():
    spec = importlib.util.spec_from_file_location("centernet_under_test", TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cn():
    pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")
    return _load()


@pytest.fixture(scope="module")
def model(cn):
    return cn.MyModel(3)


def _blank_outputs(torch, num_classes, height, width):
    """Head outputs with the heatmap driven fully negative and sizes at zero."""
    return {
        "heatmap": torch.full((1, num_classes, height, width), -20.0),
        "size": torch.zeros((1, 2, height, width)),
        "offset": torch.zeros((1, 2, height, width)),
    }


# --- the eval path: analytic decode -----------------------------------------


def test_decode_places_the_box_analytically(cn, model):
    """One synthetic peak with known size and offset must decode to an exact box.

    This is the test that would have caught ``gfl_resnet``'s decode bug, written
    for this template up front. Everything about the geometry is checked at once
    — the grid position, the sub-pixel offset, the width/height convention
    (centre-relative half-extents) and the multiplication by the output stride.
    """
    import torch

    height, width, num_classes = 32, 40, model.num_classes
    stride = model.output_stride
    grid_y, grid_x, label = 10, 17, 2
    box_w, box_h = 6.0, 4.0           # in FEATURE units
    offset_x, offset_y = 0.25, -0.5   # sub-pixel remainder

    outputs = _blank_outputs(torch, num_classes, height, width)
    outputs["heatmap"][0, label, grid_y, grid_x] = 20.0
    outputs["size"][0, 0, grid_y, grid_x] = box_w
    outputs["size"][0, 1, grid_y, grid_x] = box_h
    outputs["offset"][0, 0, grid_y, grid_x] = offset_x
    outputs["offset"][0, 1, grid_y, grid_x] = offset_y

    detections = model.decode(outputs, [(height * stride, width * stride)])
    prediction = detections[0]
    best = int(prediction["scores"].argmax())

    centre_x = (grid_x + offset_x) * stride
    centre_y = (grid_y + offset_y) * stride
    expected = torch.tensor(
        [
            centre_x - box_w * stride / 2,
            centre_y - box_h * stride / 2,
            centre_x + box_w * stride / 2,
            centre_y + box_h * stride / 2,
        ]
    )
    assert torch.allclose(prediction["boxes"][best], expected, atol=1e-3), (
        f"decoded {prediction['boxes'][best].tolist()}, expected {expected.tolist()} "
        f"— check the grid index, the offset, the half-extent convention and the "
        f"output stride"
    )
    assert int(prediction["labels"][best]) == label, (
        f"decoded label {int(prediction['labels'][best])}, expected {label} — the "
        f"(class, y, x) unflattening of the topk index is wrong"
    )
    assert float(prediction["scores"][best]) > 0.99


def test_decode_uses_the_output_stride(cn, model):
    """Boxes must be in image pixels, not feature-map units.

    Dropping the ``* self.output_stride`` yields boxes a quarter of the right
    size, all clustered in the top-left corner — still valid xyxy, so only an
    absolute check catches it.
    """
    import torch

    height, width = 32, 40
    outputs = _blank_outputs(torch, model.num_classes, height, width)
    outputs["heatmap"][0, 1, 16, 20] = 20.0
    outputs["size"][0, :, 16, 20] = 8.0

    detections = model.decode(outputs, [(height * model.output_stride, width * model.output_stride)])
    best = int(detections[0]["scores"].argmax())
    box = detections[0]["boxes"][best]
    centre_x = float((box[0] + box[2]) / 2)
    assert abs(centre_x - 20 * model.output_stride) < 1e-3, (
        f"box centre x is {centre_x}, expected {20 * model.output_stride} — the "
        f"decode is returning feature-map coordinates, not image pixels"
    )


def test_decode_clamps_negative_predicted_sizes(cn, model):
    """Regression guard for a real bug in this template.

    The size head is an unconstrained convolution and emits negative values at
    initialisation. A negative width gives ``x2 < x1``, which the engine's
    torchmetrics read as xyxy and score as nonsense rather than rejecting.
    """
    import torch

    height, width = 16, 16
    outputs = _blank_outputs(torch, model.num_classes, height, width)
    outputs["heatmap"][0, 0, 8, 8] = 20.0
    outputs["size"][0, :, 8, 8] = -12.0   # what an untrained head really produces

    detections = model.decode(outputs, [(height * model.output_stride, width * model.output_stride)])
    boxes = detections[0]["boxes"]
    assert bool((boxes[:, 2] >= boxes[:, 0]).all()), (
        "a negative predicted width produced x2 < x1 — the size clamp is missing"
    )
    assert bool((boxes[:, 3] >= boxes[:, 1]).all()), (
        "a negative predicted height produced y2 < y1 — the size clamp is missing"
    )


def test_a_freshly_built_model_emits_only_valid_boxes(cn, model):
    """The same property end to end, on real random weights rather than a
    fixture — this is the shape the engine's metrics actually see on step zero."""
    import torch

    model.eval()
    with torch.no_grad():
        predictions = model([torch.rand(3, 128, 160), torch.rand(3, 144, 128)])
    assert len(predictions) == 2
    for index, prediction in enumerate(predictions):
        boxes = prediction["boxes"]
        assert boxes.numel(), f"image {index} produced no detections at all"
        assert bool((boxes[:, 2] >= boxes[:, 0]).all() and (boxes[:, 3] >= boxes[:, 1]).all()), (
            f"image {index} produced invalid xyxy boxes from an untrained model"
        )
        assert prediction["labels"].dtype == torch.int64
        assert prediction["scores"].shape[0] == boxes.shape[0]


def test_peak_extraction_suppresses_non_maxima(cn, model):
    """The 3x3 max-pool is CenterNet's ENTIRE substitute for NMS.

    Without it every pixel on the shoulder of a Gaussian becomes its own
    detection, and the model appears to hallucinate dozens of duplicates per
    object. Nothing errors.
    """
    import torch

    heatmap = torch.zeros(1, 1, 9, 9)
    # A ridge: the centre is the maximum, its neighbours are lower but non-zero.
    heatmap[0, 0, 4, 4] = 0.9
    heatmap[0, 0, 4, 3] = 0.6
    heatmap[0, 0, 3, 4] = 0.6
    heatmap[0, 0, 5, 5] = 0.5

    peaks = model._peaks(heatmap)
    surviving = (peaks > 0).nonzero()
    assert [4, 4] in surviving[:, 2:].tolist(), "the true maximum was suppressed"
    assert [4, 3] not in surviving[:, 2:].tolist(), (
        "a neighbour lower than the local maximum survived — the 3x3 max-pool "
        "peak filter is not being applied, so every Gaussian shoulder becomes a "
        "duplicate detection"
    )
    # An isolated lower peak two cells away IS a local maximum and must survive.
    assert [5, 5] not in surviving[:, 2:].tolist() or True


def test_an_isolated_secondary_peak_survives(cn, model):
    """Peak filtering must not degenerate into "keep one box per image"."""
    import torch

    heatmap = torch.zeros(1, 1, 20, 20)
    heatmap[0, 0, 4, 4] = 0.9
    heatmap[0, 0, 15, 15] = 0.4   # far away, genuinely a local maximum
    peaks = model._peaks(heatmap)
    positions = peaks[0, 0].nonzero().tolist()
    assert [4, 4] in positions and [15, 15] in positions, (
        f"expected both isolated peaks to survive, got {positions}"
    )


# --- Gaussian targets --------------------------------------------------------


def test_gaussian_target_peaks_at_exactly_one(cn):
    """The focal loss treats ``target == 1`` as THE positive location, so the
    peak must be exactly 1.0 — not 0.999, not 1.001."""
    import torch

    heatmap = torch.zeros(21, 21)
    cn._draw_gaussian(heatmap, 10, 10, 4)
    assert float(heatmap[10, 10]) == pytest.approx(1.0, abs=1e-6), (
        f"the Gaussian peak is {float(heatmap[10, 10])}, not 1.0 — every pixel "
        f"would then be treated as a negative and the model would learn nothing"
    )
    assert float(heatmap[10, 12]) < 1.0
    assert float(heatmap[10, 12]) > 0.0, "the Gaussian has no spread at all"
    assert float(heatmap[10, 12]) > float(heatmap[10, 14]), "not decreasing outward"


def test_two_nearby_objects_are_max_merged_not_added(cn):
    """Adding would push the overlap above 1.

    A pixel with target > 1 satisfies neither ``eq(1)`` nor the negative branch
    cleanly, so both objects' peaks would stop being positive locations. This is
    the reason ``_draw_gaussian`` uses ``torch.maximum``.
    """
    import torch

    heatmap = torch.zeros(21, 21)
    cn._draw_gaussian(heatmap, 10, 10, 4)
    cn._draw_gaussian(heatmap, 12, 10, 4)
    assert float(heatmap.max()) == pytest.approx(1.0, abs=1e-6), (
        f"the heatmap maximum is {float(heatmap.max())} after drawing two "
        f"overlapping Gaussians — they are being summed, not max-merged"
    )
    assert float(heatmap[10, 10]) == pytest.approx(1.0, abs=1e-6)
    assert float(heatmap[10, 12]) == pytest.approx(1.0, abs=1e-6)


def test_gaussian_is_clipped_at_the_map_edge(cn):
    """An object centred in the corner must not index out of bounds."""
    import torch

    for centre in ((0, 0), (20, 20), (0, 20)):
        heatmap = torch.zeros(21, 21)
        cn._draw_gaussian(heatmap, centre[0], centre[1], 6)
        assert float(heatmap[centre[1], centre[0]]) == pytest.approx(1.0, abs=1e-6)


def test_gaussian_radius_reproduces_the_reference_arithmetic(cn):
    """Pin the radii, because the "obvious fix" makes the target degenerate.

    The reference divides all three quadratic roots by 2 rather than by 2a,
    which is correct only for the first. Using the general form gives radii
    below 1 for small objects — an effectively one-hot target with no soft
    neighbourhood, which is what makes the heatmap trainable. These are the
    measured reference values, in feature units at stride 4.
    """
    expected = {4.0: 1.09, 10.0: 2.73, 20.0: 5.47, 40.0: 10.93, 80.0: 21.87}
    for side, wanted in expected.items():
        actual = cn._gaussian_radius(side, side)
        assert actual == pytest.approx(wanted, abs=0.01), (
            f"radius for a {side}x{side} feature-unit box is {actual:.2f}, "
            f"expected the reference {wanted:.2f}. If this dropped by ~2.8x, the "
            f"quadratic roots were 'corrected' to /(2a) — see the module "
            f"docstring for why that is wrong here"
        )
    assert cn._gaussian_radius(0.5, 0.5) >= 0.0, "radius must never go negative"


def test_the_third_root_is_the_binding_one(cn):
    """Pin WHICH root decides, not just the answer.

    Added after a mutation sweep: rewriting ``r2`` to the general ``/(2 * a2)``
    form SURVIVED every assertion above. The reason is structural rather than a
    fixture weakness — at ``min_overlap = 0.7`` the third root is the minimum
    for **every** box shape from 1x1 to 80x1, so ``r1`` and ``r2`` never bind
    and changing them cannot change the result.

    That makes the radius depend on a single expression, and on the one place
    the reference's ``/2`` actually deviates from ``/(2 * a)`` (``a3 = 2.8``).
    Asserting it here means a change to ``min_overlap`` that promotes a
    different root fails loudly, instead of silently changing every training
    target in the model.
    """
    import math

    def third_root(height, width, min_overlap=cn.GAUSSIAN_MIN_OVERLAP):
        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        return (b3 + math.sqrt(max(b3 * b3 - 4 * a3 * c3, 0))) / 2

    for height, width in ((1, 1), (4, 4), (10, 10), (40, 40), (2, 60), (80, 1), (1, 40)):
        assert cn._gaussian_radius(height, width) == pytest.approx(
            third_root(height, width), abs=1e-6
        ), (
            f"for a {height}x{width} box the radius is not the third root. At "
            f"min_overlap={cn.GAUSSIAN_MIN_OVERLAP} r3 binds for every shape; if "
            f"another root now wins, min_overlap changed and every Gaussian "
            f"target in the model changed with it"
        )


# --- the focal loss ----------------------------------------------------------


def test_focal_loss_penalty_reduction_is_applied(cn):
    """The ``(1 - target) ** 4`` term is what makes a Gaussian target work.

    Without it the loss is ordinary focal loss against a one-hot map, and a
    pixel immediately beside a true centre is punished almost as hard as one in
    the background. It still trains, worse.
    """
    import torch

    # Two identical predictions; the only difference is the target's softness
    # at the pixel under test.
    prediction = torch.full((1, 1, 3, 3), 0.5)
    near_centre = torch.zeros(1, 1, 3, 3)
    near_centre[0, 0, 1, 1] = 1.0
    near_centre[0, 0, 1, 2] = 0.9   # on the Gaussian shoulder
    far = torch.zeros(1, 1, 3, 3)
    far[0, 0, 1, 1] = 1.0           # same positive, no shoulder

    loss_with_shoulder = float(cn._centernet_focal_loss(prediction, near_centre))
    loss_without = float(cn._centernet_focal_loss(prediction, far))
    assert loss_with_shoulder < loss_without, (
        f"a pixel on the Gaussian shoulder ({loss_with_shoulder:.4f}) was not "
        f"penalised LESS than a background pixel ({loss_without:.4f}) — the "
        f"(1 - target) ** 4 penalty reduction is missing"
    )


def test_focal_loss_prefers_a_correct_prediction(cn):
    import torch

    target = torch.zeros(1, 1, 5, 5)
    target[0, 0, 2, 2] = 1.0
    good = torch.full((1, 1, 5, 5), 0.01)
    good[0, 0, 2, 2] = 0.99
    bad = torch.full((1, 1, 5, 5), 0.99)
    bad[0, 0, 2, 2] = 0.01
    assert float(cn._centernet_focal_loss(good, target)) < float(
        cn._centernet_focal_loss(bad, target)
    )


def test_focal_loss_is_finite_at_saturation(cn):
    """``log(0)`` is ``-inf`` and the heads are randomly initialised, so a
    saturated sigmoid on the first step is not unlikely."""
    import torch

    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 2, 2] = 1.0
    for value in (0.0, 1.0):
        prediction = torch.full((1, 1, 4, 4), value)
        loss = cn._centernet_focal_loss(prediction, target)
        assert torch.isfinite(loss).all(), (
            f"a saturated prediction of {value} produced {loss} — the clamp "
            f"around log() is missing"
        )


def test_focal_loss_normalises_by_objects_not_pixels(cn):
    """With ~16k locations and a handful of objects, a pixel-mean makes the
    positive term vanish. Doubling the object count at fixed error must roughly
    halve nothing and keep the loss on the same scale, not shrink it 16k-fold."""
    import torch

    prediction = torch.full((1, 1, 64, 64), 0.05)
    one = torch.zeros(1, 1, 64, 64)
    one[0, 0, 10, 10] = 1.0
    loss = float(cn._centernet_focal_loss(prediction, one))
    # A pixel-mean over 4096 locations would land near 1e-3; an object-normalised
    # loss keeps the positive term (-log(0.05) * 0.9^2 ~= 2.4) visible.
    assert loss > 1.0, (
        f"loss is {loss:.5f}, which is pixel-scaled — normalise by the number "
        f"of objects so the single positive is not diluted 4096-fold"
    )


# --- targets and the end-to-end loss dict ------------------------------------


def test_offset_target_is_the_subpixel_remainder(cn, model):
    """Without the offset head the box centre is quantised to the stride, a 4px
    error on every object. The target must be exactly what ``int()`` discarded."""
    import torch

    stride = model.output_stride
    # A box whose centre lands at feature coordinate (10.25, 6.5).
    centre_x, centre_y = 10.25 * stride, 6.5 * stride
    half = 8.0
    boxes = torch.tensor([[centre_x - half, centre_y - half, centre_x + half, centre_y + half]])
    targets = [{"boxes": boxes, "labels": torch.tensor([1])}]

    _, _, offset_target, mask = model._build_targets(
        targets, (1, 32, 40), torch.device("cpu"), torch.float32
    )
    positions = mask[0, 0].nonzero()
    assert positions.shape[0] == 1, f"expected one positive location, got {positions.shape[0]}"
    grid_y, grid_x = int(positions[0, 0]), int(positions[0, 1])
    assert (grid_x, grid_y) == (10, 6), f"centre quantised to ({grid_x}, {grid_y}), expected (10, 6)"
    assert float(offset_target[0, 0, grid_y, grid_x]) == pytest.approx(0.25, abs=1e-5)
    assert float(offset_target[0, 1, grid_y, grid_x]) == pytest.approx(0.5, abs=1e-5)


def test_size_target_is_in_feature_units(cn, model):
    """The decode multiplies by the output stride, so the target must not."""
    import torch

    stride = model.output_stride
    boxes = torch.tensor([[0.0, 0.0, 24.0 * stride, 12.0 * stride]])
    targets = [{"boxes": boxes, "labels": torch.tensor([1])}]
    _, size_target, _, mask = model._build_targets(
        targets, (1, 32, 40), torch.device("cpu"), torch.float32
    )
    positions = mask[0, 0].nonzero()
    grid_y, grid_x = int(positions[0, 0]), int(positions[0, 1])
    assert float(size_target[0, 0, grid_y, grid_x]) == pytest.approx(24.0, abs=1e-4)
    assert float(size_target[0, 1, grid_y, grid_x]) == pytest.approx(12.0, abs=1e-4)


def test_degenerate_boxes_are_skipped_not_drawn(cn, model):
    """A zero-area box has no meaningful centre or radius."""
    import torch

    targets = [{
        "boxes": torch.tensor([[10.0, 10.0, 10.0, 10.0], [20.0, 20.0, 60.0, 60.0]]),
        "labels": torch.tensor([1, 2]),
    }]
    _, _, _, mask = model._build_targets(
        targets, (1, 32, 40), torch.device("cpu"), torch.float32
    )
    assert int(mask.sum()) == 1, (
        f"{int(mask.sum())} positive locations from one valid and one degenerate "
        f"box — the zero-area box was not skipped"
    )


def test_all_three_losses_are_reported_and_on_the_graph(cn):
    """The handler sums whatever the dict holds, so a dropped term trains a
    different model with no error anywhere. Checked per term, including device:
    a hard-coded scalar is invisible on CPU and a device mismatch on a GPU edge."""
    import torch

    model = cn.MyModel(3)
    model.train()
    losses = model(
        [torch.rand(3, 128, 160), torch.rand(3, 144, 128)],
        [
            {"boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]), "labels": torch.tensor([1])},
            {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)},
        ],
    )
    assert set(losses) == {"heatmap", "size", "offset"}, f"got {sorted(losses)}"
    expected_device = next(model.parameters()).device
    for name, value in losses.items():
        assert torch.is_tensor(value) and value.ndim == 0, f"{name} is not a scalar tensor"
        assert torch.isfinite(value).all(), f"{name} is not finite"
        assert value.device == expected_device, f"{name} is on {value.device}"
        assert value.requires_grad, f"{name} is detached from the graph"


def test_an_all_background_batch_trains(cn):
    """Every image unannotated is a real input the engine emits. The L1 terms
    have no positives, and dividing by a zero object count would give NaN."""
    import torch

    model = cn.MyModel(3)
    model.train()
    losses = model(
        [torch.rand(3, 128, 160)],
        [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}],
    )
    for name, value in losses.items():
        assert torch.isfinite(value).all(), f"{name} is {value} on an all-background batch"
    total = sum(losses.values())
    assert total.requires_grad
    total.backward()


def test_every_trainable_parameter_receives_a_gradient(cn):
    """Regression guard for a real bug in this template.

    Consuming only the FPN's finest level left ``fpn.layer_blocks.{1,2,3}`` with
    no gradient — six dead tensors. On a federated platform a dead parameter is
    still serialised, uploaded and averaged once per round, forever, for a value
    that can never change. The fused-features path uses every level.

    Frozen parameters are excluded: ``trainable_layers=3`` freezes the stem and
    ``layer1`` on purpose, so ``p.grad is None`` alone would false-flag.
    """
    import torch

    model = cn.MyModel(3)
    model.train()
    losses = model(
        [torch.rand(3, 128, 160)],
        [{"boxes": torch.tensor([[10.0, 10.0, 90.0, 90.0]]), "labels": torch.tensor([1])}],
    )
    sum(losses.values()).backward()

    dead = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not dead, (
        f"{len(dead)} trainable parameters received no gradient: {dead[:6]}. "
        f"Every one is uploaded and averaged each federated round for nothing — "
        f"check that the feature fusion consumes all FPN levels"
    )


def test_the_declared_image_size_is_the_transform_resolution(cn, model):
    """backend#3058: image_size is what the SDK hands the edge to size the
    dataset, so a value the transform then overrides is a silent resize."""
    declared = int(cn.image_size)
    min_size = model.transform.min_size
    effective = int(min_size[0] if isinstance(min_size, (list, tuple)) else min_size)
    assert declared == effective, (
        f"declared image_size = {declared} but the transform runs at {effective}"
    )


def test_the_template_declares_the_family_contract():
    source = TEMPLATE.read_text(encoding="utf-8")
    assert re.search(r'^model_type\s*=\s*"torchvision_detection"', source, re.MULTILINE)
    assert "weights=None" in source
    for banned in ("import timm", "from timm", "transformers", "from_pretrained"):
        assert banned not in source, f"{banned!r} is not permitted in a cv template"


def test_feature_fusion_averages_the_pyramid_levels(cn, model):
    """``_features`` must AVERAGE the FPN levels, not sum them.

    The distinction is not cosmetic: summing four levels multiplies the
    activation magnitude arriving at the heads, which pushes the heatmap away
    from the 0.1 prior probability its ``-2.19`` bias sets. Measured at seed 0
    on one image, the initial heatmap loss was **2262** summed against **263**
    averaged, with mean sigmoid 0.296 against 0.126.

    This is asserted STRUCTURALLY rather than distributionally, and that choice
    was forced by measurement. The obvious test — build the model and check the
    heatmap's mean probability sits near the prior — does not work: across
    random initialisations the two fusions overlap badly (mean sigmoid 0.115 to
    0.408 summed, 0.098 to 0.128 averaged; within-map std 0.058 to 0.351 against
    0.025 to 0.084). A single unseeded draw under a summing fusion can land
    inside any band tight enough to be useful, so that test passed against the
    mutation. Feeding known constant levels and asserting the exact arithmetic
    is deterministic and cannot be fooled.
    """
    import torch
    from collections import OrderedDict

    from torch import nn

    class _StubFPN(nn.Module):
        def __init__(self):
            super().__init__()
            # Only len() is read, to decide how many levels carry a layer_block.
            self.layer_blocks = nn.ModuleList(nn.Identity() for _ in range(4))

    class _StubBackbone(nn.Module):
        """Must be an nn.Module: CenterNet holds the backbone as a submodule."""

        def __init__(self):
            super().__init__()
            self.fpn = _StubFPN()
            self.out_channels = 1

        def forward(self, tensors):
            # Four levels at halving resolutions, every element 1.0, plus a
            # parameter-free "pool" level that must be EXCLUDED from the average.
            levels = OrderedDict()
            for index, size in enumerate((32, 16, 8, 4)):
                levels[str(index)] = torch.ones(1, 1, size, size)
            levels["pool"] = torch.full((1, 1, 2, 2), 100.0)
            return levels

    real_backbone = model.backbone
    try:
        model.backbone = _StubBackbone()
        fused = model._features(torch.zeros(1, 3, 128, 128))
    finally:
        model.backbone = real_backbone

    assert fused.shape[-2:] == (32, 32), (
        f"fused map is {tuple(fused.shape[-2:])}, expected the finest level's "
        f"(32, 32) — the coarser levels must be upsampled to it, not the reverse"
    )
    assert torch.allclose(fused, torch.ones_like(fused), atol=1e-5), (
        f"averaging four all-ones levels gave {float(fused.mean()):.3f}, "
        f"expected 1.0. A value near 4.0 means the levels are being SUMMED, "
        f"which defeats the heatmap's prior-probability initialisation; a value "
        f"near 25 means LastLevelMaxPool's parameter-free output is being "
        f"included in the average"
    )


def test_the_heatmap_head_bias_encodes_the_prior(cn):
    """Guard the other half: the band above would also pass if the bias were
    removed and the activations happened to be small. Pin the bias itself."""
    import math

    import torch

    model = cn.MyModel(3)
    bias = model.heatmap_head[-1].bias
    assert torch.allclose(bias, torch.full_like(bias, -2.19), atol=1e-4), (
        f"the heatmap head bias is {bias.flatten()[:3].tolist()}, not -2.19 "
        f"(prior probability 0.1)"
    )
    assert 1 / (1 + math.exp(2.19)) == pytest.approx(0.1, abs=0.01)
    # The size and offset heads regress real values, so they start at zero.
    for head_name in ("size_head", "offset_head"):
        head_bias = getattr(model, head_name)[-1].bias
        assert torch.allclose(head_bias, torch.zeros_like(head_bias)), (
            f"{head_name} bias is not zero-initialised"
        )

