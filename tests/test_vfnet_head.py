"""Contract tests for VFNet's varifocal loss, star-shaped sampling and decode
(backend#2982, Tier 2).

Why this file exists on top of the family train-step guard
----------------------------------------------------------
The loss, the deformable offset construction, the refinement and the decode are
all ours. Each has a silent failure mode:

- **The asymmetry.** VFL's positive term is deliberately *unmodulated*. Adding a
  ``(1 - p) ** gamma`` factor turns it into Quality Focal Loss — a different,
  already-shipped loss (``gfl_resnet``) — and nothing about the training run
  would look wrong.
- **The offset layout.** ``deform_conv2d`` takes ``(y, x)`` pairs relative to the
  kernel's own 3x3 grid. Swapping the pair order transposes every sampling
  point; forgetting to subtract the base grid displaces each point by its grid
  position. Both are shape-identical and produce a model that half works.
- **The refinement.** It predicts a multiplicative scale, so it must start at
  1.0. Starting at 0 makes every refined box a degenerate point while the
  initial box is fine, so the losses stay finite and the model learns to undo
  its own initialisation.
- **The eval path.** On a focal-loss detector the prior sits below
  ``score_thresh``, so a fresh model returns zero detections and every eval
  assertion passes against a well-formed empty list — how a decode bug shipped
  green in ``gfl_resnet``. The decode assertions here are analytic.
"""

import importlib.util
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parent.parent
TEMPLATE = ROOT / "model_zoo" / "object_detection" / "pytorch" / "vfnet_resnet.py"


def _load():
    spec = importlib.util.spec_from_file_location("vfnet_under_test", TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def vf():
    pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")
    return _load()


@pytest.fixture(scope="module")
def built(vf):
    import torch
    from torchvision.models.detection.image_list import ImageList

    model = vf.MyModel(3)
    model.eval()
    image = torch.rand(1, 3, 256, 320)
    with torch.no_grad():
        features = list(model.backbone(image).values())
    anchors = model.anchor_generator(ImageList(image, [(256, 320)]), features)[0]
    return model, features, anchors, list(model.anchor_generator.num_anchors_per_level)


# --- the asymmetry: VFL is NOT quality focal loss ---------------------------


def test_the_positive_term_is_unmodulated(vf):
    """A positive's weight must be exactly its target, with no focal factor.

    This is the single property that distinguishes VFL from the Quality Focal
    Loss already shipped in ``gfl_resnet``. Checked by construction: for a
    positive, ``VFL == target * BCE(logit, target)`` exactly. A ``(1 - p) **
    gamma`` factor would make the ratio depend on the prediction.
    """
    import torch
    import torch.nn.functional as F

    target = torch.tensor([[0.6]])
    for logit in (-2.0, 0.0, 0.5, 3.0):
        logits = torch.tensor([[logit]])
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
        expected = float(target) * float(bce)
        actual = float(vf._varifocal_loss(logits, target))
        assert actual == pytest.approx(expected, rel=1e-5), (
            f"at logit {logit} the positive term is {actual:.6f} but "
            f"target * BCE is {expected:.6f}. The ratio depending on the "
            f"prediction means a focal modulation has been applied to the "
            f"positive, which makes this Quality Focal Loss, not Varifocal Loss"
        )


def test_the_negative_term_keeps_the_focal_modulation(vf):
    """Negatives must be down-weighted by ``alpha * p ** gamma``.

    Without it the ~10^4 negatives per image dominate. This is the half VFL
    keeps, so an "asymmetric" implementation that dropped BOTH modulations would
    pass the test above and fail here.
    """
    import torch
    import torch.nn.functional as F

    target = torch.zeros(1, 1)
    for logit in (-1.0, 0.0, 2.0):
        logits = torch.tensor([[logit]])
        probability = float(torch.sigmoid(logits))
        bce = float(F.binary_cross_entropy_with_logits(logits, target, reduction="sum"))
        expected = vf.VFL_ALPHA * probability ** vf.VFL_GAMMA * bce
        actual = float(vf._varifocal_loss(logits, target))
        assert actual == pytest.approx(expected, rel=1e-5), (
            f"at logit {logit} the negative term is {actual:.6f}, expected "
            f"alpha * p ** gamma * BCE = {expected:.6f}"
        )


def test_vfl_differs_from_symmetric_quality_focal_loss(vf):
    """The direct comparison, so a regression to the symmetric form is caught
    even if the algebra above were somehow satisfied."""
    import torch
    import torch.nn.functional as F

    logits = torch.tensor([[2.0]])
    target = torch.tensor([[0.6]])

    varifocal = float(vf._varifocal_loss(logits, target))
    probabilities = logits.sigmoid()
    symmetric = float(
        (
            F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            * (target - probabilities).abs().pow(vf.VFL_GAMMA)
        ).sum()
    )
    assert abs(varifocal - symmetric) > 1e-3, (
        f"varifocal ({varifocal:.6f}) and symmetric quality-focal "
        f"({symmetric:.6f}) agree on a confident positive — the asymmetry is "
        f"not implemented"
    )


def test_the_negative_weight_carries_no_gradient(vf):
    """``alpha * p ** gamma`` is a weight, not a term to optimise.

    If it were differentiable the model could reduce the loss by lowering its
    own confidence on negatives through the weight rather than through the
    prediction, which is a shortcut that trains.
    """
    import torch

    logits = torch.tensor([[1.5]], requires_grad=True)
    loss = vf._varifocal_loss(logits, torch.zeros(1, 1))
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    # The weight is detached, so the gradient is exactly weight * dBCE/dlogit.
    import torch.nn.functional as F

    probability = torch.sigmoid(torch.tensor([[1.5]]))
    weight = vf.VFL_ALPHA * probability.pow(vf.VFL_GAMMA)
    fresh = torch.tensor([[1.5]], requires_grad=True)
    reference = (
        F.binary_cross_entropy_with_logits(fresh, torch.zeros(1, 1), reduction="sum")
        * weight
    )
    reference.backward()
    assert torch.allclose(logits.grad, fresh.grad, rtol=1e-5), (
        "the negative weight is not detached, so gradient flows through the "
        "modulation as well as the prediction"
    )


# --- the star-shaped deformable sampling ------------------------------------


def test_offsets_are_relative_to_the_kernels_own_grid(vf):
    """``deform_conv2d`` offsets are RELATIVE to the 3x3 sampling grid.

    With a zero-size box every sampled point is the location itself, so the
    offsets must be exactly minus the base grid. Forgetting the subtraction
    leaves each point displaced by its grid position — and the centre tap's
    displacement is zero, so the model half works and the bug is hard to see.
    """
    import torch

    star = vf._StarDeformable(8, 4)
    distances = torch.zeros(1, 4, 3, 3)
    offsets = star.star_offsets(distances)

    base = torch.tensor(
        [value for pair in star.BASE_GRID for value in pair], dtype=torch.float32
    ).view(1, 18, 1, 1)
    assert torch.allclose(offsets, -base.expand_as(offsets), atol=1e-6), (
        "with a zero-size box the offsets are not exactly minus the base grid, "
        "so they are not relative to the kernel's own sampling positions"
    )


def test_offsets_are_y_x_pairs_matching_the_measured_op_layout(vf):
    """``deform_conv2d`` reads each pair as ``(y, x)`` — measured, not assumed.

    Verified independently by feeding a one-hot input through an identity kernel
    and shifting one tap: moving the SECOND channel of a pair moves the sample
    in x. Swapping the pair order here would transpose every sampling point with
    no error raised.

    Asserted on an asymmetric box so a transposition cannot go unnoticed: a
    square box would look identical either way.
    """
    import torch

    star = vf._StarDeformable(8, 4)
    left, top, right, bottom = 1.0, 5.0, 2.0, 7.0   # deliberately all different
    distances = torch.zeros(1, 4, 1, 1)
    distances[0, 0], distances[0, 1] = left, top
    distances[0, 2], distances[0, 3] = right, bottom

    offsets = star.star_offsets(distances)[0, :, 0, 0]
    base = torch.tensor([v for pair in star.BASE_GRID for v in pair], dtype=torch.float32)
    absolute = offsets + base   # undo the relative encoding

    # Kernel raster order, as (y, x). Tap 4 is the location itself.
    expected = {
        0: (-top, -left),      # top-left corner
        1: (-top, 0.0),        # top-centre
        2: (-top, right),      # top-right corner
        3: (0.0, -left),       # left-centre
        4: (0.0, 0.0),         # the location
        5: (0.0, right),       # right-centre
        6: (bottom, -left),    # bottom-left corner
        7: (bottom, 0.0),      # bottom-centre
        8: (bottom, right),    # bottom-right corner
    }
    for tap, (want_y, want_x) in expected.items():
        got_y = float(absolute[2 * tap])
        got_x = float(absolute[2 * tap + 1])
        assert (got_y, got_x) == pytest.approx((want_y, want_x), abs=1e-5), (
            f"kernel tap {tap} samples (y={got_y}, x={got_x}), expected "
            f"(y={want_y}, x={want_x}). If y and x are swapped, every sampling "
            f"point is transposed and deform_conv2d reads the wrong pixels"
        )


def test_the_offsets_are_partially_detached(vf):
    """``GRADIENT_MUL`` lets only a fraction of the gradient into the box
    prediction through the offsets.

    The offsets are a *function of* the box prediction, so without attenuation
    the box branch is trained mostly to make sampling convenient rather than to
    be correct.

    Measured as a RATIO against the same construction at ``GRADIENT_MUL = 1``,
    rather than against a hand-counted number of non-zero taps — the offset
    construction is linear in the distances, so the ratio is exactly
    ``GRADIENT_MUL`` regardless of how many taps are populated. An earlier
    version guessed the tap count and asserted the wrong absolute value.
    """
    import torch

    def offset_gradient(multiplier):
        original = vf.GRADIENT_MUL
        vf.GRADIENT_MUL = multiplier
        try:
            star = vf._StarDeformable(8, 4)
            distances = torch.ones(1, 4, 2, 2, requires_grad=True)
            star.star_offsets(distances).sum().backward()
            return float(distances.grad.abs().sum())
        finally:
            vf.GRADIENT_MUL = original

    full = offset_gradient(1.0)
    attenuated = offset_gradient(vf.GRADIENT_MUL)
    detached = offset_gradient(0.0)

    assert full > 0.0, "the fixture itself passes no gradient through the offsets"
    assert detached == 0.0, (
        f"at GRADIENT_MUL=0 the offsets still passed {detached:.6f} of gradient, "
        f"so the detach is not covering the whole path"
    )
    assert attenuated == pytest.approx(vf.GRADIENT_MUL * full, rel=1e-4), (
        f"the offset path passed {attenuated:.6f}, but GRADIENT_MUL="
        f"{vf.GRADIENT_MUL} of the full {full:.6f} is "
        f"{vf.GRADIENT_MUL * full:.6f} — the attenuation is not being applied"
    )
    assert 0.0 < vf.GRADIENT_MUL < 1.0, (
        f"GRADIENT_MUL is {vf.GRADIENT_MUL}; it must attenuate, not disable or "
        f"pass through"
    )


def test_the_refinement_starts_as_the_identity(vf):
    """The refinement predicts a multiplicative SCALE, so its bias must be 1.0.

    A zero bias makes every refined box a degenerate point while the initial box
    is fine — the losses stay finite, the classification target (the refined
    box's IoU) is ~0, and the model spends its early training undoing its own
    initialisation.
    """
    import torch

    model = vf.MyModel(3)
    assert torch.allclose(
        model.head.reg_refine.bias, torch.ones_like(model.head.reg_refine.bias), atol=1e-6
    ), (
        f"the refinement bias is {model.head.reg_refine.bias.flatten()[:4].tolist()}, "
        f"not 1.0 — it predicts a scale factor, so it must start as the identity"
    )
    assert torch.allclose(
        model.head.reg_initial.bias, torch.ones_like(model.head.reg_initial.bias), atol=1e-6
    ), "the initial distance bias must be 1.0 so the first box is not degenerate"


def test_the_refined_box_starts_close_to_the_initial_one(vf, built):
    """The consequence of the above, measured on a real forward: at
    initialisation the refinement should be nearly the identity, not a
    different box."""
    import torch

    model, features, _, _ = built
    with torch.no_grad():
        outputs = model.head(features)
    initial = outputs["bbox_initial"]
    refined = outputs["bbox_regression"]
    assert initial.shape == refined.shape
    ratio = float((refined.sum() / initial.sum().clamp(min=1e-6)))
    assert 0.5 < ratio < 2.0, (
        f"the refined distances sum to {ratio:.3f}x the initial ones at "
        f"initialisation; the refinement should start near the identity"
    )
    assert float(refined.min()) >= 0.0, "a refined distance went negative"


def test_deform_conv2d_is_usable_in_this_environment(vf):
    """Pin the dependency claim the docstring makes.

    ``deform_conv2d`` ships compiled in torchvision's wheel and works on CPU —
    this is NOT the ``MultiScaleDeformableAttention`` custom extension the RFC
    warns about for Tier 3. If a torchvision upgrade ever changed that, this
    template must fail here rather than on an edge.
    """
    import torch
    from torchvision.ops import deform_conv2d

    x = torch.rand(1, 4, 8, 8, requires_grad=True)
    offsets = torch.zeros(1, 18, 8, 8, requires_grad=True)
    weight = torch.rand(4, 4, 3, 3, requires_grad=True)
    out = deform_conv2d(x, offsets, weight, padding=1)
    assert out.shape == (1, 4, 8, 8)
    out.sum().backward()
    assert x.grad is not None and offsets.grad is not None and weight.grad is not None


# --- ordering, losses, decode -----------------------------------------------


def test_head_flatten_is_location_major(vf, built):
    """Pinned at ``A = 2`` as well as ``A = 1``: with one anchor per location the
    location-major and anchor-major permutations are identical, so the live
    model cannot distinguish them."""
    import torch

    _, features, anchors, _ = built
    height, width = features[0].shape[-2:]
    stride = float(vf.PYRAMID_STRIDES[0])

    for row, column in ((0, 1), (1, 0), (2, 3)):
        output = torch.zeros(1, 3, height, width)
        output[0, 2, row, column] = 7.0
        flat = vf._VFNetHead._flatten(output, 3)
        expected = row * width + column
        assert (flat[0] == 7.0).nonzero().tolist() == [[expected, 2]]
        centre = vf._centres(anchors[expected : expected + 1])[0]
        assert (float(centre[0]), float(centre[1])) == (column * stride, row * stride)

    output = torch.zeros(1, 2 * 3, height, width)
    output[0, 1 * 3 + 2, 1, 3] = 7.0
    expected = (1 * width + 3) * 2 + 1
    assert (vf._VFNetHead._flatten(output, 3)[0] == 7.0).nonzero().tolist() == [[expected, 2]]


def _synthetic(vf, model, anchors, split, level, anchor_index, distance, label):
    import torch

    num_classes = model.head.num_classes
    split_anchors = list(anchors.split(split))
    cls_levels, reg_levels = [], []
    for level_id, count in enumerate(split):
        cls = torch.full((1, count, num_classes), -20.0)
        reg = torch.zeros((1, count, 4))
        if level_id == level:
            cls[0, anchor_index, label] = 20.0
            reg[0, anchor_index] = distance
        cls_levels.append(cls)
        reg_levels.append(reg)
    # head_outputs[k] is a FLAT list over levels; anchors is per image.
    return {"cls_logits": cls_levels, "bbox_regression": reg_levels}, [split_anchors]


def test_decode_uses_each_levels_own_stride_on_the_refined_box(vf, built):
    """Distances are in stride units, and it is the REFINED box that is emitted.

    Decoding a level-2 prediction with P3's stride shrinks the box fourfold
    while still producing valid xyxy, so only an analytic check catches it.
    """
    import torch

    model, features, anchors, split = built
    level, row, column, distance, label = 2, 4, 5, 3.0, 1
    grid_width = features[level].shape[-1]
    anchor_index = row * grid_width + column
    stride = vf.PYRAMID_STRIDES[level]

    head_outputs, split_anchors = _synthetic(
        vf, model, anchors, split, level, anchor_index, distance, label
    )
    prediction = model.postprocess_detections(head_outputs, split_anchors, [(256, 320)])[0]
    assert prediction["boxes"].shape[0] >= 1, "no detection survived; decode not exercised"

    best = int(prediction["scores"].argmax())
    centre = vf._centres(split_anchors[0][level][anchor_index : anchor_index + 1])[0]
    offset = distance * stride
    expected = torch.tensor([
        float(centre[0]) - offset, float(centre[1]) - offset,
        float(centre[0]) + offset, float(centre[1]) + offset,
    ])
    assert torch.allclose(prediction["boxes"][best], expected, atol=1e-3), (
        f"level {level} (stride {stride}) decoded to "
        f"{prediction['boxes'][best].tolist()}, expected {expected.tolist()}"
    )
    assert int(prediction["labels"][best]) == label


def test_decode_returns_the_handler_contract_shape(vf, built):
    import torch

    model, _, anchors, split = built
    head_outputs, split_anchors = _synthetic(vf, model, anchors, split, 1, 7, 2.0, 2)
    prediction = model.postprocess_detections(head_outputs, split_anchors, [(256, 320)])[0]
    assert {"boxes", "scores", "labels"} <= set(prediction)
    boxes = prediction["boxes"]
    assert boxes.ndim == 2 and boxes.shape[-1] == 4
    assert bool((boxes[:, 2] >= boxes[:, 0]).all() and (boxes[:, 3] >= boxes[:, 1]).all())
    assert bool((boxes[:, 0] >= 0).all() and (boxes[:, 2] <= 320).all())
    assert prediction["labels"].dtype == torch.int64


def test_all_three_losses_are_reported_on_the_graph(vf):
    """Both boxes are supervised: the refinement can only correct what the
    initial estimate roughly located, so dropping the initial term makes the
    refinement's job impossible while training perfectly happily."""
    import torch

    model = vf.MyModel(3)
    model.train()
    losses = model(
        [torch.rand(3, 128, 160), torch.rand(3, 144, 128)],
        [
            {"boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]), "labels": torch.tensor([1])},
            {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)},
        ],
    )
    assert set(losses) == {"classification", "bbox_regression", "bbox_initial"}, (
        f"expected the varifocal, refined-box and initial-box terms, got "
        f"{sorted(losses)}"
    )
    device = next(model.parameters()).device
    for name, value in losses.items():
        assert torch.is_tensor(value) and value.ndim == 0, f"{name} is not a scalar"
        assert torch.isfinite(value).all(), f"{name} is not finite"
        assert value.device == device, f"{name} is on {value.device}"
        assert value.requires_grad, f"{name} is detached from the graph"


def test_the_quality_target_does_not_leak_gradient_into_the_classifier(vf):
    """The classification target is the refined box's IoU — a label, and it must
    carry no graph.

    ⚠️ The naive form of this test is WRONG for VFNet, and this one is written
    around why. In ``gfl_resnet`` it suffices to assert that the classification
    loss puts no gradient on the box head. Here it does, **by design**:
    ``cls_star`` samples the star points of the predicted box, so its offsets
    are a function of ``reg_initial``, and ``GRADIENT_MUL`` exists precisely to
    attenuate that intended path. A test asserting zero gradient fails against
    correct code.

    So the two paths are separated by temporarily setting ``GRADIENT_MUL = 0``,
    which fully detaches the offset path. Any gradient surviving that is the
    undetached IoU target. Measured: 0.0379 at the configured 0.1, and exactly
    0.0 at 0 — so the target is detached.
    """
    import torch

    original = vf.GRADIENT_MUL
    vf.GRADIENT_MUL = 0.0
    try:
        model = vf.MyModel(3)
        model.train()
        losses = model(
            [torch.rand(3, 128, 160)],
            [{"boxes": torch.tensor([[10.0, 10.0, 90.0, 90.0]]), "labels": torch.tensor([1])}],
        )
        model.zero_grad(set_to_none=True)
        losses["classification"].backward()

        initial_grad = model.head.reg_initial.weight.grad
        leaked = 0.0 if initial_grad is None else float(initial_grad.abs().sum())
        classifier_grad = model.head.cls_star.weight.grad
    finally:
        vf.GRADIENT_MUL = original

    assert leaked == pytest.approx(0.0, abs=1e-9), (
        f"with the offset path fully detached the classification loss still put "
        f"{leaked:.6g} of gradient on head.reg_initial. The only remaining route "
        f"is the IoU target, so it is not detached — the classifier is being "
        f"used to train the box head"
    )
    assert classifier_grad is not None and bool(classifier_grad.abs().sum() > 0), (
        "the classification loss put no gradient on the classifier either, so "
        "backward did nothing and the assertion above proves nothing"
    )


def test_the_star_offsets_do_carry_their_intended_gradient(vf):
    """The other half: at the configured ``GRADIENT_MUL`` the classification loss
    SHOULD reach the box head through the sampling offsets.

    Without this, the test above would pass if the star sampling were replaced
    by a plain convolution — which would silently remove the mechanism that
    makes the classification score IoU-aware.
    """
    import torch

    model = vf.MyModel(3)
    model.train()
    losses = model(
        [torch.rand(3, 128, 160)],
        [{"boxes": torch.tensor([[10.0, 10.0, 90.0, 90.0]]), "labels": torch.tensor([1])}],
    )
    model.zero_grad(set_to_none=True)
    losses["classification"].backward()
    initial_grad = model.head.reg_initial.weight.grad
    assert initial_grad is not None and float(initial_grad.abs().sum()) > 0.0, (
        "the classification loss reached the box head not at all, so the "
        "classifier is not sampling the predicted box's star points — the "
        "IoU-aware half of the head is inert"
    )


def test_an_all_background_batch_trains(vf):
    import torch

    model = vf.MyModel(3)
    model.train()
    losses = model(
        [torch.rand(3, 128, 160)],
        [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}],
    )
    for name, value in losses.items():
        assert torch.isfinite(value).all(), f"{name} is {value}"
        assert value.requires_grad, f"{name} left the graph"
    sum(losses.values()).backward()


def test_every_trainable_parameter_receives_a_gradient(vf):
    """A dead parameter is serialised, uploaded and averaged every federated
    round for a value that can never change. Frozen parameters are excluded:
    ``trainable_layers=3`` freezes the stem on purpose."""
    import torch

    model = vf.MyModel(3)
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
    assert not dead, f"{len(dead)} trainable parameters got no gradient: {dead[:6]}"


def test_compute_loss_refuses_a_plain_anchor_generator(vf):
    import torch
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    model = vf.MyModel(3)
    model.anchor_generator = AnchorGenerator(
        sizes=tuple((8 * s,) for s in vf.PYRAMID_STRIDES),
        aspect_ratios=((1.0,),) * len(vf.PYRAMID_STRIDES),
    )
    model.train()
    with pytest.raises(RuntimeError, match="per-level anchor split"):
        model(
            [torch.rand(3, 128, 160)],
            [{"boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]), "labels": torch.tensor([1])}],
        )


# --- the duplicated ATSS assignment -----------------------------------------
#
# vfnet_resnet.py carries its own copy of _atss_assign because zoo templates
# cannot import siblings. A mutation sweep on gfl_resnet showed the consequence:
# deleting the centre-inside rule from a COPY survived the whole suite, since the
# only test covering it lived against another template. Duplicated code needs
# its guard duplicated.


def test_the_duplicated_atss_enforces_the_centre_inside_rule(vf):
    """The 0.2px fixture. An anchor clearing the adaptive threshold but centred
    OUTSIDE the object must still be rejected. Note a two-anchor fixture cannot
    test this: with exactly two candidates the mean+std threshold IS the larger
    IoU, so only the best clears it and the centre rule is never consulted."""
    import torch

    gt = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
    far = [[500.0, 500.0, 510.0, 510.0], [600.0, 600.0, 610.0, 610.0], [700.0, 700.0, 710.0, 710.0]]
    outside = torch.tensor([[10.1, 0.0, 30.1, 20.0]] + far)
    inside = torch.tensor([[9.9, 0.0, 29.9, 20.0]] + far)
    assert vf._atss_assign(inside, gt, [4], topk=4)[0] == 0
    assert vf._atss_assign(outside, gt, [4], topk=4)[0] == vf.BACKGROUND, (
        "an anchor centred just OUTSIDE the object was assigned to it — the "
        "centre-inside rule is missing from VFNet's copy of _atss_assign"
    )


def test_the_duplicated_atss_handles_a_single_candidate(vf):
    """std over one sample is NaN under the unbiased estimator, and
    ``iou >= nan`` is False everywhere, so the object silently gets no anchors."""
    import torch

    anchors = torch.tensor([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]])
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert int((vf._atss_assign(anchors, gt, [1, 1], topk=1) >= 0).sum()) >= 1


def test_the_duplicated_atss_topk_is_per_level(vf, built):
    import torch

    _, _, anchors, split = built
    gt = torch.tensor([[20.0, 20.0, 100.0, 100.0], [150.0, 60.0, 260.0, 200.0]])
    per_level = vf._atss_assign(anchors, gt, split, vf.ATSS_TOPK)
    global_topk = vf._atss_assign(anchors, gt, [anchors.shape[0]], vf.ATSS_TOPK)
    assert not torch.equal(per_level, global_topk)
    assert int((per_level >= 0).sum()) > int((global_topk >= 0).sum())
    assert int((per_level >= 0).sum()) > 0
    empty = vf._atss_assign(anchors, torch.zeros((0, 4)), split, vf.ATSS_TOPK)
    assert bool((empty == vf.BACKGROUND).all())


def test_the_template_declares_the_family_contract():
    source = TEMPLATE.read_text(encoding="utf-8")
    assert re.search(r'^model_type\s*=\s*"torchvision_detection"', source, re.MULTILINE)
    assert "weights=None" in source
    for banned in ("import timm", "from timm", "transformers", "from_pretrained"):
        assert banned not in source, f"{banned!r} is not permitted in a cv template"


def test_the_refined_distances_cannot_go_negative(vf, built):
    """The refinement's scale is ReLU'd, so a negative prediction cannot invert
    the box.

    Added after a mutation sweep: checking the freshly built model SURVIVED
    removing that ReLU, because the ``+1.0`` refinement bias keeps the scale
    positive at initialisation anyway. Driven with a strongly negative bias so
    the ReLU is the only thing in the way.
    """
    import torch

    model, features, _, _ = built
    with torch.no_grad():
        torch.nn.init.constant_(model.head.reg_refine.bias, -50.0)
        try:
            outputs = model.head(features)
        finally:
            torch.nn.init.constant_(model.head.reg_refine.bias, 1.0)

    refined = outputs["bbox_regression"]
    assert float(refined.min()) >= 0.0, (
        f"with a -50 refinement bias the refined distances reached "
        f"{float(refined.min()):.4f}. A negative distance flips an edge across "
        f"the anchor centre, giving x2 < x1 — not a valid xyxy box, and the "
        f"engine's torchmetrics read these as xyxy and would score nonsense"
    )


def test_the_quality_target_comes_from_the_refined_box(vf, built):
    """The classification target must be the IoU of the REFINED box.

    That is what inference emits, so training the score against the initial
    box's IoU would rank by a quantity the model does not report. At
    initialisation the two boxes are nearly identical (the refinement starts as
    the identity), so a mutation swapping them SURVIVED every other assertion
    here — this fixture makes them deliberately far apart and checks which one
    the loss actually used.

    ``compute_loss`` is driven directly with hand-built head outputs, so the
    expected varifocal loss can be computed both ways and compared.
    """
    import torch

    model, _, anchors, split = built
    num_anchors = anchors.shape[0]
    num_classes = model.head.num_classes
    strides = vf._anchor_strides(split, anchors.device, anchors.dtype)
    centres = vf._centres(anchors)

    initial_distance, refined_distance = 0.5, 4.0
    head_outputs = {
        "cls_logits": torch.zeros(1, num_anchors, num_classes),
        "bbox_initial": torch.full((1, num_anchors, 4), initial_distance),
        "bbox_regression": torch.full((1, num_anchors, 4), refined_distance),
    }
    targets = [{
        "boxes": torch.tensor([[20.0, 20.0, 100.0, 100.0]]),
        "labels": torch.tensor([1]),
    }]

    losses = model.compute_loss(targets, head_outputs, [anchors])
    actual = float(losses["classification"].detach())

    # Recompute the classification term both ways, using the template's own
    # assignment and loss so only the choice of box differs.
    from torchvision.ops import boxes as box_ops

    matched = vf._atss_assign(anchors, targets[0]["boxes"], split, vf.ATSS_TOPK)
    foreground = matched >= 0
    assert bool(foreground.any()), "the fixture assigned nothing"

    def expected_for(distance):
        boxes = vf._distance_to_box(
            centres, torch.full((num_anchors, 4), distance), strides
        )
        gt = targets[0]["boxes"][matched[foreground]]
        quality = box_ops.box_iou(boxes[foreground], gt).diagonal().clamp(min=0)
        scores = torch.zeros(num_anchors, num_classes)
        scores[foreground, targets[0]["labels"][matched[foreground]]] = quality
        total = float(vf._varifocal_loss(torch.zeros(num_anchors, num_classes), scores))
        return total / max(float(quality.sum()), 1.0)

    from_refined = expected_for(refined_distance)
    from_initial = expected_for(initial_distance)
    assert abs(from_refined - from_initial) > 1e-6, (
        f"the fixture's two boxes give the same loss ({from_refined:.6f}), so it "
        f"cannot distinguish them — widen the gap between the distances"
    )
    assert actual == pytest.approx(from_refined, rel=1e-4), (
        f"the classification loss is {actual:.6f}; the REFINED box gives "
        f"{from_refined:.6f} and the initial box gives {from_initial:.6f}. The "
        f"quality target must come from the refined box, because that is the "
        f"box inference emits and therefore the one the score ranks"
    )


# --- review findings on model-zoo#239 ---------------------------------------


def test_the_two_box_supervisions_are_separate(vf):
    """The refined box's loss must not train ``reg_initial`` through the product.

    Review finding. ``refined = initial * scale`` did not detach ``initial``, so
    the IoU-quality-weighted refined loss backpropagated straight into the
    initial estimate — coupling the two supervisions and training the first box
    to make the refinement's job easy rather than to be a good first estimate.
    The reference implementation detaches (``... * bbox_pred.detach()``). This
    was an oversight rather than a deliberate deviation, so it is now detached
    and the docstring says so.

    Separated the same way as the classification path, and for the same reason:
    one coupling REMAINS and is intended — ``reg_refine``'s sampling offsets are
    a function of ``initial``, attenuated by ``GRADIENT_MUL``, which is what
    makes the refinement look at the box it is correcting. So the product path
    is isolated by temporarily setting ``GRADIENT_MUL = 0``. Measured: 0.812 of
    gradient at the configured 0.1, and exactly 0.0 at 0.
    """
    import torch

    images = [torch.rand(3, 128, 160)]
    targets = [{"boxes": torch.tensor([[10.0, 10.0, 90.0, 90.0]]), "labels": torch.tensor([1])}]

    original = vf.GRADIENT_MUL
    vf.GRADIENT_MUL = 0.0
    try:
        model = vf.MyModel(3)
        model.train()
        losses = model(images, targets)
        model.zero_grad(set_to_none=True)
        losses["bbox_regression"].backward()
        grad = model.head.reg_initial.weight.grad
        leaked = 0.0 if grad is None else float(grad.abs().sum())
    finally:
        vf.GRADIENT_MUL = original

    assert leaked == pytest.approx(0.0, abs=1e-9), (
        f"with the offset path fully detached the REFINED box loss still put "
        f"{leaked:.6g} of gradient on head.reg_initial. The only remaining "
        f"route is the `initial * scale` product, so `initial` is not detached "
        f"and the two box supervisions are coupled"
    )


def test_the_initial_box_loss_does_train_the_initial_head(vf):
    """The other half: detaching the product must not orphan the initial head.

    Without this, the test above would pass if ``bbox_initial`` were dropped or
    if ``reg_initial`` were detached everywhere — which would leave the first
    box estimate untrained while the losses stayed finite.
    """
    import torch

    model = vf.MyModel(3)
    model.train()
    losses = model(
        [torch.rand(3, 128, 160)],
        [{"boxes": torch.tensor([[10.0, 10.0, 90.0, 90.0]]), "labels": torch.tensor([1])}],
    )
    model.zero_grad(set_to_none=True)
    losses["bbox_initial"].backward()
    grad = model.head.reg_initial.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0, (
        "the initial-box loss puts no gradient on head.reg_initial, so the "
        "first box estimate is never trained"
    )


def test_a_negative_per_level_scale_cannot_invert_the_boxes(vf, built):
    """The per-level scale must be inside the ReLU.

    ``reg_scales`` is an unconstrained ``Parameter`` taking gradient from the box
    losses, so it can cross zero. Applied after the clamp it would be the final
    operation and every distance on that level would go negative. This is the
    same defect ``tood_resnet`` shipped and had fixed in review; avoided here by
    construction, and pinned so it stays that way.

    Nothing downstream would catch it: measured, neither
    ``generalized_box_iou``, ``generalized_box_iou_loss`` nor ``box_iou``
    validates corner ordering — an inverted box yields a finite loss.
    """
    import torch

    model, features, _, _ = built
    with torch.no_grad():
        for scale in model.head.reg_scales:
            torch.nn.init.constant_(scale, -1.0)
        try:
            outputs = model.head(features)
        finally:
            for scale in model.head.reg_scales:
                torch.nn.init.constant_(scale, 1.0)

    for key in ("bbox_initial", "bbox_regression"):
        assert float(outputs[key].min()) >= 0.0, (
            f"with every per-level scale at -1, {key} reached "
            f"{float(outputs[key].min()):.4f}. The scale must be applied INSIDE "
            f"the ReLU so the clamp is the last word whatever the scale does"
        )

