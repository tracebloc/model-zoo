"""Contract tests for GFL's head, decode and losses (backend#2982, Tier 2).

Why this file exists on top of the family train-step guard
----------------------------------------------------------
``test_od_torchvision_family_train_step.py`` proves every
``torchvision_detection`` template completes a train step and an eval step.
For ``gfl_resnet`` that is necessary and nowhere near sufficient, because the
head, the box decode and all three losses are ours rather than torchvision's,
and every way they can be wrong produces finite losses and a clean backward:

- **Output ordering.** ``AnchorGenerator`` emits anchors location-major;
  the head's conv output is ``(N, A * K, H, W)``. Permuting those to disagree
  is *shape-identical*, so the model trains against boxes decoded at the wrong
  pixels and merely learns badly. Nothing raises, ever.
- **The integral.** Reading the distribution over the wrong axis, or against
  the wrong bin values, still yields a number in roughly the right range.
- **DFL.** Supervising the wrong pair of bins, or dropping the interpolation
  weights, still decreases.
- **The detached quality target.** QFL's target is the IoU the box achieved. If
  it is not detached, gradient reaches the boxes through the *classifier*,
  which trains and quietly optimises the wrong thing.

Each of those is pinned below. Nine mutations were registered against this file
while writing it; the ordering and detachment tests are the two that no
train-step or end-to-end assertion caught.
"""

import importlib.util
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parent.parent
TEMPLATE = ROOT / "model_zoo" / "object_detection" / "pytorch" / "gfl_resnet.py"


def _load():
    spec = importlib.util.spec_from_file_location("gfl_resnet_under_test", TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gfl():
    pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")
    return _load()


@pytest.fixture(scope="module")
def built(gfl):
    """The real model, its anchors and its level split for a known input size."""
    import torch
    from torchvision.models.detection.image_list import ImageList

    model = gfl.MyModel(3)
    model.eval()
    image = torch.rand(1, 3, 256, 320)
    with torch.no_grad():
        features = list(model.backbone(image).values())
    anchors = model.anchor_generator(ImageList(image, [(256, 320)]), features)[0]
    return model, features, anchors, list(model.anchor_generator.num_anchors_per_level)


# --- the silent one: head output index must mean the same anchor -------------


def test_head_flatten_agrees_with_the_anchor_generator_ordering(gfl, built):
    """A conv activation at grid ``(h, w)`` must land on the anchor at ``(h, w)``.

    This is the failure the module docstring calls shape-identical. It is
    checked in three linked steps rather than end to end, so a break says which
    of the three conventions moved:

    1. ``_flatten`` maps ``(k, h, w)`` to flat index ``(h * W + w) * A + a``
    2. the anchor at that flat index is centred at ``(w * stride, h * stride)``
    3. therefore decoding that anchor's distribution places the box there

    Note step 2: torchvision's anchors are **origin-aligned** — the anchor for
    grid cell ``(h, w)`` sits at ``w * stride``, not ``(w + 0.5) * stride``.
    That is half a stride off the pixel-centre convention GFL's reference
    implementation uses. It is self-consistent inside torchvision (RetinaNet
    and FCOS share this generator), so it is correct here — but it is measured,
    not assumed, and a torchvision change to it would surface as this test
    rather than as unexplained accuracy loss.
    """
    import torch

    _, features, anchors, split = built
    height, width = features[0].shape[-2:]
    num_anchors, num_channels = 1, 3
    stride = float(gfl.PYRAMID_STRIDES[0])

    for row, column in ((0, 1), (1, 0), (2, 3), (height - 1, width - 1)):
        output = torch.zeros(1, num_anchors * num_channels, height, width)
        output[0, 2, row, column] = 7.0
        flat = gfl._GFLHead._flatten(output, num_channels)

        expected_index = (row * width + column) * num_anchors
        found = (flat[0] == 7.0).nonzero()
        assert found.tolist() == [[expected_index, 2]], (
            f"grid ({row}, {column}) channel 2 landed at {found.tolist()}, "
            f"expected [[{expected_index}, 2]] — the head permutation no longer "
            f"matches AnchorGenerator's location-major ordering"
        )

        centre = gfl._centres(anchors[expected_index : expected_index + 1])[0]
        assert (float(centre[0]), float(centre[1])) == (column * stride, row * stride), (
            f"anchor {expected_index} is centred at "
            f"({float(centre[0])}, {float(centre[1])}), but grid ({row}, {column}) "
            f"at stride {stride} is ({column * stride}, {row * stride})"
        )


def test_head_flatten_is_location_major_for_more_than_one_anchor(gfl):
    """Pin the ordering at ``A > 1``, which this template never builds.

    Added after a mutation sweep: swapping ``_flatten`` to an anchor-major
    permutation SURVIVED every other test here, and the reason is that with a
    single anchor per location the two orderings are *identical* — measured,
    ``A=1`` puts the probe at flat index 8 either way, ``A=2`` puts it at 17
    versus 28. ATSS's single-anchor design means the live model cannot
    distinguish them, so the mutation is inert today and would become a silent
    wrong-pixel bug the moment anyone raised the anchor count.

    Pinning it at ``A=2`` costs nothing and makes that edit fail loudly instead.
    """
    import torch

    height, width, num_channels, num_anchors = 4, 5, 3, 2
    output = torch.zeros(1, num_anchors * num_channels, height, width)
    # anchor a=1, channel k=2, grid (1, 3)
    output[0, 1 * num_channels + 2, 1, 3] = 7.0
    flat = gfl._GFLHead._flatten(output, num_channels)

    expected_index = (1 * width + 3) * num_anchors + 1
    found = (flat[0] == 7.0).nonzero()
    assert found.tolist() == [[expected_index, 2]], (
        f"with A={num_anchors}, anchor 1 at grid (1, 3) channel 2 landed at "
        f"{found.tolist()}, expected [[{expected_index}, 2]] — _flatten is not "
        f"location-major (all anchors of a position adjacent), which is what "
        f"AnchorGenerator emits"
    )


def test_level_split_covers_every_anchor(built):
    _, _, anchors, split = built
    assert len(split) == 5, f"expected the P3..P7 pyramid, got {len(split)} levels"
    assert sum(split) == anchors.shape[0]
    assert all(size > 0 for size in split)


# --- the integral ------------------------------------------------------------


def test_integral_of_a_one_hot_distribution_is_that_bin(gfl):
    """A distribution concentrated on bin k must integrate to exactly k."""
    import torch

    for bin_index in (0, 1, 7, gfl.REG_MAX):
        logits = torch.full((1, 4, gfl.REG_MAX + 1), -50.0)
        logits[:, :, bin_index] = 50.0
        distances = gfl._integral(logits.reshape(1, -1), gfl.REG_MAX)
        assert torch.allclose(
            distances, torch.full((1, 4), float(bin_index)), atol=1e-3
        ), f"one-hot at bin {bin_index} integrated to {distances.tolist()}"


def test_integral_of_a_uniform_distribution_is_the_midpoint(gfl):
    """Uniform logits must give reg_max/2, which is also what the head produces
    at initialisation — so this doubles as the value the first train step sees."""
    import torch

    logits = torch.zeros(1, 4 * (gfl.REG_MAX + 1))
    distances = gfl._integral(logits, gfl.REG_MAX)
    assert torch.allclose(
        distances, torch.full((1, 4), gfl.REG_MAX / 2.0), atol=1e-4
    ), f"uniform integrated to {distances.tolist()}, expected {gfl.REG_MAX / 2.0}"


def test_integral_reads_the_per_side_axis_not_the_bin_axis(gfl):
    """Distinguish the four sides.

    A reshape that transposed sides and bins would still integrate to something
    plausible; this gives each side a different one-hot bin so only the correct
    axis assignment reproduces them.
    """
    import torch

    wanted = [0, 3, 9, gfl.REG_MAX]
    logits = torch.full((1, 4, gfl.REG_MAX + 1), -50.0)
    for side, bin_index in enumerate(wanted):
        logits[0, side, bin_index] = 50.0
    distances = gfl._integral(logits.reshape(1, -1), gfl.REG_MAX)
    assert torch.allclose(
        distances[0], torch.tensor([float(v) for v in wanted]), atol=1e-3
    ), f"per-side integral gave {distances[0].tolist()}, expected {wanted}"


# --- distance <-> box round trip ---------------------------------------------


def test_distance_and_box_round_trip(gfl):
    """``_box_to_distance`` then ``_distance_to_box`` must recover the box.

    Both directions are hand-written and a sign flip in either is invisible in
    a loss value; a round trip catches it.
    """
    import torch

    centres = torch.tensor([[50.0, 60.0], [200.0, 100.0]])
    boxes = torch.tensor([[30.0, 40.0, 90.0, 100.0], [180.0, 70.0, 260.0, 150.0]])
    strides = torch.tensor([8.0, 16.0])

    distances = gfl._box_to_distance(centres, boxes, strides, gfl.REG_MAX)
    recovered = gfl._distance_to_box(centres, distances, strides)
    assert torch.allclose(recovered, boxes, atol=1e-4), (
        f"round trip gave {recovered.tolist()}, expected {boxes.tolist()}"
    )


def test_distance_targets_are_clamped_into_the_representable_range(gfl):
    """An edge further than reg_max strides has no bin.

    Unclamped, DFL's ``gather`` would index past the tensor. This pins that the
    clamp exists and stays strictly below reg_max so the *upper* bin index is
    also in range.
    """
    import torch

    centres = torch.tensor([[0.0, 0.0]])
    # 10_000px away at stride 8 is 1250 strides — far past reg_max.
    boxes = torch.tensor([[-10000.0, -10000.0, 10000.0, 10000.0]])
    strides = torch.tensor([8.0])
    distances = gfl._box_to_distance(centres, boxes, strides, gfl.REG_MAX)
    assert bool((distances >= 0).all()), "a clamped distance went negative"
    assert bool((distances < gfl.REG_MAX).all()), (
        f"distances {distances.tolist()} reach or exceed reg_max={gfl.REG_MAX}; "
        f"floor()+1 would then index past the distribution"
    )
    # And the loss it feeds must be finite rather than an index error.
    logits = torch.zeros(1, 4, gfl.REG_MAX + 1)
    loss = gfl._distribution_focal_loss(logits, distances, gfl.REG_MAX)
    assert bool(torch.isfinite(loss).all())


# --- DFL ---------------------------------------------------------------------


def test_dfl_is_minimised_by_a_distribution_on_the_target(gfl):
    """The whole point of DFL: mass on the target bins must score better than
    mass anywhere else, and better than uniform."""
    import torch

    target = torch.tensor([[4.0, 4.0, 4.0, 4.0]])

    on_target = torch.full((1, 4, gfl.REG_MAX + 1), -10.0)
    on_target[:, :, 4] = 10.0
    elsewhere = torch.full((1, 4, gfl.REG_MAX + 1), -10.0)
    elsewhere[:, :, 12] = 10.0
    uniform = torch.zeros(1, 4, gfl.REG_MAX + 1)

    loss_on = gfl._distribution_focal_loss(on_target, target, gfl.REG_MAX)
    loss_off = gfl._distribution_focal_loss(elsewhere, target, gfl.REG_MAX)
    loss_uniform = gfl._distribution_focal_loss(uniform, target, gfl.REG_MAX)

    assert float(loss_on) < float(loss_uniform) < float(loss_off), (
        f"DFL ordering wrong: on-target {float(loss_on):.4f}, uniform "
        f"{float(loss_uniform):.4f}, off-target {float(loss_off):.4f}"
    )


def test_dfl_at_uniform_matches_the_analytic_value(gfl):
    """A number that can only be right if the loss is really cross-entropy over
    reg_max + 1 bins: four sides x -log(1 / (reg_max + 1))."""
    import math

    import torch

    target = torch.tensor([[4.0, 4.0, 4.0, 4.0]])
    uniform = torch.zeros(1, 4, gfl.REG_MAX + 1)
    expected = 4 * -math.log(1.0 / (gfl.REG_MAX + 1))
    actual = float(gfl._distribution_focal_loss(uniform, target, gfl.REG_MAX))
    assert abs(actual - expected) < 1e-3, (
        f"uniform DFL is {actual:.4f}, analytic value {expected:.4f}"
    )


def test_dfl_interpolation_weights_straddle_the_target(gfl):
    """A target exactly on a bin must be cheapest for that bin alone; a target
    halfway between two must be cheapest for mass split across both."""
    import torch

    def loss_for(bins, target_value):
        logits = torch.full((1, 4, gfl.REG_MAX + 1), -10.0)
        for b in bins:
            logits[:, :, b] = 10.0
        return float(
            gfl._distribution_focal_loss(
                logits, torch.full((1, 4), target_value), gfl.REG_MAX
            )
        )

    # target 4.5 sits between bins 4 and 5
    split_mass = loss_for([4, 5], 4.5)
    all_on_four = loss_for([4], 4.5)
    assert split_mass < all_on_four, (
        f"a target of 4.5 scored better with all mass on bin 4 "
        f"({all_on_four:.4f}) than split across 4 and 5 ({split_mass:.4f}) — "
        f"the interpolation weights are not being applied"
    )
    # target exactly 4.0 should prefer bin 4 alone over splitting
    assert loss_for([4], 4.0) < loss_for([4, 5], 4.0)


# --- QFL ---------------------------------------------------------------------


def test_qfl_reduces_to_focal_loss_at_a_hard_target(gfl):
    """QFL is a *generalisation*: at target 0 or 1 it must equal sigmoid focal
    loss with gamma=beta. If it does not, it is a different loss wearing the
    name."""
    import torch
    from torchvision.ops import sigmoid_focal_loss

    logits = torch.tensor([[2.0, -1.0, 0.5, -3.0]])
    for target_value in (0.0, 1.0):
        target = torch.full_like(logits, target_value)
        qfl = float(gfl._quality_focal_loss(logits, target, beta=2.0))
        focal = float(
            sigmoid_focal_loss(logits, target, alpha=-1, gamma=2.0, reduction="sum")
        )
        assert abs(qfl - focal) < 1e-5, (
            f"at target {target_value} QFL is {qfl:.6f} but focal loss is "
            f"{focal:.6f} — QFL is not reducing to focal loss"
        )


def test_qfl_prefers_a_prediction_matching_the_soft_target(gfl):
    import torch

    target = torch.tensor([[0.7]])
    close = torch.tensor([[0.847]])   # sigmoid ~= 0.70
    far = torch.tensor([[-2.0]])      # sigmoid ~= 0.12
    assert float(gfl._quality_focal_loss(close, target)) < float(
        gfl._quality_focal_loss(far, target)
    )


# --- the detached quality target --------------------------------------------


def test_the_quality_target_does_not_leak_gradient_into_the_classifier(gfl):
    """QFL's target is the IoU the predicted box achieved — a *label*.

    If it is not detached, the classification loss can be reduced by moving the
    boxes, so gradient reaches the regression branch through the classifier and
    the model quietly optimises a quantity nobody asked for. It trains, and the
    loss goes down.

    Rewritten after a mutation sweep. The first version built its own
    ``torch.no_grad()`` block and asserted the result carried no graph — which
    tests the *fixture*, not the template, and duly SURVIVED removing
    ``no_grad`` from ``compute_loss``. This drives the real ``compute_loss`` and
    asks the only question that matters: does the classification term alone put
    a gradient on the regression head?
    """
    import torch

    model = gfl.MyModel(3)
    model.train()
    losses = model(
        [torch.rand(3, 128, 160)],
        [{"boxes": torch.tensor([[10.0, 10.0, 90.0, 90.0]]), "labels": torch.tensor([1])}],
    )
    model.zero_grad(set_to_none=True)
    losses["classification"].backward()

    regression_grad = model.head.bbox_regression.weight.grad
    leaked = regression_grad is not None and bool(regression_grad.abs().sum() > 0)
    assert not leaked, (
        "the classification loss alone put a gradient on head.bbox_regression, "
        "so QFL's IoU target is not detached — the classifier is being used to "
        "train the box head"
    )
    # Control: the classification head MUST receive a gradient, or the test
    # above would pass simply because backward did nothing.
    classifier_grad = model.head.cls_logits.weight.grad
    assert classifier_grad is not None and bool(classifier_grad.abs().sum() > 0), (
        "the classification loss put no gradient on head.cls_logits either — "
        "backward did nothing, so the assertion above proves nothing"
    )


def test_compute_loss_refuses_a_plain_anchor_generator(gfl):
    """The recorded-split design has one failure mode: no record. Defaulting to
    one level would silently degrade ATSS to a global topk and give every anchor
    the P3 stride, so the template raises. This proves the raise is reachable."""
    import torch
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    model = gfl.MyModel(3)
    model.anchor_generator = AnchorGenerator(
        sizes=tuple((8 * s,) for s in gfl.PYRAMID_STRIDES),
        aspect_ratios=((1.0,),) * len(gfl.PYRAMID_STRIDES),
    )
    model.train()
    with pytest.raises(RuntimeError, match="per-level anchor split"):
        model(
            [torch.rand(3, 128, 160)],
            [{"boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]), "labels": torch.tensor([1])}],
        )


def test_all_three_losses_are_reported(gfl):
    """The handler sums whatever the dict holds, so a silently dropped loss term
    trains a different model with no error anywhere."""
    import torch

    model = gfl.MyModel(3)
    model.train()
    losses = model(
        [torch.rand(3, 128, 160), torch.rand(3, 144, 128)],
        [
            {"boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]), "labels": torch.tensor([1])},
            {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)},
        ],
    )
    assert set(losses) == {"classification", "bbox_regression", "distribution_focal"}, (
        f"expected QFL, box and DFL terms, got {sorted(losses)}"
    )
    for name, value in losses.items():
        assert torch.is_tensor(value) and value.ndim == 0, f"{name} is not a scalar"
        assert torch.isfinite(value).all(), f"{name} is not finite"


def test_an_all_background_batch_keeps_every_term_on_the_graph(gfl):
    """Every image unannotated is a real input the engine emits. The box and DFL
    terms then have no positives to average and fall back to a zero.

    Asserted **per term**, not on the total. The first version of this test
    checked ``sum(losses.values()).requires_grad`` and SURVIVED replacing the
    fallback with a bare ``torch.tensor(0.0)`` — because QFL is always computed
    and always carries a graph, so the total requires grad no matter what the
    other two terms are. The contract that actually matters is that each term is
    a graph-attached tensor on the logits' device: a detached CPU scalar is
    invisible here and a device mismatch on a GPU edge.
    """
    import torch

    model = gfl.MyModel(3)
    model.train()
    losses = model(
        [torch.rand(3, 128, 160)],
        [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}],
    )
    expected_device = next(model.parameters()).device
    for name, value in losses.items():
        assert torch.is_tensor(value) and value.ndim == 0, f"{name} is not a scalar tensor"
        assert torch.isfinite(value).all(), f"{name} is not finite"
        assert value.device == expected_device, (
            f"{name} is on {value.device} but the model is on {expected_device} — "
            f"a hard-coded scalar fallback would fail on a GPU edge"
        )
        assert value.requires_grad, (
            f"{name} is detached from the graph on an all-background batch; the "
            f"zero fallback must be derived from a tensor that carries one"
        )
    total = sum(losses.values())
    total.backward()


def test_the_template_declares_the_family_contract():
    source = TEMPLATE.read_text(encoding="utf-8")
    assert re.search(r'^model_type\s*=\s*"torchvision_detection"', source, re.MULTILINE)
    assert "weights=None" in source
    for banned in ("import timm", "from timm", "transformers", "from_pretrained"):
        assert banned not in source, f"{banned!r} is not permitted in a cv template"


# --- the duplicated ATSS assignment -----------------------------------------
#
# gfl_resnet.py carries its own copy of _atss_assign, because zoo templates
# cannot import siblings. A mutation sweep made the consequence concrete:
# deleting the centre-inside rule from THIS file's copy SURVIVED the whole suite,
# since the only test covering that rule lived in tests/test_atss_assignment.py
# against the other template. Duplicated code needs its guard duplicated, or the
# copy silently loses the protection the original has.


def test_the_duplicated_atss_assignment_enforces_the_centre_inside_rule(gfl):
    """The 0.2px fixture from tests/test_atss_assignment.py, against GFL's copy.

    An anchor that clears the adaptive threshold but is centred OUTSIDE the
    object must still be rejected. Two cases differing only by 0.2px of anchor
    position: IoU 0.329 vs 0.338, thresholds 0.225 vs 0.231 — both clear, so
    only the centre test can separate them.

    Note a two-anchor fixture cannot test this at all: with exactly two
    candidates the mean+std threshold *is* the larger IoU, so only the best
    candidate clears it and the centre rule is never consulted.
    """
    import torch

    gt = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
    far = [[500.0, 500.0, 510.0, 510.0], [600.0, 600.0, 610.0, 610.0], [700.0, 700.0, 710.0, 710.0]]
    outside = torch.tensor([[10.1, 0.0, 30.1, 20.0]] + far)
    inside = torch.tensor([[9.9, 0.0, 29.9, 20.0]] + far)

    assert gfl._atss_assign(inside, gt, [4], topk=4)[0] == 0, (
        "the anchor centred just INSIDE the object was not assigned"
    )
    assert gfl._atss_assign(outside, gt, [4], topk=4)[0] == gfl.BACKGROUND, (
        "an anchor centred just OUTSIDE the object was assigned to it — the "
        "centre-inside rule is missing from GFL's copy of _atss_assign"
    )


def test_the_duplicated_atss_assignment_assigns_and_partitions(gfl, built):
    """The all-background and assign-everything failures, against GFL's copy."""
    import torch

    _, _, anchors, split = built
    gt = torch.tensor([[20.0, 20.0, 100.0, 100.0], [150.0, 60.0, 260.0, 200.0]])
    matched = gfl._atss_assign(anchors, gt, split, gfl.ATSS_TOPK)

    positives = int((matched >= 0).sum())
    assert positives > 0, "GFL's ATSS copy assigned every anchor to background"
    assert positives < anchors.shape[0] / 2, "the threshold or centre test is not filtering"
    for index in range(gt.shape[0]):
        assert int((matched == index).sum()) > 0, f"ground-truth box {index} won no anchor"
    assert int((matched == -2).sum()) == 0, "ATSS must not emit the ignore sentinel"
    assert bool((matched == gfl.BACKGROUND).all()) is False
    empty = gfl._atss_assign(anchors, torch.zeros((0, 4)), split, gfl.ATSS_TOPK)
    assert bool((empty == gfl.BACKGROUND).all()) and empty.shape == (anchors.shape[0],)


def test_the_duplicated_atss_topk_is_per_level(gfl, built):
    """Collapsing the split to one level is a different, worse assigner that
    trains happily — so the two must be shown to disagree."""
    import torch

    _, _, anchors, split = built
    gt = torch.tensor([[20.0, 20.0, 100.0, 100.0], [150.0, 60.0, 260.0, 200.0]])
    per_level = gfl._atss_assign(anchors, gt, split, gfl.ATSS_TOPK)
    global_topk = gfl._atss_assign(anchors, gt, [anchors.shape[0]], gfl.ATSS_TOPK)
    assert not torch.equal(per_level, global_topk)
    assert int((per_level >= 0).sum()) > int((global_topk >= 0).sum())


# --- the eval path ----------------------------------------------------------
#
# A mutation sweep exposed that postprocess_detections was effectively
# untested: forcing every pyramid level to use P3's stride SURVIVED the whole
# suite. The reason is that the classification bias is initialised to prior
# probability 0.01 (sigmoid(-4.595)), which is below the default
# score_thresh of 0.05 — so a freshly built model returns ZERO detections and
# the decode is never exercised on a real box. Both tests below drive
# postprocess_detections directly with synthetic head outputs so the decode has
# something to get wrong.


def _synthetic_head_outputs(gfl, model, anchors, split, level, anchor_index, bin_index, label):
    """Head outputs that put one confident detection at a known anchor.

    Everything else is driven to a very negative logit so exactly one box
    survives the score threshold, and the surviving box's distribution is
    one-hot at ``bin_index`` on all four sides — so its geometry is analytic.
    """
    import torch

    num_classes = model.head.num_classes
    reg_width = 4 * (gfl.REG_MAX + 1)

    split_anchors = list(anchors.split(split))
    cls_levels, reg_levels = [], []
    for level_id, count in enumerate(split):
        cls = torch.full((1, count, num_classes), -20.0)
        reg = torch.full((1, count, reg_width), -20.0)
        # Uniform-ish elsewhere is irrelevant; only the chosen anchor survives.
        reg = reg.reshape(1, count, 4, gfl.REG_MAX + 1)
        reg[..., 0] = 20.0
        reg = reg.reshape(1, count, reg_width)
        if level_id == level:
            cls[0, anchor_index, label] = 20.0
            one_hot = torch.full((4, gfl.REG_MAX + 1), -20.0)
            one_hot[:, bin_index] = 20.0
            reg[0, anchor_index] = one_hot.reshape(-1)
        cls_levels.append(cls)
        reg_levels.append(reg)

    # NOTE the nesting, which is the thing this fixture exists to get right:
    # head_outputs[k] is a FLAT list over pyramid levels, each entry
    # (N, anchors_in_level, K); anchors is a list over IMAGES of lists over
    # levels. Getting this backwards is what the template itself had wrong.
    head_outputs = {"cls_logits": cls_levels, "bbox_regression": reg_levels}
    return head_outputs, [split_anchors]


def test_postprocess_decodes_with_each_levels_own_stride(gfl, built):
    """The per-level stride must be the level's own, not P3's for everything.

    An edge distance is expressed in *stride units*, so decoding a level-2
    (stride 32) prediction with P3's stride 8 shrinks every box by 4x. Nothing
    raises and the boxes are still valid xyxy, so only an analytic check catches
    it.
    """
    import torch

    model, features, anchors, split = built
    image_shape = (256, 320)

    # Level 2 is stride 32; its grid is 8x10 for a 256x320 input. Anchor (4, 5)
    # is near the centre, so a 3-stride box stays inside the image and the
    # assertion is not testing the clip.
    level, grid_width, row, column = 2, features[2].shape[-1], 4, 5
    anchor_index = row * grid_width + column
    bin_index = 3
    stride = gfl.PYRAMID_STRIDES[level]

    head_outputs, split_anchors = _synthetic_head_outputs(
        gfl, model, anchors, split, level, anchor_index, bin_index, label=1
    )
    detections = model.postprocess_detections(head_outputs, split_anchors, [image_shape])

    assert len(detections) == 1
    boxes = detections[0]["boxes"]
    assert boxes.shape[0] >= 1, (
        "no detection survived, so the decode was never exercised — check the "
        "synthetic logits against score_thresh"
    )

    centre = gfl._centres(split_anchors[0][level][anchor_index : anchor_index + 1])[0]
    offset = bin_index * stride
    expected = torch.tensor(
        [
            float(centre[0]) - offset,
            float(centre[1]) - offset,
            float(centre[0]) + offset,
            float(centre[1]) + offset,
        ]
    )
    best = boxes[detections[0]["scores"].argmax()]
    assert torch.allclose(best, expected, atol=1e-3), (
        f"level {level} (stride {stride}) decoded to {best.tolist()}, expected "
        f"{expected.tolist()} — the decode is not using this level's own stride "
        f"(P3's stride {gfl.PYRAMID_STRIDES[0]} would give a {stride // gfl.PYRAMID_STRIDES[0]}x "
        f"smaller box)"
    )
    assert int(detections[0]["labels"][detections[0]["scores"].argmax()]) == 1, (
        "the label was decoded from the wrong column of the flattened scores"
    )


def test_postprocess_returns_the_handler_contract_shape(gfl, built):
    """boxes/scores/labels, pixel xyxy, clipped to the image."""
    import torch

    model, features, anchors, split = built
    image_shape = (256, 320)
    head_outputs, split_anchors = _synthetic_head_outputs(
        gfl, model, anchors, split, level=1, anchor_index=7, bin_index=5, label=2
    )
    detections = model.postprocess_detections(head_outputs, split_anchors, [image_shape])

    prediction = detections[0]
    assert {"boxes", "scores", "labels"} <= set(prediction)
    boxes = prediction["boxes"]
    assert boxes.ndim == 2 and boxes.shape[-1] == 4
    assert bool((boxes[:, 2] >= boxes[:, 0]).all() and (boxes[:, 3] >= boxes[:, 1]).all())
    assert bool((boxes[:, 0] >= 0).all() and (boxes[:, 1] >= 0).all()), "not clipped at the origin"
    assert bool((boxes[:, 2] <= image_shape[1]).all()), "x2 past the image width"
    assert bool((boxes[:, 3] <= image_shape[0]).all()), "y2 past the image height"
    assert prediction["scores"].shape[0] == boxes.shape[0] == prediction["labels"].shape[0]
    assert prediction["labels"].dtype == torch.int64


def test_the_duplicated_atss_handles_a_single_candidate(gfl):
    """std over one sample is NaN under the unbiased estimator, and
    ``iou >= nan`` is False everywhere, so the object silently gets no anchors.

    Ported from tests/test_atss_assignment.py after a mutation sweep showed the
    GFL copy was unprotected: swapping in ``candidate_ious.std(dim=1)`` here
    SURVIVED, because the test covering it lived against the other template.
    """
    import torch

    anchors = torch.tensor([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]])
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    matched = gfl._atss_assign(anchors, gt, [1, 1], topk=1)
    assert int((matched >= 0).sum()) >= 1, (
        "a single candidate per level produced no positive — the mean+std "
        "threshold went NaN in GFL's copy of _atss_assign"
    )

