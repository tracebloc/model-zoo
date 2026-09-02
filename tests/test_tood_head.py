"""Contract tests for TOOD's T-Head, TAL assignment and decode (backend#2982, Tier 2).

Why this file exists on top of the family train-step guard
----------------------------------------------------------
The head, the assignment, the losses and the decode are all ours. The train-step
guard proves the template completes a step; it cannot see any of the following,
all of which produce finite losses and a clean backward:

- **The cold start.** This is the real bug this file caught, and it is the most
  important thing here. TAL's metric is ``t = s ** alpha * u ** beta`` with
  ``beta = 6``, and at initialisation the model cannot predict, so ``u`` is
  ~0.004 and ``t`` collapses to ~1e-15 for every anchor. A pure TAL assignment
  then selects **nothing**: the classification target is all zeros, the box loss
  has no positives, and the model trains happily while learning nothing at all.
  Measured before the fix: box loss ``4e-05`` and total alignment ``0.0``.
- **Task interaction.** TOOD's whole premise is that the two task features come
  from one shared stack. Two independent towers would train fine and not be TOOD.
- **Output ordering.** ``AnchorGenerator`` is location-major; the head's conv
  output is ``(N, A * K, H, W)``. A disagreeing permutation is shape-identical.
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
TEMPLATE = ROOT / "model_zoo" / "object_detection" / "pytorch" / "tood_resnet.py"


def _load():
    spec = importlib.util.spec_from_file_location("tood_under_test", TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tood():
    pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")
    return _load()


@pytest.fixture(scope="module")
def built(tood):
    import torch
    from torchvision.models.detection.image_list import ImageList

    model = tood.MyModel(3)
    model.eval()
    image = torch.rand(1, 3, 256, 320)
    with torch.no_grad():
        features = list(model.backbone(image).values())
    anchors = model.anchor_generator(ImageList(image, [(256, 320)]), features)[0]
    return model, features, anchors, list(model.anchor_generator.num_anchors_per_level)


# --- THE COLD START — the bug this file caught ------------------------------


def test_the_cold_start_still_assigns_positives(tood, built):
    """Degenerate predictions must still produce an assignment with a real target.

    This is the failure mode the module docstring opens with. Predicted boxes at
    initialisation are near-degenerate, so ``u ** 6`` annihilates the alignment
    metric and a faithful TAL selects nothing. The template stages per object:
    an object whose candidates are all degenerate is ranked geometrically and
    supervised with a HARD target of 1, which is what the reference
    implementation's ATSS warm-up epoch achieves.

    Driven with *exactly* zero-area predicted boxes, so the metric is identically
    zero and only the fallback can produce anything.
    """
    import torch

    _, _, anchors, split = built
    gt = torch.tensor([[20.0, 20.0, 100.0, 100.0]])
    labels = torch.tensor([1])
    scores = torch.full((anchors.shape[0], 4), 0.01)
    # Zero-area boxes at every anchor centre: IoU is 0 everywhere by construction.
    degenerate = tood._centres(anchors).repeat(1, 2)

    matched, alignment = tood._tal_assign(anchors, gt, labels, scores, degenerate, split)

    positives = int((matched >= 0).sum())
    assert positives > 0, (
        "with degenerate predictions TAL assigned NOTHING. The classification "
        "target is then all zeros and the box loss has no positives, so the "
        "model trains, reports finite losses, and never learns an object"
    )
    assert float(alignment.max()) == pytest.approx(1.0, abs=1e-6), (
        f"the cold-start target is {float(alignment.max()):.6f}, not the hard "
        f"1.0. A soft target derived from a ~0 IoU gives no positive "
        f"supervision at all"
    )
    assert float(alignment[matched >= 0].min()) > 0.0, (
        "an assigned anchor has zero alignment, so it contributes nothing to "
        "either loss — it is a positive in name only"
    )


def test_the_soft_target_takes_over_once_predictions_are_good(tood, built):
    """The other half of the staging: a well-predicting object must get the
    normalised alignment target, not the hard 1.0 forever."""
    import torch

    _, _, anchors, split = built
    gt = torch.tensor([[20.0, 20.0, 100.0, 100.0]])
    labels = torch.tensor([1])
    scores = torch.full((anchors.shape[0], 4), 0.6)
    # Predictions that genuinely overlap the object: every anchor predicts it.
    good = gt.repeat(anchors.shape[0], 1)

    matched, alignment = tood._tal_assign(anchors, gt, labels, scores, good, split)
    assert int((matched >= 0).sum()) > 0
    # The normalisation scales the metric's maximum to the best achieved IoU,
    # which here is 1.0 -- so a hard-vs-soft distinction needs a partial overlap.
    partial = gt.repeat(anchors.shape[0], 1).clone()
    partial[:, 2] = 60.0   # half the width -> IoU 0.5 with the object
    _, partial_alignment = tood._tal_assign(anchors, gt, labels, scores, partial, split)
    top = float(partial_alignment.max())
    assert 0.0 < top < 0.99, (
        f"with a best achievable IoU of ~0.5 the alignment target peaked at "
        f"{top:.4f}; a value of 1.0 means the hard cold-start target is being "
        f"used even though the metric is informative"
    )


def test_regression_starts_at_about_one_stride(tood):
    """The initialisation that makes the cold start recoverable at all.

    With a zero bias the ReLU'd distance head emits a mean of ~0.08 stride, so
    the first boxes are near-degenerate points whose IoU with anything is ~0.004
    — and ``u ** 6`` then annihilates the metric. Measured before the fix.
    """
    import torch

    model = tood.MyModel(3)
    bias = model.head.bbox_regression.bias
    assert torch.allclose(bias, torch.ones_like(bias), atol=1e-6), (
        f"the distance head bias is {bias.flatten()[:4].tolist()}, not 1.0. "
        f"Distances then start near zero, predicted boxes are degenerate "
        f"points, and TAL's u ** 6 term annihilates the alignment metric"
    )


def test_predicted_distances_are_never_negative(tood, built):
    """A negative distance flips an edge across the anchor centre, giving
    ``x2 < x1`` — not a valid xyxy box, and the engine's torchmetrics read these
    as xyxy and would score nonsense rather than raise.

    Driven by forcing the head's pre-activation strongly negative. Added after a
    mutation sweep: checking the freshly built model SURVIVED removing the ReLU,
    because the ``+1.0`` bias keeps most outputs positive at initialisation
    anyway. The guard has to make the ReLU the only thing standing between a
    negative pre-activation and the decode.
    """
    import torch

    model, features, _, _ = built
    with torch.no_grad():
        # A large negative bias makes every pre-activation negative.
        torch.nn.init.constant_(model.head.bbox_regression.bias, -50.0)
        try:
            outputs = model.head(features)
        finally:
            torch.nn.init.constant_(model.head.bbox_regression.bias, 1.0)

    distances = outputs["bbox_regression"]
    assert float(distances.min()) >= 0.0, (
        f"with a -50 bias the distance head emitted {float(distances.min()):.4f} "
        f"— the ReLU is missing, so negative distances reach the decode and "
        f"produce inverted boxes"
    )


# --- the T-Head: interaction is the point -----------------------------------


def test_both_tasks_read_the_same_shared_stack(tood):
    """TOOD's premise: one interactive stack, two task-specific reads.

    Two independent towers train perfectly well and are not TOOD. This asserts
    there is exactly one stack of inter-level convolutions and that both task
    branches consume it, by checking the module structure rather than the
    numbers — a shape assertion would pass either way.
    """
    model = tood.MyModel(3)
    head = model.head

    assert len(head.inter_convs) == tood.STACKED_CONVS, (
        f"expected {tood.STACKED_CONVS} shared inter-level convs, got "
        f"{len(head.inter_convs)}"
    )
    # There must be exactly ONE stack, not one per task.
    stacks = [name for name, _ in head.named_children() if "inter" in name]
    assert stacks == ["inter_convs"], (
        f"found {stacks} — TOOD's head must share a single stack between the "
        f"two tasks, not compute a separate tower for each"
    )
    # And two distinct attentions reading it.
    assert head.cls_attention is not head.reg_attention, (
        "the two tasks share one layer-attention module, so they cannot weight "
        "the stack differently — which is the entire mechanism"
    )
    assert head.cls_attention.num_layers == tood.STACKED_CONVS
    assert head.reg_attention.num_layers == tood.STACKED_CONVS


def test_layer_attention_weights_are_independent_not_a_distribution(tood):
    """Sigmoid, one weight per layer — NOT a softmax over layers.

    Rewritten after a mutation sweep. The first version recomputed the weights
    itself with ``torch.sigmoid(...)`` and asserted *that* was in ``[0, 1]``,
    which tests the fixture rather than the module, so swapping the module to a
    softmax SURVIVED. Same shape of defect as the detachment test in
    ``test_gfl_head.py``.

    The observable difference is what the fused output can be. Under a sigmoid
    the six weights are INDEPENDENT and can all saturate at 1, so the fused map
    approaches the plain SUM of the stack. Under a softmax they are forced to
    sum to 1, so the fused map can never exceed the largest single map. Driving
    ``fc2``'s bias high separates the two cases by a factor of six.
    """
    import torch

    channels, num_layers = 64, tood.STACKED_CONVS
    attention = tood._LayerAttention(channels, num_layers)
    # Saturate the gate: every layer weight should go to ~1 under a sigmoid.
    torch.nn.init.zeros_(attention.fc2.weight)
    torch.nn.init.constant_(attention.fc2.bias, 20.0)

    stacked = [torch.ones(1, channels, 4, 4) for _ in range(num_layers)]
    fused = attention(stacked)

    assert fused.shape == stacked[0].shape, (
        f"layer attention returned {tuple(fused.shape)}, expected "
        f"{tuple(stacked[0].shape)} — it must weight and SUM the stack"
    )
    assert float(fused.mean()) == pytest.approx(float(num_layers), abs=0.05), (
        f"with every gate saturated the fused map averages "
        f"{float(fused.mean()):.3f}; six all-ones layers each weighted ~1 must "
        f"give ~{num_layers}. A value near 1.0 means the weights are being "
        f"normalised into a distribution (softmax), so they cannot be "
        f"independent — which is not what the paper's layer attention does"
    )

    # And the gate must genuinely be bounded above: drive it the other way.
    torch.nn.init.constant_(attention.fc2.bias, -20.0)
    suppressed = attention(stacked)
    assert float(suppressed.mean()) == pytest.approx(0.0, abs=0.05), (
        f"with every gate driven off the fused map averages "
        f"{float(suppressed.mean()):.3f}, expected ~0 — the weighting is not "
        f"actually gating the stack"
    )


def test_the_two_tasks_produce_different_features(tood, built):
    """The layer attentions are separately parameterised, so on a real forward
    the two task features must not be identical. Identical features would mean
    the attention collapsed and the head is effectively single-task."""
    import torch

    model, features, _, _ = built
    head = model.head
    with torch.no_grad():
        stacked, current = [], features[0]
        for conv in head.inter_convs:
            current = conv(current)
            stacked.append(current)
        cls_feature = head.cls_reduce(head.cls_attention(stacked))
        reg_feature = head.reg_reduce(head.reg_attention(stacked))
    assert not torch.allclose(cls_feature, reg_feature), (
        "the classification and regression features are identical, so the two "
        "layer attentions are producing the same weighting — the task-specific "
        "half of the T-Head is inert"
    )


# --- output ordering ---------------------------------------------------------


def test_head_flatten_is_location_major(tood, built):
    """A conv activation at grid ``(h, w)`` must land on the anchor at ``(h, w)``.

    Pinned at ``A = 2`` as well as ``A = 1``: with a single anchor per location
    the location-major and anchor-major permutations are identical, so the live
    model cannot distinguish them and a future anchor-count change would decode
    at the wrong pixels silently.
    """
    import torch

    _, features, anchors, _ = built
    height, width = features[0].shape[-2:]
    stride = float(tood.PYRAMID_STRIDES[0])

    for row, column in ((0, 1), (1, 0), (2, 3)):
        output = torch.zeros(1, 3, height, width)
        output[0, 2, row, column] = 7.0
        flat = tood._TOODHead._flatten(output, 3)
        expected = row * width + column
        assert (flat[0] == 7.0).nonzero().tolist() == [[expected, 2]]
        centre = tood._centres(anchors[expected : expected + 1])[0]
        assert (float(centre[0]), float(centre[1])) == (column * stride, row * stride)

    output = torch.zeros(1, 2 * 3, height, width)
    output[0, 1 * 3 + 2, 1, 3] = 7.0
    flat = tood._TOODHead._flatten(output, 3)
    expected = (1 * width + 3) * 2 + 1
    assert (flat[0] == 7.0).nonzero().tolist() == [[expected, 2]], (
        "with two anchors per location the flatten is not location-major"
    )


# --- TAL assignment ---------------------------------------------------------


def test_alignment_metric_combines_both_tasks(tood, built):
    """``t = s ** alpha * u ** beta``: an anchor good at only one task must lose
    to one good at both. This is the entire point of the metric."""
    import torch

    _, _, anchors, split = built
    gt = torch.tensor([[20.0, 20.0, 100.0, 100.0]])
    labels = torch.tensor([1])

    # Everyone predicts the object well, but only some are confident.
    boxes = gt.repeat(anchors.shape[0], 1)
    scores = torch.full((anchors.shape[0], 4), 0.05)
    matched_low, alignment_low = tood._tal_assign(anchors, gt, labels, scores, boxes, split)
    scores_high = torch.full((anchors.shape[0], 4), 0.9)
    matched_high, alignment_high = tood._tal_assign(
        anchors, gt, labels, scores_high, boxes, split
    )
    # Both should assign; the metric magnitudes differ but the NORMALISED target
    # is scaled to the achieved IoU either way, so assert assignment happened
    # and the metric is score-sensitive by checking the raw product directly.
    assert int((matched_low >= 0).sum()) > 0 and int((matched_high >= 0).sum()) > 0

    from torchvision.ops import boxes as box_ops

    ious = box_ops.box_iou(gt, boxes)
    metric_low = (0.05 ** tood.TAL_ALPHA) * ious.clamp(min=0).pow(tood.TAL_BETA)
    metric_high = (0.9 ** tood.TAL_ALPHA) * ious.clamp(min=0).pow(tood.TAL_BETA)
    assert float(metric_high.max()) > float(metric_low.max()), (
        "the alignment metric is not sensitive to the classification score"
    )
    assert tood.TAL_BETA > tood.TAL_ALPHA, (
        f"beta ({tood.TAL_BETA}) must exceed alpha ({tood.TAL_ALPHA}): a "
        f"confident anchor that localises badly should be punished harder than "
        f"an unconfident one that localises well"
    )


def test_an_anchor_outside_the_object_is_never_positive(tood, built):
    """Eligibility is pool AND centre-inside. Dropping the inside test lets a
    distant anchor with a lucky prediction become a positive."""
    import torch

    _, _, anchors, split = built
    gt = torch.tensor([[20.0, 20.0, 60.0, 60.0]])
    labels = torch.tensor([1])
    scores = torch.full((anchors.shape[0], 4), 0.9)
    boxes = gt.repeat(anchors.shape[0], 1)   # everyone predicts perfectly

    matched, _ = tood._tal_assign(anchors, gt, labels, scores, boxes, split)
    centres = tood._centres(anchors)
    positive = matched >= 0
    assert bool(positive.any())
    inside = (
        (centres[:, 0] >= gt[0, 0]) & (centres[:, 0] <= gt[0, 2])
        & (centres[:, 1] >= gt[0, 1]) & (centres[:, 1] <= gt[0, 3])
    )
    assert bool((~inside[positive]).sum() == 0), (
        f"{int((~inside[positive]).sum())} positives have centres OUTSIDE the "
        f"object, even though every anchor predicted it perfectly — the "
        f"centre-inside eligibility test is not being applied"
    )


def test_at_most_topk_positives_per_object(tood, built):
    """The selection is top-m per object; without the ``ranking > 0`` filter an
    ineligible anchor is promoted whenever an object has fewer than m
    candidates, and without topk every eligible anchor becomes positive."""
    import torch

    _, _, anchors, split = built
    gt = torch.tensor([[20.0, 20.0, 200.0, 200.0], [40.0, 40.0, 120.0, 120.0]])
    labels = torch.tensor([1, 2])
    scores = torch.full((anchors.shape[0], 4), 0.5)
    boxes = gt[:1].repeat(anchors.shape[0], 1)

    matched, _ = tood._tal_assign(anchors, gt, labels, scores, boxes, split)
    for index in range(gt.shape[0]):
        count = int((matched == index).sum())
        assert count <= tood.TAL_TOPK, (
            f"object {index} won {count} anchors, above TAL_TOPK "
            f"({tood.TAL_TOPK})"
        )


def test_every_positive_carries_a_non_zero_alignment(tood, built):
    """A positive with zero alignment is a positive in name only.

    It weights the box loss at 0 and trains the classifier towards 0 for its own
    class. Added after a mutation sweep: dropping the ``ranking > 0`` filter
    survived, because the per-object topk returns ``k`` columns even when an
    object has fewer than ``k`` candidates with a non-zero metric, so zero-metric
    anchors get promoted.

    The first version of this fixture ALSO failed to catch it, and the reason is
    worth recording: it gave good predictions to ``anchors[:3]``, which are the
    first anchors of P3 — at the image's top-left corner, outside the object and
    therefore never eligible. The metric stayed zero everywhere, the object
    counted as degenerate, and the hard-target cold-start path fired and made
    every alignment 1.0. The fixture has to pick anchors that are genuinely
    ELIGIBLE, or it tests the cold-start branch instead of the one it names.
    """
    import torch

    _, _, anchors, split = built
    gt = torch.tensor([[20.0, 20.0, 100.0, 100.0]])
    labels = torch.tensor([1])
    scores = torch.full((anchors.shape[0], 4), 0.5)

    # Anchors genuinely inside the object AND in the geometric pool.
    centres = tood._centres(anchors)
    eligible = (
        tood._candidate_pool(centres, tood._centres(gt), split, tood.POOL_TOPK)
        & tood._inside_mask(centres, gt)
    )[0]
    eligible_idx = eligible.nonzero().flatten()
    assert eligible_idx.numel() > 3, (
        f"only {eligible_idx.numel()} eligible anchors; the fixture needs more "
        f"than 3 to leave the object non-degenerate with few informative ones"
    )

    # Only three eligible anchors predict the object, and they predict it at a
    # PARTIAL overlap; every other prediction is a zero-area box, so their
    # metric is 0 while the object is NOT degenerate.
    #
    # The partial overlap matters. An exact prediction gives IoU 1.0, and the
    # normalisation legitimately scales the best candidate's target to the
    # object's best achieved IoU — so the target would be 1.0 on the
    # non-degenerate path too, and indistinguishable from the hard cold-start
    # value. Half the object's width gives IoU 0.5, which separates them.
    half_width = gt.clone()
    half_width[:, 2] = 60.0                     # IoU 0.5 with the 20..100 object
    predicted = centres.repeat(1, 2)
    predicted[eligible_idx[:3]] = half_width.repeat(3, 1)

    matched, alignment = tood._tal_assign(anchors, gt, labels, scores, predicted, split)
    positive = matched >= 0
    assert bool(positive.any()), "the fixture produced no positives at all"
    assert float(alignment.max()) == pytest.approx(0.5, abs=0.05), (
        f"the alignment target peaked at {float(alignment.max()):.6f}, expected "
        f"~0.5 — the object's best achieved IoU, which is what the "
        f"normalisation scales to. A value of 1.0 means the hard cold-start "
        f"target fired and this fixture is testing the wrong branch"
    )
    zero_weighted = int((alignment[positive] == 0).sum())
    assert zero_weighted == 0, (
        f"{zero_weighted} of {int(positive.sum())} positives have zero "
        f"alignment. They weight the box loss at 0 and push their own class "
        f"towards 0 in the classification loss — the `ranking > 0` filter is "
        f"not excluding candidates the metric rejected"
    )


def test_a_small_object_cannot_promote_ineligible_anchors(tood, built):
    """When an object has fewer eligible anchors than ``TAL_TOPK``.

    ``metric.topk(k)`` always returns ``k`` columns. For a small object with
    only a handful of anchors inside it, the remaining columns are anchors with
    a zero ranking — ineligible ones — and without the filter they become
    positives. The larger fixtures above cannot show this because they have more
    than ``TAL_TOPK`` eligible anchors, so the topk never has to reach outside.
    """
    import torch

    _, _, anchors, split = built
    # Small enough that far fewer than TAL_TOPK anchor centres fall inside it.
    gt = torch.tensor([[100.0, 100.0, 118.0, 118.0]])
    labels = torch.tensor([1])
    scores = torch.full((anchors.shape[0], 4), 0.7)
    predicted = gt.repeat(anchors.shape[0], 1)

    centres = tood._centres(anchors)
    inside = tood._inside_mask(centres, gt)[0]
    assert int(inside.sum()) < tood.TAL_TOPK, (
        f"{int(inside.sum())} anchor centres fall inside this object, which is "
        f"not fewer than TAL_TOPK ({tood.TAL_TOPK}) — pick a smaller box or the "
        f"test cannot exercise the case it exists for"
    )

    matched, _ = tood._tal_assign(anchors, gt, labels, scores, predicted, split)
    positive = matched >= 0
    outside_positives = int((~inside[positive]).sum())
    assert outside_positives == 0, (
        f"{outside_positives} anchors OUTSIDE a small object were made positive. "
        f"topk returns TAL_TOPK columns regardless of how many candidates are "
        f"eligible, so the zero-ranking filter is what keeps the surplus out"
    )
    assert int(positive.sum()) <= int(inside.sum()), (
        f"{int(positive.sum())} positives for an object with only "
        f"{int(inside.sum())} anchors inside it"
    )


def test_zero_object_image_is_all_background(tood, built):
    import torch

    _, _, anchors, split = built
    matched, alignment = tood._tal_assign(
        anchors, torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64),
        torch.zeros((anchors.shape[0], 4)), anchors.clone(), split,
    )
    assert matched.shape == (anchors.shape[0],)
    assert bool((matched == tood.BACKGROUND).all())
    assert float(alignment.abs().sum()) == 0.0


def test_the_assignment_does_not_leak_gradient(tood):
    """The assignment reads the model's own predictions, so it is a
    label-producing step. If it carried a graph, the classification loss could
    be reduced by moving the boxes and the classifier would train the box head.
    """
    import torch

    model = tood.MyModel(3)
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
        "so the TAL assignment is not detached"
    )
    classifier_grad = model.head.cls_logits.weight.grad
    assert classifier_grad is not None and bool(classifier_grad.abs().sum() > 0), (
        "the classification loss put no gradient on head.cls_logits either, so "
        "backward did nothing and the assertion above proves nothing"
    )


def test_compute_loss_refuses_a_plain_anchor_generator(tood):
    import torch
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    model = tood.MyModel(3)
    model.anchor_generator = AnchorGenerator(
        sizes=tuple((8 * s,) for s in tood.PYRAMID_STRIDES),
        aspect_ratios=((1.0,),) * len(tood.PYRAMID_STRIDES),
    )
    model.train()
    with pytest.raises(RuntimeError, match="per-level anchor split"):
        model(
            [torch.rand(3, 128, 160)],
            [{"boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]), "labels": torch.tensor([1])}],
        )


# --- the eval path, analytically --------------------------------------------


def _synthetic(tood, model, anchors, split, level, anchor_index, distance, label):
    """Head outputs putting one confident detection at a known anchor."""
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


def test_decode_uses_each_levels_own_stride(tood, built):
    """Distances are in stride units, so decoding a level-2 prediction with P3's
    stride shrinks the box fourfold while still producing valid xyxy."""
    import torch

    model, features, anchors, split = built
    level, row, column, distance, label = 2, 4, 5, 3.0, 1
    grid_width = features[level].shape[-1]
    anchor_index = row * grid_width + column
    stride = tood.PYRAMID_STRIDES[level]

    head_outputs, split_anchors = _synthetic(
        tood, model, anchors, split, level, anchor_index, distance, label
    )
    detections = model.postprocess_detections(head_outputs, split_anchors, [(256, 320)])
    prediction = detections[0]
    assert prediction["boxes"].shape[0] >= 1, "no detection survived; the decode was not exercised"

    best = int(prediction["scores"].argmax())
    centre = tood._centres(split_anchors[0][level][anchor_index : anchor_index + 1])[0]
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


def test_decode_returns_the_handler_contract_shape(tood, built):
    import torch

    model, features, anchors, split = built
    head_outputs, split_anchors = _synthetic(tood, model, anchors, split, 1, 7, 2.0, 2)
    prediction = model.postprocess_detections(head_outputs, split_anchors, [(256, 320)])[0]
    assert {"boxes", "scores", "labels"} <= set(prediction)
    boxes = prediction["boxes"]
    assert boxes.ndim == 2 and boxes.shape[-1] == 4
    assert bool((boxes[:, 2] >= boxes[:, 0]).all() and (boxes[:, 3] >= boxes[:, 1]).all())
    assert bool((boxes[:, 0] >= 0).all() and (boxes[:, 2] <= 320).all())
    assert prediction["labels"].dtype == torch.int64


# --- the loss dict -----------------------------------------------------------


def test_both_losses_are_reported_on_the_graph(tood):
    import torch

    model = tood.MyModel(3)
    model.train()
    losses = model(
        [torch.rand(3, 128, 160), torch.rand(3, 144, 128)],
        [
            {"boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]), "labels": torch.tensor([1])},
            {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)},
        ],
    )
    assert set(losses) == {"classification", "bbox_regression"}, f"got {sorted(losses)}"
    device = next(model.parameters()).device
    for name, value in losses.items():
        assert torch.is_tensor(value) and value.ndim == 0, f"{name} is not a scalar"
        assert torch.isfinite(value).all(), f"{name} is not finite"
        assert value.device == device, f"{name} is on {value.device}"
        assert value.requires_grad, f"{name} is detached from the graph"
    assert float(losses["bbox_regression"].detach()) > 0.0, (
        "the box loss is exactly zero on a batch with an annotated image, which "
        "means no positive was assigned or every alignment weight was zero"
    )


def test_an_all_background_batch_trains(tood):
    import torch

    model = tood.MyModel(3)
    model.train()
    losses = model(
        [torch.rand(3, 128, 160)],
        [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}],
    )
    for name, value in losses.items():
        assert torch.isfinite(value).all(), f"{name} is {value}"
        assert value.requires_grad, f"{name} left the graph"
    sum(losses.values()).backward()


def test_every_trainable_parameter_receives_a_gradient(tood):
    """A dead parameter is still serialised, uploaded and averaged every
    federated round, forever, for a value that can never change. Frozen
    parameters are excluded: trainable_layers=3 freezes the stem on purpose."""
    import torch

    model = tood.MyModel(3)
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


def test_the_template_declares_the_family_contract():
    source = TEMPLATE.read_text(encoding="utf-8")
    assert re.search(r'^model_type\s*=\s*"torchvision_detection"', source, re.MULTILINE)
    assert "weights=None" in source
    for banned in ("import timm", "from timm", "transformers", "from_pretrained"):
        assert banned not in source, f"{banned!r} is not permitted in a cv template"


# --- review findings on model-zoo#238 ---------------------------------------
#
# Two real defects, both found by review and by Bugbot independently, and both
# quiet: they degrade supervision rather than breaking training. The gap that
# let the first through was named in the review and is worth stating plainly:
# NOTHING in the suite exercised a non-unit `scales[level]`.


def test_a_negative_per_level_scale_cannot_invert_the_boxes(vf_unused=None):
    """The per-level scale must not be able to make distances negative.

    ``scales[level]`` is an unconstrained ``Parameter`` taking gradient from the
    GIoU loss, so it can cross zero. With the scale applied AFTER the ReLU it
    was the final operation and nothing clamped the result: measured, a scale of
    -1 inverted **every** box on that level (428 of 428 with ``x2 < x1``).
    ``generalized_box_iou_loss`` does not assert on degenerate boxes the way
    ``generalized_box_iou`` does, so training just continued on inverted boxes.

    ``test_predicted_distances_are_never_negative`` could not catch it — it
    drives the conv bias to -50 and leaves the scale at 1. This drives the
    scale, which is the axis that was untested.
    """
    import torch

    module = _load()
    model = module.MyModel(3)
    model.eval()
    image = torch.rand(1, 3, 128, 160)
    with torch.no_grad():
        features = list(model.backbone(image).values())
        for scale in model.head.scales:
            torch.nn.init.constant_(scale, -1.0)
        outputs = model.head(features)

    distances = outputs["bbox_regression"]
    assert float(distances.min()) >= 0.0, (
        f"with every per-level scale at -1 the distances reached "
        f"{float(distances.min()):.4f}. The scale must be applied INSIDE the "
        f"ReLU so the clamp is the last word whatever the scale does"
    )


def test_a_negative_scale_also_leaves_the_decoded_boxes_valid(vf_unused=None):
    """The consequence, end to end: the decode must still emit valid xyxy.

    Asserted separately from the distances because this is the property the
    engine's torchmetrics actually depend on, and they read boxes as xyxy pixels
    without validating them.
    """
    import torch
    from torchvision.models.detection.image_list import ImageList

    module = _load()
    model = module.MyModel(3)
    model.eval()
    image = torch.rand(1, 3, 128, 160)
    with torch.no_grad():
        features = list(model.backbone(image).values())
        for scale in model.head.scales:
            torch.nn.init.constant_(scale, -2.5)
        outputs = model.head(features)
        anchors = model.anchor_generator(ImageList(image, [(128, 160)]), features)[0]

    split = model.anchor_generator.num_anchors_per_level
    strides = module._anchor_strides(split, anchors.device, anchors.dtype)
    boxes = module._distance_to_box(
        module._centres(anchors), outputs["bbox_regression"][0], strides
    )
    inverted = int(((boxes[:, 2] < boxes[:, 0]) | (boxes[:, 3] < boxes[:, 1])).sum())
    assert inverted == 0, (
        f"{inverted} of {boxes.shape[0]} decoded boxes are inverted with a "
        f"negative per-level scale"
    )


def test_the_alignment_target_comes_from_the_matched_object(vf_unused=None):
    """The soft label must come from the SAME object whose class channel it is
    written into.

    ``matched`` resolves a multi-claimed anchor by IoU. Taking
    ``normalised.max(dim=0)`` for the alignment decouples the two, so an anchor
    selected by both a well-predicted object and a degenerate one gets the
    degenerate one's hard 1.0 written onto the *good* object's channel. Review
    finding on model-zoo#238.

    An earlier version of this test used the real model's anchors with two
    overlapping objects and SURVIVED the mutation, because neither object ended
    up degenerate — every anchor predicted both of them at least a little, so
    both took the soft path and the max happened to be the matched one. The
    scenario has to be constructed exactly.

    This fixture is fully synthetic and pins the numbers:

        gt0 = [0, 0, 100, 100]      gt1 = [90, 0, 200, 100]   (overlap: x 90-100)
        anchor A centred (95, 50)   -> inside BOTH
        anchor B centred (150, 50)  -> inside gt1 only
        anchor C centred (50, 50)   -> inside gt0 only

        A predicts [0, 0, 70, 71]   -> IoU 0.497 with gt0, EXACTLY 0 with gt1
        B predicts a zero-area point -> IoU 0 with both
        C predicts [0, 0, 50, 50]   -> IoU 0.25 with gt0

    So gt1's metric is zero for every eligible anchor and it is DEGENERATE,
    giving its positives the hard target 1.0 — while gt0 is not, and A is gt0's
    best candidate, so A's own target is gt0's best achieved IoU, 0.497.

    A is matched to gt0 (0.497 > 0). Gathering from the matched object gives
    0.497; a max over all objects gives gt1's 1.0.
    """
    import torch

    module = _load()
    gt = torch.tensor([[0.0, 0.0, 100.0, 100.0], [90.0, 0.0, 200.0, 100.0]])
    labels = torch.tensor([1, 2])
    anchors = torch.tensor([
        [91.0, 46.0, 99.0, 54.0],      # A, centre (95, 50)
        [146.0, 46.0, 154.0, 54.0],    # B, centre (150, 50)
        [46.0, 46.0, 54.0, 54.0],      # C, centre (50, 50)
    ])
    predicted = torch.tensor([
        [0.0, 0.0, 70.0, 71.0],        # IoU 0.497 with gt0, 0 with gt1
        [150.0, 50.0, 150.0, 50.0],    # zero area: 0 with both
        [0.0, 0.0, 50.0, 50.0],        # IoU 0.25 with gt0
    ])
    scores = torch.full((3, 4), 0.6)

    matched, alignment = module._tal_assign(
        anchors, gt, labels, scores, predicted, [3], topk=3, pool_topk=3
    )

    assert int(matched[0]) == 0, (
        f"anchor A was matched to object {int(matched[0])}, expected 0 — it has "
        f"IoU 0.497 with gt0 and 0 with gt1, so the fixture is not set up as "
        f"described"
    )
    assert float(alignment[1]) == pytest.approx(1.0, abs=1e-6), (
        f"anchor B's target is {float(alignment[1]):.6f}, expected the hard 1.0 "
        f"— gt1 is meant to be degenerate here, or the fixture no longer "
        f"exercises the leak"
    )
    assert float(alignment[0]) == pytest.approx(0.497, abs=0.01), (
        f"anchor A carries an alignment of {float(alignment[0]):.6f}, expected "
        f"~0.497 — gt0's best achieved IoU, which is A's own target. A value of "
        f"1.0 is gt1's hard cold-start target leaking across, which means the "
        f"alignment is taken as a max over all objects rather than gathered "
        f"from the matched one"
    )
    assert float(alignment[0]) < 1.0 - 1e-6, (
        "anchor A's target reached the hard 1.0 belonging to a different object"
    )
