"""Matching, proposal-learning and decode tests for
``object_detection/pytorch/sparse_rcnn.py``.

What is worth testing here, and what is not
-------------------------------------------
``tests/test_od_torchvision_family_train_step.py`` proves the template trains
and evaluates, and ``tests/test_model_contract.py`` proves it constructs.
Neither can tell an exact Hungarian matching from a greedy one, learned
proposals from 100 frozen full-image boxes, dynamic instance interaction from a
plain concatenation, or six independent stages from one stage run six times.
Every one of those is a shape-identical substitution that leaves the existing
assertions green while removing what Sparse R-CNN is.

Every test below names the mutation it exists for.

Four rules the tests obey, each learned from a real bug in this roster
---------------------------------------------------------------------
**An independent oracle, not a second copy of the algorithm.** Two tests in
this repo's history recomputed the quantity under test and asserted on their own
copy, so swapping the real code for something else survived. The matcher here is
checked against **brute force over every permutation** — a different algorithm
with the same answer — not against another shortest-augmenting-path
implementation.

**Drive the decode at batch >= 2, above threshold, with per-image differences.**
A real bug shipped by iterating a list of pyramid levels as though it were
images, with ``zip`` truncating instead of raising: right for image 0, silently
wrong past it. A fresh focal-loss head also cannot catch it, because its scores
sit below any sensible threshold and the assertions pass against a well-formed
empty list.

**Every positive control must be able to react.** A test that would pass if the
code did nothing is not a test. The proposal-movement checks assert both that
the parameter moves and that it moves *differently per row*; the dynamic-conv
check asserts both that changing the proposal feature changes the output and
that not changing it does not.

**No statistic of randomly initialised weights.** Measured bands overlap.
Structural claims are tested structurally, with known constants.
"""

import importlib.util
import itertools
import pathlib
import re
from collections import OrderedDict

import pytest

ROOT = pathlib.Path(__file__).parent.parent
TEMPLATE = ROOT / "model_zoo" / "object_detection" / "pytorch" / "sparse_rcnn.py"

pytest.importorskip("torch", reason="pytorch not installed in this CI job")
pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")

import torch  # noqa: E402 — after importorskip, deliberately
from torch import nn  # noqa: E402
from torchvision.models.detection.transform import GeneralizedRCNNTransform  # noqa: E402
from torchvision.ops import MultiScaleRoIAlign  # noqa: E402


def _module():
    spec = importlib.util.spec_from_file_location(
        re.sub(r"\W", "_", f"sparse_{TEMPLATE.stem}"), TEMPLATE
    )
    assert spec and spec.loader, f"{TEMPLATE}: importlib could not build a spec"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _module()

#: Small-model dimensions for the tests that do not need the real ResNet-50.
#: The full template is ~106M parameters and ~17s per CPU step; the pieces under
#: test here are the head, the matcher and the decode, none of which care what
#: produced the feature maps.
SMALL_CHANNELS = 16
SMALL_PROPOSALS = 6
SMALL_STAGES = 2
SMALL_CLASSES = 4
#: Deliberately non-square, so a height/width transposition is a shape error
#: rather than a silent pass.
SMALL_FEATURE_SHAPE = (5, 9)
SMALL_IMAGE_SHAPE = (40, 72)


class _StubBackbone(nn.Module):
    """Four pyramid levels of the right width, from a single trainable conv.

    Trainable on purpose: the gradient-reachability tests need the backbone to
    be part of the graph, and a constant would make them vacuous.
    """

    def __init__(self, channels=SMALL_CHANNELS):
        super().__init__()
        self.out_channels = channels
        self.stem = nn.Conv2d(3, channels, 3, padding=1)

    def forward(self, x):
        base = self.stem(x)
        levels = OrderedDict()
        for index in range(4):
            stride = 2 ** (index + 2)
            levels[str(index)] = nn.functional.adaptive_avg_pool2d(
                base, (max(1, x.shape[-2] // stride), max(1, x.shape[-1] // stride))
            )
        return levels


def _small_model(num_classes=SMALL_CLASSES, num_proposals=SMALL_PROPOSALS,
                 num_stages=SMALL_STAGES, min_size=32, max_size=64):
    return MODULE._SparseRCNN(
        _StubBackbone(),
        num_classes,
        GeneralizedRCNNTransform(
            min_size=min_size,
            max_size=max_size,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        ),
        MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=3, sampling_ratio=2),
        num_proposals=num_proposals,
        num_stages=num_stages,
        d_model=SMALL_CHANNELS,
        num_heads=4,
        dim_feedforward=32,
        dim_dynamic=4,
        roi_output_size=3,
    )


def _images_and_targets():
    images = [torch.rand(3, 32, 48), torch.rand(3, 40, 32)]
    targets = [
        {
            "boxes": torch.tensor([[2.0, 2.0, 16.0, 16.0], [18.0, 18.0, 30.0, 28.0]]),
            "labels": torch.tensor([1, 2]),
        },
        {"boxes": torch.tensor([[1.0, 1.0, 12.0, 20.0]]), "labels": torch.tensor([3])},
    ]
    return images, targets


# --- the matcher, against an independent oracle ----------------------------


def _brute_force_minimum(cost):
    """Optimal total cost by enumerating every matching. Independent of the
    algorithm under test — a different method with the same answer, which is
    what makes it an oracle rather than a second copy."""
    rows, cols = cost.shape
    if rows <= cols:
        return min(
            sum(float(cost[i, permutation[i]]) for i in range(rows))
            for permutation in itertools.permutations(range(cols), rows)
        )
    return min(
        sum(float(cost[permutation[j], j]) for j in range(cols))
        for permutation in itertools.permutations(range(rows), cols)
    )


@pytest.mark.parametrize("seed", range(12))
def test_the_matcher_is_optimal_on_wide_cost_matrices(seed):
    """MUTATION: greedy nearest-cost pairing, or ``argmin`` per row.

    Greedy agrees with the optimum on most inputs and disagrees on exactly the
    ambiguous ones a detector's matcher meets, so spot-checking cannot catch it.
    Brute force over every permutation can.
    """
    torch.manual_seed(seed)
    rows = int(torch.randint(1, 5, (1,)))
    cols = int(torch.randint(rows, 7, (1,)))
    cost = torch.randn(rows, cols)
    row_index, col_index = MODULE._linear_sum_assignment(cost)

    assert sorted(row_index.tolist()) == list(range(rows)), (
        f"every row must be matched exactly once, got {row_index.tolist()}"
    )
    assert len(set(col_index.tolist())) == col_index.numel(), (
        f"a column was matched twice: {col_index.tolist()}"
    )
    total = float(cost[row_index, col_index].sum())
    assert total == pytest.approx(_brute_force_minimum(cost), abs=1e-5), (
        f"total cost {total:.6f} is not the brute-force optimum "
        f"{_brute_force_minimum(cost):.6f} on\n{cost}"
    )


@pytest.mark.parametrize("seed", range(8))
def test_the_matcher_is_optimal_on_tall_cost_matrices(seed):
    """More ground truth than proposals — a configuration the shipped template
    does not build (100 proposals against a handful of objects), which per
    trap 24 is exactly why it is worth a guard: the transpose branch would
    otherwise never run."""
    torch.manual_seed(100 + seed)
    cols = int(torch.randint(1, 5, (1,)))
    rows = int(torch.randint(cols, 7, (1,)))
    cost = torch.randn(rows, cols)
    row_index, col_index = MODULE._linear_sum_assignment(cost)
    assert sorted(col_index.tolist()) == list(range(cols))
    assert len(set(row_index.tolist())) == row_index.numel()
    assert float(cost[row_index, col_index].sum()) == pytest.approx(
        _brute_force_minimum(cost), abs=1e-5
    )


def test_the_matcher_beats_greedy_on_a_case_where_they_differ():
    """The mutation made concrete, so the parametrized tests above cannot be the
    only thing standing between greedy and optimal.

    Greedy takes ``(0, 0)`` at cost 1 and is then forced into ``(1, 1)`` at
    cost 100, for 101. The optimum is ``(0, 1)`` + ``(1, 0)`` at 2 + 3 = 5.
    """
    cost = torch.tensor([[1.0, 2.0], [3.0, 100.0]])
    row_index, col_index = MODULE._linear_sum_assignment(cost)
    total = float(cost[row_index, col_index].sum())
    assert total == pytest.approx(5.0), (
        f"total cost {total} — greedy gives 101.0 and the optimum is 5.0"
    )
    assert col_index[0] == 1 and col_index[1] == 0


def test_the_matcher_handles_an_empty_cost_matrix():
    """An unannotated image is an explicit engine output, so zero ground truth
    reaches the matcher on real data."""
    for shape in ((0, 5), (3, 0), (0, 0)):
        rows, cols = MODULE._linear_sum_assignment(torch.zeros(shape))
        assert rows.numel() == 0 and cols.numel() == 0
        assert rows.dtype == torch.int64


def test_a_non_matrix_cost_is_rejected():
    with pytest.raises(ValueError, match="2-D cost matrix"):
        MODULE._linear_sum_assignment(torch.zeros(3))


# --- what the matcher selects at initialisation ----------------------------


def test_the_matcher_selects_exactly_one_proposal_per_object_at_init():
    """The cold-start question, asked of the real model.

    A cold-start metric can starve an assigner while every loss stays finite —
    TOOD's ``t = s^alpha * u^beta`` measures ~1e-15 at initialisation, so a
    faithful implementation selects nothing and learns nothing. Hungarian
    matching cannot do that, and this pins WHY: the assignment is
    cardinality-forced, not threshold-gated. It returns ``min(num_gt,
    num_proposals)`` pairs whatever the costs are.
    """
    torch.manual_seed(0)
    model = _small_model()
    model.train()
    images, targets = _images_and_targets()
    transformed, transformed_targets = model.transform(images, targets)
    features = model.backbone(transformed.tensors)
    stage_logits, stage_boxes = model._run_stages(
        features,
        transformed.image_sizes,
        transformed.tensors.dtype,
        transformed.tensors.device,
    )
    matches = model._match(
        stage_logits[0], stage_boxes[0], transformed_targets, transformed.image_sizes
    )
    assert len(matches) == 2
    for image_index, (gt_index, proposal_index) in enumerate(matches):
        expected = int(transformed_targets[image_index]["boxes"].shape[0])
        assert gt_index.numel() == expected, (
            f"image {image_index} has {expected} objects and the matcher "
            f"selected {gt_index.numel()} — on a freshly built model the "
            f"positive set must never be empty"
        )
        assert sorted(gt_index.tolist()) == list(range(expected))
        assert len(set(proposal_index.tolist())) == expected, (
            f"image {image_index}: the matcher gave two objects the same "
            f"proposal ({proposal_index.tolist()}) — one-to-one is the whole "
            f"point of set prediction"
        )


def test_an_unannotated_image_matches_nothing_and_still_produces_finite_losses():
    """MUTATION: assume at least one object per image.

    The engine emits an explicit zero-object target for an unannotated image, so
    this is a real input, and a division by a zero object count would produce
    NaN losses that the handler sums without complaint.
    """
    torch.manual_seed(0)
    model = _small_model()
    model.train()
    images = [torch.rand(3, 32, 48), torch.rand(3, 40, 32)]
    targets = [
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)},
    ]
    losses = model(images, targets)
    assert losses, "an all-unannotated batch produced no losses at all"
    for name, value in losses.items():
        assert torch.isfinite(value).all(), f"{name} is not finite on a zero-object batch"


# --- the learned proposals -------------------------------------------------


def test_the_proposals_are_trainable_parameters_reachable_from_parameters():
    """MUTATION: register the proposals as buffers, or ``.detach()`` them before
    use.

    Either gives a detector with 100 fixed full-image boxes that can never
    specialise, and it trains and evaluates without complaint.
    """
    model = _small_model()
    for name in ("init_proposal_boxes", "init_proposal_features"):
        parameter = getattr(model, name)
        assert isinstance(parameter, nn.Parameter), (
            f"{name} is a {type(parameter).__name__}, not an nn.Parameter"
        )
        assert parameter.requires_grad, f"{name} is frozen"
        assert any(p is parameter for p in model.parameters()), (
            f"{name} is not reachable from model.parameters() — an optimizer "
            f"built from parameters() would never update it"
        )
    assert model.init_proposal_boxes.shape == (SMALL_PROPOSALS, 4)
    assert model.init_proposal_features.shape == (SMALL_PROPOSALS, SMALL_CHANNELS)


def test_the_proposal_boxes_start_as_the_whole_image():
    """The reference initialisation: normalized cxcywh ``(0.5, 0.5, 1, 1)``, so
    every proposal covers the whole image and the asymmetry that lets them
    differentiate comes from the FEATURES, which are random."""
    model = _small_model()
    boxes = model.init_proposal_boxes.detach()
    assert torch.allclose(boxes[:, :2], torch.full_like(boxes[:, :2], 0.5))
    assert torch.allclose(boxes[:, 2:], torch.ones_like(boxes[:, 2:]))
    features = model.init_proposal_features.detach()
    assert float((features - features[0]).abs().max()) > 0.0, (
        "the proposal features are identical across proposals, so all 100 "
        "proposals would receive the same gradient forever"
    )


def test_the_proposal_boxes_move_under_training_and_differentiate():
    """MUTATION: use a detached copy of the proposals, or drop the box loss.

    Both halves matter. That the parameter moves rules out "not in the graph";
    that different rows move by different amounts rules out a shared-gradient
    path that would leave all 100 proposals identical forever, which is what
    ``init_proposal_boxes`` starting out identical makes possible.
    """
    torch.manual_seed(0)
    model = _small_model()
    model.train()
    before = model.init_proposal_boxes.detach().clone()
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.5
    )
    images, targets = _images_and_targets()

    for _ in range(3):
        optimizer.zero_grad()
        losses = model(images, targets)
        sum(losses.values()).backward()
        assert model.init_proposal_boxes.grad is not None, (
            "init_proposal_boxes received no gradient — the proposals are not "
            "in the graph"
        )
        assert float(model.init_proposal_boxes.grad.abs().max()) > 0.0
        optimizer.step()

    after = model.init_proposal_boxes.detach()
    delta = (after - before).abs()
    assert float(delta.max()) > 0.0, "the learned proposal boxes did not move"
    assert float((after - after[0]).abs().max()) > 0.0, (
        "the proposal boxes are still identical to each other after training — "
        "they can never specialise into a spatial prior"
    )


def test_the_proposal_features_move_under_training():
    torch.manual_seed(0)
    model = _small_model()
    model.train()
    before = model.init_proposal_features.detach().clone()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    images, targets = _images_and_targets()
    optimizer.zero_grad()
    sum(model(images, targets).values()).backward()
    assert model.init_proposal_features.grad is not None
    optimizer.step()
    assert float((model.init_proposal_features.detach() - before).abs().max()) > 0.0


# --- dynamic instance interaction -----------------------------------------


def test_the_dynamic_convolution_generates_its_parameters_from_the_proposal():
    """Structural half: the generator's output width must be exactly the two
    ``d_model x dim_dynamic`` blocks the interaction applies. A generator sized
    for one block, or for a bias only, is a different mechanism."""
    conv = MODULE._DynamicConv(d_model=16, dim_dynamic=4, roi_output_size=3)
    assert conv.dynamic_layer.in_features == 16
    assert conv.dynamic_layer.out_features == 2 * 16 * 4, (
        f"the dynamic generator emits {conv.dynamic_layer.out_features} numbers; "
        f"two 16x4 parameter blocks need {2 * 16 * 4}"
    )


def test_the_roi_filter_depends_on_the_proposal_feature():
    """MUTATION: concatenate the proposal feature instead of generating the
    filter from it, or ignore it entirely.

    Behavioural, in both directions: the SAME ROI with DIFFERENT proposal
    features must give different outputs (otherwise the proposal feature does
    nothing), and the same ROI with the same proposal feature must give the
    same output (otherwise the test is measuring noise).
    """
    torch.manual_seed(0)
    conv = MODULE._DynamicConv(d_model=16, dim_dynamic=4, roi_output_size=3)
    conv.eval()
    roi = torch.rand(1, 9, 16)
    first = torch.rand(1, 16)
    second = torch.rand(1, 16)
    with torch.no_grad():
        a = conv(first, roi)
        a_again = conv(first.clone(), roi.clone())
        b = conv(second, roi)
    assert torch.allclose(a, a_again, atol=1e-6), (
        "the same proposal feature and ROI gave two different outputs — this "
        "test cannot distinguish anything"
    )
    assert not torch.allclose(a, b, atol=1e-6), (
        "changing the proposal feature left the filtered ROI unchanged — the "
        "interaction is not dynamic"
    )


def test_the_roi_filter_depends_on_the_roi_as_well():
    """The mirror: a "dynamic" filter that ignored its ROI would pass the test
    above while being a function of the proposal feature alone."""
    torch.manual_seed(0)
    conv = MODULE._DynamicConv(d_model=16, dim_dynamic=4, roi_output_size=3)
    conv.eval()
    proposal = torch.rand(1, 16)
    with torch.no_grad():
        a = conv(proposal, torch.rand(1, 9, 16))
        b = conv(proposal, torch.rand(1, 9, 16))
    assert not torch.allclose(a, b, atol=1e-6), (
        "two different ROIs gave the same output — the interaction ignores the "
        "pooled features"
    )


# --- six independent stages -----------------------------------------------


def test_the_stages_are_independent_parameter_subtrees():
    """MUTATION: build one stage and run it six times, or ``[stage] * 6``.

    A recurrence has the same forward shape, the same loss keys and a sixth of
    the capacity.
    """
    model = _small_model(num_stages=4)
    stages = model.stages
    assert len(stages) == 4
    assert len({id(stage) for stage in stages}) == 4, "the stages are one object"
    tensors = [{id(p) for p in stage.parameters()} for stage in stages]
    for first, second in itertools.combinations(range(4), 2):
        assert not (tensors[first] & tensors[second]), (
            f"stages {first} and {second} share parameter tensors"
        )


def test_the_shipped_template_builds_six_stages_and_a_hundred_proposals():
    """The declared configuration, read off the built model rather than the
    module constants — a constant that is declared and not wired through would
    otherwise pass."""
    model = MODULE.MyModel(3)
    assert len(model.stages) == MODULE.NUM_STAGES == 6
    assert model.num_proposals == MODULE.NUM_PROPOSALS == 100
    assert model.init_proposal_boxes.shape == (100, 4)


def test_deep_supervision_produces_a_loss_at_every_stage():
    """MUTATION: supervise the last stage only.

    Sparse R-CNN converges badly without deep supervision, and a loss dict
    holding only the last stage's three keys is entirely plausible-looking. The
    engine calls ``sum(losses.values())``, so a missing stage errors nowhere.
    """
    torch.manual_seed(0)
    model = _small_model(num_stages=3)
    model.train()
    images, targets = _images_and_targets()
    losses = model(images, targets)
    expected = {
        f"{name}_stage{index}"
        for index in range(3)
        for name in ("classification", "bbox_regression", "giou")
    }
    assert set(losses) == expected, f"loss keys {sorted(losses)} != {sorted(expected)}"
    for value in losses.values():
        assert torch.isfinite(value).all()


def test_boxes_are_detached_between_stages_but_not_from_their_own_loss():
    """MUTATION (either direction): drop the ``.detach()``, so stage k+1's loss
    reaches stage k's regressor; or detach the boxes the loss uses, so the box
    losses have no gradient at all and only the classifier trains.

    Both leave every loss finite and every key present, and they are opposite
    mistakes in the same two lines — which is why both are asserted here.
    """
    torch.manual_seed(0)
    model = _small_model(num_stages=3)
    model.train()
    images, targets = _images_and_targets()
    transformed, _ = model.transform(images, targets)
    features = model.backbone(transformed.tensors)
    seen = []
    original = model.box_roi_pool.forward

    def recording(feature_maps, boxes, image_shapes):
        seen.append(boxes)
        return original(feature_maps, boxes, image_shapes)

    model.box_roi_pool.forward = recording
    _, stage_boxes = model._run_stages(
        features,
        transformed.image_sizes,
        transformed.tensors.dtype,
        transformed.tensors.device,
    )
    assert len(seen) == 3, f"expected one pooling call per stage, saw {len(seen)}"
    for stage_index in (1, 2):
        for boxes in seen[stage_index]:
            assert not boxes.requires_grad, (
                f"the boxes stage {stage_index} pools still carry a gradient — "
                f"the refinement was not detached"
            )
    for stage_index, boxes in enumerate(stage_boxes):
        assert boxes.requires_grad, (
            f"stage {stage_index}'s predicted boxes are detached, so its box "
            f"and GIoU losses can never train the regressor"
        )


# --- the decode -----------------------------------------------------------


def test_the_decode_keeps_the_images_apart_at_batch_two():
    """MUTATION: index the per-image logits or boxes on the wrong axis, or let a
    ``zip`` truncate to one image.

    The real bug this rule comes from was right for image 0 and silently wrong
    past it. Here each image's logits favour a DIFFERENT class and each image's
    boxes occupy a disjoint region, so a mix-up shows up as the wrong label or a
    box from the other image.
    """
    model = _small_model()
    logits = torch.full((2, SMALL_PROPOSALS, SMALL_CLASSES), -10.0)
    logits[0, :, 1] = 4.0  # sigmoid ~0.982
    logits[1, :, 3] = 4.0
    boxes = torch.zeros(2, SMALL_PROPOSALS, 4)
    boxes[0] = torch.tensor([2.0, 2.0, 10.0, 10.0])
    boxes[1] = torch.tensor([25.0, 20.0, 35.0, 30.0])

    results = model._detections(logits, boxes, [SMALL_IMAGE_SHAPE] * 2)
    assert len(results) == 2, f"batch of 2 produced {len(results)} results"

    assert set(results[0]["labels"].tolist()) >= {1}
    assert int(results[0]["labels"][0]) == 1, (
        f"image 0's top detection is class {int(results[0]['labels'][0])}, "
        f"expected the stubbed class 1"
    )
    assert int(results[1]["labels"][0]) == 3, (
        f"image 1's top detection is class {int(results[1]['labels'][0])}, "
        f"expected the stubbed class 3 — the decode read image 0's row"
    )
    assert float(results[0]["scores"].max()) == pytest.approx(
        float(torch.sigmoid(torch.tensor(4.0))), abs=1e-5
    )
    assert float(results[0]["boxes"][0, 0]) == pytest.approx(2.0, abs=1e-4)
    assert float(results[1]["boxes"][0, 0]) == pytest.approx(25.0, abs=1e-4)


def test_the_background_column_never_wins_a_detection_slot():
    """MUTATION: score over all ``num_classes`` columns including index 0.

    Column 0 is the dataset's background id, which set prediction never targets
    — "no object" is expressed by a proposal going unmatched — so a background
    detection is meaningless and would displace a real one from the top-k. Here
    background is given the HIGHEST logit, so including it is unmissable.
    """
    model = _small_model()
    logits = torch.full((1, SMALL_PROPOSALS, SMALL_CLASSES), -10.0)
    logits[0, :, 0] = 9.0
    logits[0, :, 2] = 1.0
    boxes = torch.zeros(1, SMALL_PROPOSALS, 4)
    boxes[0] = torch.tensor([2.0, 2.0, 10.0, 10.0])
    results = model._detections(logits, boxes, [SMALL_IMAGE_SHAPE])
    labels = set(results[0]["labels"].tolist())
    assert 0 not in labels, f"the background class was returned as a detection: {labels}"
    assert labels == {1, 2, 3}, f"expected only foreground classes, got {labels}"
    assert int(results[0]["labels"][0]) == 2


def test_the_decode_applies_no_nms():
    """MUTATION: add NMS back "to clean up the duplicates".

    One-to-one training is what removes duplicates in this architecture; NMS
    would suppress genuinely overlapping objects that set prediction can keep
    apart. Two IDENTICAL boxes with identical high scores must both survive.
    """
    model = _small_model(num_proposals=2)
    logits = torch.full((1, 2, SMALL_CLASSES), -10.0)
    logits[0, :, 2] = 5.0
    boxes = torch.tensor([[[3.0, 3.0, 20.0, 20.0], [3.0, 3.0, 20.0, 20.0]]])
    results = model._detections(logits, boxes, [SMALL_IMAGE_SHAPE])
    top = results[0]["labels"] == 2
    assert int(top.sum()) == 2, (
        f"two identical class-2 predictions produced {int(top.sum())} class-2 "
        f"detections — something is suppressing duplicates"
    )


def test_eval_boxes_come_back_in_original_image_coordinates():
    """MUTATION: skip ``transform.postprocess``.

    The transform resizes the input, so the model's boxes are in resized
    coordinates and the metrics compare them against ground truth in ORIGINAL
    coordinates. Nothing raises: the boxes are simply scaled wrong, which reads
    as poor accuracy. A tall input forces a non-unit scale factor, so the
    identity mapping cannot pass.
    """
    torch.manual_seed(0)
    model = _small_model(min_size=32, max_size=64)
    model.eval()
    image = torch.rand(3, 96, 48)
    with torch.no_grad():
        results = model([image])
    boxes = results[0]["boxes"]
    assert boxes.numel(), "no detections to check"
    # The proposals start as the whole (resized) image, and postprocess maps
    # them back — so at least one box must reach beyond the resized extent.
    assert float(boxes[:, 3].max()) > 64.0, (
        f"the largest box reaches y={float(boxes[:, 3].max()):.1f} on a "
        f"96-pixel-tall image resized to at most 64 — the boxes were not mapped "
        f"back to the original coordinates"
    )
    assert float(boxes[:, 3].max()) <= 96.0 + 1e-3


# --- contract-level checks -------------------------------------------------


#: The resolution Sparse R-CNN trains at in the paper, and torchvision's default
#: for the two-stage family. A LITERAL transcribed from outside this repository.
PUBLISHED_TRAIN_RESOLUTION = 800


def test_the_declared_image_size_is_the_published_resolution():
    """MUTATION: declare 448.

    ⚠️ THIS TEST EXISTS BECAUSE THE OBVIOUS ONE IS CIRCULAR, and the circularity
    has already bitten twice in this roster. ``MyModel`` passes
    ``min_size=image_size``, so "declared equals the built model's transform" —
    what ``tests/test_od_declared_resolution.py`` checks family-wide — moves
    BOTH sides when the declaration is mutated and can never fail here. On the
    sibling ``efficientdet_d0`` branch exactly that left a wrong 448 on disk
    with twenty tests green.

    Trap 31 in miniature, inside a test: a number compared against itself.
    """
    assert int(MODULE.image_size) == PUBLISHED_TRAIN_RESOLUTION, (
        f"declared image_size={MODULE.image_size}; Sparse R-CNN trains at "
        f"{PUBLISHED_TRAIN_RESOLUTION}"
    )


def test_the_declared_image_size_is_wired_into_the_transform():
    """The #3058 half, and NOT redundant with the test above — each is
    individually mutable-to-red, which is the trap-29 test. Measured off the
    BUILT model, never asserted from the source."""
    model = MODULE.MyModel(3)
    min_size = model.transform.min_size
    effective = int(min_size[0] if isinstance(min_size, (list, tuple)) else min_size)
    assert effective == int(MODULE.image_size), (
        f"declared image_size={MODULE.image_size} but the transform runs at "
        f"{effective}"
    )


# --- external oracles: the numbers, derived from outside this file ---------
#
# TRAP 31 (SELF-CONSISTENT-NUMBER). `yolox_s` shipped an exact parameter count
# as proof its architecture was real; it was exact for the model as built, while
# a wrong expand ratio left the backbone ~1.15M parameters narrow. A count
# measured off your own model proves only internal consistency.
#
# ⚠️ HONEST LIMIT, STATED RATHER THAN PAPERED OVER. Unlike EfficientNet-B0
# (where torchvision ships an independent implementation of the same published
# table) there is no library implementation of Sparse R-CNN to compare against,
# and I could not find an authoritative published INTEGER for its R50 parameter
# count — the figure usually quoted is "~106M", which is a magnitude, not a
# check. So the total is NOT asserted against a published number. What is
# asserted instead:
#
#   * the ResNet-50-FPN trunk against torchvision's own Faster R-CNN backbone,
#     which is DIFFERENTLY CONSTRUCTED (its builder, not these two calls);
#   * every stage against hand arithmetic from the published Sparse R-CNN R50
#     configuration — d_model 256, dim_dynamic 64, 2 dynamic blocks, FFN 2048,
#     1 classification layer, 3 regression layers, 7x7 RoI, 100 proposals,
#     6 stages;
#   * an exact decomposition of the total into those two plus the learned
#     proposals, with no fudge term.
#
# That covers ~75% of the parameters with a genuinely external expectation and
# names the remainder's source. A stated absence of an oracle beats a circular
# number.

#: Published Sparse R-CNN R50 configuration. Transcribed, not read from the
#: template — that is the whole point.
PUBLISHED_D_MODEL = 256
PUBLISHED_DIM_DYNAMIC = 64
PUBLISHED_NUM_DYNAMIC = 2
PUBLISHED_FEEDFORWARD = 2048
PUBLISHED_CLS_LAYERS = 1
PUBLISHED_REG_LAYERS = 3
PUBLISHED_ROI_EDGE = 7
PUBLISHED_PROPOSALS = 100
PUBLISHED_STAGES = 6
#: ResNet-50 normalises this many channels. Referenced only to SIZE the
#: regression the backbone comparison below would see if the backbone went back
#: to FrozenBatchNorm2d, which holds the two-per-channel affine values as
#: buffers instead of parameters (backend#3093).
RESNET50_NORMALISED_CHANNELS = 26560


def _layer_norm_parameters(width):
    return 2 * width


def _derived_stage_parameters(num_classes):
    """Parameters in ONE Sparse R-CNN stage, from the published R50 config."""
    d = PUBLISHED_D_MODEL
    positions = PUBLISHED_ROI_EDGE * PUBLISHED_ROI_EDGE

    # nn.MultiheadAttention: a packed 3d x d in-projection with bias, plus a
    # d x d output projection with bias.
    attention = 3 * d * d + 3 * d + (d * d + d)
    # Dynamic instance interaction: the generator emits two d x dim_dynamic
    # parameter blocks per proposal, then a Linear from the flattened ROI.
    dynamic = (
        (d * (PUBLISHED_NUM_DYNAMIC * d * PUBLISHED_DIM_DYNAMIC)
         + PUBLISHED_NUM_DYNAMIC * d * PUBLISHED_DIM_DYNAMIC)
        + _layer_norm_parameters(PUBLISHED_DIM_DYNAMIC)
        + _layer_norm_parameters(d)
        + (d * positions * d + d)
        + _layer_norm_parameters(d)
    )
    feed_forward = (d * PUBLISHED_FEEDFORWARD + PUBLISHED_FEEDFORWARD) + (
        PUBLISHED_FEEDFORWARD * d + d
    )
    # Each MLP block is Linear(bias=False) + LayerNorm + ReLU.
    def mlp(depth):
        return depth * (d * d + _layer_norm_parameters(d))

    return (
        attention
        + _layer_norm_parameters(d)          # attention_norm
        + dynamic
        + _layer_norm_parameters(d)          # interaction_norm
        + feed_forward
        + _layer_norm_parameters(d)          # feed_forward_norm
        + mlp(PUBLISHED_CLS_LAYERS)
        + mlp(PUBLISHED_REG_LAYERS)
        + (d * num_classes + num_classes)    # class_logits
        + (d * 4 + 4)                        # boxes_delta
    )


def test_the_backbone_matches_torchvisions_faster_rcnn_backbone():
    """MUTATION: a wrong backbone width, depth, or ``returned_layers``.

    THE yolox_s CLASS OF BUG. Compared against torchvision's own Faster R-CNN
    backbone, reached through its BUILDER rather than through the same two calls
    this template makes — so the comparison is not just "did I pass the same
    arguments twice".

    The two backbones' parameter counts must now be EQUAL, with no correction
    term. They used to differ by exactly ``2 x 26,560``: torchvision's
    untrained builder uses live ``BatchNorm2d`` where this template used
    ``FrozenBatchNorm2d``, which holds weight and bias as buffers. backend#3093
    replaced frozen BN with GroupNorm, whose weight and bias ARE parameters of
    the same shapes, so the difference is zero and the comparison is exact.
    Both halves are still asserted below in their post-fix form -- equal
    parameters, and no buffers at all -- so a re-introduced norm mismatch fails
    here rather than being absorbed.
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    model = MODULE.MyModel(3)
    oracle = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=4)

    def conv_shapes(module):
        return sorted(tuple(p.shape) for p in module.parameters() if p.dim() == 4)

    assert conv_shapes(model.backbone) == conv_shapes(oracle.backbone), (
        "the backbone's convolution shapes differ from torchvision's "
        "ResNet-50-FPN — a width, a depth or returned_layers is wrong. First "
        "divergence: "
        f"{next((a, b) for a, b in zip(conv_shapes(model.backbone), conv_shapes(oracle.backbone)) if a != b)}"
    )

    mine = sum(p.numel() for p in model.backbone.parameters())
    theirs = sum(p.numel() for p in oracle.backbone.parameters())
    assert mine == theirs, (
        f"the backbone has {mine} parameters against torchvision's {theirs}; "
        f"GroupNorm holds weight/bias as parameters of the same shapes as "
        f"BatchNorm2d, so these must be equal. A shortfall of exactly "
        f"{2 * RESNET50_NORMALISED_CHANNELS} is FrozenBatchNorm2d holding them "
        f"as buffers instead (backend#3093)"
    )
    assert sum(b.numel() for b in model.backbone.buffers()) == 0, (
        "a GroupNorm backbone holds no buffers: no running statistics for the "
        "averaging service to ship each federated round, and none of "
        "FrozenBatchNorm2d's four-per-channel either"
    )
    # P2 is load-bearing for this architecture: the proposals start as the whole
    # image, so the finest level is what a specialised proposal eventually pools
    # from. returned_layers=[2,3,4] would drop it and still train.
    assert model.box_roi_pool.featmap_names == ["0", "1", "2", "3"], (
        f"pooling from {model.box_roi_pool.featmap_names}; Sparse R-CNN pools "
        f"from P2..P5"
    )


def test_each_stage_matches_the_published_configuration():
    """MUTATION: a wrong ``dim_dynamic``, FFN width, or MLP depth.

    Every one is shape-consistent end to end and invisible to the identity and
    disjointness checks above. The expectation comes from the published R50
    configuration, with nothing read from the template.
    """
    for num_classes in (4, 11):
        expected = _derived_stage_parameters(num_classes)
        model = MODULE.MyModel(num_classes - 1)
        for index, stage in enumerate(model.stages):
            measured = sum(p.numel() for p in stage.parameters())
            assert measured == expected, (
                f"at num_classes={num_classes}, stage {index} has {measured} "
                f"parameters; the published R50 configuration gives {expected}"
            )
        del model


def test_the_whole_model_reconciles_as_trunk_plus_stages_plus_proposals():
    """The end-to-end decomposition, exact and with no fudge term.

    ``total = trunk + 6 x derived stage + (100 x 4 boxes + 100 x 256 features)``

    Deliberately NOT asserted against the "~106M" figure usually quoted for
    Sparse R-CNN R50: that is a magnitude, not a check, and asserting agreement
    with it would be exactly the reassuring-looking non-evidence trap 31
    describes. See the honest-limit note above.
    """
    num_classes = 4
    model = MODULE.MyModel(num_classes - 1)
    trunk = sum(p.numel() for p in model.backbone.parameters())
    stages = PUBLISHED_STAGES * _derived_stage_parameters(num_classes)
    proposals = PUBLISHED_PROPOSALS * 4 + PUBLISHED_PROPOSALS * PUBLISHED_D_MODEL

    measured = sum(p.numel() for p in model.parameters())
    assert measured == trunk + stages + proposals, (
        f"the model has {measured} parameters; trunk {trunk} + {PUBLISHED_STAGES} "
        f"stages {stages} + learned proposals {proposals} gives "
        f"{trunk + stages + proposals}"
    )
    assert proposals == 26000, "100 boxes of 4 plus 100 features of 256"


def test_the_backbone_stem_and_first_stage_are_frozen():
    """MUTATION: ``trainable_layers=5`` (or 0).

    The one claim the torchvision oracle cannot check — its builder warns that
    ``trainable_backbone_layers`` has no effect without pretrained weights and
    falls back to 5, and the setting changes ``requires_grad`` only, never a
    count. Checked directly against torchvision's documented semantics:
    ``trainable_layers=3`` trains ``layer2/3/4`` and freezes the stem and
    ``layer1``.

    It also matters for the reachability test below, which is
    ``requires_grad``-aware precisely because these are meant to be untouched.
    """
    model = MODULE.MyModel(3)
    frozen, trainable = [], []
    for name, parameter in model.backbone.body.named_parameters():
        (trainable if parameter.requires_grad else frozen).append(name)
    assert frozen, "no backbone parameter is frozen — trainable_layers is not 3"
    unexpected_frozen = [n for n in frozen if not n.startswith(("conv1", "bn1", "layer1"))]
    assert not unexpected_frozen, f"unexpectedly frozen: {unexpected_frozen[:5]}"
    unexpected_trainable = [
        n for n in trainable if not n.startswith(("layer2", "layer3", "layer4"))
    ]
    assert not unexpected_trainable, f"unexpectedly trainable: {unexpected_trainable[:5]}"


def test_no_trainable_parameter_is_unreachable_by_the_loss():
    """MUTATION: build a stage, or a head, and never call it.

    ``requires_grad``-aware on purpose: ``p.grad is None`` alone false-flags the
    deliberately frozen ResNet stem and ``layer1`` -- whose GroupNorm affine
    values are parameters since backend#3093, not buffers. The
    defect is a TRAINABLE parameter the loss never reaches.
    """
    torch.manual_seed(0)
    model = _small_model(num_stages=3)
    model.train()
    images, targets = _images_and_targets()
    sum(model(images, targets).values()).backward()
    unreachable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not unreachable, (
        f"{len(unreachable)} trainable parameters received no gradient: "
        f"{unreachable[:8]}"
    )


def test_the_finest_pyramid_level_is_reached_once_a_proposal_is_small_enough():
    """The known cold start, pinned rather than left as folklore.

    At initialisation every proposal is the whole image, so ``MultiScaleRoIAlign``
    routes them all to the coarsest level and the finest level's FPN
    convolutions receive a gradient of exactly zero on step one. That is a
    property of the initialisation, not a dead parameter — and the way to show
    it is to make the proposals small and require the gradient to appear. If
    this ever fails, the finest level is genuinely unreachable and should be
    dropped from ``featmap_names`` rather than trained on nothing.
    """
    torch.manual_seed(0)
    model = _small_model(num_stages=1)
    model.train()
    with torch.no_grad():
        # A small, off-centre proposal: 1/16 of each edge.
        model.init_proposal_boxes[:, :2] = 0.25
        model.init_proposal_boxes[:, 2:] = 0.0625
    images, targets = _images_and_targets()
    sum(model(images, targets).values()).backward()
    finest = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.stem")
    ]
    assert finest, "the stub backbone exposes no convolution to check"
    for name, parameter in finest:
        assert parameter.grad is not None and float(parameter.grad.abs().max()) > 0.0, (
            f"{name} received no gradient even with small proposals"
        )


# --- the matching COST, not just the matching ------------------------------
#
# ⚠️ THESE TWO TESTS WERE ADDED BECAUSE A MUTATION SURVIVED. Computing the L1
# matching cost on raw pixels instead of normalized boxes passed the whole file
# (67 green). Everything here checked the matcher's CARDINALITY — exactly one
# proposal per object, all distinct — and cardinality is invariant to any
# reweighting of the cost, so nothing was watching WHICH proposal got picked.
#
# That is the trap-24 shape in a different place: the fixtures could not
# exercise the rule. A one-to-one matcher always returns min(num_gt,
# num_proposals) pairs; asking only "how many?" can never see a cost bug.


def _matching_fixture(image_edge):
    """One object, two proposals, chosen so the L1 term decides the winner.

    Proposal 0 has the BETTER class score and slightly WORSE geometry; proposal
    1 has perfect geometry and a poor class score. With the L1 term normalized
    the class advantage wins; with it computed on raw pixels the geometry term
    is inflated by the image edge and swamps everything, so proposal 1 wins.

    Returns ``(logits, boxes, targets, image_shape)`` at the requested edge, so
    the same relative geometry can be posed at two scales.
    """
    scale = image_edge / 100.0
    ground_truth = torch.tensor([[10.0, 10.0, 50.0, 50.0]]) * scale
    proposals = torch.stack([
        torch.tensor([12.0, 12.0, 52.0, 52.0]) * scale,   # offset, better class
        torch.tensor([10.0, 10.0, 50.0, 50.0]) * scale,   # exact, worse class
    ])
    logits = torch.full((1, 2, SMALL_CLASSES), -6.0)
    logits[0, 0, 1] = 2.0
    logits[0, 1, 1] = -2.0
    targets = [{"boxes": ground_truth, "labels": torch.tensor([1])}]
    return logits, proposals.unsqueeze(0), targets, (int(image_edge), int(image_edge))


def test_the_l1_matching_cost_is_scale_normalized():
    """MUTATION: ``torch.cdist(predicted, gt_boxes)`` on raw pixels.

    THE SURVIVOR. Normalizing is what makes the L1 term comparable to the class
    and GIoU terms, which are both already scale-free — on an 800px image a raw
    L1 of 8 pixels becomes a cost of 40 against a class cost of order 1, so the
    matcher stops being able to trade geometry against classification at all and
    just picks the closest box.

    Asserted on ``_match``'s real return value, at two image scales: the choice
    must be the same at both, and must be the proposal the class score favours.
    """
    model = _small_model()
    for edge in (100.0, 800.0):
        logits, boxes, targets, image_shape = _matching_fixture(edge)
        (gt_index, proposal_index), = model._match(logits, boxes, targets, [image_shape])
        assert gt_index.tolist() == [0]
        assert proposal_index.tolist() == [0], (
            f"at a {int(edge)}px edge the matcher chose proposal "
            f"{proposal_index.tolist()}, expected [0] — proposal 0 has the "
            f"better class score and only a 2%-of-edge geometry offset, so a "
            f"scale-normalized cost must prefer it. Choosing 1 means the L1 "
            f"term is in raw pixels and has swamped the class cost"
        )


def test_the_matching_cost_reacts_when_the_class_advantage_is_removed():
    """The positive control the test above needs.

    Without it, ``proposal_index == [0]`` could be satisfied by a matcher that
    always returns the first proposal — or by one that ignores the class cost
    entirely. Levelling the two class scores leaves geometry as the only
    signal, and then the exact-geometry proposal 1 must win.
    """
    model = _small_model()
    logits, boxes, targets, image_shape = _matching_fixture(100.0)
    # Same class score for both proposals: geometry is now the only difference.
    logits[0, 0, 1] = -2.0
    (_, proposal_index), = model._match(logits, boxes, targets, [image_shape])
    assert proposal_index.tolist() == [1], (
        f"with the class scores levelled the matcher chose "
        f"{proposal_index.tolist()}, expected [1] — proposal 1 matches the "
        f"ground truth exactly, so a matcher that reads geometry at all must "
        f"pick it. Still choosing 0 means the geometry terms are inert"
    )


def test_the_giou_term_decides_the_match_where_l1_cannot():
    """MUTATION: drop the GIoU term from the matching cost.

    ⚠️ ALSO A SURVIVOR FIRST TIME ROUND, and worth explaining because the reason
    is instructive. In the matching cost, GIoU is *largely* redundant with the
    L1 term: for xyxy boxes both grow with displacement, so on most fixtures
    removing GIoU changes the cost without changing the ranking. My first
    attempt at a cost test was one of those, and dropping GIoU passed 69 green.

    Where the two genuinely diverge is DISJOINT boxes. L1-on-xyxy measures
    corner displacement; GIoU measures how much empty space the enclosing box
    has to add, which depends on the proposal's extent and not just its corner.
    So a large box ADJACENT to the object and a tiny box FAR from it can sit at
    almost the same L1 while being very differently close in the GIoU sense.

    The fixture is exactly that, and the numbers are chosen so the term is the
    decider with margin on both sides — not a tie, which the assignment would
    break arbitrarily:

        ground truth  [40, 40, 60, 60]
        proposal 0    [ 0,  0, 40, 40]   L1 1.200,  GIoU -0.444,  logit 0.0
        proposal 1    [16, 16, 19, 19]   L1 1.300,  GIoU -0.789,  logit 1.0

    Proposal 1 has the better class score. With GIoU in the cost, proposal 0
    wins by 0.320; with it removed, proposal 1 wins by 0.368. Deterministic
    arithmetic, so those margins are enormous next to float error.

    ⚠️ THIS FIXTURE WAS RETUNED, and the reason is a third failure mode worth
    naming. Its earlier proposal 1 was [10, 20, 18, 28], and the margins held
    only because the focal matching cost had its alpha SWAPPED -- FOCAL_ALPHA
    on the negative term instead of the positive. That made the class term
    1.77x weaker than the loss it is supposed to mirror (measured: class delta
    0.245 swapped vs 0.434 correct), which is what let a 0.324 GIoU gap
    outweigh it. Correcting the alpha (model-zoo#246, Bugbot) flipped this test
    to proposal 1.
    
    So the assertion was right and the MARGIN depended on the bug: not a
    degenerate fixture in the usual sense -- it could fail -- but one calibrated
    against wrong behaviour, so fixing the behaviour broke it. When a
    weight-correction breaks a passing test, check whether the test was tuned
    to the old weight before concluding the correction is wrong.
    """
    model = _small_model()
    ground_truth = torch.tensor([[40.0, 40.0, 60.0, 60.0]])
    proposals = torch.tensor([[[0.0, 0.0, 40.0, 40.0], [16.0, 16.0, 19.0, 19.0]]])
    logits = torch.full((1, 2, SMALL_CLASSES), -6.0)
    logits[0, 0, 1] = 0.0
    logits[0, 1, 1] = 1.0
    targets = [{"boxes": ground_truth, "labels": torch.tensor([1])}]

    (gt_index, proposal_index), = model._match(logits, proposals, targets, [(100, 100)])
    assert gt_index.tolist() == [0]
    assert proposal_index.tolist() == [0], (
        f"the matcher chose proposal {proposal_index.tolist()}, expected [0]. "
        f"Proposal 1 has the better class score and a near-identical L1, so "
        f"only the GIoU term can prefer proposal 0 — choosing 1 means GIoU is "
        f"not in the cost"
    )


def test_the_giou_loss_term_responds_to_overlap():
    """MUTATION: return a constant (or the L1 value) for the ``giou`` loss key.

    The loss-side companion to the test above, and here GIoU is NOT redundant:
    it is scale-invariant where L1 is not, which is what makes box regression
    behave the same on a large and a small object. A perfectly-matched box must
    give a GIoU loss of zero, and a disjoint one strictly more than 1.
    """
    model = _small_model(num_proposals=1)
    ground_truth = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    targets = [{"boxes": ground_truth, "labels": torch.tensor([1])}]
    logits = torch.zeros(1, 1, SMALL_CLASSES)

    exact = model._stage_loss(
        logits, ground_truth.clone().unsqueeze(0), targets, [(100, 100)], 1
    )
    assert float(exact["giou"]) == pytest.approx(0.0, abs=1e-5), (
        f"a box identical to the ground truth gave a GIoU loss of "
        f"{float(exact['giou'])}, expected 0"
    )

    # Disjoint, and far: 1 - GIoU exceeds 1 once GIoU goes negative.
    disjoint = torch.tensor([[[200.0, 200.0, 240.0, 240.0]]])
    apart = model._stage_loss(logits, disjoint, targets, [(100, 100)], 1)
    assert float(apart["giou"]) > float(exact["giou"]), (
        "a disjoint box gave no more GIoU loss than an exact one — the term is "
        "inert"
    )
    assert float(apart["giou"]) > MODULE.COST_GIOU * 1.0, (
        f"a disjoint box gave a GIoU loss of {float(apart['giou'])}; 1 - GIoU "
        f"exceeds 1 for a negative GIoU, weighted by {MODULE.COST_GIOU}"
    )

    # ⚠️ THE UPPER BOUND IS THE ONE THAT MATTERS, and it was missing — a
    # mutation returning the L1 value in place of the GIoU loss passed
    # everything above (71 green), because L1 is also 0 for an exact box and
    # also large for a far one. BOUNDEDNESS is what separates them: GIoU lies in
    # [-1, 1], so `1 - GIoU` lies in [0, 2] no matter how far away the box is,
    # while L1 grows without limit. That bound is the reason GIoU is usable as a
    # loss at all — an unbounded geometry term lets one hopeless proposal
    # dominate the gradient early in training, which is exactly when every
    # proposal is hopeless.
    very_far = torch.tensor([[[9000.0, 9000.0, 9040.0, 9040.0]]])
    remote = model._stage_loss(logits, very_far, targets, [(100, 100)], 1)
    assert float(remote["giou"]) <= MODULE.COST_GIOU * 2.0 + 1e-5, (
        f"a box 9000 pixels away gave a GIoU loss of {float(remote['giou'])}, "
        f"above the {MODULE.COST_GIOU * 2.0} ceiling that `1 - GIoU` cannot "
        f"exceed — this term is not GIoU, it is something unbounded"
    )
    assert float(remote["giou"]) > float(apart["giou"]), (
        "moving the box further gave no more GIoU loss — the term saturated "
        "before its ceiling"
    )


def test_matcher_cost_weights_match_the_loss():
    """The focal matching cost must weight positives and negatives the way
    ``sigmoid_focal_loss`` does — asserted against torchvision's own formula,
    not against a value copied out of this template.

    ⚠️ WHY THIS EXISTS AND WHY NOTHING ELSE HERE COULD CATCH IT. Every other
    matcher test in this file asserts CARDINALITY — how many proposals are
    selected, or that the selection is optimal for a *given* cost matrix. All
    of that is invariant to any reweighting of the cost, so a swapped alpha
    changes which proposal wins on a close call while every assertion stays
    green. It shipped to review that way: FOCAL_ALPHA on the negative term and
    (1 - FOCAL_ALPHA) on the positive, i.e. 0.75/0.25 instead of 0.25/0.75.

    torchvision computes ``alpha_t = alpha * targets + (1 - alpha) * (1 -
    targets)``, so a POSITIVE is weighted ``alpha`` and a NEGATIVE
    ``1 - alpha``. The oracle below is that formula evaluated directly — an
    independent source rather than a restatement of the module's constants.
    """
    from torchvision.ops import sigmoid_focal_loss

    module = _module()
    alpha, gamma = module.FOCAL_ALPHA, module.FOCAL_GAMMA

    # A single logit, and the two terms the matcher builds from it.
    logit = torch.tensor([[0.7]])
    probability = logit.sigmoid()

    # torchvision's loss for this logit as a positive and as a negative. Ratio
    # of the two isolates alpha_t, since the focal modulation and the log term
    # are identical between the matcher and the loss.
    pos_loss = float(sigmoid_focal_loss(logit, torch.ones_like(logit),
                                        alpha=alpha, gamma=gamma, reduction="sum"))
    neg_loss = float(sigmoid_focal_loss(logit, torch.zeros_like(logit),
                                        alpha=alpha, gamma=gamma, reduction="sum"))

    # The same two quantities, unweighted, so the expected weights can be
    # recovered from torchvision without assuming them.
    pos_unweighted = float((1 - probability).pow(gamma) * -probability.clamp(min=1e-8).log())
    neg_unweighted = float(probability.pow(gamma) * -(1 - probability).clamp(min=1e-8).log())

    expected_pos_weight = pos_loss / pos_unweighted
    expected_neg_weight = neg_loss / neg_unweighted

    assert abs(expected_pos_weight - alpha) < 1e-5, (
        f"the oracle disagrees with the module's own alpha ({alpha}); "
        f"torchvision weights a positive by {expected_pos_weight:.6f}. If this "
        f"fires, the oracle is wrong, not the template."
    )
    assert abs(expected_neg_weight - (1 - alpha)) < 1e-5, (
        f"torchvision weights a negative by {expected_neg_weight:.6f}, not "
        f"{1 - alpha}"
    )

    # Now the template's own cost terms, read off the source it actually runs.
    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    positive_block = re.search(
        r"positive_cost = \(\s*(.+?)\s*\*", source, re.DOTALL)
    negative_block = re.search(
        r"negative_cost = \(\s*(.+?)\s*\*", source, re.DOTALL)
    assert positive_block and negative_block, (
        "could not locate the matcher's positive/negative cost terms — the "
        "shape of this function changed and this guard stopped reaching it"
    )
    positive_weight, negative_weight = positive_block.group(1), negative_block.group(1)

    assert positive_weight.strip() == "FOCAL_ALPHA", (
        f"the POSITIVE matching term is weighted {positive_weight.strip()!r}; "
        f"torchvision weights a positive by alpha, so it must be FOCAL_ALPHA. "
        f"Swapping these makes the assignment minimise a different objective "
        f"than the loss that trains the matched pair."
    )
    assert negative_weight.strip() == "(1 - FOCAL_ALPHA)", (
        f"the NEGATIVE matching term is weighted {negative_weight.strip()!r}; "
        f"torchvision weights a negative by (1 - alpha)."
    )


def test_the_dynamic_interaction_builds_exactly_two_blocks():
    """The dynamic conv generates exactly two parameter blocks, asserted on the
    BUILT module rather than on a constant.

    ⚠️ WHY THIS IS NOT COVERED BY THE PARAMETER ORACLE. That oracle derives its
    expectation from ``PUBLISHED_NUM_DYNAMIC``, transcribed in *this* file from
    the paper. So it validates the template against the published spec — which
    is the right thing — but it cannot notice the template *declaring* a
    different number than it builds, because it never reads the template's own
    constant. A `NUM_DYNAMIC = 2` module constant existed and reached nothing:
    `_DynamicConv.__init__` never accepted it and used a literal 2, so setting
    it to 3 changed neither the model nor any test. It has been removed; this
    guard is what keeps the removal honest, by pinning the structural fact to
    the built module instead of to a name.

    Caught in review on model-zoo#246 as the "constant that lies" shape.
    """
    module = _module()
    model = _small_model()

    # Reach the first stage's dynamic interaction on the built model.
    interaction = None
    for candidate in model.modules():
        if type(candidate).__name__ == "_DynamicConv":
            interaction = candidate
            break
    assert interaction is not None, (
        "no _DynamicConv in the built model — this guard stopped reaching the "
        "module it names, so it would pass by checking nothing"
    )

    expected = 2 * interaction.num_params
    assert interaction.dynamic_layer.out_features == expected, (
        f"the dynamic layer generates {interaction.dynamic_layer.out_features} "
        f"parameters; two blocks of num_params={interaction.num_params} is "
        f"{expected}. The pair is down-project then up-project and is "
        f"structural, not a configurable count."
    )

    # And no resurrected constant: a name implying configurability must not
    # come back without actually reaching the module above.
    assert not hasattr(module, "NUM_DYNAMIC"), (
        "the template declares NUM_DYNAMIC again. If it is genuinely threaded "
        "through to _DynamicConv now, assert that here instead of removing "
        "this check — a constant that does not reach the model it describes is "
        "documentation pretending to be configuration."
    )

