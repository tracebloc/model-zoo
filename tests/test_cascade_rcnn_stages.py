"""Structural and decode tests for ``object_detection/pytorch/cascade_rcnn.py``.

What is worth testing here, and what is not
-------------------------------------------
``tests/test_od_torchvision_family_train_step.py`` already proves the template
trains and evaluates, and ``tests/test_model_contract.py`` proves it constructs.
Neither can tell a three-stage cascade from one head run three times, or from
three heads that all match at the same IoU — and those are precisely the
mutations that make the file *not Cascade R-CNN* while leaving every existing
assertion green.

So every test below is chosen because it fails under a specific,
shape-preserving mutation, and each names the mutation it exists for.

Two rules the tests obey, both learned from real bugs in this roster
-------------------------------------------------------------------
**Drive the decode directly, above threshold, at batch >= 2.** A fresh
detector's scores can sit below ``score_thresh``, so an assertion against a
well-formed EMPTY detection list passes no matter what the decode does. A real
``zip``-truncation bug shipped through every guard that way: it iterated a list
of pyramid levels as though it were images, and ``zip`` truncated instead of
raising, so it processed one image and was wrong at batch > 1. The eval tests
here stub the stage outputs to known above-threshold logits, run **two** images
with **different** proposal counts, and check each image's detections came from
its own proposals.

**Assert on the real function's return value, never on a recomputation.** Two
tests in this repo's history recomputed the quantity themselves (their own
``torch.sigmoid``, their own ``no_grad`` block) and asserted on their own copy,
so swapping the real module to a softmax survived. Where an exact number is
needed here it is a **literal**, chosen by picking stub logits whose arithmetic
is exact — 0.5, from stage scores of 0.25 / 0.5 / 0.75.
"""

import importlib.util
import pathlib
import re
from collections import OrderedDict

import pytest

ROOT = pathlib.Path(__file__).parent.parent
TEMPLATE = ROOT / "model_zoo" / "object_detection" / "pytorch" / "cascade_rcnn.py"

pytest.importorskip("torch", reason="pytorch not installed in this CI job")
pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")

import torch  # noqa: E402 — after importorskip, deliberately
from torch import nn  # noqa: E402
from torchvision.ops import MultiScaleRoIAlign  # noqa: E402


def _module():
    spec = importlib.util.spec_from_file_location(
        re.sub(r"\W", "_", f"cascade_{TEMPLATE.stem}"), TEMPLATE
    )
    assert spec and spec.loader, f"{TEMPLATE}: importlib could not build a spec"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _module()

#: Channel width and feature-map size for the standalone ROI-head tests. A
#: single non-square level: ``MultiScaleRoIAlign`` needs no more, and a square
#: map cannot distinguish a transposed index from a correct one.
STUB_CHANNELS = 8
STUB_FEATURE_SHAPE = (5, 9)
#: Post-transform image size the stub features are said to come from. 8x the
#: feature map, i.e. a stride-8 level.
STUB_IMAGE_SHAPE = (40, 72)


class _StubStage(nn.Module):
    """A cascade stage whose class logits and box deltas are dictated.

    Replaces a real ``_CascadeStage`` so the tests drive ``_refine`` and
    ``_postprocess`` — the two pieces of decode logic this template owns — with
    known inputs, rather than hoping a randomly initialised head produces
    something informative. ``logits_row`` is broadcast over proposals;
    ``delta_row`` is the four ``(dx, dy, dw, dh)`` deltas used for EVERY class,
    so a test does not have to also model the per-class column selection unless
    that is what it is testing.
    """

    def __init__(self, num_classes, logits_row, delta_row):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("logits_row", torch.as_tensor(logits_row, dtype=torch.float32))
        self.register_buffer("delta_row", torch.as_tensor(delta_row, dtype=torch.float32))

    def forward(self, pooled_features):
        count = pooled_features.shape[0]
        logits = self.logits_row.unsqueeze(0).expand(count, -1).clone()
        deltas = self.delta_row.repeat(self.num_classes).unsqueeze(0).expand(count, -1).clone()
        return logits, deltas


class _PerImageStubStage(nn.Module):
    """A cascade stage whose class logits differ PER IMAGE.

    ⚠️ TRAP 24, and the batch-two decode test had it. It used ``_StubStage``,
    which broadcasts one logits row over every proposal of every image — so
    image 0's scores and image 1's scores were bit-identical and a mutation
    that sliced image 0's scores for BOTH images passed with 59 green tests.
    The fixture made the rule unreachable.

    ``counts`` is the per-image proposal count, which is what lets the stage
    split the flat ``(sum(counts), C, H, W)`` pooled tensor back into images.
    Deliberately allowed to differ per image: equal counts are their own
    degeneracy, because then an off-by-one-image slice lands on a
    correctly-sized row.
    """

    def __init__(self, num_classes, rows, counts, delta_row=(0.0, 0.0, 0.0, 0.0)):
        super().__init__()
        self.num_classes = num_classes
        self.counts = tuple(counts)
        self.register_buffer("rows", torch.as_tensor(rows, dtype=torch.float32))
        self.register_buffer("delta_row", torch.as_tensor(delta_row, dtype=torch.float32))

    def forward(self, pooled_features):
        total = pooled_features.shape[0]
        assert total == sum(self.counts), (
            f"_PerImageStubStage was built for counts {self.counts} "
            f"(={sum(self.counts)} proposals) but was pooled {total} — the "
            f"fixture and the model disagree about the batch"
        )
        logits = torch.cat(
            [self.rows[index].unsqueeze(0).expand(count, -1)
             for index, count in enumerate(self.counts)]
        ).clone()
        deltas = (
            self.delta_row.repeat(self.num_classes).unsqueeze(0).expand(total, -1).clone()
        )
        return logits, deltas


def _roi_heads(num_classes=4, **kwargs):
    return MODULE._CascadeRoIHeads(
        MultiScaleRoIAlign(featmap_names=["0"], output_size=3, sampling_ratio=2),
        STUB_CHANNELS,
        num_classes,
        representation_size=16,
        roi_output_size=3,
        **kwargs,
    )


def _features(batch):
    return OrderedDict(
        [("0", torch.rand(batch, STUB_CHANNELS, *STUB_FEATURE_SHAPE))]
    )


def _spy_on_pooling(roi_heads):
    """Record the proposals handed to ``box_roi_pool`` at each stage.

    The refinement chain is only observable at this seam: ``forward`` does not
    return the intermediate proposals, and asserting on the returned detections
    conflates refinement with post-processing.

    ⚠️ TRAP 32, and this function had it. It used to record
    ``[b.detach().clone() for b in boxes]`` and hand only that back — so
    ``test_refined_proposals_are_detached_from_the_previous_stage`` asserted
    ``not boxes.requires_grad`` on tensors THE SPY had just detached. It was
    unconditionally true, and it passed against a template with no ``.detach()``
    in it at all. Two lists now: detached clones for the geometry (the values
    have to survive the rest of the forward pass) and the ``requires_grad`` flag
    captured BEFORE detaching, which is the only thing that can answer the
    question.
    """
    seen = []
    requires_grad = []
    original = roi_heads.box_roi_pool.forward

    def recording(features, boxes, image_shapes):
        requires_grad.append([bool(b.requires_grad) for b in boxes])
        seen.append([b.detach().clone() for b in boxes])
        return original(features, boxes, image_shapes)

    roi_heads.box_roi_pool.forward = recording
    return seen, requires_grad


# --- the cascade is really three stages ------------------------------------


def test_three_stages_are_three_independent_parameter_subtrees():
    """MUTATION: build one stage and run it three times (or ``[stage] * 3``).

    A recurrent single head has the same forward-pass shape, the same losses and
    the same detections structure. It is caught only by counting parameters that
    are not the same tensors.
    """
    model = MODULE.MyModel(3)
    names = [n for n, _ in model.roi_heads.named_parameters()]
    per_stage = {
        index: {n for n in names if n.startswith(f"stages.{index}.")} for index in range(3)
    }
    for index, subtree in per_stage.items():
        assert subtree, f"stages.{index} contributes no parameters"
    assert not (per_stage[0] & per_stage[1] & per_stage[2]), "stage subtrees overlap by name"

    # Names being distinct is not enough: three ModuleList entries holding the
    # SAME module object also get three name prefixes. Identity is what settles
    # it — shared parameters would be one tensor under three names.
    stages = model.roi_heads.stages
    assert len({id(stage) for stage in stages}) == 3, "the three stages are the same object"
    tensors = [{id(p) for p in stage.parameters()} for stage in stages]
    assert not (tensors[0] & tensors[1]), "stages 0 and 1 share parameter tensors"
    assert not (tensors[1] & tensors[2]), "stages 1 and 2 share parameter tensors"
    assert not (tensors[0] & tensors[2]), "stages 0 and 2 share parameter tensors"


def test_stage_iou_thresholds_are_distinct_and_increasing():
    """MUTATION: give every stage the same IoU threshold.

    That is what makes a cascade a cascade — three heads at 0.5 is an ensemble
    of three heads at 0.5. Read off the BUILT model's matchers, not the module
    constant, so a constant that is declared and then not wired through fails.
    """
    model = MODULE.MyModel(3)
    matchers = model.roi_heads.proposal_matchers
    assert len(matchers) == 3, f"expected three matchers, got {len(matchers)}"
    thresholds = [m.high_threshold for m in matchers]
    assert thresholds == [0.5, 0.6, 0.7], (
        f"stage IoU thresholds are {thresholds}, expected the paper's "
        f"[0.5, 0.6, 0.7] — read from the built model's matchers"
    )
    assert thresholds[0] < thresholds[1] < thresholds[2]
    # The matcher's low threshold must track the high one: a stage whose
    # background threshold stayed at 0.5 while its foreground threshold rose
    # would silently create a BETWEEN_THRESHOLDS band that fastrcnn_loss
    # treats as background.
    assert [m.low_threshold for m in matchers] == thresholds


def test_stage_box_coder_weights_tighten_with_the_stage():
    """MUTATION: reuse stage 0's ``(10, 10, 5, 5)`` for all three.

    A later stage's residuals are smaller, so identical weights leave stages 2
    and 3 regressing deltas near zero. Nothing raises and the losses look fine.
    """
    coders = MODULE.MyModel(3).roi_heads.box_coders
    weights = [tuple(float(w) for w in c.weights) for c in coders]
    assert weights == [(10.0, 10.0, 5.0, 5.0), (20.0, 20.0, 10.0, 10.0), (30.0, 30.0, 15.0, 15.0)]
    for earlier, later in zip(weights, weights[1:]):
        assert all(a < b for a, b in zip(earlier, later)), (
            f"box-coder weights must tighten with the stage, got {weights}"
        )


def test_a_stage_description_of_inconsistent_length_is_rejected():
    """A configuration the shipped template does not build, per trap 24: the
    three per-stage lists are independent arguments, so nothing but this check
    stops a four-threshold / three-weight cascade from silently running three
    stages and ignoring the fourth threshold."""
    with pytest.raises(ValueError, match="same number of stages"):
        _roi_heads(iou_thresholds=(0.5, 0.6, 0.7, 0.8))


def test_the_cascade_generalises_beyond_three_stages():
    """Also trap 24: the shipped template builds exactly three stages, so a loop
    that had been hard-coded to three would never be caught. Two and four are
    configurations this template does not build, and both must work."""
    for count in (2, 4):
        thresholds = tuple(0.5 + 0.1 * index for index in range(count))
        heads = _roi_heads(
            iou_thresholds=thresholds,
            bbox_reg_weights=tuple(
                (10.0 * (index + 1),) * 2 + (5.0 * (index + 1),) * 2 for index in range(count)
            ),
            stage_loss_weights=tuple(1.0 / 2**index for index in range(count)),
        )
        assert len(heads.stages) == count
        heads.train()
        proposals = [torch.tensor([[1.0, 1.0, 20.0, 20.0], [4.0, 4.0, 30.0, 26.0]])]
        targets = [{"boxes": torch.tensor([[2.0, 2.0, 22.0, 21.0]]), "labels": torch.tensor([1])}]
        _, losses = heads(_features(1), proposals, [STUB_IMAGE_SHAPE], targets)
        assert len(losses) == 2 * count, (
            f"a {count}-stage cascade produced {sorted(losses)} — expected two "
            f"loss keys per stage"
        )
        for value in losses.values():
            assert torch.isfinite(value).all()


# --- refinement actually feeds the next stage ------------------------------


def test_zero_deltas_leave_the_next_stage_proposals_unchanged():
    """Half of the refinement test: with the decode fed zero deltas, the boxes
    the next stage pools must be the SAME boxes.

    On its own this passes if refinement is a pass-through, which is why it is
    paired with the test below. Together they pin that the next stage's
    proposals are a FUNCTION OF the deltas, in both directions — a single
    positive test could be satisfied by any change at all.
    """
    heads = _roi_heads()
    for index in range(3):
        heads.stages[index] = _StubStage(4, [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    heads.eval()
    seen, seen_requires_grad = _spy_on_pooling(heads)
    proposals = [torch.tensor([[2.0, 3.0, 18.0, 21.0], [10.0, 5.0, 34.0, 30.0]])]
    with torch.no_grad():
        heads(_features(1), proposals, [STUB_IMAGE_SHAPE])
    assert len(seen) == 3, f"expected one pooling call per stage, saw {len(seen)}"
    assert torch.allclose(seen[1][0], seen[0][0], atol=1e-4)
    assert torch.allclose(seen[2][0], seen[1][0], atol=1e-4)


def test_a_positive_width_delta_widens_the_next_stage_proposals():
    """MUTATION: drop the refinement and pass the original proposals on.

    The positive control the test above needs. Direction only — asserting the
    exact decoded width would mean reimplementing ``BoxCoder.decode_single``
    here and asserting against my own copy, which is how two tests in this repo
    came to pass while never calling the code they named.
    """
    heads = _roi_heads()
    for index in range(3):
        # dw > 0 only: the box must get wider and no taller.
        heads.stages[index] = _StubStage(4, [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.0])
    heads.eval()
    seen, seen_requires_grad = _spy_on_pooling(heads)
    proposals = [torch.tensor([[2.0, 3.0, 18.0, 21.0], [10.0, 5.0, 24.0, 30.0]])]
    with torch.no_grad():
        heads(_features(1), proposals, [STUB_IMAGE_SHAPE])

    def widths(boxes):
        return boxes[:, 2] - boxes[:, 0]

    def heights(boxes):
        return boxes[:, 3] - boxes[:, 1]

    assert bool((widths(seen[1][0]) > widths(seen[0][0])).all()), (
        "a positive dw did not widen the boxes handed to stage 1 — refinement "
        "is not feeding the next stage"
    )
    assert bool((widths(seen[2][0]) > widths(seen[1][0])).all()), (
        "stage 2 received stage 0's boxes, not stage 1's refinement"
    )
    assert torch.allclose(heights(seen[1][0]), heights(seen[0][0]), atol=1e-3), (
        "a dw-only delta changed the box heights — the delta columns are "
        "being read in the wrong order"
    )


def test_refined_proposals_are_detached_from_the_previous_stage():
    """MUTATION: drop the ``.detach()``.

    Stage k+1's loss would then flow back into stage k's regressor, which trains
    each stage on a distribution it can reach back and change. Finite losses,
    same keys, quietly the wrong objective.
    """
    heads = _roi_heads()
    heads.train()
    seen, seen_requires_grad = _spy_on_pooling(heads)
    proposals = [torch.tensor([[1.0, 1.0, 20.0, 20.0], [4.0, 4.0, 30.0, 26.0]])]
    targets = [{"boxes": torch.tensor([[2.0, 2.0, 22.0, 21.0]]), "labels": torch.tensor([1])}]
    heads(_features(1), proposals, [STUB_IMAGE_SHAPE], targets)
    assert len(seen) == 3

    # Stage 0 is handed the caller's proposals, which never carried a gradient,
    # so it says nothing either way — asserting on it would be the degenerate
    # fixture of trap 24. Stages 1 and 2 are handed REFINED boxes, decoded from
    # a head output that does require a gradient, so False there can only come
    # from the detach.
    assert seen_requires_grad[0] == [False], "the fixture's own proposals should carry no grad"
    for stage_index in (1, 2):
        assert seen_requires_grad[stage_index] == [False], (
            f"the proposals stage {stage_index} pools still carry a gradient "
            f"({seen_requires_grad[stage_index]}) — the refinement was not "
            f"detached, so stage {stage_index}'s loss reaches stage "
            f"{stage_index - 1}'s regressor"
        )


# --- the eval decode, driven above threshold, at batch >= 2 ---------------


def test_inference_averages_the_three_stage_scores_exactly():
    """MUTATION: score from the last stage only (or from the first).

    Stub logits are chosen so the three stages' softmax probabilities for class
    1 are exactly 0.25, 0.5 and 0.75 — ``exp(a) / (exp(a) + 3)`` at
    ``a = 0, ln 3, ln 9`` over four classes. Their mean is **0.5**, a literal,
    and it collides with neither single-stage value.
    """
    heads = _roi_heads(num_classes=4, score_thresh=0.01, nms_thresh=1.0)
    import math

    for index, logit in enumerate((0.0, math.log(3.0), math.log(9.0))):
        heads.stages[index] = _StubStage(4, [0.0, logit, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    heads.eval()
    proposals = [torch.tensor([[2.0, 3.0, 30.0, 33.0]])]
    with torch.no_grad():
        results, _ = heads(_features(1), proposals, [STUB_IMAGE_SHAPE])

    scores = results[0]["scores"]
    labels = results[0]["labels"]
    assert scores.numel(), (
        "the decode returned nothing above a 0.01 threshold on 0.5-probability "
        "stub logits — that is the vacuous-eval path, not a pass"
    )
    best = int(scores.argmax())
    assert int(labels[best]) == 1, f"expected class 1 to win, got {int(labels[best])}"
    assert float(scores[best]) == pytest.approx(0.5, abs=1e-6), (
        f"the winning score is {float(scores[best]):.6f}; the mean of the three "
        f"stages' 0.25/0.5/0.75 is 0.5. Last-stage-only gives 0.75, "
        f"first-stage-only 0.25"
    )


def test_stage_score_averaging_is_order_independent():
    """The other half of the averaging claim: a mean does not care which stage
    produced which score, and any single-stage or weighted read does."""
    import math

    logits = (0.0, math.log(3.0), math.log(9.0))
    outcomes = []
    for order in (logits, tuple(reversed(logits))):
        heads = _roi_heads(num_classes=4, score_thresh=0.01, nms_thresh=1.0)
        for index, logit in enumerate(order):
            heads.stages[index] = _StubStage(4, [0.0, logit, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
        heads.eval()
        with torch.no_grad():
            results, _ = heads(
                _features(1), [torch.tensor([[2.0, 3.0, 30.0, 33.0]])], [STUB_IMAGE_SHAPE]
            )
        outcomes.append(float(results[0]["scores"].max()))
    assert outcomes[0] == pytest.approx(outcomes[1], abs=1e-6), (
        f"reversing the stage order changed the ensembled score "
        f"({outcomes[0]:.6f} vs {outcomes[1]:.6f}) — the scores are not being "
        f"averaged"
    )


def test_eval_decode_keeps_the_images_apart_at_batch_two():
    """MUTATION: index a per-image slice by the wrong offset, or let ``zip``
    truncate to one image.

    The failure this is written for produced a well-formed result for image 0
    and garbage (or image 0's answer again) for image 1, and was invisible at
    batch 1 and invisible on a fresh model whose scores never cleared the
    threshold. All three degeneracies are removed here: two images, DIFFERENT
    proposal counts, DIFFERENT winning classes per image, and stub logits well
    above threshold.

    The per-image class difference is the one that matters most and was missing:
    with both images stubbed to the same class, ``scores[0:count]`` for every
    image passed with 59 tests green.
    """
    heads = _roi_heads(num_classes=4, score_thresh=0.2, nms_thresh=1.0)

    # Disjoint proposal regions, a different NUMBER of them per image, and a
    # different winning class per image.
    first = torch.tensor([[1.0, 1.0, 9.0, 9.0], [2.0, 2.0, 10.0, 10.0], [3.0, 1.0, 11.0, 9.0]])
    second = torch.tensor([[30.0, 25.0, 39.0, 34.0], [31.0, 26.0, 40.0, 35.0]])
    expected_class = {0: 2, 1: 3}
    rows = [[0.0, 0.0, 6.0, 0.0], [0.0, 0.0, 0.0, 6.0]]
    counts = (first.shape[0], second.shape[0])
    for index in range(3):
        heads.stages[index] = _PerImageStubStage(4, rows, counts)
    heads.eval()

    with torch.no_grad():
        results, _ = heads(_features(2), [first, second], [STUB_IMAGE_SHAPE] * 2)

    assert len(results) == 2, f"batch of 2 produced {len(results)} results"
    for index, (result, source) in enumerate(zip(results, (first, second))):
        assert result["scores"].numel(), (
            f"image {index} produced no detections from stub logits at "
            f"probability ~0.998 — the vacuous-eval path"
        )
        assert set(result["labels"].tolist()) == {expected_class[index]}, (
            f"image {index} returned labels "
            f"{sorted(set(result['labels'].tolist()))}, expected only "
            f"{expected_class[index]} — the decode is reading another image's "
            f"score slice"
        )
        # Zero deltas mean the decoded boxes ARE the proposals (clipped), so
        # every detection must lie inside this image's own proposal region.
        boxes = result["boxes"]
        assert float(boxes[:, 0].min()) >= float(source[:, 0].min()) - 1e-3, (
            f"image {index}'s detections start left of its own proposals — the "
            f"decode read another image's box slice"
        )
        assert float(boxes[:, 2].max()) <= float(source[:, 2].max()) + 1e-3, (
            f"image {index}'s detections extend past its own proposals — the "
            f"decode read another image's box slice"
        )


def test_per_class_delta_columns_are_selected_by_the_predicted_class():
    """MUTATION: index the ``(P, num_classes, 4)`` deltas at a fixed column, or
    at the argmax over ALL classes including background.

    Only the predicted class's four deltas are used, and taking another class's
    produces a valid-looking box decoded from the wrong regressor. Here class 2
    is the argmax and only class 2's delta column is non-zero, so a correct
    selection moves the box and any other selection leaves it where it was.
    """
    heads = _roi_heads(num_classes=4, score_thresh=0.01, nms_thresh=1.0)

    class _ClassKeyedStage(nn.Module):
        def forward(self, pooled_features):
            count = pooled_features.shape[0]
            logits = torch.zeros(count, 4)
            logits[:, 2] = 6.0
            deltas = torch.zeros(count, 4, 4)
            deltas[:, 2, 2] = 0.5  # class 2, dw
            return logits, deltas.reshape(count, -1)

    for index in range(3):
        heads.stages[index] = _ClassKeyedStage()
    heads.eval()
    seen, seen_requires_grad = _spy_on_pooling(heads)
    proposals = [torch.tensor([[2.0, 3.0, 18.0, 21.0]])]
    with torch.no_grad():
        heads(_features(1), proposals, [STUB_IMAGE_SHAPE])

    widths = [float(boxes[0][0, 2] - boxes[0][0, 0]) for boxes in seen]
    assert widths[1] > widths[0] + 1e-3, (
        f"stage 1 pooled a box of width {widths[1]:.3f} against stage 0's "
        f"{widths[0]:.3f} — only class 2's dw was non-zero and class 2 is the "
        f"argmax, so an unchanged width means the wrong delta column was read"
    )


# --- training-side selection ----------------------------------------------


def test_ground_truth_is_added_to_the_proposals_at_every_stage():
    """MUTATION: add the ground truth at stage 0 only.

    This is the difference between a cold start and no start. On a freshly
    initialised RPN the best proposal reaches a low-single-digit IoU, so a
    0.7-threshold stage with no ground-truth proposal selects ZERO positives,
    contributes a zero box loss and never learns to localise — with every loss
    finite the whole time. The ground-truth box is the only IoU-1.0 candidate
    available, so its presence in the pooled set at the LAST stage is what is
    checked.
    """
    heads = _roi_heads()
    heads.train()
    seen, seen_requires_grad = _spy_on_pooling(heads)
    ground_truth = torch.tensor([[2.0, 2.0, 22.0, 21.0]])
    # Proposals deliberately far from the ground truth: at IoU 0 no stage would
    # find a positive without the ground-truth box being added.
    proposals = [torch.tensor([[30.0, 28.0, 39.0, 38.0], [32.0, 30.0, 38.0, 36.0]])]
    targets = [{"boxes": ground_truth, "labels": torch.tensor([1])}]
    heads(_features(1), proposals, [STUB_IMAGE_SHAPE], targets)

    for stage_index, boxes in enumerate(seen):
        matches = (boxes[0] - ground_truth[0]).abs().sum(dim=1) < 1e-4
        assert bool(matches.any()), (
            f"stage {stage_index} pooled no box equal to the ground truth — "
            f"without it a high-IoU stage has no positive to learn from"
        )


def test_every_stage_contributes_both_of_its_losses():
    """MUTATION: accumulate the loss for the last stage only, or overwrite the
    dict key each stage instead of suffixing it.

    Either leaves a plausible loss dict that trains one stage. The engine calls
    ``sum(losses.values())``, so a missing stage is not an error anywhere.
    """
    heads = _roi_heads()
    heads.train()
    proposals = [torch.tensor([[1.0, 1.0, 20.0, 20.0], [4.0, 4.0, 30.0, 26.0]])]
    targets = [{"boxes": torch.tensor([[2.0, 2.0, 22.0, 21.0]]), "labels": torch.tensor([1])}]
    _, losses = heads(_features(1), proposals, [STUB_IMAGE_SHAPE], targets)
    expected = {
        f"loss_{kind}_stage{index}"
        for index in range(3)
        for kind in ("classifier", "box_reg")
    }
    assert set(losses) == expected, f"loss keys {sorted(losses)} != {sorted(expected)}"


def test_stage_loss_weights_are_applied():
    """MUTATION: drop the per-stage weighting.

    Compare the shipped weights against an all-ones cascade on identical inputs:
    the weighted classification losses must be strictly smaller for the two
    down-weighted stages and identical for stage 0, whose weight is 1.0.
    """
    proposals = [torch.tensor([[1.0, 1.0, 20.0, 20.0], [4.0, 4.0, 30.0, 26.0]])]
    targets = [{"boxes": torch.tensor([[2.0, 2.0, 22.0, 21.0]]), "labels": torch.tensor([1])}]
    features = _features(1)

    weighted = _roi_heads()
    for index in range(3):
        weighted.stages[index] = _StubStage(4, [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    unweighted = _roi_heads(stage_loss_weights=(1.0, 1.0, 1.0))
    for index in range(3):
        unweighted.stages[index] = _StubStage(4, [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])

    weighted.train()
    unweighted.train()
    torch.manual_seed(0)
    _, weighted_losses = weighted(features, proposals, [STUB_IMAGE_SHAPE], targets)
    torch.manual_seed(0)
    _, plain_losses = unweighted(features, proposals, [STUB_IMAGE_SHAPE], targets)

    assert float(weighted_losses["loss_classifier_stage0"]) == pytest.approx(
        float(plain_losses["loss_classifier_stage0"]), rel=1e-6
    ), "stage 0's weight is 1.0, so its loss must be unchanged"
    for index, weight in ((1, 0.5), (2, 0.25)):
        assert float(weighted_losses[f"loss_classifier_stage{index}"]) == pytest.approx(
            weight * float(plain_losses[f"loss_classifier_stage{index}"]), rel=1e-5
        ), f"stage {index} is not weighted by {weight}"


#: The resolution Faster R-CNN / Cascade R-CNN train at in the literature, and
#: torchvision's own default for the family. A LITERAL transcribed from outside
#: this repository — see the note on circularity below.
PUBLISHED_TRAIN_RESOLUTION = 800


def test_the_declared_image_size_is_the_published_resolution():
    """MUTATION: declare 448.

    ⚠️ THIS TEST EXISTS BECAUSE THE OBVIOUS ONE IS CIRCULAR. The natural guard —
    "declared ``image_size`` equals the built model's transform resolution",
    which is what ``tests/test_od_declared_resolution.py`` checks family-wide —
    CANNOT FAIL for this template, because ``MyModel`` passes
    ``min_size=image_size`` into the transform. Mutating the declaration moves
    both sides and the equality still holds. Measured: mutating ``image_size``
    to 448 left 59 tests green.

    That is trap 31 in miniature, inside a test: a number compared against
    itself. So the declaration is pinned against a literal from the published
    architecture instead, and the equality assertion below is kept for what it
    is actually worth.
    """
    assert int(MODULE.image_size) == PUBLISHED_TRAIN_RESOLUTION, (
        f"declared image_size={MODULE.image_size}; Cascade R-CNN trains at "
        f"{PUBLISHED_TRAIN_RESOLUTION}, which is also torchvision's default for "
        f"this family"
    )


def test_the_declared_image_size_is_wired_into_the_transform():
    """The #3058 half, and it is NOT redundant with the test above — each is
    individually mutable-to-red, which is the trap-29 test.

    Mutating ``image_size`` alone moves both sides here and is caught by the
    published-value assertion. Mutating the ``min_size=image_size`` wiring (to a
    hard-coded 448, say) leaves the declaration at 800 and is caught here. So
    neither assertion is doing the other's job.

    Measured off the BUILT model, never asserted from the source.
    """
    model = MODULE.MyModel(3)
    min_size = model.transform.min_size
    effective = int(min_size[0] if isinstance(min_size, (list, tuple)) else min_size)
    assert effective == int(MODULE.image_size), (
        f"declared image_size={MODULE.image_size} but the transform runs at "
        f"{effective}"
    )
    assert model.transform.fixed_size is None, (
        "this template is a min_size model, not a fixed-size one; a fixed_size "
        "would make the declared value a different promise"
    )


def test_no_trainable_parameter_is_unreachable_by_the_loss():
    """MUTATION: build a head, or a whole stage, and never call it.

    ``requires_grad``-aware on purpose: ``p.grad is None`` alone false-flags the
    deliberately frozen ResNet stem and the FrozenBatchNorm statistics, which
    are meant to be untouched. The defect is a TRAINABLE parameter the loss
    never reaches.
    """
    torch.manual_seed(0)
    model = MODULE.MyModel(3)
    model.train()
    images = [torch.rand(3, 96, 120), torch.rand(3, 112, 96)]
    targets = [
        {
            "boxes": torch.tensor([[8.0, 8.0, 48.0, 48.0], [56.0, 56.0, 88.0, 90.0]]),
            "labels": torch.tensor([1, 2]),
        },
        {"boxes": torch.tensor([[4.0, 4.0, 32.0, 40.0]]), "labels": torch.tensor([2])},
    ]
    losses = model(images, targets)
    sum(losses.values()).backward()
    unreachable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not unreachable, (
        f"{len(unreachable)} trainable parameters received no gradient: "
        f"{unreachable[:8]}"
    )


# --- external oracles: the numbers, derived from outside this file ---------
#
# TRAP 31 (SELF-CONSISTENT-NUMBER). `yolox_s` shipped for review with an exact
# parameter count offered as proof the architecture was real. It WAS exact — for
# the model as built — while a wrong `expansion` default left the backbone
# ~1.15M parameters narrower than published YOLOX-S. Thirty guards and
# thirty-three mutations missed it, because every one of them was checked
# against the same self-derived table: `id()`-disjointness and
# `data_ptr`-identity hold at ANY width.
#
# So no expectation below is measured off this template. Two independent
# sources, and nothing imported from `model_zoo`:
#
#   * torchvision's own `fasterrcnn_resnet50_fpn`, a different implementation of
#     the same published ResNet-50-FPN + RPN, for everything this template
#     REUSES;
#   * hand arithmetic from the published Faster R-CNN head widths
#     (256 * 7 * 7 -> 1024 -> 1024, then num_classes and 4 * num_classes), for
#     the three stage heads this template WRITES.
#
# Guard count is not defence depth: thirty guards sharing one circular oracle
# are one guard wearing thirty hats.


#: Published Faster R-CNN box-head widths. Transcribed from the architecture,
#: not read from the template — that is the whole point.
ROI_POOLED_EDGE = 7
FPN_OUT_CHANNELS = 256
BOX_HEAD_WIDTH = 1024


def _derived_stage_parameters(num_classes):
    """Parameters in ONE cascade stage, from the published head widths.

    ``TwoMLPHead``: ``fc6`` maps the flattened 256x7x7 ROI to 1024, ``fc7``
    1024 to 1024, both with bias. ``FastRCNNPredictor``: a ``num_classes``
    classifier and a ``4 * num_classes`` box regressor, both with bias.
    """
    flattened = FPN_OUT_CHANNELS * ROI_POOLED_EDGE * ROI_POOLED_EDGE
    fc6 = flattened * BOX_HEAD_WIDTH + BOX_HEAD_WIDTH
    fc7 = BOX_HEAD_WIDTH * BOX_HEAD_WIDTH + BOX_HEAD_WIDTH
    classifier = BOX_HEAD_WIDTH * num_classes + num_classes
    regressor = BOX_HEAD_WIDTH * 4 * num_classes + 4 * num_classes
    return fc6 + fc7 + classifier + regressor


def _oracle():
    """torchvision's Faster R-CNN R50-FPN — an independent implementation of the
    backbone and RPN this template reuses.

    Built with ``weights=None`` so nothing is fetched. torchvision warns that
    ``trainable_backbone_layers`` has no effect without pretrained weights and
    falls back to 5; that changes ``requires_grad`` only, never a parameter
    count, so the shape comparisons below are unaffected. What it DOES mean is
    that this oracle cannot check the template's ``trainable_layers=3`` — that
    is checked structurally in its own test instead.
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    return fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=4)


def test_backbone_convolution_shapes_match_torchvisions_resnet50_fpn():
    """MUTATION: a wrong backbone width, depth, or ``returned_layers``.

    THE yolox_s CLASS OF BUG, and the reason this test compares shapes rather
    than a count: a narrower backbone is still internally consistent, still
    trains, still evaluates, and still passes every identity and disjointness
    check in this file. The multiset of 4-D parameter shapes pins width AND
    depth AND the FPN's level count in one comparison, against a library
    implementation of the same published architecture.

    ``returned_layers`` is genuinely covered here: torchvision's default is
    ``[1, 2, 3, 4]``, so a template built on ``[2, 3, 4]`` produces a different
    shape multiset and fails.
    """
    mine = MODULE.MyModel(3)
    oracle = _oracle()

    def conv_shapes(module):
        return sorted(tuple(p.shape) for p in module.parameters() if p.dim() == 4)

    mine_shapes = conv_shapes(mine.backbone)
    oracle_shapes = conv_shapes(oracle.backbone)
    assert mine_shapes == oracle_shapes, (
        f"the backbone's convolution shapes differ from torchvision's "
        f"ResNet-50-FPN: {len(mine_shapes)} tensors vs {len(oracle_shapes)}. "
        f"First divergence: "
        f"{next((a, b) for a, b in zip(mine_shapes, oracle_shapes) if a != b)}"
    )
    mine_elements = sum(p.numel() for p in mine.backbone.parameters() if p.dim() == 4)
    oracle_elements = sum(p.numel() for p in oracle.backbone.parameters() if p.dim() == 4)
    assert mine_elements == oracle_elements == 26797248, (
        f"backbone convolution weights total {mine_elements} against "
        f"torchvision's {oracle_elements}"
    )


def test_rpn_parameter_count_matches_torchvisions():
    """MUTATION: a wrong anchor count, or an RPN head at the wrong width.

    The RPN head's output channels are ``num_anchors`` and ``4 * num_anchors``,
    so a mis-sized anchor generator changes this number. Compared against
    torchvision's, not against a literal I chose.
    """
    mine = MODULE.MyModel(3)
    oracle = _oracle()
    mine_total = sum(p.numel() for p in mine.rpn.parameters())
    oracle_total = sum(p.numel() for p in oracle.rpn.parameters())
    assert mine_total == oracle_total == 593935, (
        f"RPN has {mine_total} parameters against torchvision's {oracle_total}"
    )
    per_location = mine.rpn.anchor_generator.num_anchors_per_location()
    assert per_location == [3] * 5, (
        f"anchors per location {per_location}, expected three aspect ratios at "
        f"each of five pyramid levels"
    )


def test_each_stage_head_matches_the_published_head_arithmetic():
    """MUTATION: a stage head at the wrong width — 512 instead of 1024, or a
    pooled edge of 5 instead of 7.

    Both are shape-consistent end to end and invisible to every structural
    check in this file. The expectation is computed from the published widths
    above, with nothing read from the template.
    """
    for num_classes, expected in ((4, _derived_stage_parameters(4)),
                                  (11, _derived_stage_parameters(11))):
        model = MODULE.MyModel(num_classes - 1)
        for index, stage in enumerate(model.roi_heads.stages):
            measured = sum(p.numel() for p in stage.parameters())
            assert measured == expected, (
                f"at num_classes={num_classes}, stage {index} has {measured} "
                f"parameters; the published head widths give {expected}"
            )
        del model


def test_the_whole_model_reconciles_against_the_two_oracles():
    """The end-to-end number, with every term sourced from outside this file.

    ``total = oracle backbone + oracle RPN + 3 x derived stage - 53,120``.

    The correction is exact and explained, which is what makes it admissible
    rather than a fudge: this template uses ``FrozenBatchNorm2d``, which holds
    weight and bias as BUFFERS, while torchvision's untrained builder uses live
    ``BatchNorm2d``, which holds them as PARAMETERS. ResNet-50 normalises 26,560
    channels, so exactly ``2 x 26,560 = 53,120`` affine values move from
    parameters to buffers. Both halves of that are asserted below, so the
    correction cannot silently absorb a real discrepancy.
    """
    mine = MODULE.MyModel(3)
    oracle = _oracle()

    normalised_channels = 26560
    frozen_affine = 2 * normalised_channels

    oracle_backbone = sum(p.numel() for p in oracle.backbone.parameters())
    oracle_rpn = sum(p.numel() for p in oracle.rpn.parameters())
    expected = oracle_backbone + oracle_rpn + 3 * _derived_stage_parameters(4) - frozen_affine

    measured = sum(p.numel() for p in mine.parameters())
    assert measured == expected, (
        f"the model has {measured} parameters; torchvision's backbone + RPN "
        f"plus three published-width stages minus the {frozen_affine} frozen "
        f"affine values gives {expected} (difference {measured - expected})"
    )

    # The correction, both halves, so it cannot be a coincidence of the right
    # size: the affine values must be ABSENT from the parameters and PRESENT in
    # the buffers, and FrozenBatchNorm2d keeps four buffers per channel
    # (weight, bias, running_mean, running_var) where BatchNorm2d keeps two
    # plus a scalar num_batches_tracked.
    assert oracle_backbone - sum(p.numel() for p in mine.backbone.parameters()) == frozen_affine
    assert sum(b.numel() for b in mine.backbone.buffers()) == 4 * normalised_channels, (
        "FrozenBatchNorm2d should hold four buffers per normalised channel"
    )


def test_the_backbone_stem_and_first_stage_are_frozen():
    """MUTATION: ``trainable_layers=5`` (or 0).

    This is the one claim the torchvision oracle CANNOT check — it warns that
    ``trainable_backbone_layers`` has no effect without pretrained weights and
    falls back to 5, and the setting changes ``requires_grad`` only, never a
    parameter count. So it is checked directly, against torchvision's
    documented semantics: ``trainable_layers=3`` trains ``layer2``, ``layer3``
    and ``layer4`` and freezes the stem and ``layer1``.

    It matters beyond fidelity: a frozen stem is why
    ``test_no_trainable_parameter_is_unreachable_by_the_loss`` has to be
    ``requires_grad``-aware, and if the stem were trainable that test would
    start reporting the frozen layers as defects.
    """
    model = MODULE.MyModel(3)
    frozen, trainable = [], []
    for name, parameter in model.backbone.body.named_parameters():
        (trainable if parameter.requires_grad else frozen).append(name)

    assert frozen, "no backbone parameter is frozen — trainable_layers is not 3"
    assert all(
        name.startswith(("conv1", "bn1", "layer1")) for name in frozen
    ), f"unexpected frozen parameters: {[n for n in frozen if not n.startswith(('conv1', 'bn1', 'layer1'))][:5]}"
    assert all(
        name.startswith(("layer2", "layer3", "layer4")) for name in trainable
    ), f"unexpected trainable parameters: {[n for n in trainable if not n.startswith(('layer2', 'layer3', 'layer4'))][:5]}"
    assert all(
        parameter.requires_grad for parameter in model.backbone.fpn.parameters()
    ), "the FPN must be trainable"
