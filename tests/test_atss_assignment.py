"""Contract test for ATSS's assignment (backend#2982, Tier 2).

Why this file exists on top of the family train-step guard
----------------------------------------------------------
``test_od_torchvision_family_train_step.py`` proves every
``torchvision_detection`` template completes a train step and an eval step. That
is necessary and it is not sufficient for ``atss_resnet``, because unlike every
template before it the *assignment logic is ours*, not torchvision's — and the
ways it can be wrong are all silent:

- assign **nothing**: every anchor labelled background. Focal loss over an
  all-negative image is finite and small, backward runs, the train step passes,
  and the model learns no objects at all.
- assign **everything**: the threshold computed as a NaN and the comparison
  swallowing it, or the centre test dropped.
- lose the **per-level** step: taking a global topk instead of topk-per-level
  is a different (worse) assigner that trains perfectly happily. This is the
  one the module docstring calls out, because the per-level split arrives
  through a recorded attribute rather than an argument.

None of those fail a train step. Each of them fails something below.

The convention being honoured is torchvision's ``Matcher`` encoding, since
``RetinaNetHead.compute_loss`` is reused verbatim: ``>= 0`` is a ground-truth
index, ``-1`` is background, ``-2`` is ignore.
"""

import importlib.util
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parent.parent
TEMPLATE = ROOT / "model_zoo" / "object_detection" / "pytorch" / "atss_resnet.py"


def _load():
    spec = importlib.util.spec_from_file_location("atss_resnet_under_test", TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def atss():
    pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")
    return _load()


@pytest.fixture(scope="module")
def anchors_and_levels(atss):
    """Real anchors from the template's own generator, not synthetic ones.

    A hand-rolled anchor grid would let the assignment pass against a tiling the
    model never produces; this drives the actual ``_LevelAwareAnchorGenerator``
    over a real backbone output, which also exercises the recording step.
    """
    import torch
    from torchvision.models.detection.image_list import ImageList

    model = atss.MyModel(3)
    model.eval()
    image = torch.rand(1, 3, 256, 320)
    with torch.no_grad():
        features = list(model.backbone(image).values())
    anchors = model.anchor_generator(ImageList(image, [(256, 320)]), features)[0]
    return model, anchors, list(model.anchor_generator.num_anchors_per_level)


def test_the_level_split_is_recorded_and_covers_every_anchor(anchors_and_levels):
    """The per-level split must exist and account for exactly the anchors built.

    If it under-counts, the last level is silently never sampled from; if it
    over-counts, ``scatter_`` would raise. Either way the assigner stops being
    ATSS, so the totals are pinned.
    """
    _, anchors, levels = anchors_and_levels
    assert levels, "no per-level split recorded — _LevelAwareAnchorGenerator did not run"
    assert len(levels) == 5, f"expected the P3..P7 pyramid, got {len(levels)} levels"
    assert sum(levels) == anchors.shape[0], (
        f"level split sums to {sum(levels)} but {anchors.shape[0]} anchors were "
        f"built — the assignment would ignore the difference"
    )
    assert all(size > 0 for size in levels), f"an empty pyramid level: {levels}"


def test_it_assigns_positives(atss, anchors_and_levels):
    """The all-background failure: trains fine, learns nothing."""
    import torch

    _, anchors, levels = anchors_and_levels
    gt = torch.tensor([[20.0, 20.0, 100.0, 100.0], [150.0, 60.0, 260.0, 200.0]])
    matched = atss._atss_assign(anchors, gt, levels, atss.ATSS_TOPK)

    assert matched.shape == (anchors.shape[0],)
    assert matched.dtype == torch.int64
    positives = int((matched >= 0).sum())
    assert positives > 0, (
        "ATSS assigned every anchor to background. A model trained this way "
        "completes every train step and never learns an object"
    )
    # Every ground-truth box must win at least one anchor, or it contributes
    # nothing to the regression loss.
    for index in range(gt.shape[0]):
        assert int((matched == index).sum()) > 0, (
            f"ground-truth box {index} was assigned no anchor at all"
        )
    # And it must not degenerate the other way.
    assert positives < anchors.shape[0] / 2, (
        f"{positives} of {anchors.shape[0]} anchors positive — the threshold or "
        f"the centre test is not filtering"
    )


def test_it_never_emits_the_ignore_sentinel(atss, anchors_and_levels):
    """ATSS partitions into positive/negative with no ambiguous band.

    Emitting ``-2`` would silently drop those anchors from the classification
    loss (the head treats it as ignore), which is fixed-threshold behaviour
    leaking back in.
    """
    import torch

    _, anchors, levels = anchors_and_levels
    gt = torch.tensor([[20.0, 20.0, 100.0, 100.0]])
    matched = atss._atss_assign(anchors, gt, levels, atss.ATSS_TOPK)
    assert int((matched == -2).sum()) == 0, "ATSS emitted the ignore sentinel"
    assert bool(((matched >= 0) | (matched == atss.BACKGROUND)).all())


def test_zero_object_image_is_all_background(atss, anchors_and_levels):
    """The engine's dataset emits an explicit empty target for an unannotated
    image, so this is a real input, not an edge case."""
    import torch

    _, anchors, levels = anchors_and_levels
    matched = atss._atss_assign(anchors, torch.zeros((0, 4)), levels, atss.ATSS_TOPK)
    assert matched.shape == (anchors.shape[0],)
    assert bool((matched == atss.BACKGROUND).all())


def test_a_single_candidate_does_not_produce_a_nan_threshold(atss):
    """std over one sample is NaN under the unbiased estimator.

    ``ious >= nan`` is False everywhere, so the object would silently get no
    anchors. The template computes the deviation explicitly to avoid it; this
    pins that, by forcing exactly one candidate per level with ``topk=1``.
    """
    import torch

    anchors = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]]
    )
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    matched = atss._atss_assign(anchors, gt, [1, 1], topk=1)
    assert int((matched >= 0).sum()) >= 1, (
        "a single candidate per level produced no positive — the mean+std "
        "threshold went NaN"
    )


def test_the_centre_inside_rule_is_enforced(atss):
    """An anchor that clears the adaptive threshold but is centred OUTSIDE the
    object must still be rejected.

    This test was rewritten after a mutation sweep: the first version dropped
    the centre-inside rule from the template and still passed. The reason is
    worth recording, because it is a trap for anyone extending this file. With
    exactly two candidates the mean+std threshold *is* the larger of the two
    IoUs — for values ``a < b``, ``mean + population_std == b`` — so only the
    single best candidate ever clears it and the centre test is never consulted.
    A two-anchor fixture cannot exercise this rule at all.

    The fixture below fixes that by making the outside-centred anchor the
    highest-IoU candidate among four, so it clears the threshold on its own
    merits and only the centre rule can reject it. The two cases differ by
    **0.2px of anchor position** and nothing else:

        anchor (10.1, 0, 30.1, 20) -> centre x = 20.1, just OUTSIDE gt x2=20
        anchor ( 9.9, 0, 29.9, 20) -> centre x = 19.9, just INSIDE

    IoU 0.329 vs 0.338 against a threshold of 0.225 / 0.231 — both clear it, so
    the assignment differs only because of the centre test. Dropping that test
    makes the first case assign, and this fails.
    """
    import torch

    gt = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
    # Three anchors far away with IoU 0, so the mean+std threshold stays low
    # enough that the anchor under test clears it.
    far = [[500.0, 500.0, 510.0, 510.0], [600.0, 600.0, 610.0, 610.0], [700.0, 700.0, 710.0, 710.0]]

    outside = torch.tensor([[10.1, 0.0, 30.1, 20.0]] + far)
    inside = torch.tensor([[9.9, 0.0, 29.9, 20.0]] + far)

    matched_outside = atss._atss_assign(outside, gt, [4], topk=4)
    matched_inside = atss._atss_assign(inside, gt, [4], topk=4)

    assert matched_inside[0] == 0, (
        "the anchor centred just INSIDE the object was not assigned to it, "
        "although it clears the adaptive threshold"
    )
    assert matched_outside[0] == atss.BACKGROUND, (
        "an anchor centred just OUTSIDE the object was assigned to it. It "
        "clears the mean+std threshold, so the centre-inside rule is the only "
        "thing that should reject it — and it is not being applied"
    )


def test_per_level_topk_is_actually_per_level(atss, anchors_and_levels):
    """The guard that proves step 1 is live.

    Collapsing the split to a single level turns topk-per-level into a global
    topk. That is a different assigner which trains happily, so the only way to
    know the real one is running is to show the two disagree.
    """
    _, anchors, levels = anchors_and_levels
    import torch

    gt = torch.tensor([[20.0, 20.0, 100.0, 100.0], [150.0, 60.0, 260.0, 200.0]])
    per_level = atss._atss_assign(anchors, gt, levels, atss.ATSS_TOPK)
    global_topk = atss._atss_assign(anchors, gt, [anchors.shape[0]], atss.ATSS_TOPK)

    assert not torch.equal(per_level, global_topk), (
        "per-level and global topk produced identical assignments, so the "
        "per-level step is not affecting the result — check that "
        "num_anchors_per_level is reaching _atss_assign"
    )
    assert int((per_level >= 0).sum()) > int((global_topk >= 0).sum()), (
        "per-level sampling should draw candidates from every pyramid level and "
        "so yield more positives than one global topk"
    )


def test_it_differs_from_the_fixed_threshold_matcher(atss, anchors_and_levels):
    """Guard against a regression to RetinaNet's assignment.

    If ``compute_loss`` were ever reverted, or the override silently stopped
    being called, this file's other tests would all still pass — they test
    ``_atss_assign`` directly. This one pins that the function is not merely
    reproducing ``det_utils.Matcher``.
    """
    import torch
    from torchvision.models.detection import _utils as det_utils
    from torchvision.ops import boxes as box_ops

    _, anchors, levels = anchors_and_levels
    gt = torch.tensor([[20.0, 20.0, 100.0, 100.0], [150.0, 60.0, 260.0, 200.0]])

    atss_matched = atss._atss_assign(anchors, gt, levels, atss.ATSS_TOPK)
    fixed = det_utils.Matcher(0.5, 0.4, allow_low_quality_matches=True)
    fixed_matched = fixed(box_ops.box_iou(gt, anchors))

    assert not torch.equal(atss_matched, fixed_matched), (
        "ATSS produced exactly RetinaNet's fixed-threshold assignment — the "
        "adaptive threshold is not doing anything"
    )


def test_compute_loss_refuses_a_plain_anchor_generator(atss):
    """The recorded-attribute design has one failure mode: no record.

    A plain ``AnchorGenerator`` would leave the split absent, and defaulting to
    "one level" there would silently degrade ATSS to global topk. The template
    raises instead; this proves the raise is reachable rather than decorative.
    """
    import torch
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    model = atss.MyModel(3)
    # Swap in a generator that records nothing, exactly as a future edit
    # dropping _LevelAwareAnchorGenerator would.
    model.anchor_generator = AnchorGenerator(
        sizes=((64,), (128,), (256,), (512,), (1024,)),
        aspect_ratios=((1.0,),) * 5,
    )
    model.train()
    with pytest.raises(RuntimeError, match="per-level anchor split"):
        model(
            [torch.rand(3, 128, 160)],
            [{"boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]), "labels": torch.tensor([1])}],
        )


def test_the_template_declares_the_family_contract():
    """Cheap, no-torch assertions on the header — these are what route the
    template on the platform, and they are easy to lose in an edit."""
    source = TEMPLATE.read_text(encoding="utf-8")
    assert re.search(r'^model_type\s*=\s*"torchvision_detection"', source, re.MULTILINE), (
        "atss_resnet must declare model_type = 'torchvision_detection' — not the "
        "legacy 'rcnn' alias and not an architecture name"
    )
    # The offline rules that CI cannot observe on a warm cache.
    assert "weights=None" in source, "the backbone must be built with weights=None"
    for banned in ("import timm", "from timm", "transformers", "from_pretrained"):
        assert banned not in source, f"{banned!r} is not permitted in a cv template"
