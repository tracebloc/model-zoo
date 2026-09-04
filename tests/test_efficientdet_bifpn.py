"""Structural and decode tests for ``object_detection/pytorch/efficientdet_d0.py``.

What is worth testing here, and what is not
-------------------------------------------
``tests/test_od_torchvision_family_train_step.py`` proves the template trains
and evaluates, and ``tests/test_model_contract.py`` proves it constructs.
Neither can tell BiFPN's **learnable weighted** fusion from a plain sum, a
shared head from five per-level heads, or a depthwise-separable conv from a
dense one — all three are shape-identical substitutions that leave every
existing assertion green while removing what the architecture is.

Every test below names the mutation it exists for.

Three rules the tests obey, each learned from a real bug in this roster
----------------------------------------------------------------------
**Never assert a statistic of randomly initialised weights.** Measured bands
overlap, so "the fusion weights look learnable" cannot separate correct from
incorrect. Where a property is structural, the structure is tested: known
constant inputs through a stub, and exact arithmetic on the result.

**Drive the decode above threshold, at batch >= 2, on a non-square map with
several anchors per location.** A fresh focal-loss detector initialises its
classification prior at 0.01, below the 0.05 ``score_thresh``, so it returns
**zero detections** and every eval assertion passes against a well-formed empty
list. That is not a hypothetical: a real decode bug shipped through every guard
this way. Square feature maps and one-anchor-per-location are equally
disqualifying, because they make anchor-major and location-major orderings
indistinguishable.

**Assert on the real function's return value.** No test here recomputes the
quantity it is checking; the fusion arithmetic is pinned to a literal (1.75),
and every positive control has to react for the test to be able to pass.
"""

import importlib.util
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parent.parent
TEMPLATE = ROOT / "model_zoo" / "object_detection" / "pytorch" / "efficientdet_d0.py"

pytest.importorskip("torch", reason="pytorch not installed in this CI job")
pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")

import torch  # noqa: E402 — after importorskip, deliberately
from torch import nn  # noqa: E402


def _module():
    spec = importlib.util.spec_from_file_location(
        re.sub(r"\W", "_", f"effdet_{TEMPLATE.stem}"), TEMPLATE
    )
    assert spec and spec.loader, f"{TEMPLATE}: importlib could not build a spec"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _module()


def _fusion_nodes(model):
    return [m for m in model.modules() if isinstance(m, MODULE._WeightedFusion)]


# --- fast normalized fusion ------------------------------------------------


def test_fusion_arithmetic_is_the_weighted_normalized_sum():
    """MUTATION: an unweighted sum, or a weighted sum with no normalisation.

    Known weights (1, 3) on known inputs (all-ones, all-twos) give exactly
    ``(1*1 + 3*2) / (1 + 3) = 1.75``. An unweighted sum gives 3.0, an
    unnormalised weighted sum 7.0, a mean 1.5, and taking only the first input
    1.0 — every plausible substitution lands on a different number, which is
    why the literal is worth more here than any structural check.
    """
    node = MODULE._WeightedFusion(2)
    with torch.no_grad():
        node.weight.copy_(torch.tensor([1.0, 3.0]))
    inputs = [torch.ones(1, 2, 3, 3), torch.full((1, 2, 3, 3), 2.0)]
    with torch.no_grad():
        out = node(inputs)
    assert out.shape == inputs[0].shape
    # eps = 1e-4 in the denominator, hence the tolerance rather than equality.
    assert float(out.min()) == pytest.approx(1.75, abs=1e-4)
    assert float(out.max()) == pytest.approx(1.75, abs=1e-4)


def test_fusion_weights_are_clamped_non_negative():
    """MUTATION: drop the ``relu`` on the weights.

    A negative weight makes the denominator shrink towards zero (and can cross
    it), so the node's output explodes rather than being merely wrong. With the
    relu, a negative weight simply removes that input: weights ``(-5, 1)`` must
    give exactly the second input.
    """
    node = MODULE._WeightedFusion(2)
    with torch.no_grad():
        node.weight.copy_(torch.tensor([-5.0, 1.0]))
        out = node([torch.ones(1, 1, 2, 2), torch.full((1, 1, 2, 2), 7.0)])
    assert float(out.max()) == pytest.approx(7.0, abs=1e-3)
    assert float(out.min()) == pytest.approx(7.0, abs=1e-3)


def test_a_fusion_node_rejects_the_wrong_number_of_inputs():
    """A configuration the shipped topology does not build, per trap 24. Node
    arity is fixed by the BiFPN wiring (2 inputs top-down, 3 in the middle of
    the bottom-up path); a mis-wired node would otherwise broadcast against a
    shorter weight vector and produce something plausible."""
    node = MODULE._WeightedFusion(3)
    with pytest.raises(ValueError, match="was built for 3 inputs"):
        node([torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2)])


def test_every_fusion_weight_is_a_trainable_parameter():
    """MUTATION: register the fusion weights as buffers, or detach them.

    Either leaves the arithmetic above intact and the architecture's headline
    claim gone: BiFPN's F is for *fast normalized fusion*, and without a
    gradient it is a fixed average with extra steps.
    """
    model = MODULE.MyModel(3)
    nodes = _fusion_nodes(model)
    # 3 BiFPN repeats x (4 top-down + 4 bottom-up) nodes.
    assert len(nodes) == 3 * 8, f"expected 24 fusion nodes, found {len(nodes)}"

    parameter_ids = {id(p) for p in model.parameters()}
    for index, node in enumerate(nodes):
        assert isinstance(node.weight, nn.Parameter), (
            f"fusion node {index}'s weight is a {type(node.weight).__name__}, "
            f"not an nn.Parameter"
        )
        assert node.weight.requires_grad, f"fusion node {index}'s weight is frozen"
        assert id(node.weight) in parameter_ids, (
            f"fusion node {index}'s weight is not reachable from "
            f"model.parameters() — an optimizer built from parameters() would "
            f"never update it"
        )


def test_every_fusion_weight_receives_a_non_zero_gradient():
    """The positive control the test above needs: being a parameter is not the
    same as being reached. This runs a REAL forward and backward through the
    real loss, then requires a non-zero gradient at all 24 nodes."""
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

    for index, node in enumerate(_fusion_nodes(model)):
        assert node.weight.grad is not None, (
            f"fusion node {index} received no gradient — it is a parameter the "
            f"loss never reaches"
        )
        assert float(node.weight.grad.abs().max()) > 0.0, (
            f"fusion node {index}'s gradient is exactly zero, so it can never "
            f"move away from the unweighted mean it initialises to"
        )


def test_the_bifpn_is_bidirectional():
    """MUTATION: keep the top-down path and drop the bottom-up one.

    That is an FPN with learnable weights, not a BiFPN. The bottom-up path is
    what lets a fine level inform a coarse one, so the test perturbs P3 (the
    finest input) and requires the OUTPUT at P7 (the coarsest) to change — a
    dependency only the bottom-up path can create.
    """
    torch.manual_seed(0)
    layer = MODULE._BiFPNLayer(channels=8)
    layer.eval()
    base = [
        torch.rand(1, 8, 20, 28),
        torch.rand(1, 8, 10, 14),
        torch.rand(1, 8, 5, 7),
        torch.rand(1, 8, 3, 4),
        torch.rand(1, 8, 2, 2),
    ]
    with torch.no_grad():
        before = layer([t.clone() for t in base])[-1]
        perturbed = [t.clone() for t in base]
        perturbed[0] = perturbed[0] + 5.0
        after = layer(perturbed)[-1]
    assert not torch.allclose(before, after, atol=1e-6), (
        "changing the finest input left the coarsest output identical — the "
        "bottom-up path is not wired, so this is an FPN not a BiFPN"
    )


def test_the_bifpn_is_top_down_as_well():
    """The mirror image: perturb P7 and require P3's output to change. Together
    with the test above this pins BOTH directions, so neither can be removed
    without a failure."""
    torch.manual_seed(0)
    layer = MODULE._BiFPNLayer(channels=8)
    layer.eval()
    base = [
        torch.rand(1, 8, 20, 28),
        torch.rand(1, 8, 10, 14),
        torch.rand(1, 8, 5, 7),
        torch.rand(1, 8, 3, 4),
        torch.rand(1, 8, 2, 2),
    ]
    with torch.no_grad():
        before = layer([t.clone() for t in base])[0]
        perturbed = [t.clone() for t in base]
        perturbed[-1] = perturbed[-1] + 5.0
        after = layer(perturbed)[0]
    assert not torch.allclose(before, after, atol=1e-6), (
        "changing the coarsest input left the finest output identical — the "
        "top-down path is not wired"
    )


def test_bifpn_repeats_are_independent_parameter_subtrees():
    """MUTATION: one BiFPN layer applied three times.

    Compound scaling grows the number of BiFPN *layers*; reusing one is a
    recurrence with a third of the capacity, and it is invisible in the forward
    pass.
    """
    model = MODULE.MyModel(3)
    layers = model.backbone.bifpn
    assert len(layers) == MODULE.BIFPN_REPEATS == 3
    assert len({id(layer) for layer in layers}) == 3, "the BiFPN repeats are one object"
    ids = [{id(p) for p in layer.parameters()} for layer in layers]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2]), (
        "BiFPN repeats share parameter tensors"
    )


# --- the shared, separable head -------------------------------------------


def test_the_head_towers_are_shared_across_levels():
    """MUTATION: build one tower per pyramid level.

    Sharing is the reason the head's parameter count does not grow with the
    pyramid. Identity is what settles it — five towers with identical structure
    would pass any shape check.
    """
    head = MODULE.MyModel(3).head
    assert len(head.cls_tower) == MODULE.HEAD_CONVS
    assert len(head.box_tower) == MODULE.HEAD_CONVS
    # One tower object, not five: applied to each level by the forward loop.
    assert not isinstance(head.cls_tower[0], nn.ModuleList), (
        "the classification tower is nested per level — it is not shared"
    )


def test_the_shared_head_gives_identical_logits_for_identical_levels():
    """The behavioural half of sharing, which the structural test cannot reach:
    feed the SAME tensor as two different pyramid levels and the head must
    produce identical outputs for both. Per-level heads (with per-level random
    init, or per-level norm statistics) cannot."""
    torch.manual_seed(0)
    head = MODULE._EfficientDetHead(8, 9, 4, num_convs=2)
    head.eval()
    feature = torch.rand(2, 8, 5, 7)
    with torch.no_grad():
        out = head([feature, feature.clone()])
    half = out["cls_logits"].shape[1] // 2
    assert torch.allclose(
        out["cls_logits"][:, :half], out["cls_logits"][:, half:], atol=1e-6
    ), "the same feature map at two levels produced different logits"
    assert torch.allclose(
        out["bbox_regression"][:, :half], out["bbox_regression"][:, half:], atol=1e-6
    )


def test_head_and_bifpn_convolutions_are_depthwise_separable():
    """MUTATION: a dense 3x3 in place of the depthwise + pointwise pair.

    This is where EfficientDet's parameter budget goes, and a dense conv is a
    drop-in with the same output shape and ~8x the parameters. Checked
    structurally: the 3x3 must have ``groups == in_channels`` and be followed by
    a 1x1.
    """
    model = MODULE.MyModel(3)
    separables = [
        m
        for m in model.modules()
        if isinstance(m, nn.Sequential)
        and len(m) == 2
        and all(isinstance(c, nn.Conv2d) for c in m)
    ]
    assert separables, "found no separable-conv blocks at all"
    for block in separables:
        depthwise, pointwise = block
        assert depthwise.kernel_size == (3, 3)
        assert depthwise.groups == depthwise.in_channels, (
            f"a 3x3 with groups={depthwise.groups} against "
            f"in_channels={depthwise.in_channels} is a dense convolution, not "
            f"a depthwise one"
        )
        assert pointwise.kernel_size == (1, 1)
        assert pointwise.groups == 1


def test_the_squeeze_excite_gate_is_present_and_multiplicative():
    """MUTATION: drop the squeeze-excite block from the MBConv.

    An inverted residual without it is a different block, and the model still
    trains. Behavioural check: the gate is multiplicative and channel-wise, so
    zeroing its expand weights and biasing them strongly negative must drive
    the block's output to (nearly) zero — an additive or absent gate cannot.
    """
    torch.manual_seed(0)
    block = MODULE._MBConv(8, 8, 6, 1, 3)
    gates = [m for m in block.modules() if isinstance(m, MODULE._SqueezeExcite)]
    assert len(gates) == 1, f"expected one squeeze-excite per MBConv, found {len(gates)}"
    gate = gates[0]
    # The gate sits INSIDE the expansion, so its width is the MBConv's hidden
    # width (in_channels * expand_ratio), not the block's input width.
    hidden = gate.expand.out_channels
    assert hidden == 8 * 6, f"squeeze-excite width {hidden}, expected the 48-wide expansion"
    x = torch.rand(2, hidden, 6, 8)
    block.eval()
    with torch.no_grad():
        gate_open = gate(x)
        gate.expand.weight.zero_()
        gate.expand.bias.fill_(-30.0)
        gate_shut = gate(x)
    assert float(gate_shut.abs().max()) < 1e-6, (
        "shutting the squeeze-excite gate did not zero its output — the gate is "
        "not multiplicative"
    )
    assert float(gate_open.abs().max()) > 0.0


def test_the_model_carries_no_batchnorm_running_buffers():
    """MUTATION: swap the GroupNorms back to BatchNorm2d.

    Two reasons this matters and neither is style: BN running statistics average
    poorly across non-IID federated clients (``CLAUDE.md``), and
    ``FrozenBatchNorm2d`` — the escape hatch the rest of this family uses — does
    not normalise anything on a from-scratch trunk, because its running
    statistics are the untouched 0/1 defaults.
    """
    model = MODULE.MyModel(3)
    batchnorms = [
        name
        for name, module in model.named_modules()
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm))
    ]
    assert not batchnorms, f"BatchNorm modules present: {batchnorms[:5]}"
    buffers = [name for name, _ in model.named_buffers()]
    assert not buffers, f"expected no buffers at all, found {buffers[:5]}"


# --- the pyramid ----------------------------------------------------------


def test_the_backbone_emits_five_levels_at_the_declared_strides():
    """MUTATION: tap the wrong EfficientNet stages, giving a pyramid whose
    strides do not match ``PYRAMID_STRIDES`` — which is what the anchor sizes
    and the level split are computed from. Nothing raises; the anchors are
    simply the wrong size for their level.

    Deliberately run on a NON-SQUARE input, so a height/width transposition in
    a reshape shows up as a shape mismatch rather than passing.
    """
    backbone = MODULE._EfficientDetBackbone(channels=16, repeats=1)
    backbone.eval()
    height, width = 256, 384
    with torch.no_grad():
        features = backbone(torch.rand(1, 3, height, width))
    assert list(features) == ["0", "1", "2", "3", "4"]
    for index, (name, stride) in enumerate(zip(features, MODULE.PYRAMID_STRIDES)):
        feature = features[name]
        assert feature.shape[1] == 16, f"level {name} has {feature.shape[1]} channels"
        expected = (-(-height // stride), -(-width // stride))
        assert tuple(feature.shape[-2:]) == expected, (
            f"level {name} is {tuple(feature.shape[-2:])} on a {height}x{width} "
            f"input; stride {stride} implies {expected}"
        )
        del index


def test_nine_anchors_per_location():
    """Three octaves x three aspect ratios. Load-bearing beyond fidelity: at one
    anchor per location the head's anchor-major and location-major flattenings
    are the same tensor, so the ordering test below could not fail."""
    model = MODULE.MyModel(3)
    per_location = model.anchor_generator.num_anchors_per_location()
    assert per_location == [9] * 5, f"anchors per location: {per_location}"


def test_head_flatten_ordering_is_location_major():
    """MUTATION: permute the head output as ``(N, K, A, H, W)`` instead of
    ``(N, A, K, H, W)``, or skip the permute and reshape directly.

    Every variant is the same shape, so nothing raises — the model just trains
    against boxes decoded at the wrong pixels. Pinned with a stub: one channel
    of the raw conv output is set to a marker value at one spatial position, and
    the flattened output must carry that marker at the index
    ``(row * W + column) * A + anchor``, which is the ordering
    ``AnchorGenerator`` emits.

    Non-square map (5x7) and 9 anchors per location, both required for the
    assertion to be able to fail.
    """
    anchors, classes, height, width = 9, 4, 5, 7
    raw = torch.zeros(2, anchors * classes, height, width)
    row, column, anchor, klass = 3, 5, 4, 2
    marker = 12.5
    # Channel layout of the conv output is (A, K) flattened, so this is the
    # channel belonging to anchor `anchor`, class `klass`.
    raw[1, anchor * classes + klass, row, column] = marker

    flat = MODULE._EfficientDetHead._flatten(raw, classes)
    assert flat.shape == (2, height * width * anchors, classes)
    index = (row * width + column) * anchors + anchor
    assert float(flat[1, index, klass]) == pytest.approx(marker), (
        f"the marker did not land at location-major index {index}; the head's "
        f"flatten does not agree with AnchorGenerator's ordering"
    )
    # And nowhere else — a wrong permute would also put a marker somewhere.
    assert float(flat.abs().sum()) == pytest.approx(marker), (
        "the marker appears more than once in the flattened output"
    )
    assert float(flat[0].abs().sum()) == 0.0, "image 0 was contaminated by image 1"


def test_eval_decode_returns_per_image_detections_above_threshold():
    """MUTATION: iterate the split head outputs as if the outer list were
    images when it is pyramid LEVELS — the real bug this rule comes from, which
    ``zip`` truncated instead of raising, so it processed level 0 of image 0 and
    was silently wrong at batch > 1.

    A fresh model cannot catch it: the 0.01 classification prior is below the
    0.05 ``score_thresh``, so the decode returns an empty list and every
    assertion passes vacuously. So the head is replaced with a stub whose logits
    are ABOVE threshold and DIFFER per image, at batch 2, and each image's
    detections are required to be its own.
    """
    model = MODULE.MyModel(3)
    real_head = model.head
    target_class = {0: 1, 1: 3}

    class _StubHead(nn.Module):
        """Per-image logits: image 0 favours class 1, image 1 favours class 3,
        and only the anchors in the top-left corner of the finest level score
        at all — so a decode that mixes the images up returns the wrong label,
        and one that mixes the levels up returns boxes at the wrong scale."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, features):
            out = self.inner(features)
            logits = torch.full_like(out["cls_logits"], -10.0)
            for image_index, klass in target_class.items():
                logits[image_index, :20, klass] = 4.0
            return {
                "cls_logits": logits,
                "bbox_regression": torch.zeros_like(out["bbox_regression"]),
            }

    model.head = _StubHead(real_head)
    model.eval()
    with torch.no_grad():
        results = model([torch.rand(3, 96, 128), torch.rand(3, 112, 96)])

    assert len(results) == 2, f"batch of 2 produced {len(results)} results"
    for image_index, result in enumerate(results):
        assert result["scores"].numel(), (
            f"image {image_index} produced no detections from logits at "
            f"sigmoid(4.0) ~= 0.98 — that is the vacuous-eval path, not a pass"
        )
        assert set(result["labels"].tolist()) == {target_class[image_index]}, (
            f"image {image_index} returned labels "
            f"{sorted(set(result['labels'].tolist()))}, expected only "
            f"{target_class[image_index]} — the decode is reading another "
            f"image's row"
        )
        boxes = result["boxes"]
        assert bool((boxes[:, 2] >= boxes[:, 0]).all() and (boxes[:, 3] >= boxes[:, 1]).all())


#: EfficientDet-D0's input resolution, from the paper's compound-scaling table
#: (D0 512, D1 640, D2 768, D3 896, D4 1024, D5 1280, D6/D7 1280/1536). A
#: LITERAL transcribed from outside this repository — see the circularity note.
PUBLISHED_D0_RESOLUTION = 512


def test_the_declared_image_size_is_the_published_d0_resolution():
    """MUTATION: declare 448.

    ⚠️ THIS TEST EXISTS BECAUSE THE OBVIOUS ONE IS CIRCULAR, and the circularity
    already bit: this template spent a while on disk declaring 448, with all
    twenty tests in this file green. ``MyModel`` passes
    ``fixed_size=(image_size, image_size)``, so "declared equals the built
    model's transform" — what ``tests/test_od_declared_resolution.py`` checks
    family-wide — moves BOTH sides when the declaration is mutated and can never
    fail here.

    That is trap 31 in miniature, inside a test: a number compared against
    itself. Resolution is not cosmetic for EfficientDet — compound scaling ties
    it to the width and depth coefficients, so 448 is not a smaller D0, it is
    not a D-anything.
    """
    assert int(MODULE.image_size) == PUBLISHED_D0_RESOLUTION, (
        f"declared image_size={MODULE.image_size}; D0's compound-scaling "
        f"resolution is {PUBLISHED_D0_RESOLUTION}"
    )


def test_the_declared_image_size_is_wired_into_the_transform():
    """The #3058 half, and NOT redundant with the test above — each is
    individually mutable-to-red, which is the trap-29 test. Mutating
    ``image_size`` alone moves both sides and is caught above; removing the
    ``fixed_size`` wiring leaves the declaration at 512 while the transform
    treats it as a lower bound, and is caught here.

    Measured off the BUILT model, never asserted from the source.
    """
    model = MODULE.MyModel(3)
    fixed = model.transform.fixed_size
    assert fixed is not None, (
        "efficientdet_d0 is a fixed-resolution model; the transform has no "
        "fixed_size, so it would treat image_size as a lower bound and let a "
        "larger image through at its own size"
    )
    assert int(fixed[0]) == int(fixed[1]) == int(MODULE.image_size), (
        f"declared image_size={MODULE.image_size} but the transform runs at "
        f"{tuple(int(v) for v in fixed)}"
    )


# --- external oracles: the numbers, derived from outside this file ---------
#
# TRAP 31 (SELF-CONSISTENT-NUMBER). `yolox_s` shipped for review with an exact
# parameter count offered as proof the architecture was real. It WAS exact — for
# the model as built — while a wrong `expansion` default left the backbone
# ~1.15M parameters narrower than published YOLOX-S. Thirty guards and
# thirty-three mutations missed it, because all of them were checked against the
# same self-derived table.
#
# EfficientDet is where this matters most in this roster: the whole architecture
# is a claim about parameter efficiency, so a count is the headline number, and
# "3,887,485, and the published D0 is 3.9M" is exactly the reassuring-looking
# non-evidence the trap describes.
#
# Two independent sources, and nothing imported from `model_zoo`:
#
#   * `torchvision.models.efficientnet_b0(weights=None)` — a different
#     implementation of the same published B0 table. Its `features[:8]` is the
#     stem plus the seven MBConv stages, i.e. exactly this template's trunk
#     (`features[8]` is the final 1x1 to 1280, which a detector does not use).
#     The comparison is valid despite this template using GroupNorm: GroupNorm
#     and BatchNorm both hold one weight and one bias PER CHANNEL, so their
#     PARAMETER counts are identical — only BatchNorm's running statistics
#     differ, and those are buffers.
#   * hand arithmetic from the published D0 coefficients (W=64, 3 BiFPN
#     repeats, 3 head convs, 9 anchors per location) for the parts this
#     template writes.
#
# NOTE ON THE PUBLISHED HEADLINE FIGURE. EfficientDet-D0 is usually quoted at
# 3.9M parameters. This template is NOT directly comparable to that: the quoted
# figure is at 90 COCO classes with per-level BatchNorm in the heads, where this
# template shares one GroupNorm across levels. So the headline number is
# deliberately NOT asserted — a number that needs three caveats to match is not
# evidence. The two sources above are.

#: Published D0 compound-scaling coefficients. Transcribed, not read from the
#: template.
PUBLISHED_BIFPN_WIDTH = 64
PUBLISHED_BIFPN_REPEATS = 3
PUBLISHED_HEAD_CONVS = 3
PUBLISHED_ANCHORS_PER_LOCATION = 9
#: EfficientNet-B0's P3/P4/P5 tap widths, from the published stage table.
PUBLISHED_TAP_CHANNELS = (40, 112, 320)


def _separable_parameters(width):
    """A depthwise 3x3 (no bias) plus a pointwise 1x1 (with bias), at constant
    width — one conv "layer" in the BiFPN and the heads."""
    return width * 9 + width * width + width


def _group_norm_parameters(width):
    """GroupNorm holds one weight and one bias per channel — the same count as
    BatchNorm's affine parameters, which is what makes the B0 oracle valid."""
    return 2 * width


def _oracle_trunk_parameters():
    """torchvision's EfficientNet-B0 stem + seven MBConv stages."""
    from torchvision.models import efficientnet_b0

    return sum(p.numel() for p in efficientnet_b0(weights=None).features[:8].parameters())


def test_the_backbone_trunk_matches_torchvisions_efficientnet_b0():
    """MUTATION: a wrong expand ratio, width, repeat count or kernel anywhere in
    the B0 table — THE yolox_s CLASS OF BUG.

    A narrower trunk is internally consistent, trains, evaluates, and passes
    every identity and disjointness check in this file. Compared here against a
    library implementation of the same published table.
    """
    model = MODULE.MyModel(3)
    mine = sum(p.numel() for p in model.backbone.body.parameters())
    oracle = _oracle_trunk_parameters()
    assert mine == oracle == 3595388, (
        f"the EfficientNet-B0 trunk has {mine} parameters against "
        f"torchvision's {oracle} — a divergence here means the stage table, a "
        f"width, an expand ratio or a kernel size is wrong"
    )
    assert model.backbone.body.tap_channels == PUBLISHED_TAP_CHANNELS, (
        f"P3/P4/P5 tap widths {model.backbone.body.tap_channels}, published B0 "
        f"gives {PUBLISHED_TAP_CHANNELS}"
    )


def test_the_backbone_trunk_shapes_match_torchvisions_tensor_for_tensor():
    """The count above could in principle be hit by a compensating pair of
    errors. The shape multiset cannot."""
    from torchvision.models import efficientnet_b0

    model = MODULE.MyModel(3)

    def shapes(module):
        return sorted(tuple(p.shape) for p in module.parameters())

    mine = shapes(model.backbone.body)
    oracle = shapes(efficientnet_b0(weights=None).features[:8])
    assert len(mine) == len(oracle), (
        f"the trunk has {len(mine)} parameter tensors against torchvision's "
        f"{len(oracle)}"
    )
    assert mine == oracle, (
        "first divergence: "
        f"{next((a, b) for a, b in zip(mine, oracle) if a != b)}"
    )


def test_the_whole_model_reconciles_against_the_published_coefficients():
    """The end-to-end number, every term from outside this file.

    ``total = oracle trunk + laterals + extra levels + BiFPN + heads``, where
    everything after the trunk is hand arithmetic from the published D0
    coefficients. No fudge term — this is an exact equality.
    """
    num_classes = 4
    width = PUBLISHED_BIFPN_WIDTH
    anchors = PUBLISHED_ANCHORS_PER_LOCATION
    separable = _separable_parameters(width)

    # P3/P4/P5 projected to the BiFPN width: 1x1 conv (no bias) + norm.
    laterals = sum(tap * width + _group_norm_parameters(width) for tap in PUBLISHED_TAP_CHANNELS)
    # P6 and P7: strided 3x3 (no bias) + norm.
    extra = 2 * (width * width * 9 + _group_norm_parameters(width))
    # One BiFPN layer: 4 two-input top-down fusion weights, 4 top-down convs,
    # 3 three-input + 1 two-input bottom-up fusion weights, 4 bottom-up convs,
    # and 8 norms.
    per_layer = (
        4 * 2
        + 4 * separable
        + (3 * 3 + 2)
        + 4 * separable
        + 8 * _group_norm_parameters(width)
    )
    bifpn = PUBLISHED_BIFPN_REPEATS * per_layer
    # Shared class and box towers, then separable prediction convs.
    heads = (
        2 * PUBLISHED_HEAD_CONVS * separable
        + 2 * PUBLISHED_HEAD_CONVS * _group_norm_parameters(width)
        + (width * 9 + width * anchors * num_classes + anchors * num_classes)
        + (width * 9 + width * anchors * 4 + anchors * 4)
    )

    expected = _oracle_trunk_parameters() + laterals + extra + bifpn + heads
    measured = sum(p.numel() for p in MODULE.MyModel(num_classes - 1).parameters())
    assert measured == expected == 3851773, (
        f"the model has {measured} parameters; the published D0 coefficients "
        f"over torchvision's B0 trunk give {expected} "
        f"(trunk {_oracle_trunk_parameters()} + laterals {laterals} + extra "
        f"{extra} + bifpn {bifpn} + heads {heads})"
    )


def test_the_head_parameter_count_scales_only_with_the_class_head():
    """MUTATION: a dense 3x3 prediction conv instead of a separable one.

    Shape-identical, and at 90 classes it is 467,370 parameters against 52,506.
    The published class_net is separable, so the count must grow by exactly
    ``width * anchors * delta + anchors * delta`` per added class — a dense conv
    grows nine times faster.
    """
    width = PUBLISHED_BIFPN_WIDTH
    anchors = PUBLISHED_ANCHORS_PER_LOCATION
    small = sum(p.numel() for p in MODULE.MyModel(3).parameters())
    large = sum(p.numel() for p in MODULE.MyModel(13).parameters())
    added_classes = 13 - 3
    expected_growth = added_classes * (width * anchors + anchors)
    assert large - small == expected_growth, (
        f"adding {added_classes} classes added {large - small} parameters; a "
        f"separable prediction conv adds {expected_growth}, a dense 3x3 would "
        f"add {added_classes * (width * anchors * 9 + anchors)}"
    )


def test_no_trainable_parameter_is_unreachable_by_the_loss():
    """MUTATION: build a module and never call it — a BiFPN repeat left out of
    the loop, a head tower applied to only the first level.

    ``requires_grad``-aware on purpose: ``p.grad is None`` alone false-flags a
    deliberately frozen stem. The defect is a TRAINABLE parameter the loss never
    reaches, and this template has no frozen parameters at all, so the expected
    count is zero.
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


def test_the_between_thresholds_band_is_excluded_from_the_loss():
    """MUTATION: treat ``BETWEEN_THRESHOLDS`` (-2) anchors as negatives.

    EfficientDet assigns at 0.5/0.4 with the band between the two ignored;
    counting it as background trains the model to suppress the anchors it is
    least sure about. Both models are given the same everything, so the only
    difference is the mask — and the losses must differ, which they cannot if
    the band is empty. So the fixture is checked for a non-empty band first,
    which is what stops this passing vacuously.
    """
    from torchvision.models.detection import _utils as det_utils

    torch.manual_seed(0)
    head = MODULE._EfficientDetHead(8, 9, 4, num_convs=1)
    anchors = [torch.tensor([[0.0, 0.0, 10.0, 10.0]] * 4)]
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])}]
    matched = [torch.tensor([0, -1, det_utils.Matcher.BETWEEN_THRESHOLDS, -1])]
    assert int((matched[0] == det_utils.Matcher.BETWEEN_THRESHOLDS).sum()) > 0, (
        "the fixture has no BETWEEN_THRESHOLDS anchor, so this test could not "
        "distinguish the two behaviours"
    )
    head_outputs = {
        "cls_logits": torch.zeros(1, 4, 4),
        "bbox_regression": torch.zeros(1, 4, 4),
    }
    with_band_excluded = float(
        head.compute_loss(targets, head_outputs, anchors, matched)["classification"]
    )
    # The same anchor relabelled a plain negative (-1). If the band were being
    # counted as background, these two calls would agree exactly.
    all_negative = [torch.tensor([0, -1, -1, -1])]
    with_band_as_negative = float(
        head.compute_loss(targets, head_outputs, anchors, all_negative)["classification"]
    )
    assert with_band_excluded != pytest.approx(with_band_as_negative, rel=1e-9), (
        "excluding the BETWEEN_THRESHOLDS anchor gave the same classification "
        "loss as counting it a negative — the band is not being excluded"
    )
