"""Guards for ``object_detection/pytorch/yolov10_s.py``, each proven able to go
red by a mutation that is kept in the suite.

Why this file exists
--------------------
``tests/test_od_torchvision_family_train_step.py`` proves a template returns a
loss dict and a ``List[Dict]`` of xyxy predictions. For a template that wraps a
torchvision builder that is a real assertion: the loss is the library's. For
``yolov10_s.py`` the backbone, neck, dual head, both assigners and all six
losses are **our own code**, so "returns a loss dict" proves only that our code
returns a dict.

THE ONE THAT MATTERS MOST, STATED FIRST
---------------------------------------
The one2one head is fed ``feature.detach()``. Delete that ``.detach()`` and:

* the model constructs, and every shape is unchanged;
* the parameter count is **identical** — a detach is not a parameter, so the
  published-architecture guard cannot see it;
* the same six finite loss keys come back;
* the losses still fall, and the template still overfits one object past every
  threshold in ``guard_overfits_a_single_object``;
* the family train-step test stays green.

Nothing about the *value* of anything is wrong; the **gradient graph** is. So
``guard_one2one_head_is_detached_from_the_backbone`` measures the gradient graph
— it backpropagates the one2one losses ALONE and requires every backbone and
neck parameter to come back with no gradient or an exactly-zero one, then
backpropagates the one2many losses alone and requires the trunk gradient to be
non-zero, so "isolated" cannot be satisfied by a model where nothing trains.
Both halves are needed: the first alone is satisfied by a detached *everything*.

The rest of the silent surface
------------------------------
* ``SCDown`` with the stride on its pointwise conv instead of its depthwise
  one. Same output shape, **same parameter count** (a 1x1's count does not
  depend on its stride), and the block degenerates to point-sampling;
* ``Attention``'s head count, which is **parameter-invariant**: ``num_heads *
  int((dim / num_heads) * 0.5) == dim * 0.5`` for any head count dividing
  ``dim``, so a hardcoded 8 changes no parameter, no shape and no published
  figure. The textbook "constant that reaches nothing";
* ``PSA`` applying attention to both halves, or dropping either residual —
  identical shapes, identical parameters, trains;
* ``RepVGGDW`` activating each branch before summing, which is identical in
  count and shape and is no longer re-parameterisable;
* the two heads collapsing onto ONE assignment (``ONE2ONE_TOPK`` replaced by
  ``TAL_TOPK``), which trains happily and produces a model that still needs the
  NMS this architecture exists to remove;
* eval decoding the **one2many** branch, whose ten positives per object are
  exactly what needs suppressing — duplicate-heavy output, healthy losses;
* ``C2f`` fanning out as a ``C3``; the assigner matching nothing, or the wrong
  anchors; the DFL decode taking an argmax; predictions never mapped back to
  original image coordinates.

None of those fail a train step. So each is a named guard here, and each guard
is paired with a **mutation** — an exact textual edit to the shipped template
that the guard must catch. ``_mutate`` asserts its anchor appears exactly once,
so a mutation that no longer applies is a RED, not a survivor reported as
"passed"; and ``test_no_mutation_baseline`` runs the whole guard table against
the unmutated file so a sweep always carries its own zero row.

Three traps this file is shaped around
--------------------------------------
**The eval path is vacuous on a fresh model.** The DFL head's class prior is
``log(5 / nc / (640 / stride) ** 2)`` — roughly ``sigmoid(-9)`` — so a freshly
built model's detections are indistinguishable from noise. And this template
cannot even be rescued by a score threshold, because ``SCORE_THRESH`` is 0.0 by
design: it returns a full ``max_det`` ranked candidates whatever they are, so
``model(images)`` at initialisation returns 300 boxes of noise and "it returned
boxes" is worth nothing at all. ``guard_decode_is_per_image_and_aligned``
therefore drives ``_predictions`` **directly**, with synthetic head outputs that
name which anchor is confident, **at batch two** — a per-image bug is invisible
at batch one by construction — plus a second fixture where the *background*
channel is the strongest, which the first cannot see.

**Cardinality is invariant to cost.** Asserting *how many* anchors an assigner
selects proves nothing about the metric that ranks them: a swapped focal
``alpha`` hid in a sibling template through a full mutation sweep because every
assertion counted proposals. So the assigner guards here assert **which**
anchor or ground truth is selected, and with what soft target — and each one
first asserts that its fixture can distinguish the two answers.

**A self-measured number is not evidence.** ``_PINNED_TOTALS`` is a tripwire,
labelled as one. The parameter count is asserted against
``_reference_parameters``, derived from the published architecture with nothing
from ``model_zoo/`` imported, and anchored to figures from outside this repo —
see ``_PUBLISHED``. That reference is checked at **two published scales** AND
**per yaml layer** (``_REFERENCE_LAYERS``), so a disagreement names the layer
that drifted rather than only a total; and the second scale is re-measured on
the BUILT model by rebuilding this module with YOLOv10-N's table.

⚠️ ONE GUARD THIS FILE DELIBERATELY DOES NOT CONTAIN
-----------------------------------------------------
There is no guard asserting the decode's two-stage top-k differs from a single
flat top-k over all ``(anchor, class)`` pairs, because **it does not**. If pair
``(i, c)`` scores ``s`` then anchor ``i``'s maximum is at least ``s``, so anchor
``i`` can only miss the top ``max_det`` anchors when ``max_det`` anchors score
above ``s`` — each contributing a better pair — and ``(i, c)`` is outside the
top ``max_det`` pairs either way. Measured as well as argued: 4000 randomised
``(anchors, classes, max_det)`` fixtures, zero differences in the returned score
multiset. The staging is an efficiency factorisation.

A guard on it would pass for the wrong reason and read as coverage of the
decode, which is why the absence is documented rather than left to be
rediscovered. What IS guarded is the property that genuinely distinguishes this
decode from its two NMS siblings: ``guard_decode_is_nms_free`` feeds several
anchors that decode to the same box at the same class and requires all of them
back, where any NMS collapses them to one.
"""

import importlib.util
import pathlib
import re
import tempfile

import pytest

ROOT = pathlib.Path(__file__).parent.parent
OD_PYTORCH = ROOT / "model_zoo" / "object_detection" / "pytorch"
TEMPLATE = OD_PYTORCH / "yolov10_s.py"

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="pytorch not installed in this CI job",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("torchvision") is None,
        reason="torchvision not installed in this CI job",
    ),
]


# --------------------------------------------------------------------------
# loading, mutating
# --------------------------------------------------------------------------


def _exec_source(source: str, stem: str):
    """Import ``source`` as a fresh module.

    Written to a real file rather than ``exec``-ed into a dict so tracebacks
    point somewhere, and so the module behaves exactly as the uploaded template
    does — a template is a file, and the SDK loads it as one.
    """
    with tempfile.TemporaryDirectory(prefix="tb-mutate-") as raw:
        target = pathlib.Path(raw) / f"{stem}.py"
        target.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(stem, target)
        assert spec and spec.loader, f"cannot build a spec for {target}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Kept so a guard that needs a SECOND, independently-configured copy of
        # the module under test can re-exec the same source. Reading the file
        # again would silently hand the guard the PRISTINE template while the
        # mutation sweep believed it was testing a mutated one — a guard pointed
        # at unmutated code reports "passed", which is indistinguishable from a
        # survivor and reads as the opposite of what it means.
        module.__tb_source__ = source
        return module


def _load():
    return _exec_source(TEMPLATE.read_text(encoding="utf-8"), TEMPLATE.stem)


def _reload(module):
    """A fresh copy of whatever source ``module`` was built from.

    Used by the guards that rebuild at another published scale: they mutate
    module-level tables, so they need their own copy — and it must be the same
    source, mutated or not.
    """
    source = getattr(module, "__tb_source__", None)
    assert source, (
        f"{module.__name__}: no recorded source to re-exec. A guard rebuilding "
        f"at another scale must not fall back to reading the template file, or "
        f"it would test pristine code under a mutation."
    )
    return _exec_source(source, f"reloaded_{module.__name__}")


def _mutate(anchor: str, replacement: str):
    """Load the template with ``anchor`` replaced by ``replacement``.

    THE ANCHOR MUST APPEAR EXACTLY ONCE, and that assertion is the whole point.
    A patch whose anchor has drifted silently changes nothing, the guard is then
    pointed at a pristine template, and the mutation reports "passed" — which is
    indistinguishable from a genuine survivor and reads as the opposite of what
    it means. So a drifted anchor fails here instead.
    """
    source = TEMPLATE.read_text(encoding="utf-8")
    occurrences = source.count(anchor)
    assert occurrences == 1, (
        f"{TEMPLATE.name}: mutation anchor occurs {occurrences} times, expected "
        f"1. The template moved under the mutation; re-anchor it rather than "
        f"loosening this check.\nanchor:\n{anchor}"
    )
    mutated = source.replace(anchor, replacement)
    assert mutated != source, "the replacement is identical to the anchor"
    return _exec_source(mutated, f"mutated_{TEMPLATE.stem}")


def _build(module, num_classes: int, input_size=None):
    entry_name = getattr(module, "main_class", None) or getattr(
        module, "main_method", None
    )
    assert entry_name, f"{module.__name__}: no main_class / main_method"
    entry = getattr(module, entry_name)
    if input_size is None:
        return entry(num_classes)
    return entry(num_classes, input_size)


def _apply_scale(module, scale) -> None:
    """Point a loaded template at another published scale.

    ⚠️ A SCALE IS NOT ONLY MULTIPLIERS. The v10 yamls differ in a *block kind*
    as well: ``yolov10n`` puts a plain ``C2f`` at the stride-32 backbone stage
    where ``yolov10s`` puts a ``C2fCIB``. Setting the multipliers alone
    reproduces neither scale, which is what makes the rebuild in
    ``guard_architecture_table_is_a_live_knob`` a real second measurement.
    """
    module.WIDTH_MULT = scale["width"]
    module.DEPTH_MULT = scale["depth"]
    module.MAX_CHANNELS = scale["max_channels"]
    module.BACKBONE_P5_BLOCK = scale["p5_block"]


# --------------------------------------------------------------------------
# the published architecture, re-derived
# --------------------------------------------------------------------------
#
# WHY THIS IS RE-DERIVED RATHER THAN MEASURED.
#
# A parameter count taken off the model under test can only ever prove the code
# is consistent with itself. That is not a theoretical caveat: `yolox_s` shipped
# for review citing a self-measured 7,788,886 parameters as proof "the design is
# real" while a wrong `expansion` default left the whole backbone and neck about
# 1.15M parameters too narrow. Thirty guards and thirty-three mutations missed it
# because every one of them shared the same self-derived table.
#
# So the reference below is arithmetic on (in, out, kernel, groups), transcribed
# from `ultralytics/cfg/models/v10/yolov10{n,s}.yaml` and
# `ultralytics/nn/modules/{block,conv,head}.py`, with NOTHING under `model_zoo/`
# imported. And the transcription is itself anchored to figures from outside
# this repo, so it cannot drift into agreeing with a wrong template.

#: Published parameter totals, from the model summary Ultralytics prints.
#: ``(total, gradients)``. TWO scales, so the anchor pins the width multiplier,
#: the depth multiplier AND the stride-32 block kind rather than one arithmetic
#: total:
#:
#:   YOLOv10n summary: 385 layers, 2775520 parameters, 2775504 gradients, 8.7 GFLOPs
#:   YOLOv10s summary: 402 layers, 8128272 parameters, 8128256 gradients
#:
#: The n line is quoted verbatim from THU-MIG/yolov10 issue #226; both were
#: reproduced independently against the THU-MIG fork at commit 453c6e38 and
#: against ultralytics==8.4.138, which agree to the parameter.
#:
#: ⚠️ THESE ARE THE **DUAL-HEAD, UNFUSED** FIGURES, and picking the wrong row of
#: the three that exist per scale is the single easiest way to conclude this
#: template is 20% too big. Also measured upstream:
#:
#:   dual head, fused:      2,762,608 / 8,096,880
#:   one2one only, fused:   2,299,264 / 7,248,960   <- the docs' "2.3M" / "7.2M"
#:
#: The docs table quotes the LAST of those. Its own footnote says so — "Params
#: and FLOPs values are for the fused model after model.fuse(), which merges
#: Conv and BatchNorm layers and removes the auxiliary one-to-many detection
#: head" — and THU-MIG/yolov10#13's author confirms it ("the one-to-many head is
#: not needed during inference, this part of params and FLOPs can be ignored");
#: their flops.py deletes cv2/cv3 before measuring. The FLOPs corroborate
#: independently: the published 6.7G / 21.6G match the one2one-only graph
#: (6.84 / 21.86) and not the dual-head one (8.74 / 25.11).
#:
#: This template anchors to the dual-head unfused row because that is the graph
#: it BUILDS: both heads are in the module tree, both are shipped and averaged
#: every federated round, and GroupNorm does not fuse into a conv the way BN
#: does — so neither fused figure describes anything this repo produces.
_PUBLISHED = {
    "yolov10n": (2_775_520, 2_775_504),
    "yolov10s": (8_128_272, 8_128_256),
}
#: Parameters upstream stores for the DFL bin vector and this template does not:
#: a frozen ``Conv2d`` weight, ``requires_grad=False``, hence "16 parameters, 0
#: gradients". This template builds the bins with ``torch.arange`` in the decode
#: and stores nothing, so ITS TOTAL IS THE PUBLISHED GRADIENT COUNT exactly.
_DFL_PROJECTION_CONSTANTS = 16
#: The class count the published figures are quoted at.
_PUBLISHED_CLASSES = 80
#: The scale this template ships.
_SHIPPED_SCALE = "yolov10s"

#: ⚠️ FIGURES THAT ARE **NOT** ANCHORS, recorded so nobody corroborates against
#: one. Almost every YOLOv10 summary line in the wild is a custom-class
#: fine-tune, with an "Overriding model.yaml nc=80 with nc=<N>" line above it
#: that is easy to miss. These two are known non-COCO and are asserted below to
#: be reproducible ONLY at their own class counts.
_NOT_COCO_FIGURES = {1: 2_707_430, 3: 2_708_210}

#: Parameters this head gains per additional class, at any ``nc <= 100``.
#: ``cls_hidden`` is ``max(128, min(nc, 100)) = 128`` there, so each extra class
#: adds one 1x1 row plus a bias on three levels of TWO branches: 3 * 129 * 2.
#: The cheap independent check on any figure found anywhere.
_PARAMS_PER_CLASS = 774

#: The published scales, transcribed from the yaml ``scales:`` block plus the
#: one block-kind difference between the n and s files.
_SCALES = {
    "yolov10n": {
        "width": 0.25,
        "depth": 0.33,
        "max_channels": 1024,
        # yolov10n.yaml layer 8: `[-1, 3, C2f, [1024, True]]`
        "p5_block": "c2f",
    },
    "yolov10s": {
        "width": 0.50,
        "depth": 0.33,
        "max_channels": 1024,
        # yolov10s.yaml layer 8: `[-1, 3, C2fCIB, [1024, True, True]]` — the
        # ONLY line on which the two yamls differ.
        "p5_block": "c2f_cib",
    },
}


def _conv(cin, cout, kernel, groups=1, bias=False):
    return (cin // groups) * cout * kernel * kernel + (cout if bias else 0)


def _norm(channels):
    """An affine normalisation layer: one scale and one shift per channel.

    GroupNorm and BatchNorm are IDENTICAL here, which is what makes comparing a
    GroupNorm build against a published BatchNorm count legitimate — see the
    federated note in the template. What differs is the BUFFERS, and those are
    pinned separately in ``guard_no_stateful_normalisation``.
    """
    return 2 * channels


def _cna(cin, cout, kernel, groups=1):
    """conv -> norm (affine); the activation has no parameters."""
    return _conv(cin, cout, kernel, groups) + _norm(cout)


def _bottleneck(channels):
    """BOTH convs 3x3, at FULL branch width (upstream e=1.0, k=(3, 3))."""
    return _cna(channels, channels, 3) + _cna(channels, channels, 3)


def _repvggdw(channels):
    """Depthwise 7x7 + depthwise 3x3, summed before one activation."""
    return _cna(channels, channels, 7, groups=channels) + _cna(
        channels, channels, 3, groups=channels
    )


def _cib(cin, cout, large_kernel, expansion=1.0):
    """Compact inverted block: dw3 -> pw expand -> mixer -> pw project -> dw3.

    ``expansion`` is 1.0 because ``C2fCIB`` calls it at the already-halved
    branch width, so the inverted expansion runs at TWICE that.
    """
    hidden = int(cout * expansion)
    expanded = 2 * hidden
    mixer = (
        _repvggdw(expanded)
        if large_kernel
        else _cna(expanded, expanded, 3, groups=expanded)
    )
    return (
        _cna(cin, cin, 3, groups=cin)
        + _cna(cin, expanded, 1)
        + mixer
        + _cna(expanded, cout, 1)
        + _cna(cout, cout, 3, groups=cout)
    )


def _c2f_shell(cin, cout, blocks, block):
    half = int(cout * 0.5)
    # (2 + n), not 2: C2f fuses EVERY intermediate block output.
    return _cna(cin, 2 * half, 1) + _cna((2 + blocks) * half, cout, 1) + blocks * block(
        half
    )


def _c2f(cin, cout, blocks):
    return _c2f_shell(cin, cout, blocks, _bottleneck)


def _c2f_cib(cin, cout, blocks, large_kernel=True):
    return _c2f_shell(cin, cout, blocks, lambda h: _cib(h, h, large_kernel))


def _sppf(channels, repeats=3):
    """One shared max-pool, so only the two 1x1s carry parameters."""
    half = channels // 2
    return _cna(channels, half, 1) + _cna(half * (repeats + 1), channels, 1)


def _scdown(cin, cout, kernel=3):
    """Pointwise 1x1 at full resolution, then a DEPTHWISE strided kxk."""
    return _cna(cin, cout, 1) + _cna(cout, cout, kernel, groups=cout)


def _attention(dim, head_dim=64, attn_ratio=0.5):
    """``qkv`` / ``proj`` / depthwise ``pe``.

    ⚠️ PARAMETER-INVARIANT IN ``num_heads``: ``num_heads * int((dim /
    num_heads) * attn_ratio)`` is ``dim * attn_ratio`` for any head count
    dividing ``dim``, so ``qkv``'s width — and every parameter here — is the
    same whatever the head count is. That is exactly why the head count needs a
    guard the parameter comparison cannot provide.
    """
    num_heads = max(1, dim // head_dim)
    key_dim = int((dim // num_heads) * attn_ratio)
    qkv_channels = dim + 2 * key_dim * num_heads
    return (
        _cna(dim, qkv_channels, 1)
        + _cna(dim, dim, 1)
        + _cna(dim, dim, 3, groups=dim)
    )


def _psa(channels, ratio=0.5):
    """Half the channels attend; the other half bypasses."""
    inner = int(channels * ratio)
    return (
        _cna(channels, 2 * inner, 1)
        + _cna(2 * inner, channels, 1)
        + _attention(inner)
        + _cna(inner, 2 * inner, 1)
        + _cna(2 * inner, inner, 1)
    )


def _v10_detect(head_channels, class_channels, reg_max=16, dual=True):
    """The dual head.

    ⚠️ The class tower is NOT YOLOv8's two dense 3x3 convs — v10 replaces them
    with two DEPTHWISE-SEPARABLE pairs, which is most of the head's saving. And
    the one2one branch is a full copy of both towers and both predictors
    (upstream's ``copy.deepcopy``), so the whole head counts TWICE.
    """
    box_hidden = max(16, head_channels[0] // 4, reg_max * 4)
    cls_hidden = max(head_channels[0], min(class_channels, 100))
    branch = 0
    for channels in head_channels:
        branch += _cna(channels, box_hidden, 3) + _cna(box_hidden, box_hidden, 3)
        branch += _conv(box_hidden, 4 * reg_max, 1, bias=True)
        branch += _cna(channels, channels, 3, groups=channels)
        branch += _cna(channels, cls_hidden, 1)
        branch += _cna(cls_hidden, cls_hidden, 3, groups=cls_hidden)
        branch += _cna(cls_hidden, cls_hidden, 1)
        branch += _conv(cls_hidden, class_channels, 1, bias=True)
    return branch * (2 if dual else 1)


def _widths(scale):
    """The five stage widths, width-scaled the way the yaml is parsed."""
    import math as _math

    def scale_width(channels):
        scaled = min(channels, scale["max_channels"]) * scale["width"]
        return max(8, int(_math.ceil(scaled / 8)) * 8)

    return tuple(scale_width(c) for c in (64, 128, 256, 512, 1024))


def _depths(scale):
    def scale_depth(blocks):
        return max(int(round(blocks * scale["depth"])), 1)

    return scale_depth(3), scale_depth(6), scale_depth(3)


def _reference_layers(class_channels, scale, reg_max=16, dual=True):
    """Per-YAML-LAYER parameter counts, derived from the published spec alone.

    Keyed by the yaml's own layer index, so a disagreement says WHICH layer
    drifted instead of only that a total did. Every one of these 18 numbers was
    cross-checked against a layer-by-layer measurement of upstream at both
    published scales — 36 comparisons, zero mismatches — which is a far stronger
    check on the transcription than the two totals alone.
    """
    stem, p2, p3, p4, p5 = _widths(scale)
    shallow, deep, neck = _depths(scale)
    p5_cib = scale["p5_block"] == "c2f_cib"
    return {
        0: _cna(3, stem, 3),
        1: _cna(stem, p2, 3),
        2: _c2f(p2, p2, shallow),
        3: _cna(p2, p3, 3),
        4: _c2f(p3, p3, deep),
        5: _scdown(p3, p4),
        6: _c2f(p4, p4, deep),
        7: _scdown(p4, p5),
        8: _c2f_cib(p5, p5, shallow) if p5_cib else _c2f(p5, p5, shallow),
        9: _sppf(p5),
        10: _psa(p5),
        13: _c2f(p5 + p4, p4, neck),
        16: _c2f(p4 + p3, p3, neck),
        17: _cna(p3, p3, 3),
        19: _c2f(p3 + p4, p4, neck),
        20: _scdown(p4, p4),
        22: _c2f_cib(p4 + p5, p5, neck),
        23: _v10_detect((p3, p4, p5), class_channels, reg_max, dual),
    }


def _reference_parameters(class_channels, scale, reg_max=16, dual=True):
    return sum(_reference_layers(class_channels, scale, reg_max, dual).values())


#: Published per-layer structure at the shipped scale, independent of the total:
#: it says WHAT drifted when the count disagrees. Measured against upstream.
_REFERENCE_LAYERS_SHIPPED = {
    0: 928,
    1: 18_560,
    2: 29_056,
    3: 73_984,
    4: 197_632,
    5: 36_096,
    6: 788_480,
    7: 137_728,
    8: 958_464,
    9: 656_896,
    10: 990_976,
    13: 591_360,
    16: 148_224,
    17: 147_712,
    19: 493_056,
    20: 68_864,
    22: 1_089_536,
    # 1,700,720 upstream, of which 16 are the stored DFL projection.
    23: 1_700_704,
}

#: Published per-stage structure at the shipped scale.
_REFERENCE_STRUCTURE = {
    "backbone_out": (128, 256, 512),
    # YOLOv10 keeps the backbone's widths at the head — unlike RTMDet, which
    # projects all three levels to a common width.
    "neck_out": (128, 256, 512),
    "backbone_blocks": (1, 2, 2, 1),
    "neck_blocks": (1, 1, 1, 1),
    # The stride-32 backbone stage is the ONE line the n and s yamls disagree
    # on, so its TYPE is pinned rather than only its block count.
    "backbone_stage_kinds": ("C2f", "C2f", "C2f", "C2fCIB"),
    # The two DEEPEST backbone transitions are SCDown, the two shallowest a
    # plain strided 3x3; in the neck it is the stride-32 one only.
    "backbone_downsample_kinds": ("ConvNormAct", "ConvNormAct", "SCDown", "SCDown"),
    "neck_downsample_kinds": ("ConvNormAct", "SCDown"),
    "psa_hidden": 256,
    "attention_heads": 4,
    "attention_head_dim": 64,
    "attention_key_dim": 32,
    "box_hidden": 64,
    "cls_hidden": 128,
    "strides": (8, 16, 32),
    "reg_max": 16,
}


def test_the_reference_derivation_matches_the_published_figures() -> None:
    """The transcription, checked against the numbers it is transcribed from.

    Runs before anything is built, and needs no torch: if this fails, the
    reference is wrong and every comparison against it is worthless. TWO scales
    are pinned, so the check covers the width multiplier, the depth multiplier
    AND the stride-32 block kind rather than one arithmetic accident. The DFL
    gap is asserted as an exact constant rather than hidden in a tolerance.
    """
    assert _SHIPPED_SCALE in _PUBLISHED, "the shipped scale must be anchored"
    assert len(_PUBLISHED) >= 2, (
        "one published figure cannot distinguish a wrong table from a wrong "
        "transcription of it — that is the whole reason this dict has more than "
        "one row"
    )
    for scale, (total, gradients) in sorted(_PUBLISHED.items()):
        assert total - gradients == _DFL_PROJECTION_CONSTANTS, (
            f"published {scale}: total {total:,} minus gradients "
            f"{gradients:,} is {total - gradients}, not the "
            f"{_DFL_PROJECTION_CONSTANTS} frozen DFL projection constants. One "
            f"of the two figures is mis-transcribed."
        )
        derived = _reference_parameters(_PUBLISHED_CLASSES, _SCALES[scale])
        assert derived == gradients, (
            f"the spec transcription derives {derived:,} parameters for {scale} "
            f"at {_PUBLISHED_CLASSES} classes, but the published summary "
            f"reports {gradients:,} gradients ({total:,} total, of which "
            f"{_DFL_PROJECTION_CONSTANTS} are the frozen DFL projection this "
            f"template does not store) — off by {derived - gradients:+,}. Fix "
            f"the transcription against the yaml before trusting any comparison "
            f"that uses it."
        )


def test_the_reference_matches_the_published_figures_per_layer() -> None:
    """The same claim, per yaml layer.

    Two totals agreeing is a much weaker statement than eighteen layers
    agreeing: a pair of compensating errors survives the total and cannot
    survive this. It is also what makes a failure diagnosable — the message
    names the layer.
    """
    derived = _reference_layers(_PUBLISHED_CLASSES, _SCALES[_SHIPPED_SCALE])
    assert set(derived) == set(_REFERENCE_LAYERS_SHIPPED), (
        f"layer index sets differ: derived {sorted(derived)} against pinned "
        f"{sorted(_REFERENCE_LAYERS_SHIPPED)}"
    )
    wrong = {
        index: (derived[index], _REFERENCE_LAYERS_SHIPPED[index])
        for index in derived
        if derived[index] != _REFERENCE_LAYERS_SHIPPED[index]
    }
    assert not wrong, (
        "the transcription disagrees with the measured upstream layer table at "
        + ", ".join(
            f"layer {i} (derived {d:,}, upstream {u:,}, off by {d - u:+,})"
            for i, (d, u) in sorted(wrong.items())
        )
    )


def test_the_two_scales_are_actually_different() -> None:
    """Guard the guard above: two rows that happened to describe the same
    architecture would read as a two-way anchor and be a one-way one."""
    totals = {
        scale: _reference_parameters(_PUBLISHED_CLASSES, spec)
        for scale, spec in _SCALES.items()
    }
    assert len(set(totals.values())) == len(totals), (
        f"the anchored scales derive the same parameter count: {totals}"
    )
    kinds = {spec["p5_block"] for spec in _SCALES.values()}
    assert kinds == {"c2f", "c2f_cib"}, (
        f"the anchored scales must cover BOTH stride-32 block kinds, got "
        f"{sorted(kinds)} — otherwise the C2fCIB/C2f distinction (the one line "
        f"on which the two published yamls differ) is never checked against a "
        f"published figure"
    )
    widths = {spec["width"] for spec in _SCALES.values()}
    assert len(widths) == len(_SCALES), (
        f"the anchored scales must differ in width, got {sorted(widths)}"
    )


def test_the_block_kind_is_not_absorbed_by_the_multipliers() -> None:
    """The stride-32 block kind must be independently load-bearing.

    If YOLOv10-N's multipliers reproduced N's published count with EITHER block
    kind, then ``p5_block`` would be decoration and the cross-scale rebuild
    would prove only that the multipliers work. Measured: the wrong kind is
    210,432 parameters out, so the kind is genuinely part of the scale.
    """
    scale = dict(_SCALES["yolov10n"])
    right = _reference_parameters(_PUBLISHED_CLASSES, scale)
    scale["p5_block"] = "c2f_cib"
    wrong = _reference_parameters(_PUBLISHED_CLASSES, scale)
    assert right == _PUBLISHED["yolov10n"][1]
    assert wrong != right, (
        "the stride-32 block kind changes no parameter at the YOLOv10-N "
        "multipliers, so it is decoration and `_apply_scale` is not really "
        "selecting a scale"
    )


def test_the_head_is_linear_in_the_class_count() -> None:
    """The +774/class slope, and the two non-COCO figures it disqualifies.

    This is the cheap independent check on any parameter figure found in the
    wild, and it is why the two known custom-``nc`` figures in
    ``_NOT_COCO_FIGURES`` are recorded rather than left as plausible anchors: a
    reviewer corroborating against ``2,707,430`` would be corroborating against
    a one-class fine-tune.
    """
    scale = _SCALES[_SHIPPED_SCALE]
    base = _reference_parameters(_PUBLISHED_CLASSES, scale)
    step = _reference_parameters(_PUBLISHED_CLASSES + 1, scale) - base
    assert step == _PARAMS_PER_CLASS, (
        f"this head gains {step} parameters per class, not the "
        f"{_PARAMS_PER_CLASS} recorded. cls_hidden is capped at "
        f"max(128, min(nc, 100)) = 128 for nc <= 100, so the slope is "
        f"3 levels * (128 + 1 bias) * 2 branches. If it moved, either the head "
        f"width or the dual-branch duplication changed."
    )
    narrow = _SCALES["yolov10n"]
    for classes, figure in sorted(_NOT_COCO_FIGURES.items()):
        assert _reference_parameters(_PUBLISHED_CLASSES, narrow) != figure, (
            f"{figure:,} is reproducible at {_PUBLISHED_CLASSES} classes, but "
            f"it is recorded as an nc={classes} fine-tune — one of the two is "
            f"wrong and the anchors need re-checking"
        )
        assert _reference_parameters(classes, narrow) + (
            _DFL_PROJECTION_CONSTANTS
        ) == figure, (
            f"{figure:,} is recorded as YOLOv10-N at nc={classes}, but the "
            f"derivation gives "
            f"{_reference_parameters(classes, narrow) + _DFL_PROJECTION_CONSTANTS:,} "
            f"there. Either the note is wrong or the derivation is."
        )


# --------------------------------------------------------------------------
# structure guards
# --------------------------------------------------------------------------


def _backbone_blocks(model):
    return tuple(len(stage.m) for stage in model.backbone.stages)


def _neck_blocks(model):
    neck = model.neck
    return tuple(
        len(getattr(neck, name).m) for name in ("td_p4", "td_p3", "bu_p4", "bu_p5")
    )


def guard_matches_the_published_architecture(module) -> None:
    """The built module tree must match the PUBLISHED architecture, re-derived.

    The independent half of the evidence. ``module_tree_size`` pins the totals
    this repo measured, so it can only catch a regression away from whatever was
    shipped; this one re-computes the count from the published spec and
    compares, so it catches shipping the wrong architecture in the first place —
    which is what happened on a sibling template.
    """
    class_channels = module.output_classes + 1  # the deliberate label-space +1
    expected = _reference_parameters(class_channels, _SCALES[_SHIPPED_SCALE])
    model = _build(module, module.output_classes)
    actual = sum(p.numel() for p in model.parameters())

    assert actual == expected, (
        f"{module.__name__}: built model has {actual:,} parameters; YOLOv10-S "
        f"re-derived from its published spec has {expected:,} at the same "
        f"{class_channels} class channels — a difference of "
        f"{actual - expected:+,}. Something in the multipliers, the block "
        f"kinds, the kernel sizes, the depthwise groupings, the attention "
        f"shape or the dual head does not match the design this template "
        f"claims to implement. This is the check a parameter count measured off "
        f"the model itself CANNOT make.\n"
        f"If the gap is a multiple of {_PARAMS_PER_CLASS}, it is a CLASS COUNT "
        f"problem, not an architecture one: the head is +{_PARAMS_PER_CLASS} "
        f"per class and this template deliberately allocates output_classes + 1 "
        f"channels for the family handler's background index."
    )

    reference = _REFERENCE_STRUCTURE
    assert tuple(model.backbone.out_channels) == reference["backbone_out"], (
        f"{module.__name__}: backbone emits {tuple(model.backbone.out_channels)} "
        f"channels, published design has {reference['backbone_out']}"
    )
    assert tuple(model.neck.out_channels) == reference["neck_out"], (
        f"{module.__name__}: neck emits {tuple(model.neck.out_channels)} "
        f"channels, published design has {reference['neck_out']}"
    )
    assert _backbone_blocks(model) == reference["backbone_blocks"], (
        f"{module.__name__}: backbone stages hold {_backbone_blocks(model)} "
        f"blocks, published design has {reference['backbone_blocks']}"
    )
    assert _neck_blocks(model) == reference["neck_blocks"], (
        f"{module.__name__}: neck stages hold {_neck_blocks(model)} blocks, "
        f"published design has {reference['neck_blocks']}"
    )

    kinds = tuple(type(stage).__name__ for stage in model.backbone.stages)
    assert kinds == reference["backbone_stage_kinds"], (
        f"{module.__name__}: backbone stages are {kinds}, published design has "
        f"{reference['backbone_stage_kinds']} — the stride-32 stage is a plain "
        f"C2f in yolov10n.yaml and a C2fCIB in yolov10s.yaml, and that one line "
        f"is the ONLY place the two published yamls differ"
    )
    downs = tuple(type(d).__name__ for d in model.backbone.downsamples)
    assert downs == reference["backbone_downsample_kinds"], (
        f"{module.__name__}: backbone transitions are {downs}, published design "
        f"has {reference['backbone_downsample_kinds']} — SCDown's saving scales "
        f"with in * out, so it is published only at the two WIDE transitions "
        f"and using it everywhere (or nowhere) misses the count"
    )
    neck_downs = (
        type(model.neck.bu_down3).__name__,
        type(model.neck.bu_down4).__name__,
    )
    assert neck_downs == reference["neck_downsample_kinds"], (
        f"{module.__name__}: the bottom-up transitions are {neck_downs}, "
        f"published design has {reference['neck_downsample_kinds']}"
    )
    assert type(model.neck.bu_p5).__name__ == "C2fCIB", (
        f"{module.__name__}: the bottom-up stride-32 fusion is "
        f"{type(model.neck.bu_p5).__name__}; every published v10 scale puts a "
        f"C2fCIB there"
    )

    attention = model.backbone.psa.attn
    assert model.backbone.psa.hidden == reference["psa_hidden"]
    assert attention.num_heads == reference["attention_heads"], (
        f"{module.__name__}: PSA's attention runs {attention.num_heads} heads, "
        f"published design has {reference['attention_heads']} "
        f"(hidden {model.backbone.psa.hidden} / "
        f"{reference['attention_head_dim']} per head)"
    )
    assert attention.head_dim == reference["attention_head_dim"]
    assert attention.key_dim == reference["attention_key_dim"]
    assert model.head.box_hidden == reference["box_hidden"]
    assert model.head.cls_hidden == reference["cls_hidden"]
    assert tuple(model.head.strides) == reference["strides"]
    assert model.head.reg_max == reference["reg_max"]


def guard_architecture_table_is_a_live_knob(module) -> None:
    """The scale table must REACH the built model — and the proof is a second
    published parameter count, measured on the rebuild.

    The failure mode this is written against is a declared table that is read
    once and then contradicted by a hardcoded literal deeper in the builder:
    the shipped scale still comes out right, the constant reads as
    configuration, and it is decoration.

    YOLOv10 makes this a two-part claim, which is why the rebuild is worth more
    here than on ``yolov8_s.py``. A v10 scale is multipliers **plus a block
    kind** — ``yolov10n`` puts a plain ``C2f`` at the stride-32 backbone stage
    where ``yolov10s`` puts a ``C2fCIB`` — so setting the multipliers alone
    reproduces neither published figure. The rebuild sets both and asserts the
    result carries YOLOv10-N's published gradient count exactly, and then
    asserts that reverting the block kind alone breaks it, so neither half can
    be decoration.
    """
    import torch

    shipped = _build(module, 3)
    shipped_total = sum(p.numel() for p in shipped.parameters())
    assert type(shipped.backbone.stages[-1]).__name__ == "C2fCIB", (
        f"{module.__name__}: the SHIPPED build's stride-32 stage is "
        f"{type(shipped.backbone.stages[-1]).__name__}, but YOLOv10-S publishes "
        f"a C2fCIB there — so the rebuild below would not be exercising a "
        f"different code path and this guard would be checking nothing"
    )

    scale = "yolov10n"
    rebuilt_module = _reload(module)
    _apply_scale(rebuilt_module, _SCALES[scale])
    try:
        model = _build(rebuilt_module, _PUBLISHED_CLASSES - 1)
    except Exception as error:  # noqa: BLE001 — any build failure is the bug
        raise AssertionError(
            f"{module.__name__}: rebuilding with the published {scale} scale "
            f"failed with {type(error).__name__}: {error}. The scale table is "
            f"meant to be the scale selector; a builder that only works for the "
            f"shipped table has hardcoded something it declares."
        ) from error

    actual = sum(p.numel() for p in model.parameters())
    expected = _PUBLISHED[scale][1]
    assert actual == expected, (
        f"{module.__name__}: rebuilt with the published {scale} scale the model "
        f"has {actual:,} parameters, but that scale's published summary reports "
        f"{_PUBLISHED[scale][0]:,} — i.e. {expected:,} once the "
        f"{_DFL_PROJECTION_CONSTANTS} frozen DFL projection constants this "
        f"template does not store are removed (off by {actual - expected:+,}). "
        f"Either the table does not reach the builder, or a literal deeper in "
        f"it overrides the table for every scale but the shipped one."
    )
    assert actual != shipped_total, (
        f"fixture is degenerate: the {scale} rebuild has the same parameter "
        f"count as the shipped build"
    )
    assert type(model.backbone.stages[-1]).__name__ == "C2f", (
        f"{module.__name__}: rebuilt at the {scale} scale the stride-32 stage "
        f"is {type(model.backbone.stages[-1]).__name__}, but yolov10n.yaml "
        f"declares a plain C2f there. BACKBONE_P5_BLOCK is not reaching the "
        f"builder."
    )

    model.eval()
    with torch.no_grad():
        model([torch.rand(3, 96, 96)])

    # And the block kind alone must be load-bearing, or the rebuild above proved
    # only that the multipliers work.
    half_applied = _reload(module)
    _apply_scale(half_applied, _SCALES[scale])
    half_applied.BACKBONE_P5_BLOCK = "c2f_cib"
    wrong = sum(
        p.numel() for p in _build(half_applied, _PUBLISHED_CLASSES - 1).parameters()
    )
    assert wrong != expected, (
        f"{module.__name__}: the {scale} multipliers reproduce that scale's "
        f"published count with EITHER stride-32 block kind, so "
        f"BACKBONE_P5_BLOCK is decoration and _apply_scale is not really "
        f"selecting a scale"
    )


def guard_head_scales_linearly_with_the_class_count(module) -> None:
    """The built head must gain exactly ``_PARAMS_PER_CLASS`` per class.

    Two things at once. It pins the DUAL branch on the built model without
    reading the constructor — a single-branch head gains half as much — and it
    is the property that makes any externally-found parameter figure checkable,
    which is what stopped a custom-``nc`` fine-tune figure being adopted as an
    anchor here.
    """
    low = sum(p.numel() for p in _build(module, 7).parameters())
    high = sum(p.numel() for p in _build(module, 8).parameters())
    step = high - low
    assert step == _PARAMS_PER_CLASS, (
        f"{module.__name__}: the built model gains {step} parameters for one "
        f"extra class, expected {_PARAMS_PER_CLASS} = 3 levels * (cls_hidden "
        f"128 + 1 bias) * 2 branches. Half that ({_PARAMS_PER_CLASS // 2}) "
        f"means the one2one branch is missing or shared; anything else means "
        f"the class tower's width moved."
    )


#: Buffer and tensor totals measured off this repo's own build, as a cheap
#: regression tripwire.
#:
#: ⚠️ SELF-MEASURED. They prove the code is consistent with itself and nothing
#: more — see the block comment above ``_reference_parameters`` for the sibling
#: template where exactly such a number was cited as evidence and was wrong.
#: Parameters are asserted against the re-derived published spec in
#: ``guard_matches_the_published_architecture`` and
#: ``guard_architecture_table_is_a_live_knob``; what lives here is only what
#: those derivations do not cover.
#:
#: Updating these is legitimate when the architecture changes on purpose; state
#: the intended change in the commit message.
_PINNED_TOTALS = {"buffers": 0, "tensors": 321}


def guard_module_tree_size_is_pinned(module) -> None:
    """The built model's buffer and tensor totals are exact.

    Measured at the template's declared ``output_classes`` so the number is
    reproducible from the file alone.
    """
    model = _build(module, module.output_classes)
    actual = {
        "buffers": sum(b.numel() for b in model.buffers()),
        "tensors": len(model.state_dict()),
    }
    assert actual == _PINNED_TOTALS, (
        f"{module.__name__}: module tree is {actual}, pinned at "
        f"{_PINNED_TOTALS} (at output_classes={module.output_classes}). Some "
        f"norm layer, block count or head shape moved. This is a SELF-MEASURED "
        f"tripwire, not evidence the architecture is right — parameters are "
        f"checked against the re-derived published spec. If the change was "
        f"deliberate, update the row and say so in the commit."
    )


def guard_no_stateful_normalisation(module) -> None:
    """No BatchNorm anywhere, and zero buffer elements.

    ``running_mean``/``running_var`` are buffers the averaging service ships
    and averages every federated round, and they average badly across non-IID
    clients. Asserted two ways because they fail differently: a module-type scan
    names the offending layer, while the buffer total also catches a stateful
    norm this scan has never heard of.

    NOT satisfied by ``FrozenBatchNorm2d`` either, and that is the point of
    preferring GroupNorm: Frozen BN moves ``weight``/``bias`` into buffers,
    which would change the parameter count and silently invalidate the
    published-architecture comparison — and on a ``weights=None`` backbone it is
    a bit-exact identity anyway (backend#3093), i.e. it normalises nothing.
    """
    from torch import nn
    from torch.nn.modules.batchnorm import _BatchNorm

    model = _build(module, 3)

    stateful = sorted(
        name for name, sub in model.named_modules() if isinstance(sub, _BatchNorm)
    )
    assert not stateful, (
        f"{module.__name__}: {len(stateful)} BatchNorm layer(s) in the tree "
        f"({stateful[:4]}). Their running statistics are buffers the averaging "
        f"service ships and averages every round. Use GroupNorm."
    )

    group_norms = [sub for sub in model.modules() if isinstance(sub, nn.GroupNorm)]
    assert group_norms, (
        f"{module.__name__}: no GroupNorm in the tree at all — this probe is "
        f"checking nothing"
    )

    buffered = sorted(
        f"{name} ({tensor.numel()})"
        for name, tensor in model.named_buffers()
        if tensor.numel()
    )
    assert not buffered, (
        f"{module.__name__}: {len(buffered)} buffer tensor(s) carry elements "
        f"({buffered[:4]}). Every one is shipped and averaged each federated "
        f"round. This template is written to carry none: GroupNorm has no "
        f"running statistics, and the DFL bin vector is built with "
        f"torch.arange in the decode rather than stored."
    )


def guard_norm_groups_are_derived_from_the_channel_count(module) -> None:
    """``_norm_groups`` must derive the group count, not assume 32.

    A hardcoded 32 works at the shipped width and crashes as soon as the width
    multiplier is lowered: YOLOv10-N's 0.25 puts 16 channels in the stem and
    ``GroupNorm(32, 16)`` raises. So this rebuilds at the published YOLOv10-N
    scale and asserts a sub-32 group count is genuinely produced there, rather
    than asserting it from the shipped build where it would be vacuous.

    It matters more on this template than on either sibling: ``CIB`` and
    ``SCDown`` are built from DEPTHWISE convolutions, so the tree carries norms
    at every intermediate width the inverted blocks produce.
    """
    import torch
    from torch import nn

    assert module._norm_groups(48) == 24, "48 takes 24 groups, not 32"
    assert module._norm_groups(16) == 16, "16 channels cannot take 32 groups"
    assert module._norm_groups(24) == 24
    assert module._norm_groups(3) == 3
    assert module._norm_groups(1) == 1
    assert module._norm_groups(64) == 32, (
        "64 channels should take the full 32 groups — if this changed, the "
        "shipped build's norms changed with it"
    )

    shipped = _build(module, 3)
    shipped_pairs = {
        (sub.num_groups, sub.num_channels)
        for sub in shipped.modules()
        if isinstance(sub, nn.GroupNorm)
    }
    assert shipped_pairs, "no GroupNorm in the shipped build — nothing checked"
    for groups, channels in sorted(shipped_pairs):
        assert channels % groups == 0, (
            f"{module.__name__}: GroupNorm({groups}, {channels}) does not divide"
        )

    narrow = _reload(module)
    _apply_scale(narrow, _SCALES["yolov10n"])
    try:
        model = _build(narrow, 3)
    except Exception as error:  # noqa: BLE001 — any build failure is the bug
        raise AssertionError(
            f"{module.__name__}: rebuilding at the published YOLOv10-N scale "
            f"failed with {type(error).__name__}: {error}. A hardcoded "
            f"GroupNorm group count crashes here — the stem is 16 channels at "
            f"that width — which is why the count is derived."
        ) from error

    pairs = {
        (sub.num_groups, sub.num_channels)
        for sub in model.modules()
        if isinstance(sub, nn.GroupNorm)
    }
    assert pairs, "no GroupNorm at the narrower scale — nothing was checked"
    assert any(groups < 32 for groups, _ in pairs), (
        f"fixture is degenerate: every GroupNorm at the YOLOv10-N scale still "
        f"takes 32 groups ({sorted(pairs)}), so a hardcoded 32 would pass this "
        f"guard too and the derivation is not being exercised"
    )
    for groups, channels in sorted(pairs):
        assert channels % groups == 0, (
            f"{module.__name__}: GroupNorm({groups}, {channels}) at the "
            f"YOLOv10-N scale does not divide"
        )

    model.eval()
    with torch.no_grad():
        model([torch.rand(3, 96, 96)])


# --------------------------------------------------------------------------
# the v10-specific blocks
# --------------------------------------------------------------------------


def guard_scdown_downsamples_with_the_depthwise_conv(module) -> None:
    """``SCDown`` must change channels at FULL resolution, then downsample
    depthwise — not the other way round.

    ⚠️ SWAPPING THE STRIDE IS SHAPE-IDENTICAL **AND** PARAMETER-IDENTICAL. A
    1x1 conv's parameter count does not depend on its stride, and both
    arrangements emit the same spatial size, so the published-architecture
    guard, the pinned totals and every loss are blind to it. What is lost is the
    whole block: with the stride on the pointwise conv, three quarters of the
    input pixels are discarded before the channel transform ever sees them and
    the depthwise 3x3 then mixes an already-subsampled map.

    So this measures the tensor each conv RECEIVES, and asserts the
    shape-and-parameter silence explicitly so nobody assumes a cheaper check
    would have caught it.
    """
    import torch

    block = module.SCDown(8, 16)
    block.eval()
    probe = torch.rand(1, 8, 12, 12)

    assert block.cv1.conv.stride == (1, 1), (
        f"{module.__name__}: SCDown's POINTWISE conv has stride "
        f"{block.cv1.conv.stride}; it must be 1 so the channel transform sees "
        f"the full-resolution map. The depthwise conv does the downsampling."
    )
    assert block.cv2.conv.stride == (2, 2), (
        f"{module.__name__}: SCDown's DEPTHWISE conv has stride "
        f"{block.cv2.conv.stride}, expected 2 — it is what performs the "
        f"spatial reduction"
    )
    assert block.cv2.conv.groups == block.cv2.conv.in_channels, (
        f"{module.__name__}: SCDown's second conv has "
        f"{block.cv2.conv.groups} groups for "
        f"{block.cv2.conv.in_channels} channels — it must be DEPTHWISE, which "
        f"is the entire parameter saving over a plain strided 3x3"
    )

    # State the silence: a swapped-stride build has the same count and shape.
    swapped = module.SCDown(8, 16)
    swapped.cv1.conv.stride = (2, 2)
    swapped.cv2.conv.stride = (1, 1)
    swapped.eval()
    with torch.no_grad():
        correct_out = block(probe)
        swapped_out = swapped(probe)
    assert correct_out.shape == swapped_out.shape, (
        f"fixture is degenerate: the correct block emits "
        f"{tuple(correct_out.shape)} and the swapped-stride one "
        f"{tuple(swapped_out.shape)}. They must MATCH — the whole point is that "
        f"a swapped stride is invisible to a shape check."
    )
    assert sum(p.numel() for p in block.parameters()) == sum(
        p.numel() for p in swapped.parameters()
    ), "fixture is degenerate: the two arrangements differ in parameter count"

    seen = {}
    handles = [
        block.cv1.conv.register_forward_pre_hook(
            lambda _m, i: seen.__setitem__("cv1", i[0].shape)
        ),
        block.cv2.conv.register_forward_pre_hook(
            lambda _m, i: seen.__setitem__("cv2", i[0].shape)
        ),
    ]
    try:
        with torch.no_grad():
            block(probe)
    finally:
        for handle in handles:
            handle.remove()

    assert set(seen) == {"cv1", "cv2"}, "both pre-hooks did not fire"
    assert tuple(seen["cv1"][-2:]) == (12, 12), (
        f"{module.__name__}: SCDown's pointwise conv received a "
        f"{tuple(seen['cv1'][-2:])} map for a 12x12 input — it must see the "
        f"FULL resolution"
    )
    assert tuple(seen["cv2"][-2:]) == (12, 12), (
        f"{module.__name__}: SCDown's depthwise conv received a "
        f"{tuple(seen['cv2'][-2:])} map, so the downsample happened BEFORE it — "
        f"the stride is on the pointwise conv. Same shape out, same parameter "
        f"count, and the block is now point-sampling every other pixel."
    )


def guard_repvggdw_sums_its_branches_before_one_activation(module) -> None:
    """``RepVGGDW`` must sum its depthwise 7x7 and 3x3 and THEN activate.

    That single activation over the sum is what makes the block
    re-parameterisable: both branches are conv-plus-norm and so affine, and
    their sum collapses into one 7x7 depthwise kernel at deployment (upstream
    pads the 3x3 to 7x7 and adds the weights). Activating each branch first
    keeps the parameter count, the shapes and the loss keys identical and
    destroys that property — the sum of two SiLU outputs is not affine.
    """
    import torch
    import torch.nn.functional as F

    block = module.RepVGGDW(6)
    block.eval()
    probe = torch.rand(1, 6, 9, 9) * 4.0 - 2.0

    large, small = module.REPVGGDW_KERNELS
    assert large > small, (
        f"{module.__name__}: REPVGGDW_KERNELS is {module.REPVGGDW_KERNELS}; the "
        f"first must be the LARGE kernel the block is named for"
    )
    assert block.conv.conv.kernel_size == (large, large)
    assert block.conv1.conv.kernel_size == (small, small)
    for name, conv in (("conv", block.conv.conv), ("conv1", block.conv1.conv)):
        assert conv.groups == conv.in_channels, (
            f"{module.__name__}: RepVGGDW.{name} has {conv.groups} groups for "
            f"{conv.in_channels} channels — both branches must be DEPTHWISE, "
            f"which is what makes a 7x7 affordable here"
        )

    with torch.no_grad():
        actual = block(probe)
        branch_large = block.conv(probe)
        branch_small = block.conv1(probe)
        summed_then_activated = F.silu(branch_large + branch_small)
        activated_then_summed = F.silu(branch_large) + F.silu(branch_small)

    assert not torch.allclose(
        summed_then_activated, activated_then_summed, atol=1e-4
    ), (
        "fixture is degenerate: the two orders agree on this input, so the rule "
        "cannot fire — the probe needs values where SiLU is genuinely nonlinear"
    )
    assert torch.allclose(actual, summed_then_activated, atol=1e-5), (
        f"{module.__name__}: RepVGGDW is not summing its branches before the "
        f"activation. Max deviation from sum-then-activate is "
        f"{float((actual - summed_then_activated).abs().max()):.4g}, against "
        f"{float((actual - activated_then_summed).abs().max()):.4g} from "
        f"activate-then-sum. The parameter count, the shapes and every loss are "
        f"unchanged either way; what is lost is the re-parameterisation the "
        f"'Rep' names."
    )


def guard_cib_is_inverted_and_mostly_depthwise(module) -> None:
    """``CIB`` must be the published five-conv inverted block, with only its two
    pointwise convs dense and its inner mixer at TWICE the branch width.

    The expansion is the silent one. Upstream calls ``CIB`` with ``e=1.0`` from
    inside ``C2fCIB``, where the width handed in is *already* the halved branch,
    so the inverted expansion runs at ``2 * branch``. A plausible-looking
    ``e=0.5`` halves the inner mixer, keeps every shape legal, leaves every loss
    finite — and misses the published count, which is the only thing that would
    notice.
    """
    import torch

    channels = 16
    block = module.CIB(channels, channels, 1.0, True)
    block.eval()
    convs = [
        sub for sub in block.block if hasattr(sub, "conv")
    ]
    mixers = [sub for sub in block.block if not hasattr(sub, "conv")]

    assert len(block.block) == 5, (
        f"{module.__name__}: CIB has {len(block.block)} stages, published "
        f"design has 5 (dw3 -> pw expand -> mixer -> pw project -> dw3)"
    )
    first, expand, mixer, project, last = block.block

    for name, cna in (("first", first), ("last", last)):
        assert cna.conv.groups == cna.conv.in_channels, (
            f"{module.__name__}: CIB's {name} conv has {cna.conv.groups} groups "
            f"for {cna.conv.in_channels} channels — the outer convs must be "
            f"DEPTHWISE, which is what makes the block compact"
        )
        assert cna.conv.kernel_size == (3, 3)
    assert expand.conv.kernel_size == (1, 1) and expand.conv.groups == 1
    assert project.conv.kernel_size == (1, 1) and project.conv.groups == 1

    expanded = expand.conv.out_channels
    assert expanded == 2 * channels, (
        f"{module.__name__}: CIB at expansion 1.0 on {channels} channels "
        f"expands to {expanded}, expected {2 * channels}. Inside C2fCIB the "
        f"width handed in is ALREADY the halved branch, so the inverted "
        f"expansion runs at twice it — an expansion of 0.5 here gives "
        f"{channels} and every shape still lines up."
    )
    assert project.conv.in_channels == expanded

    assert type(mixer).__name__ == "RepVGGDW", (
        f"{module.__name__}: CIB's inner mixer is {type(mixer).__name__} with "
        f"large_kernel set, expected RepVGGDW — the yaml's third positional "
        f"argument (`C2fCIB, [1024, True, True]`) selects it, and every "
        f"published v10 C2fCIB sets it"
    )

    # And the non-large-kernel path must be the plain depthwise 3x3, or the
    # selector is not really a selector.
    plain = module.CIB(channels, channels, 1.0, True, large_kernel=False)
    plain_mixer = plain.block[2]
    assert type(plain_mixer).__name__ != "RepVGGDW", (
        f"{module.__name__}: large_kernel=False still builds a "
        f"{type(plain_mixer).__name__} — the flag reaches nothing"
    )
    assert sum(p.numel() for p in plain.parameters()) != sum(
        p.numel() for p in block.parameters()
    ), "fixture is degenerate: the two mixer kinds have the same parameter count"

    with torch.no_grad():
        out = block(torch.rand(1, channels, 8, 8))
    assert tuple(out.shape) == (1, channels, 8, 8), (
        f"{module.__name__}: CIB changed its shape to {tuple(out.shape)}"
    )
    assert block.use_add, (
        f"{module.__name__}: CIB at equal in/out widths with shortcut=True must "
        f"take its identity branch"
    )
    del convs, mixers


def guard_c2f_fuses_every_intermediate_block(module) -> None:
    """``C2f`` must fuse EVERY intermediate block output, not just the last.

    That ``2 + n`` fusion is the difference from YOLOv5/YOLOX's ``C3``, which
    concatenates only the final block output and the skip. A C3-shaped
    implementation trains happily at a slightly smaller parameter count, so this
    reconstructs the expected branch list and compares it against the tensor
    ``cv2`` is actually handed rather than reading the constructor.
    """
    import torch

    blocks = 3
    stage = module.C2f(12, 12, n=blocks, shortcut=False)
    stage.eval()
    assert len(stage.m) == blocks, (
        f"fixture is degenerate: asked for {blocks} blocks, built "
        f"{len(stage.m)}"
    )
    assert blocks >= 2, (
        "fixture is degenerate: at one block the C2f and C3 branch lists "
        "contain the same tensors"
    )
    assert stage.cv2.conv.in_channels == (2 + blocks) * stage.hidden, (
        f"{module.__name__}: C2f's fusion conv takes "
        f"{stage.cv2.conv.in_channels} channels for {blocks} blocks at hidden "
        f"width {stage.hidden}; C2f's shape is (2 + n) * hidden = "
        f"{(2 + blocks) * stage.hidden}. A C3 takes 2 * hidden."
    )

    probe = torch.rand(1, 12, 8, 8) + 0.5
    captured = []
    handle = stage.cv2.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        with torch.no_grad():
            stage(probe)
    finally:
        handle.remove()
    assert len(captured) == 1, "the fusion conv's pre-hook did not fire once"

    with torch.no_grad():
        branches = list(stage.cv1(probe).chunk(2, dim=1))
        for block in stage.m:
            branches.append(block(branches[-1]))
        expected = torch.cat(branches, dim=1)

    assert captured[0].shape == expected.shape, (
        f"C2f hands its fusion conv {tuple(captured[0].shape)}; the branch list "
        f"is {tuple(expected.shape)}"
    )
    assert torch.allclose(captured[0], expected, atol=1e-5), (
        "C2f's fusion conv is not receiving [split_0, split_1, m0(split_1), "
        "m1(m0(...)), ...]. The channel count is right and the model trains, so "
        "the likely shape is a C3 padded to the C2f width — only the final "
        "block output is being kept."
    )


def guard_sppf_pools_in_series(module) -> None:
    """``SPPF``'s max-pools must be applied **in series**, each on the previous
    output.

    Series is what gives the 5/9/13 effective receptive field from one 5x5
    kernel — the whole reason the block is "fast". Applying all three to
    ``cv1``'s output instead produces identical shapes, identical parameters and
    three identical branches, so the block widens the deepest stage's receptive
    field by 5 rather than 13 and nothing else notices.
    """
    import torch

    block = module.SPPF(8, 8)
    block.eval()
    assert block.repeats >= 2, (
        f"fixture is degenerate: {block.repeats} repeat(s). At one repeat the "
        f"series and parallel arrangements contain the same tensors"
    )

    probe = torch.rand(1, 8, 9, 9)
    captured = []
    handle = block.cv2.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        with torch.no_grad():
            block(probe)
    finally:
        handle.remove()
    assert len(captured) == 1, "the fusion conv's pre-hook did not fire once"

    with torch.no_grad():
        base = block.cv1(probe)
        series = [base]
        for _ in range(block.repeats):
            series.append(block.pool(series[-1]))
        expected = torch.cat(series, dim=1)
        parallel = torch.cat(
            [base] + [block.pool(base) for _ in range(block.repeats)], dim=1
        )

    assert not torch.allclose(expected, parallel, atol=1e-5), (
        "fixture is degenerate: series and parallel pooling agree on this "
        "input, so the rule cannot fire"
    )
    assert captured[0].shape == expected.shape, (
        f"SPPF hands its fusion conv {tuple(captured[0].shape)}, expected "
        f"{tuple(expected.shape)}"
    )
    assert torch.allclose(captured[0], expected, atol=1e-5), (
        "SPPF's pools are not applied in series. The three branches are the "
        "same 5x5 pooling of cv1's output rather than 5 / 9 / 13, so the "
        "deepest stage's receptive field is a third of the design's — and every "
        "shape, parameter and loss is unchanged."
    )


def guard_psa_attends_to_only_half_the_channels(module) -> None:
    """``PSA`` must send only ONE half of its split through attention, and the
    other half through untouched.

    That is what "partial self-attention" names, and it is why attention is
    affordable at the stride-32 stage at all. Two silent errors are covered:

    * attending to both halves (or to the bypass half instead), which keeps
      every shape and every parameter and simply doubles the token mixing;
    * dropping either residual, which is likewise shape- and count-identical.

    Checked by reconstructing the expected fusion input and comparing against
    the tensor ``cv2`` is actually handed.
    """
    import torch

    channels = 128
    block = module.PSA(channels, channels)
    block.eval()
    assert block.hidden == channels // 2, (
        f"{module.__name__}: PSA splits {channels} channels into halves of "
        f"{block.hidden}, expected {channels // 2}"
    )
    assert block.attn.dim == block.hidden, (
        f"{module.__name__}: PSA's attention is sized {block.attn.dim} for a "
        f"{block.hidden}-channel half. If it were sized for the full width the "
        f"attention would not be PARTIAL — and the parameter count would move, "
        f"so this is belt-and-braces with the published figure."
    )

    probe = torch.rand(1, channels, 6, 6)
    captured = []
    handle = block.cv2.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        with torch.no_grad():
            block(probe)
    finally:
        handle.remove()
    assert len(captured) == 1, "the fusion conv's pre-hook did not fire once"

    with torch.no_grad():
        bypass, attending = block.cv1(probe).split(
            (block.hidden, block.hidden), dim=1
        )
        residual = attending + block.attn(attending)
        residual = residual + block.ffn(residual)
        expected = torch.cat((bypass, residual), dim=1)
        # The two orders this guard has to be able to tell apart.
        both_attend = torch.cat(
            (bypass + block.attn(bypass), residual), dim=1
        )
        no_residual = torch.cat((bypass, block.ffn(block.attn(attending))), dim=1)

    assert not torch.allclose(expected, both_attend, atol=1e-4), (
        "fixture is degenerate: attending to the bypass half produces the same "
        "tensor, so the rule cannot fire"
    )
    assert not torch.allclose(expected, no_residual, atol=1e-4), (
        "fixture is degenerate: dropping the residuals produces the same "
        "tensor, so the rule cannot fire"
    )
    assert captured[0].shape == expected.shape, (
        f"PSA hands its fusion conv {tuple(captured[0].shape)}, expected "
        f"{tuple(expected.shape)}"
    )
    assert torch.allclose(captured[0], expected, atol=1e-5), (
        "PSA's fusion conv is not receiving [bypass_half, attended_half]. The "
        "bypass half must reach the fusion UNCHANGED — that partial "
        "application is what the block is named for — and the attending half "
        "must get attention and the feed-forward each as a residual. Every "
        "shape and every parameter is identical under all of these."
    )
    # The bypass half really is untouched, asserted directly as well as via the
    # reconstruction, because it is the load-bearing half of the claim.
    assert torch.allclose(captured[0][:, : block.hidden], bypass, atol=1e-6), (
        "PSA's bypass half is not reaching the fusion unchanged"
    )


def guard_attention_matches_the_manual_formula(module) -> None:
    """``Attention`` routes through SDPA, and must compute what the explicit
    matmul-softmax-matmul computed.

    ``tests/test_sdpa_equivalence.py`` is the acceptance bar backend#2090 set
    for this conversion: run the pre-conversion formula and the SDPA path on the
    SAME weights and the SAME input, and require them to agree in fp32 at tight
    tolerance. This is that bar applied to this template's attention, and it is
    the reason the conversion is safe to make at all — a silently different
    attention leaves every structural guard in this file green.

    Two ways the conversion can go wrong, and they fail DIFFERENTLY:

    * **the q/k/v layout.** Upstream holds them channels-first — ``(batch,
      heads, width, tokens)`` — while SDPA's contract is ``(batch, heads,
      tokens, width)``. ⚠️ MEASURED, not assumed: at this template's shape the
      wrong layout **raises** rather than differing quietly, because ``key`` and
      ``value`` then disagree on their sequence length (``key_dim`` 32 against
      ``head_dim`` 64) and the downstream reshape gets the wrong element count.
      That is luck rather than design — it depends on ``ATTN_RATIO`` making the
      two widths differ — so this guard converts any exception into its own
      assertion instead of letting the mutation sweep record an ERROR, and the
      claim made here is the narrow one: it is caught, not that it is subtle;
    * **the scale.** SDPA's default is ``1 / sqrt(query.size(-1))``, which
      equals ``self.scale`` exactly today, so relying on the default would make
      ``self.scale`` a constant that reaches nothing and a future
      ``ATTN_RATIO`` change would be silently ignored by the kernel. That one
      IS numerically silent — it just scales the logits — and is what the
      tolerance below actually catches.

    The manual formula is written out longhand rather than called, so it cannot
    drift into being the same code it is checking.
    """
    import torch

    torch.manual_seed(0)
    dim = 128
    attention = module.Attention(dim)
    attention.eval()
    probe = torch.rand(2, dim, 5, 5)

    def manual(block, x):
        """The pre-conversion formula, verbatim."""
        batch, channels, height, width = x.shape
        tokens = height * width
        qkv = block.qkv(x).view(
            batch, block.num_heads, 2 * block.key_dim + block.head_dim, tokens
        )
        query, key, value = qkv.split(
            [block.key_dim, block.key_dim, block.head_dim], dim=2
        )
        scores = (query.transpose(-2, -1) @ key) * block.scale
        scores = scores.softmax(dim=-1)
        attended = (value @ scores.transpose(-2, -1)).reshape(
            batch, channels, height, width
        )
        return block.proj(
            attended + block.pe(value.reshape(batch, channels, height, width))
        )

    with torch.no_grad():
        expected = manual(attention, probe)
        # WRAPPED: at this shape a transposed layout raises rather than
        # returning wrong numbers, and an unwrapped exception would make the
        # mutation sweep record an ERROR instead of a caught mutation.
        try:
            actual = attention(probe)
        except Exception as error:  # noqa: BLE001 — any failure here is the bug
            raise AssertionError(
                f"{module.__name__}: the attention forward raised "
                f"{type(error).__name__}: {error}. The SDPA call wants q/k/v as "
                f"(batch, heads, tokens, width); upstream holds them "
                f"channels-first, so the transposes around the call are part of "
                f"the call and not tidiness."
            ) from error

    assert expected.dtype == torch.float32
    assert actual.shape == expected.shape, (
        f"{module.__name__}: SDPA path emits {tuple(actual.shape)}, the manual "
        f"formula {tuple(expected.shape)}"
    )
    deviation = float((actual - expected).abs().max())
    assert deviation < 1e-4, (
        f"{module.__name__}: the SDPA attention path deviates from the explicit "
        f"matmul-softmax-matmul by {deviation:.3g} (bar 1e-4, measured 6.3e-07 "
        f"on the shipped template). The likely cause is the SCALE: SDPA's "
        f"default is 1 / sqrt(query.size(-1)) and this call passes self.scale "
        f"explicitly so the declared value cannot be silently ignored — if "
        f"those two ever disagree, this is where it shows."
    )
    # The tolerance is only meaningful if the formula is doing real mixing:
    # a near-uniform attention would agree with almost anything.
    assert float(expected.abs().max()) > 1e-3, (
        f"fixture is degenerate: the manual formula's output is ~0 "
        f"({float(expected.abs().max()):.3g}), so agreeing with it proves "
        f"nothing"
    )


def guard_sppf_and_psa_are_applied_to_the_deepest_map(module) -> None:
    """``SPPF`` and ``PSA`` must actually be APPLIED, in that order, to the
    stride-32 feature map the backbone returns.

    ⚠️ A BYPASSED MODULE IS INVISIBLE TO EVERY COUNT IN THIS FILE. If
    ``forward`` stops calling ``self.psa``, the module is still CONSTRUCTED, so
    the parameter total, the buffer total and the state_dict key set are all
    unchanged and ``guard_matches_the_published_architecture`` passes — measured,
    not assumed: that mutation DID NOT RAISE against the published-count guard.
    Worse, the bypassed parameters are still shipped and averaged every
    federated round, so the model pays for a block it does not use.

    This is the module-scale version of "a constant that reaches nothing", and
    it needs a functional check: the two modules' forward hooks must fire
    exactly once each, ``PSA`` must receive ``SPPF``'s output, and the
    backbone's third return value must be ``PSA``'s output.
    """
    import torch

    model = _build(module, 3, 64)
    model.eval()
    backbone = model.backbone

    seen = {}
    handles = [
        backbone.sppf.register_forward_hook(
            lambda _m, i, o: seen.setdefault("sppf", []).append(
                (i[0].detach().clone(), o.detach().clone())
            )
        ),
        backbone.psa.register_forward_hook(
            lambda _m, i, o: seen.setdefault("psa", []).append(
                (i[0].detach().clone(), o.detach().clone())
            )
        ),
    ]
    try:
        with torch.no_grad():
            outputs = backbone(torch.rand(1, 3, 64, 64))
    finally:
        for handle in handles:
            handle.remove()

    assert "sppf" in seen, (
        f"{module.__name__}: SPPF was never called during a backbone forward. "
        f"It is still constructed, so the parameter count, the buffer count and "
        f"every state_dict key are UNCHANGED and no count in this file can see "
        f"it — while its parameters are still shipped and averaged every "
        f"federated round for nothing."
    )
    assert "psa" in seen, (
        f"{module.__name__}: PSA was never called during a backbone forward. "
        f"It is still constructed, so no parameter count can see it. PSA is "
        f"YOLOv10's partial self-attention on the deepest stage — bypassing it "
        f"leaves a plain YOLOv8 backbone that trains perfectly well and is not "
        f"the published architecture."
    )
    assert len(seen["sppf"]) == 1 and len(seen["psa"]) == 1, (
        f"{module.__name__}: SPPF fired {len(seen['sppf'])} time(s) and PSA "
        f"{len(seen['psa'])} — each must be applied exactly once"
    )

    sppf_out = seen["sppf"][0][1]
    psa_in, psa_out = seen["psa"][0]
    assert torch.allclose(psa_in, sppf_out, atol=1e-6), (
        f"{module.__name__}: PSA did not receive SPPF's output. The published "
        f"order is SPPF then PSA on the deepest map (yaml layers 9 then 10); "
        f"swapping them or feeding both from the stage output changes no shape "
        f"and no parameter."
    )
    assert torch.allclose(outputs[2], psa_out, atol=1e-6), (
        f"{module.__name__}: the backbone's stride-32 output is not PSA's "
        f"output, so PSA runs and is then discarded — which is the same defect "
        f"as never calling it, and equally invisible to every count here."
    )


def guard_attention_head_count_is_derived_and_reaches_the_attention(module) -> None:
    """``Attention``'s head count must be DERIVED from ``ATTENTION_HEAD_DIM``,
    and the derivation must reach the attention it configures.

    ⚠️ THIS KNOB IS PARAMETER-INVARIANT, which is what makes it the textbook
    "constant that reaches nothing" and why it needs a guard the published count
    cannot provide. ``num_heads * int((dim / num_heads) * ATTN_RATIO)`` equals
    ``dim * ATTN_RATIO`` for any head count that divides ``dim``, so ``qkv``'s
    output width — and therefore EVERY parameter in the module — is identical
    whatever the head count is. A hardcoded ``num_heads = 8`` changes no
    parameter, no shape, no loss key and no published figure; it only
    re-factorises the attention into eight narrower heads.

    A sibling shipped exactly this shape of defect: ``NUM_DYNAMIC = 2`` threaded
    through as a kwarg while the module hardcoded the literal.

    So three things are asserted: the count is the derivation at two different
    widths, the parameter invariance is stated (so nobody later "simplifies"
    this guard away believing the count guard covers it), and the head count is
    shown to CHANGE THE OUTPUT — which is the only observable consequence.
    """
    import torch

    head_dim = module.ATTENTION_HEAD_DIM
    for dim in (128, 256, 512):
        attention = module.Attention(dim)
        assert attention.num_heads == dim // head_dim, (
            f"{module.__name__}: Attention({dim}) runs "
            f"{attention.num_heads} heads; derived from "
            f"ATTENTION_HEAD_DIM={head_dim} it is {dim // head_dim}. A "
            f"hardcoded count changes NO parameter and NO shape, so nothing "
            f"else in this suite can see it."
        )
        assert attention.head_dim == head_dim, (
            f"{module.__name__}: Attention({dim}) has head_dim "
            f"{attention.head_dim}, expected {head_dim}"
        )

    # State the invariance, so this guard is not later deleted as redundant.
    reference = module.Attention(256)
    counts = {}
    for probe_heads in (2, 4, 8, 16):
        alternative = module.Attention(256, head_dim=256 // probe_heads)
        assert alternative.num_heads == probe_heads
        counts[probe_heads] = sum(p.numel() for p in alternative.parameters())
    assert len(set(counts.values())) == 1, (
        f"the head count is NO LONGER parameter-invariant ({counts}). That "
        f"makes this guard less necessary, not more — but the reasoning in its "
        f"docstring is now wrong and the published-count guard may already "
        f"cover it. Re-read both before changing either."
    )
    assert counts[4] == sum(p.numel() for p in reference.parameters())

    # And the count must actually change the computation, or it is inert even
    # though it is "derived".
    torch.manual_seed(0)
    four = module.Attention(256, head_dim=64)
    sixteen = module.Attention(256, head_dim=16)
    sixteen.load_state_dict(four.state_dict())
    four.eval()
    sixteen.eval()
    probe = torch.rand(1, 256, 4, 4)
    with torch.no_grad():
        a, b = four(probe), sixteen(probe)
    assert not torch.allclose(a, b, atol=1e-4), (
        f"{module.__name__}: attention with 4 heads and with 16 heads produce "
        f"the same output from identical weights, so the head count is inert. "
        f"It is meant to decide how the channels are partitioned before the "
        f"softmax."
    )


# --------------------------------------------------------------------------
# the dual head — the reason this template exists
# --------------------------------------------------------------------------

#: The parameter-name prefixes of the feature trunk: everything the one2one
#: head's gradient must NOT reach.
_TRUNK_PREFIXES = ("backbone.", "neck.")
#: ...and of the one2one head, which it MUST reach.
_ONE2ONE_PREFIXES = (
    "head.one2one_box_convs.",
    "head.one2one_box_preds.",
    "head.one2one_cls_convs.",
    "head.one2one_cls_preds.",
)


def _overfit_targets(torch, edge, label=1):
    return [
        {
            "boxes": torch.tensor([[edge * 0.25, edge * 0.25, edge * 0.7, edge * 0.7]]),
            "labels": torch.tensor([label], dtype=torch.int64),
        }
    ]


def _nonzero_grads(model, prefixes):
    return [
        name
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
        and parameter.grad is not None
        and float(parameter.grad.abs().sum()) > 0.0
    ]


def guard_one2one_head_is_detached_from_the_backbone(module) -> None:
    """⚠️ THE HEADLINE GUARD. The one2one head's gradient must not reach the
    feature trunk.

    ``YOLOv10Head.forward`` feeds the one2one branch ``feature.detach()``, so
    only the one2many branch trains the backbone and neck. Removing that
    ``.detach()`` is invisible to everything else in this suite and to the
    family train-step test: the parameter count is identical (a detach is not a
    parameter), every shape is unchanged, the same six finite losses come back,
    they still fall, and the template still overfits one object past every
    threshold in ``guard_overfits_a_single_object``. Only the gradient GRAPH
    differs — so this guard measures the gradient graph.

    Both directions are asserted, and the second is not decoration:

    1. backpropagating the ONE2ONE losses alone must leave every trunk
       parameter with no gradient or an exactly-zero one. This is the rule;
    2. backpropagating the ONE2MANY losses alone must leave the trunk with a
       NON-ZERO gradient. Without this, "the one2one loss does not reach the
       trunk" is also satisfied by a model where nothing reaches the trunk —
       a template that detached its features entirely, or whose losses were
       disconnected, would pass (1) and be completely broken.

    And the one2one head's own parameters must receive a real gradient from (1),
    or the branch is not being trained at all and (1) is vacuous for a third
    reason.
    """
    import torch

    torch.manual_seed(0)
    edge = 128
    model = _build(module, 3, edge)
    model.train()
    losses = model([torch.rand(3, edge, edge)], _overfit_targets(torch, edge))

    one2one_keys = sorted(k for k in losses if "one2one" in k)
    one2many_keys = sorted(k for k in losses if "one2one" not in k)
    assert one2one_keys and one2many_keys, (
        f"fixture is degenerate: the loss dict must carry BOTH branches' terms "
        f"separately for this guard to separate them, got {sorted(losses)}"
    )
    trunk_names = [
        name
        for name, _ in model.named_parameters()
        if name.startswith(_TRUNK_PREFIXES)
    ]
    assert trunk_names, (
        "fixture is degenerate: no parameter matches the trunk prefixes "
        f"{_TRUNK_PREFIXES} — a rename must fail the lookup rather than "
        f"quietly narrowing this probe to nothing"
    )
    one2one_names = [
        name
        for name, _ in model.named_parameters()
        if name.startswith(_ONE2ONE_PREFIXES)
    ]
    assert one2one_names, (
        "fixture is degenerate: no parameter matches the one2one prefixes "
        f"{_ONE2ONE_PREFIXES}"
    )

    # (1) the one2one losses ALONE.
    model.zero_grad(set_to_none=True)
    sum(losses[key] for key in one2one_keys).backward(retain_graph=True)

    leaked = _nonzero_grads(model, _TRUNK_PREFIXES)
    assert not leaked, (
        f"{module.__name__}: backpropagating ONLY the one2one losses "
        f"({one2one_keys}) put a non-zero gradient on "
        f"{len(leaked)}/{len(trunk_names)} backbone/neck parameters "
        f"({leaked[:5]}). The one2one head must be fed DETACHED features, so "
        f"its sparse one-to-one supervision cannot compete with the one2many "
        f"branch for the features that branch exists to provide.\n"
        f"This is the single most silent defect available in this template: the "
        f"parameter count is unchanged, every shape is unchanged, all six "
        f"losses stay finite and fall, and the model still overfits one object. "
        f"Nothing else in this suite — or in "
        f"tests/test_od_torchvision_family_train_step.py — can see it."
    )
    trained = _nonzero_grads(model, _ONE2ONE_PREFIXES)
    assert trained, (
        f"{module.__name__}: the one2one losses put NO gradient on any of the "
        f"{len(one2one_names)} one2one head parameters, so the branch is not "
        f"being trained and the isolation assertion above is vacuous. The "
        f"detach must stop at the head's INPUT, not detach the head itself."
    )

    # (2) the mirror: the one2many losses MUST reach the trunk.
    model.zero_grad(set_to_none=True)
    sum(losses[key] for key in one2many_keys).backward()
    reached = _nonzero_grads(model, _TRUNK_PREFIXES)
    assert reached, (
        f"{module.__name__}: backpropagating the one2many losses "
        f"({one2many_keys}) left EVERY backbone and neck parameter with a zero "
        f"or absent gradient. Nothing trains the feature trunk at all, which "
        f"would also satisfy the isolation check above — that is exactly why "
        f"this second direction is asserted."
    )


def guard_dual_head_branches_are_independent(module) -> None:
    """The one2many and one2one branches must share NO parameters.

    Upstream builds the one2one branch with ``copy.deepcopy``, so the two are
    independent copies. Sharing them would train and would defeat the entire
    design: the two branches are supervised with DIFFERENT assignments from the
    same features, so a shared tower receives both gradients and learns
    neither assignment.

    Also asserts they start IDENTICAL, which is what deepcopy gives and what a
    second fresh construction would not — the two heads reading the same
    features under different assignments is the design; starting them apart adds
    a difference it does not intend.
    """
    import torch

    head = _build(module, 3).head
    groups = (
        ("box_convs", "one2one_box_convs"),
        ("box_preds", "one2one_box_preds"),
        ("cls_convs", "one2one_cls_convs"),
        ("cls_preds", "one2one_cls_preds"),
    )
    for many_name, one_name in groups:
        many = getattr(head, many_name, None)
        one = getattr(head, one_name, None)
        assert many is not None and one is not None, (
            f"{module.__name__}: the head is missing {many_name} or {one_name} "
            f"— a rename must fail here rather than narrowing this probe to "
            f"nothing"
        )
        many_ids = {id(p) for p in many.parameters()}
        one_ids = {id(p) for p in one.parameters()}
        assert many_ids and one_ids, (
            f"{module.__name__}: {many_name}/{one_name} hold "
            f"{len(many_ids)}/{len(one_ids)} parameters"
        )
        shared = many_ids & one_ids
        assert not shared, (
            f"{module.__name__}: {many_name} and {one_name} share "
            f"{len(shared)} parameter tensor(s) — the dual head is one head "
            f"wearing two names. It would train, report the same six loss keys, "
            f"and learn neither assignment, because a shared tower receives "
            f"both the one2many and the one2one gradient."
        )
        many_params = list(many.parameters())
        one_params = list(one.parameters())
        assert len(many_params) == len(one_params), (
            f"{module.__name__}: {many_name} has {len(many_params)} tensors and "
            f"{one_name} has {len(one_params)}; the one2one branch is a copy of "
            f"the one2many branch and must have the same shape"
        )
        for left, right in zip(many_params, one_params):
            assert left.shape == right.shape, (
                f"{module.__name__}: {many_name}/{one_name} disagree on a "
                f"tensor shape: {tuple(left.shape)} against "
                f"{tuple(right.shape)}"
            )
            assert torch.equal(left, right), (
                f"{module.__name__}: {one_name} does not start identical to "
                f"{many_name}. Upstream builds it with copy.deepcopy AFTER the "
                f"one2many branch is initialised (including the bias init), so "
                f"the two heads begin from the same weights and differ only in "
                f"their assignment."
            )


def guard_dual_assignment_is_consistent(module) -> None:
    """The two branches must use the SAME alignment metric and differ ONLY in
    the number of candidates per ground truth.

    That is what "consistent dual assignments" means, and it is the whole
    mechanism: because both heads rank candidates by
    ``score ** TAL_ALPHA * iou ** TAL_BETA``, the one2one head's single positive
    is the anchor the one2many head also ranked first, so the dense gradient
    that shapes the features is not pulling them away from what the deployed
    head is asked to do.

    Two silent failures are covered:

    * the one2one branch assigned with ``TAL_TOPK`` too, which collapses the
      dual assignment into two identical ones. Six finite losses, falling
      normally, and a deployed head with ten positives per object — i.e. a
      model that still needs the NMS this architecture removes;
    * ``ONE2ONE_TOPK`` set above 1, same thing by degree.

    Checked by spying on the ``topk`` argument each ``assign`` call receives,
    then by asserting the functional consequence: at ``ONE2ONE_TOPK`` exactly
    one anchor per ground truth is selected.
    """
    import torch

    assert module.ONE2ONE_TOPK == 1, (
        f"{module.__name__}: ONE2ONE_TOPK is {module.ONE2ONE_TOPK}; the "
        f"one2one head is one-to-one BY DEFINITION — one object, one positive "
        f"anchor — and that is the only reason its output needs no NMS"
    )
    assert module.TAL_TOPK > module.ONE2ONE_TOPK, (
        f"{module.__name__}: TAL_TOPK ({module.TAL_TOPK}) must exceed "
        f"ONE2ONE_TOPK ({module.ONE2ONE_TOPK}) — the one2many branch exists to "
        f"supply the DENSE supervision one-to-one assignment cannot"
    )

    edge = 128
    model = _build(module, 3, edge)
    model.train()

    seen = []
    original = model.assign

    def spy(*args, **kwargs):
        # `topk` is positional in the template's own call sites; accept both so
        # a signature change fails the ASSERTION below rather than this wrapper.
        if "topk" in kwargs:
            seen.append(kwargs["topk"])
        else:
            seen.append(args[-1])
        return original(*args, **kwargs)

    model.assign = spy
    try:
        model([torch.rand(3, edge, edge)], _overfit_targets(torch, edge))
    finally:
        model.assign = original

    assert seen, (
        f"{module.__name__}: `assign` was never called with a recordable topk, "
        f"so this guard measured nothing. Both branches must route through the "
        f"one shared assigner."
    )
    assert sorted(set(seen)) == sorted({module.TAL_TOPK, module.ONE2ONE_TOPK}), (
        f"{module.__name__}: the assigner was called with topk values "
        f"{sorted(set(seen))}; the dual assignment requires exactly "
        f"{sorted({module.TAL_TOPK, module.ONE2ONE_TOPK})} — "
        f"{module.TAL_TOPK} for the one2many branch and "
        f"{module.ONE2ONE_TOPK} for the one2one branch. A single value means "
        f"the two branches share one assignment: it trains, all six losses stay "
        f"finite, and the deployed head keeps ten positives per object, so the "
        f"model still needs the NMS this architecture exists to remove."
    )

    # The functional consequence, on a fixture with many valid candidates.
    from torchvision.ops import box_iou

    label = 1
    count = 12
    assert count > module.TAL_TOPK, (
        f"fixture is degenerate: {count} candidates against TAL_TOPK "
        f"{module.TAL_TOPK}"
    )
    gt = [[0.0, 0.0, 400.0, 400.0]]
    pred_boxes = [[0.0, 0.0, 380.0 - 10.0 * i, 380.0 - 10.0 * i] for i in range(count)]
    scores = [{label: 0.5} for _ in range(count)]
    points = [[10.0 + 15.0 * i, 200.0] for i in range(count)]
    ious = box_iou(torch.tensor(gt), torch.tensor(pred_boxes))[0].tolist()
    assert all(a > b for a, b in zip(ious, ious[1:])) and min(ious) > 0.0, (
        f"fixture is degenerate: the metric must be strictly decreasing and "
        f"every candidate positive — {ious}"
    )

    sparse = _assign(
        model, torch, gt, [label], scores, pred_boxes, points, module.ONE2ONE_TOPK
    )[0]
    dense = _assign(
        model, torch, gt, [label], scores, pred_boxes, points, module.TAL_TOPK
    )[0]
    assert int(sparse.sum()) == module.ONE2ONE_TOPK, (
        f"{module.__name__}: at topk={module.ONE2ONE_TOPK} the assigner "
        f"selected {int(sparse.sum())} anchors for one ground truth, expected "
        f"{module.ONE2ONE_TOPK}. The one2one branch's output is only "
        f"NMS-free because its supervision is one-to-one."
    )
    assert int(dense.sum()) == module.TAL_TOPK, (
        f"{module.__name__}: at topk={module.TAL_TOPK} the assigner selected "
        f"{int(dense.sum())} anchors, expected {module.TAL_TOPK}"
    )
    assert bool(sparse[int(torch.tensor(ious).argmax())]), (
        f"{module.__name__}: the one2one branch's single positive is not the "
        f"best-ranked candidate. CONSISTENCY is the point: both branches rank "
        f"by the same metric, so the one2one positive must be the anchor the "
        f"one2many branch also ranks first."
    )


def guard_eval_decodes_the_one2one_branch(module) -> None:
    """Eval must decode the ONE2ONE branch, not the one2many one.

    The one2many branch was taught ten positives per object; decoding it with
    no NMS anywhere in the pipeline returns ten near-duplicate boxes per object
    and mAP collapses, while every loss stays healthy and every structural guard
    stays green. There is nothing downstream to notice: the engine's metrics
    just score the duplicates.

    Driven by rewiring the two branches' class predictors to fire on DIFFERENT
    channels, so the label of the top detection names which branch was decoded.
    A fresh model cannot be used for this — its prior is far below any useful
    score and its detections are noise.
    """
    import torch
    from torch import nn

    model = _build(module, 4, 64)
    head = model.head
    one2many_channel, one2one_channel = 1, 3
    assert one2many_channel != one2one_channel, "fixture: channels must differ"
    assert one2one_channel < model.num_classes

    # Only the PREDICTORS are rewired, not the towers: a zeroed weight makes the
    # output a constant whatever the tower emits, so the branch's own channel
    # widths are left intact and this fixture cannot be broken by a width change.
    for level in range(len(head.strides)):
        for preds, channel in (
            (head.cls_preds, one2many_channel),
            (head.one2one_cls_preds, one2one_channel),
        ):
            replacement = nn.Conv2d(preds[level].in_channels, model.num_classes, 1)
            nn.init.constant_(replacement.weight, 0.0)
            nn.init.constant_(replacement.bias, -20.0)
            with torch.no_grad():
                replacement.bias[channel] = 20.0
            preds[level] = replacement

    model.eval()
    with torch.no_grad():
        prediction = model([torch.rand(3, 64, 64)])[0]

    labels = prediction["labels"]
    scores = prediction["scores"]
    assert labels.numel(), (
        f"{module.__name__}: the rewired model produced no detections, so this "
        f"guard is vacuous"
    )
    # ⚠️ NOT a claim about the label SET. SCORE_THRESH is 0.0 on this template,
    # so `_predictions` returns a full ranked budget and every label appears
    # somewhere in it — asserting `set(labels) == {channel}` fails on the
    # SHIPPED template, which is the trap recorded on `_synthetic_head_output`.
    # The rewiring puts sigmoid(+20) on one channel and sigmoid(-20) on the
    # rest, so the discriminating question is which channel owns the CONFIDENT
    # detections.
    confident = scores > 0.5
    assert bool(confident.any()), (
        f"{module.__name__}: no detection scored above 0.5 although a class "
        f"bias of +20 was wired in (sigmoid(20) ~ 1.0), so this fixture is not "
        f"exercising the decode. Top score was {float(scores.max()):.6f}."
    )
    owners = sorted(set(labels[confident].tolist()))
    assert owners == [one2one_channel], (
        f"{module.__name__}: the confident detections carry labels {owners}. "
        f"The one2one branch was rewired to fire on channel {one2one_channel} "
        f"and the one2many branch on channel {one2many_channel}, so a correct "
        f"decode is confident about {one2one_channel} alone. Decoding the "
        f"one2many branch instead returns its TEN positives per object with no "
        f"suppression anywhere in this pipeline — duplicate-heavy predictions "
        f"and collapsed mAP, while every loss and every structural guard stays "
        f"green."
    )


def guard_class_tower_is_depthwise_separable(module) -> None:
    """The class tower must be YOLOv10's depthwise-separable pair, not YOLOv8's
    two dense 3x3 convolutions.

    This is most of the v10 head's parameter saving, and a copy of
    ``yolov8_s.py``'s tower here type-checks, trains, reports the same loss keys
    and is about 1.2M parameters too heavy at this scale — visible only to the
    published count, which is why the SHAPE is also pinned here where the
    failure is diagnosable.
    """
    from torch import nn

    head = _build(module, 3).head
    for group_name in ("cls_convs", "one2one_cls_convs"):
        tower = getattr(head, group_name)[0]
        convs = [sub for sub in tower.modules() if isinstance(sub, nn.Conv2d)]
        assert len(convs) == 4, (
            f"{module.__name__}: {group_name}[0] holds {len(convs)} "
            f"convolutions, published design has 4 — two depthwise-separable "
            f"pairs (depthwise 3x3 then pointwise 1x1, twice). YOLOv8's tower "
            f"has 2 DENSE 3x3s and is ~1.2M parameters heavier here."
        )
        depthwise = [c for c in convs if c.groups > 1 and c.groups == c.in_channels]
        pointwise = [c for c in convs if c.kernel_size == (1, 1) and c.groups == 1]
        assert len(depthwise) == 2, (
            f"{module.__name__}: {group_name}[0] has {len(depthwise)} depthwise "
            f"convolution(s), expected 2. A DENSE 3x3 in either position is "
            f"YOLOv8's head and trains identically."
        )
        assert len(pointwise) == 2, (
            f"{module.__name__}: {group_name}[0] has {len(pointwise)} pointwise "
            f"convolution(s), expected 2"
        )
        for conv in depthwise:
            assert conv.kernel_size == (3, 3), (
                f"{module.__name__}: a depthwise conv in {group_name}[0] has "
                f"kernel {conv.kernel_size}, expected 3x3"
            )


def guard_head_is_decoupled(module) -> None:
    """The classification and box towers must share no parameters, in BOTH
    branches.

    Checked by parameter identity rather than by reading the constructor, and
    rather than by relying on a crash. On this head the two towers happen to be
    different widths, so the crudest coupling raises a shape error on the first
    forward — that is luck, not design, and a width-matched coupling would be
    silent. Identity catches it either way.
    """
    head = _build(module, 3).head
    for cls_name, box_name in (
        ("cls_convs", "box_convs"),
        ("one2one_cls_convs", "one2one_box_convs"),
    ):
        cls_ids = {id(p) for p in getattr(head, cls_name).parameters()}
        box_ids = {id(p) for p in getattr(head, box_name).parameters()}
        assert cls_ids and box_ids, (
            f"expected both towers to hold parameters, got {len(cls_ids)} "
            f"{cls_name} / {len(box_ids)} {box_name}"
        )
        shared = cls_ids & box_ids
        assert not shared, (
            f"{module.__name__}: {cls_name} and {box_name} share "
            f"{len(shared)} parameter tensor(s) — the head is COUPLED, not "
            f"decoupled. It would train and log identical loss keys either way."
        )


# --------------------------------------------------------------------------
# head geometry and decode
# --------------------------------------------------------------------------


#: Non-square, and all three DIFFERENT. On a square feature map the row-major
#: and column-major flattenings coincide exactly, so a transposed reshape is
#: invisible — and the template never builds a non-square map, because the
#: transform pads every batch to a square. That is precisely why the ordering
#: has to be driven by hand.
_ORDERING_SHAPES = ((3, 5), (2, 3), (1, 2))


def _positional_feature(torch, channels, height, width):
    """A feature map whose channel 0 holds ``y * 100 + x`` at every location, so
    an output value identifies the cell that produced it."""
    ys = torch.arange(height, dtype=torch.float32).unsqueeze(1)
    xs = torch.arange(width, dtype=torch.float32).unsqueeze(0)
    feature = torch.zeros(1, channels, height, width)
    feature[0, 0] = ys * 100.0 + xs
    return feature


def _zero_conv(nn, in_channels, out_channels):
    conv = nn.Conv2d(in_channels, out_channels, 1)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


def _select_channel_zero(nn, in_channels, out_channels):
    """A 1x1 conv that copies input channel 0 into output channel 0."""
    conv = _zero_conv(nn, in_channels, out_channels)
    conv.weight.data[0, 0, 0, 0] = 1.0
    return conv


def guard_head_flatten_order_matches_the_anchor_table(module) -> None:
    """Each branch's flattened predictions and the anchor table returned
    alongside them must agree, cell for cell.

    Rewires level 0 so its class channel 0 is literally its input feature's
    channel 0, feeds a map coding ``y * 100 + x``, and reads the codes back in
    the order the head emitted them — then compares against the cell
    coordinates the head returned in the same call. If the two are flattened in
    different orders, every anchor is matched against another cell's prediction:
    the assigner's geometry and the classifier's evidence come from different
    places, every loss stays finite, and the model simply cannot learn to
    localise.

    Checked on the ONE2ONE branch as well as the one2many one. The template
    routes both through one ``_branch_forward``, so today they cannot disagree —
    but that is a property of the current factoring, and a future split into two
    methods would be invisible here otherwise. Each branch is only ever compared
    against its own anchor table, so a divergence between them has nothing else
    to notice it.
    """
    import torch
    from torch import nn

    model = _build(module, 2)
    head = model.head
    channels_per_level = [
        tower[0][0].conv.in_channels for tower in head.cls_convs
    ]

    features = []
    for level, (height, width) in enumerate(_ORDERING_SHAPES):
        channels = channels_per_level[level]
        for convs, preds in (
            (head.cls_convs, head.cls_preds),
            (head.one2one_cls_convs, head.one2one_cls_preds),
        ):
            convs[level] = nn.Identity()
            preds[level] = _select_channel_zero(nn, channels, model.num_classes)
        for convs, preds in (
            (head.box_convs, head.box_preds),
            (head.one2one_box_convs, head.one2one_box_preds),
        ):
            convs[level] = nn.Identity()
            preds[level] = _zero_conv(nn, channels, 4 * head.reg_max)
        features.append(_positional_feature(torch, channels, height, width))

    with torch.no_grad():
        one2many, one2one = head(tuple(features))

    height, width = _ORDERING_SHAPES[0]
    assert height != width, (
        "fixture is degenerate: level 0's feature map is square, so the two "
        "flattening orders coincide and a transposed reshape cannot be seen"
    )
    count = height * width
    for branch_name, (cls_logits, _, anchors) in (
        ("one2many", one2many),
        ("one2one", one2one),
    ):
        # The anchor table holds cell centres, so subtract the half-cell offset
        # to recover the integer cell the code was written into.
        cells = [(float(a[0]) - 0.5, float(a[1]) - 0.5) for a in anchors[:count]]
        expected = [y * 100.0 + x for x, y in cells]
        actual = [float(v) for v in cls_logits[0, :count, 0]]
        assert actual == expected, (
            f"{module.__name__}: the {branch_name} branch's flattened "
            f"predictions do not line up with the anchor table returned "
            f"alongside them. Cell (y, x) codes read back as {actual}, expected "
            f"{expected}. The predictions and the anchor coordinates are "
            f"flattened in DIFFERENT orders, which is invisible on the square "
            f"feature maps the template actually builds."
        )


def guard_decode_scales_by_each_levels_stride(module) -> None:
    """A one-cell distance must decode to a box the size of **that anchor's**
    stride, so the three levels cover three object scales.

    Using one stride everywhere leaves every loss finite and the train step
    green; the model just cannot represent small objects.
    """
    import torch

    reg_max = module.REG_MAX
    anchors = torch.tensor([[0.5, 0.5, 8.0], [0.5, 0.5, 32.0]])
    # One-hot on bin 1 -> a distance of exactly one cell on all four edges.
    dist = torch.full((1, 2, 4, reg_max), -50.0)
    dist[..., 1] = 50.0
    decoded = module._decode_boxes(dist, anchors)

    widths = [
        float(decoded[0, 0, 2] - decoded[0, 0, 0]),
        float(decoded[0, 1, 2] - decoded[0, 1, 0]),
    ]
    assert widths == pytest.approx([16.0, 64.0], abs=1e-3), (
        f"{module.__name__}: a one-cell distance decoded to widths {widths} at "
        f"strides (8, 32); expected [16.0, 64.0] — two cells across at each "
        f"level's own stride. The per-anchor stride column is not reaching the "
        f"decode, so every level predicts at one scale."
    )
    assert widths[0] != widths[1], "fixture is degenerate: the two strides agree"


def guard_dfl_decode_is_the_softmax_expectation(module) -> None:
    """The box distribution must be decoded by its **expectation under a
    softmax**, not by an argmax and not from raw logits.

    Both mistakes train: the DFL loss is computed on the logits, so it never
    sees this function. An argmax quantises every box edge to whole cells (and
    kills the gradient through the decode); dropping the softmax lets the logit
    magnitude scale the distance arbitrarily.
    """
    import torch

    reg_max = module.REG_MAX
    peak = 5
    assert peak != 0, (
        "fixture is degenerate: a one-hot on bin 0 decodes to 0.0 with or "
        "without the softmax, because the bin index it is multiplied by is zero"
    )

    one_hot = torch.full((1, 1, 4, reg_max), -50.0)
    one_hot[..., peak] = 50.0
    assert module._distribution_to_distance(one_hot).flatten().tolist() == (
        pytest.approx([float(peak)] * 4, abs=1e-3)
    ), (
        f"{module.__name__}: a distribution concentrated on bin {peak} must "
        f"decode to a distance of {peak} cells"
    )

    split = torch.full((1, 1, 4, reg_max), -50.0)
    split[..., peak] = 50.0
    split[..., peak + 1] = 50.0
    halves = module._distribution_to_distance(split).flatten().tolist()
    assert halves == pytest.approx([peak + 0.5] * 4, abs=1e-3), (
        f"{module.__name__}: an even split between bins {peak} and {peak + 1} "
        f"decoded to {halves}, expected {peak + 0.5}. An argmax over the "
        f"distribution gives {peak} or {peak + 1} and quantises every box edge "
        f"to whole cells — the loss is computed on the logits and never sees it."
    )

    uniform = torch.full((1, 1, 4, reg_max), 3.0)
    mean_bin = (reg_max - 1) / 2.0
    flat = module._distribution_to_distance(uniform).flatten().tolist()
    assert flat == pytest.approx([mean_bin] * 4, abs=1e-3), (
        f"{module.__name__}: uniform logits of 3.0 decoded to {flat}, expected "
        f"{mean_bin} (the mean bin index). Without the softmax this is "
        f"3.0 * sum(range({reg_max})) = "
        f"{3.0 * sum(range(reg_max))} — finite, trainable, and nonsense."
    )


def guard_dfl_target_is_in_cell_units_of_its_own_level(module) -> None:
    """The DFL regression target must be expressed in **that anchor's** cell
    units, so the same pixel box is a different target at a different stride.

    Dividing by one stride everywhere leaves every loss finite and simply asks
    the coarse levels to predict distances they cannot represent — the top bin
    is ``REG_MAX - 1 = 15`` cells.
    """
    import torch

    reg_max = module.REG_MAX
    anchors = torch.tensor([[1.5, 1.5, 8.0], [1.5, 1.5, 32.0]])
    boxes = torch.tensor([[0.0, 0.0, 48.0, 48.0], [0.0, 0.0, 48.0, 48.0]])
    distances = module._boxes_to_distance(boxes, anchors, reg_max)

    fine = [float(v) for v in distances[0]]
    coarse = [float(v) for v in distances[1]]
    assert fine == pytest.approx([1.5, 1.5, 4.5, 4.5]), (
        f"{module.__name__}: a [0, 0, 48, 48] box against a stride-8 anchor at "
        f"cell centre (1.5, 1.5) gives distances {fine}; in cell units it is "
        f"[1.5, 1.5, 4.5, 4.5]"
    )
    assert coarse == pytest.approx([1.5, 1.5, 0.0, 0.0]), (
        f"{module.__name__}: the SAME box against a stride-32 anchor gives "
        f"{coarse}; in that level's cell units it is [1.5, 1.5, 0.0, 0.0]. One "
        f"stride is being used for every level, so the target no longer means "
        f"'cells at this anchor's own scale'."
    )
    assert fine != coarse, (
        "fixture is degenerate: the two strides produce the same target, so a "
        "single-stride bug cannot be seen"
    )


def guard_dfl_loss_interpolates_between_the_two_bracketing_bins(module) -> None:
    """The DFL loss must be cross-entropy against the two bins a real-valued
    target falls between, weighted by its distance to each.

    That is what makes the distribution learn a *sharp* mode at the right
    distance: the expectation decode is many-to-one, so a loss on the decoded
    distance alone leaves the distribution's shape unconstrained. Both plausible
    simplifications — collapsing the two bins into one, or averaging them
    without the weights — train perfectly happily and are checked here.
    """
    import torch

    reg_max = module.REG_MAX

    def one_hot(bin_index):
        logits = torch.full((1, 4, reg_max), -20.0)
        logits[..., bin_index] = 20.0
        return logits

    def even_split(low, high):
        logits = torch.full((1, 4, reg_max), -20.0)
        logits[..., low] = 20.0
        logits[..., high] = 20.0
        return logits

    integer_target = torch.full((1, 4), 3.0)
    at_three = float(module._distribution_focal_loss(one_hot(3), integer_target))
    at_four = float(module._distribution_focal_loss(one_hot(4), integer_target))
    assert at_three < at_four, (
        f"{module.__name__}: for an integer target of 3.0 the loss is "
        f"{at_three:.4f} on a distribution peaked at bin 3 and {at_four:.4f} at "
        f"bin 4. The two bracketing bins are being weighted equally instead of "
        f"by the target's distance to each, so the loss cannot tell 3 from 4."
    )

    half_target = torch.full((1, 4), 3.5)
    peak_low = float(module._distribution_focal_loss(one_hot(3), half_target))
    peak_high = float(module._distribution_focal_loss(one_hot(4), half_target))
    mixed = float(module._distribution_focal_loss(even_split(3, 4), half_target))
    assert peak_low == pytest.approx(peak_high, abs=1e-4), (
        f"{module.__name__}: a target of 3.5 sits exactly between bins 3 and 4, "
        f"so a distribution peaked at either must cost the same — got "
        f"{peak_low:.4f} and {peak_high:.4f}"
    )
    assert mixed < peak_low - 1e-3, (
        f"{module.__name__}: for a target of 3.5 an even split over bins 3 and "
        f"4 costs {mixed:.4f}, no better than concentrating on one of them "
        f"({peak_low:.4f}). The upper bin is not the lower one plus one, so the "
        f"loss has no fractional resolution at all and every box edge is pinned "
        f"to a whole cell."
    )


def _synthetic_head_output(torch, model, cells, stride, confident):
    """``(cls_logits, dist_logits, anchors)`` with named confident detections.

    ``confident`` is ``{(image, cell): channel}``. Everything else is set to
    ``-10``, i.e. ``sigmoid(-10) = 4.5e-5``, so the named entries dominate the
    ranking by four orders of magnitude and every assertion downstream is about
    them. ⚠️ Note this template's ``SCORE_THRESH`` is 0.0, so the quiet entries
    are NOT filtered out the way they would be on the NMS siblings — they are
    simply ranked below. Assertions here are therefore about the TOP detection,
    never about how many came back.
    """
    images = 1 + max(image for image, _ in confident)
    anchors = torch.tensor(
        [[x + 0.5, y + 0.5, float(stride)] for x, y in cells], dtype=torch.float32
    )
    cls_logits = torch.full((images, len(cells), model.num_classes), -10.0)
    dist_logits = torch.full((images, len(cells), 4, model.head.reg_max), -50.0)
    # One-hot on bin 1: a two-cell box centred on its own anchor.
    dist_logits[..., 1] = 50.0
    for (image, cell), channel in confident.items():
        cls_logits[image, cell, channel] = 10.0
    return cls_logits, dist_logits, anchors


def guard_decode_is_per_image_and_aligned(module) -> None:
    """Decoding is driven directly, with named confident anchors, at batch two.

    A freshly built DFL head predicts around ``sigmoid(-9)`` on every class, and
    this template's ``SCORE_THRESH`` is 0.0 by design — so ``model(images)`` at
    initialisation returns a full 300 boxes of pure noise and any assertion
    downstream of it is worse than vacuous: it looks like coverage. That is how
    a real defect shipped through every guard on a sibling template — its
    post-processing iterated the wrong axis and ``zip`` truncated silently
    instead of raising, so it processed one image only and was broken outright
    at batch > 1. Both halves are addressed here: synthetic head outputs with
    one confident detection per image, and **two** images.
    """
    import torch

    model = _build(module, 2)
    classes = model.num_classes
    assert classes >= 3, "need a background channel and two real classes"
    cells = [(index, 0) for index in range(6)]

    cls_logits, dist_logits, anchors = _synthetic_head_output(
        torch, model, cells, 8, {(0, 1): 2, (1, 4): 1}
    )
    results = model._predictions(
        cls_logits, dist_logits, anchors, [(64, 64), (64, 64)]
    )

    assert isinstance(results, list) and len(results) == 2, (
        f"{module.__name__}: decoding two images returned "
        f"{len(results) if isinstance(results, list) else type(results).__name__}"
        f" result(s), expected 2. The engine's handler indexes predictions per "
        f"image; a truncating zip over the wrong axis loses images silently and "
        f"is invisible at batch one."
    )
    # Channel 0 is sliced off, so head channel c is emitted as label c.
    for index, (expected_label, expected_x) in enumerate([(2, 12.0), (1, 36.0)]):
        prediction = results[index]
        assert prediction["boxes"].numel(), (
            f"{module.__name__}: image {index} produced no detection at all, so "
            f"the fixture is not exercising the decode"
        )
        best = int(prediction["scores"].argmax())
        label = int(prediction["labels"][best])
        box = prediction["boxes"][best]
        centre_x = float((box[0] + box[2]) / 2.0)
        assert label == expected_label, (
            f"{module.__name__}: image {index}'s top detection is class "
            f"{label}, expected {expected_label}. The score, label and box "
            f"columns are flattened independently and have come apart."
        )
        assert abs(centre_x - expected_x) < 1.0, (
            f"{module.__name__}: image {index}'s top detection is centred at "
            f"x={centre_x:.1f}, expected {expected_x:.1f} — the confident "
            f"anchor was paired with another anchor's box."
        )

    # ⚠️ SECOND SCENARIO, same driver: the BACKGROUND channel strongest.
    # Channel 0 is never a positive target -- since backend#3062 the family
    # handler hands this template model space [1, C], so channel 0 trains only
    # as a negative. Emitting it spends a detection slot the real object should
    # have had, and yields dataset label -1 once the handler shifts back. It
    # binds HARDER here than on the NMS siblings: the whole decode is a top-k
    # budget and nothing downstream can recover the slot.
    #
    # This needs its own fixture because the one above deliberately puts its
    # confident scores on REAL classes, so it cannot see a channel-0 leak.
    background_logit, real_logit = 10.0, 8.0
    bg_cls, bg_dist, bg_anchors = _synthetic_head_output(
        torch, model, cells, 8, {(0, 2): 0}
    )
    bg_cls[0, 2, classes - 1] = real_logit  # a real class too, but weaker
    bg_results = model._predictions(bg_cls, bg_dist, bg_anchors, [(64, 64)])
    bg_labels = bg_results[0]["labels"]
    assert bg_labels.numel(), (
        f"{module.__name__}: the background-channel fixture decoded to nothing, "
        f"so the assertions below are vacuous"
    )
    assert not bool((bg_labels == 0).any()), (
        f"{module.__name__}: decode returned label 0, the background channel: "
        f"{sorted(set(bg_labels.tolist()))}. It is trained only as a negative "
        f"and must be dropped BEFORE the top-k, not left to the engine — the "
        f"detection budget is spent here."
    )

    # ⚠️ AND NOT ONLY BY ITS LABEL. Asserting "no label 0" catches the decode
    # that emits channel 0 AS label 0 and nothing else: a decode that keeps the
    # channel but maps it to some OTHER label — rotating the background column
    # to the end, say — passes that assertion, passes the first fixture above
    # (every real class still lands on its own label), and still spends a
    # detection slot on a channel trained only as a negative. So the SCORE is
    # asserted too: the strongest surviving detection must be the real class's
    # sigmoid(8), never the background channel's larger sigmoid(10).
    leaked = float(torch.sigmoid(torch.tensor(background_logit)))
    kept = float(torch.sigmoid(torch.tensor(real_logit)))
    assert leaked > kept, (
        "fixture is degenerate: the background channel must be the STRONGER of "
        "the two, or its leak is indistinguishable from the real detection"
    )
    best_score = float(bg_results[0]["scores"].max())
    assert best_score == pytest.approx(kept, abs=1e-4), (
        f"{module.__name__}: the strongest detection on the background fixture "
        f"scores {best_score:.6f}. The cell carries +{background_logit} on the "
        f"background channel and +{real_logit} on a real class, so a correct "
        f"decode returns {kept:.6f} (the real class) and a decode that lets "
        f"channel 0 through returns {leaked:.6f} under SOME label. Its label is "
        f"not evidence on its own — the channel can be kept and renamed."
    )


def guard_decode_is_nms_free(module) -> None:
    """The decode must apply NO suppression.

    This is the property that genuinely separates this template from its two
    NMS siblings, and the one worth guarding — see the note in the module
    docstring about why the two-stage top-k is NOT separately guardable (it is
    an efficiency factorisation of a single flat top-k, proven and measured).

    The fixture gives several anchors that decode to the **same box** at the
    **same class** with the same score. Any NMS, at any IoU threshold, collapses
    those to one; a one-to-one head returns all of them, because its
    supervision guarantees it will not produce them in the first place and there
    is nothing in this pipeline that would clean them up if it did.
    """
    import torch

    model = _build(module, 2)
    duplicates = 5
    # Every anchor at the SAME point, so every decoded box is identical.
    cells = [(0, 0)] * duplicates
    confident = {(0, index): 1 for index in range(duplicates)}
    cls_logits, dist_logits, anchors = _synthetic_head_output(
        torch, model, cells, 8, confident
    )
    results = model._predictions(cls_logits, dist_logits, anchors, [(64, 64)])
    boxes = results[0]["boxes"]
    labels = results[0]["labels"]

    top = float(results[0]["scores"].max())
    at_top = int((results[0]["scores"] >= top - 1e-6).sum())
    unique_boxes = {tuple(round(float(v), 3) for v in box) for box in boxes}
    assert len(unique_boxes) < len(boxes) or len(boxes) == 1, (
        "fixture is degenerate: the fixture did not actually produce duplicate "
        f"boxes ({len(unique_boxes)} unique of {len(boxes)})"
    )
    assert at_top == duplicates, (
        f"{module.__name__}: {duplicates} anchors decoding to the SAME box at "
        f"the same class and score returned {at_top} detection(s) at the top "
        f"score, expected {duplicates}. Something is suppressing duplicates — "
        f"but this head is one-to-one and its output needs no suppression. An "
        f"NMS here is not a safety net: at the IoU thresholds the sibling "
        f"templates use it merges genuinely distinct overlapping objects, and "
        f"it re-introduces the latency this architecture exists to remove."
    )
    assert sorted(set(labels.tolist()))[:1] == [1], (
        f"{module.__name__}: the duplicated detections came back under labels "
        f"{sorted(set(labels.tolist()))[:3]}, expected the fixture's class 1"
    )


def guard_predictions_are_in_original_image_coordinates(module) -> None:
    """Eval predictions must be mapped back to each input image's own frame.

    The engine's metrics compare predictions against targets in the dataset's
    pixel space, so a detector returning boxes in its internal resized frame
    scores near-zero mAP while the training loss falls normally — invisible to
    every other check here. The fixture is deliberately **non-square and much
    smaller than the transform's edge**, so the internal frame is several times
    larger and an unmapped box lands far outside the image.
    """
    import torch

    edge = 128
    height, width = 24, 32
    model = _build(module, 3, edge)
    scale = min(edge / height, edge / width)
    assert scale >= 3.0, (
        f"fixture is degenerate: the internal frame is only {scale:.1f}x the "
        f"input, so an unmapped box could still land inside the image"
    )

    model.eval()
    with torch.no_grad():
        prediction = model([torch.rand(3, height, width)])[0]

    boxes = prediction["boxes"]
    assert boxes.numel(), f"{module.__name__}: no predictions to check"
    max_x = float(boxes[:, 0::2].max())
    max_y = float(boxes[:, 1::2].max())
    assert max_x <= width + 1.0 and max_y <= height + 1.0, (
        f"{module.__name__}: predictions reach x={max_x:.1f}, y={max_y:.1f} for "
        f"a {height}x{width} input — they are still in the model's internal "
        f"{int(height * scale)}x{int(width * scale)} frame. The metrics read "
        f"them as dataset pixels, so mAP would be near zero while the loss fell "
        f"normally. Call transform.postprocess()."
    )


def guard_declared_image_size_is_the_measured_edge(module) -> None:
    """The declared ``image_size`` must be the resolution the backbone receives.

    The **family-wide** version of this is
    ``tests/test_od_declared_resolution.py`` (backend#3058), which reads the
    effective resolution off the transform's configured ``min_size``. What this
    adds, for this template only, is a stronger measurement: a forward hook
    reports the spatial size of the tensor the backbone is **actually handed**,
    after the pad to ``size_divisible=32``, and asserts it is **square** — which
    the family guard does not, since it compares a single edge.
    """
    import torch

    declared = module.image_size
    assert isinstance(declared, int) and declared > 0, (
        f"{module.__name__}: image_size must be a positive int, got {declared!r}"
    )
    model = _build(module, 3)

    seen = []

    def record(_module, _inputs, output):
        tensors = output[0].tensors
        seen.append((int(tensors.shape[-2]), int(tensors.shape[-1])))

    handle = model.transform.register_forward_hook(record)
    try:
        model.eval()
        with torch.no_grad():
            model([torch.rand(3, declared, declared)])
    finally:
        handle.remove()

    assert seen, f"{module.__name__}: the transform hook never fired"
    height, width = seen[0]
    assert (height, width) == (declared, declared), (
        f"{module.__name__}: declares image_size={declared} but the backbone "
        f"receives {height}x{width} for a square input at the declared edge. "
        f"The SDK hands image_size to the edge to size the dataset, so a "
        f"template that then rescales trains on the declared resolution "
        f"stretched to another one — paying the resize twice and losing the "
        f"detail it could have had (backend#3058). Build the transform with "
        f"min_size == max_size == image_size, or declare {height}."
    )


def guard_seed_excluded_prefixes_are_exactly_the_class_shaped_keys(module) -> None:
    """``SEED_EXCLUDED_PREFIXES`` must name every class-count-dependent tensor
    and nothing else.

    Re-derives the declaration the way ``tools/derive_seed_excluded.py`` does —
    build twice at different class counts and diff the state_dict shapes — so a
    head that grows a second class-shaped tensor, or a declared prefix that has
    gone stale, is red here rather than an edge-only strict-load failure
    (backend#2642).

    ⚠️ THIS TEMPLATE HAS TWICE THE CLASS-SHAPED TENSORS OF ITS SIBLINGS. Both
    the one2many and the one2one branch carry a per-level class predictor and
    both are sized from ``output_classes``, so a prefix list copied from
    ``yolov8_s.py`` or ``yolov9_s.py`` covers exactly half of them — and a
    hosted seed would ship the other three, which is the precise shape mismatch
    backend#2642 exists to remove.

    It also pins the property the declaration DEPENDS on: the class tower's
    width, ``max(in_channels[0], min(num_classes, 100))``, is 128 for every
    class count at this width because ``in_channels[0]`` is 128 and the class
    term is capped at 100. That is a property of THIS scale, not of the formula
    — rebuild at YOLOv10-N's width (64 at P3) and ``cls_hidden`` becomes 80 at
    80 classes, i.e. class-count dependent, at which point the seed would carry
    more tensors per level than the declaration admits.
    """
    low, high = 7, 61
    a = _build(module, low)
    b = _build(module, high)

    assert a.head.cls_hidden == b.head.cls_hidden, (
        f"{module.__name__}: the class tower is {a.head.cls_hidden} channels "
        f"wide at {low} classes and {b.head.cls_hidden} at {high}. The seed "
        f"contract assumes only the 1x1 predictors depend on the class count; a "
        f"class-count-dependent tower belongs in SEED_EXCLUDED_PREFIXES too."
    )

    left, right = a.state_dict(), b.state_dict()
    assert set(left) == set(right), (
        f"{module.__name__}: the two builds have different KEY SETS, not just "
        f"different shapes — something other than the head moved with the class "
        f"count, and a prefix list would paper over it: "
        f"{sorted(set(left) ^ set(right))[:6]}"
    )
    changed = sorted(key for key in left if left[key].shape != right[key].shape)
    assert changed, (
        f"{module.__name__}: NO tensor changed shape between {low} and {high} "
        f"classes, so this guard is checking nothing — the head is not sized "
        f"from the class count at all"
    )

    declared = tuple(module.SEED_EXCLUDED_PREFIXES)
    assert declared, f"{module.__name__}: SEED_EXCLUDED_PREFIXES is empty"
    uncovered = [
        key for key in changed if not any(key.startswith(p) for p in declared)
    ]
    assert not uncovered, (
        f"{module.__name__}: class-shaped tensor(s) {uncovered} are not covered "
        f"by SEED_EXCLUDED_PREFIXES {declared}. A hosted seed would carry them, "
        f"and they fit only the one class count it was built with — the exact "
        f"shape mismatch backend#2642 exists to remove. NOTE this template's "
        f"DUAL head means there are two class predictors per level, so a "
        f"prefix list copied from a single-head sibling covers half of them. "
        f"Re-run tools/derive_seed_excluded.py and tools/seed_contract.py "
        f"apply; do not hand-edit the constant."
    )
    dead = [
        prefix
        for prefix in declared
        if not any(key.startswith(prefix) for key in changed)
    ]
    assert not dead, (
        f"{module.__name__}: declared prefix(es) {dead} match no "
        f"class-shaped tensor. A stale prefix silently allows a real key to go "
        f"missing from a seed. Re-derive it."
    )
    # Both branches must be represented, stated separately so the failure names
    # the one that is missing rather than only "a key is uncovered".
    for marker in ("head.cls_preds.", "head.one2one_cls_preds."):
        assert any(prefix.startswith(marker) for prefix in declared), (
            f"{module.__name__}: SEED_EXCLUDED_PREFIXES names no {marker}* "
            f"prefix. Both the one2many and the one2one class predictor are "
            f"sized from output_classes and both must be excluded from a seed."
        )


def guard_reg_max_reaches_the_head_and_the_decode(module) -> None:
    """``REG_MAX`` must be a live knob, not a constant that reaches nothing.

    The failure mode this is written against is a declared parameter that is
    passed to a stage and never used: changing it alters neither the model nor
    any test, so it reads as configuration and is decoration. Here it decides
    both branches' box channel counts, the head's reshape, the decode's bin
    vector and the DFL target's clamp — so the guard rebuilds at a DIFFERENT
    value and asserts all of them moved with it.
    """
    import torch

    assert module.REG_MAX == 16, (
        f"{module.__name__}: REG_MAX is {module.REG_MAX}; YOLOv10 publishes 16, "
        f"and the pinned totals and reference count assume it"
    )
    shipped = _build(module, 3)
    assert shipped.head.box_preds[0].out_channels == 4 * shipped.head.reg_max

    saved = module.REG_MAX
    probe = 8
    try:
        module.REG_MAX = probe
        model = _build(module, 3)
        assert model.reg_max == probe, (
            f"{module.__name__}: REG_MAX = {probe} but the model reports "
            f"reg_max = {model.reg_max}"
        )
        assert model.head.reg_max == probe, (
            f"{module.__name__}: REG_MAX = {probe} but the HEAD reports "
            f"reg_max = {model.head.reg_max}. A default argument is evaluated "
            f"once at class-definition time, so a head reading REG_MAX from its "
            f"own signature keeps the import-time value while the model tracks "
            f"the current one."
        )
        for group in ("box_preds", "one2one_box_preds"):
            for level, predictor in enumerate(getattr(model.head, group)):
                assert predictor.out_channels == 4 * probe, (
                    f"{module.__name__}: {group}[{level}] emits "
                    f"{predictor.out_channels} channels at REG_MAX = {probe}; "
                    f"the DFL head's width is 4 * reg_max = {4 * probe}. A "
                    f"hardcoded channel count leaves the knob decorative, and "
                    f"BOTH branches have to move with it."
                )

        model.eval()
        with torch.no_grad():
            (_, dist_logits, anchors), _ = model.head(
                model.neck(model.backbone(torch.rand(1, 3, 64, 64)))
            )
        assert dist_logits.shape[-2:] == (4, probe), (
            f"{module.__name__}: head emits {tuple(dist_logits.shape[-2:])} "
            f"per anchor at REG_MAX = {probe}, expected (4, {probe})"
        )
        distance = module._distribution_to_distance(dist_logits)
        assert float(distance.max()) <= probe - 1 + 1e-4, (
            f"{module.__name__}: the decode produced a distance of "
            f"{float(distance.max())} cells from {probe} bins; the expectation "
            f"cannot exceed {probe - 1}"
        )

        huge = torch.tensor([[-10_000.0, -10_000.0, 10_000.0, 10_000.0]])
        clamped = module._boxes_to_distance(huge, anchors[:1], probe)
        assert float(clamped.max()) == pytest.approx(probe - 1 - 0.01), (
            f"{module.__name__}: an enormous box's DFL target clamps to "
            f"{float(clamped.max())} at REG_MAX = {probe}; the top bin is "
            f"{probe - 1} and the target must stay strictly below it or the "
            f"upper interpolation bin is out of range"
        )
    finally:
        module.REG_MAX = saved


# --------------------------------------------------------------------------
# task-aligned assignment
#
# EVERY GUARD BELOW ASSERTS *WHICH* ANCHOR OR GROUND TRUTH IS SELECTED, AND
# WITH WHAT SOFT TARGET — never how many. Cardinality is invariant to any
# reweighting of the alignment metric: swap the two exponents and exactly the
# same number of anchors is chosen, from the wrong end of the ranking. That gap
# hid a swapped focal alpha in `sparse_rcnn` through a full mutation sweep and
# two review passes.
#
# These rules are SHARED BY BOTH BRANCHES — `assign` is one function called at
# two `topk` values — so a defect here breaks the one2many supervision AND the
# deployed head at once. The tests drive it at TAL_TOPK unless the rule is about
# the one-to-one case, which `dual_assignment_is_consistent` owns.
# --------------------------------------------------------------------------


def _assign(
    model, torch, gt_boxes, gt_labels, scores_by_anchor, pred_boxes, points, topk=None
):
    """Call ``assign`` with explicit per-anchor class probabilities.

    ``scores_by_anchor`` is ``[{channel: probability}, ...]``, one dict per
    anchor; unset channels are 0. Building the score matrix by hand is what lets
    a fixture separate "best classified" from "best localised".

    ``topk`` defaults to the one2many value because that is the branch these
    fixtures exercise; it is passed EXPLICITLY (the template's own ``assign``
    has no default, deliberately) so a fixture can also ask for the one2one
    assignment.
    """
    if topk is None:
        topk = 10
    scores = torch.zeros((len(scores_by_anchor), model.num_classes))
    for anchor, mapping in enumerate(scores_by_anchor):
        for channel, probability in mapping.items():
            scores[anchor, channel] = probability
    return model.assign(
        torch.tensor(gt_boxes),
        torch.tensor(gt_labels, dtype=torch.int64),
        scores,
        torch.tensor(pred_boxes),
        torch.tensor(points),
        topk,
    )


def guard_tal_metric_weights_localisation_over_classification(module) -> None:
    """The alignment metric must be ``score ** TAL_ALPHA * iou ** TAL_BETA``,
    with the published 0.5 and 6.0 — **in that order**.

    The two exponents are wildly asymmetric, so IoU dominates: a well-localised
    but poorly-classified anchor beats a confidently-classified but badly-boxed
    one. Swapping them reverses the ranking, selects exactly the same NUMBER of
    anchors, and leaves every loss finite — which is why this asserts which
    anchor receives the larger soft target rather than counting anything.

    On THIS template the metric carries extra weight: both branches rank by it,
    so its direction is also what makes the one2one head's single positive
    agree with the one2many head's best — the "consistent" in "consistent dual
    assignments".

    The fixture states its own discriminating property: under the published
    exponents anchor B wins clearly, under the swap anchor A does.
    """
    import torch
    from torchvision.ops import box_iou

    model = _build(module, 3)
    label = 1
    gt = [[0.0, 0.0, 100.0, 100.0]]
    # A: badly boxed, confidently classified. B: well boxed, barely classified.
    pred_boxes = [[19.0, 19.0, 119.0, 119.0], [6.0, 6.0, 106.0, 106.0]]
    scores = [{label: 0.9}, {label: 0.3}]
    points = [[25.0, 25.0], [75.0, 75.0]]

    ious = box_iou(torch.tensor(gt), torch.tensor(pred_boxes))[0]
    iou_a, iou_b = float(ious[0]), float(ious[1])
    published = [
        0.9**module.TAL_ALPHA * iou_a**module.TAL_BETA,
        0.3**module.TAL_ALPHA * iou_b**module.TAL_BETA,
    ]
    swapped = [
        0.9**module.TAL_BETA * iou_a**module.TAL_ALPHA,
        0.3**module.TAL_BETA * iou_b**module.TAL_ALPHA,
    ]
    assert iou_b > iou_a and published[1] > 2.0 * published[0], (
        f"fixture is degenerate: anchor B must be the better-localised one and "
        f"must win clearly under the published exponents — IoUs "
        f"({iou_a:.3f}, {iou_b:.3f}), metrics {published}"
    )
    assert swapped[0] > 2.0 * swapped[1], (
        f"fixture is degenerate: under swapped exponents anchor A must win, or "
        f"this guard cannot tell the two orders apart — {swapped}"
    )

    fg_mask, labels, _, aligned = _assign(
        model, torch, gt, [label], scores, pred_boxes, points
    )
    assert bool(fg_mask.all()), (
        f"fixture is degenerate: both anchors must be selected so the "
        f"comparison is about the soft TARGET rather than about cardinality — "
        f"got {fg_mask.tolist()}"
    )
    assert labels.tolist() == [label, label]
    better = int(aligned.argmax())
    assert better == 1, (
        f"{module.__name__}: the larger soft target went to anchor {better}, "
        f"the confidently-classified but badly-boxed one. The task-aligned "
        f"metric is score ** {module.TAL_ALPHA} * iou ** {module.TAL_BETA}, so "
        f"localisation dominates and anchor 1 must win: soft targets "
        f"{aligned.tolist()} against IoUs ({iou_a:.3f}, {iou_b:.3f}). Swapping "
        f"the exponents selects the same NUMBER of anchors from the wrong end "
        f"of the ranking and every loss stays finite."
    )


def guard_tal_requires_the_anchor_point_inside_the_box(module) -> None:
    """A candidate's anchor POINT must lie inside the ground-truth box.

    Without that prefilter a box that merely overlaps well is assigned to an
    anchor sitting somewhere else entirely — the classifier is then trained to
    fire at a location the object does not cover, every loss stays finite, and
    the detector learns a systematic offset. The fixture gives the OUTSIDE
    anchor both the better IoU and the better score, so it would win on metric
    alone.
    """
    import torch
    from torchvision.ops import box_iou

    model = _build(module, 3)
    label = 1
    gt = [[0.0, 0.0, 100.0, 100.0]]
    pred_boxes = [[20.0, 20.0, 80.0, 80.0], [5.0, 5.0, 105.0, 105.0]]
    scores = [{label: 0.4}, {label: 0.9}]
    points = [[50.0, 50.0], [150.0, 150.0]]  # second is OUTSIDE the box

    ious = box_iou(torch.tensor(gt), torch.tensor(pred_boxes))[0]
    assert float(ious[1]) > float(ious[0]), (
        f"fixture is degenerate: the outside anchor must have the BETTER IoU, "
        f"or dropping the inside rule would change nothing — {ious.tolist()}"
    )

    fg_mask, _, _, _ = _assign(model, torch, gt, [label], scores, pred_boxes, points)
    assert bool(fg_mask[0]), (
        f"{module.__name__}: the anchor inside the box was not selected at all, "
        f"so this guard is not testing what it claims"
    )
    assert not bool(fg_mask[1]), (
        f"{module.__name__}: an anchor whose point is at (150, 150) — outside "
        f"the ground-truth box [0, 0, 100, 100] — was assigned to it, on the "
        f"strength of a better IoU ({float(ious[1]):.3f} against "
        f"{float(ious[0]):.3f}) and a better score. The geometric prefilter is "
        f"gone: the classifier is now trained to fire where the object is not, "
        f"and every loss stays finite."
    )


def guard_tal_selects_the_topk_best_ranked_candidates(module) -> None:
    """Exactly the ``topk`` **highest-metric** candidates per ground truth.

    Both halves matter and only one of them is a count. A fixture with more
    valid candidates than ``TAL_TOPK`` and a strictly monotone metric pins WHICH
    ones are chosen, so selecting the worst ``k``, or every candidate, is red
    even though "how many" is unchanged in the second case only.
    """
    import torch
    from torchvision.ops import box_iou

    model = _build(module, 3)
    label = 1
    count = 20
    assert count > module.TAL_TOPK, (
        f"fixture is degenerate: {count} candidates against TAL_TOPK = "
        f"{module.TAL_TOPK} — the bound cannot bind"
    )
    gt = [[0.0, 0.0, 400.0, 400.0]]
    # Strictly shrinking predictions, all inside the ground truth, so the IoU
    # (and therefore the metric) is strictly decreasing in the anchor index.
    pred_boxes = [
        [0.0, 0.0, 380.0 - 10.0 * i, 380.0 - 10.0 * i] for i in range(count)
    ]
    scores = [{label: 0.5} for _ in range(count)]
    points = [[10.0 + 15.0 * i, 200.0] for i in range(count)]

    ious = box_iou(torch.tensor(gt), torch.tensor(pred_boxes))[0].tolist()
    assert all(a > b for a, b in zip(ious, ious[1:])) and min(ious) > 0.0, (
        f"fixture is degenerate: the metric must be strictly decreasing and "
        f"every candidate positive, or 'which k' is not determined — {ious}"
    )

    fg_mask, _, _, _ = _assign(
        model, torch, gt, [label], scores, pred_boxes, points, module.TAL_TOPK
    )
    selected = sorted(int(i) for i in fg_mask.nonzero().flatten())
    expected = list(range(module.TAL_TOPK))
    assert selected == expected, (
        f"{module.__name__}: selected anchors {selected}, expected {expected} — "
        f"the {module.TAL_TOPK} with the highest alignment metric out of "
        f"{count} valid candidates whose metric is strictly decreasing in the "
        f"index. Note this asserts WHICH, not how many: taking the worst k "
        f"selects the same number of anchors."
    )


def guard_tal_target_is_the_normalised_alignment_metric(module) -> None:
    """A positive's soft target must be its alignment metric rescaled so the
    best-aligned anchor lands on its ground truth's best IoU.

    Neither head has an objectness branch: the score ranked at inference is the
    classifier's, so the classifier is what has to carry localisation quality. A
    hard 1.0 target trains happily and simply removes the model's ability to say
    "this is a car, but I have it badly boxed"; dropping the normalisation
    leaves the raw metric, which at ``iou ** 6`` is a small number and quietly
    rescales the whole classification loss.

    The fixture's best IoU is deliberately neither 0 nor 1 — at 1.0 the
    normalised target and a hard target are the same number and the rule cannot
    fire.
    """
    import torch
    from torchvision.ops import box_iou

    model = _build(module, 3)
    label = 1
    gt = [[0.0, 0.0, 100.0, 100.0]]
    pred_boxes = [
        [0.0, 0.0, 60.0, 60.0],
        [0.0, 0.0, 80.0, 80.0],
        [0.0, 0.0, 70.0, 70.0],
    ]
    scores = [{label: 0.5}, {label: 0.5}, {label: 0.5}]
    points = [[30.0, 30.0], [50.0, 50.0], [65.0, 65.0]]

    ious = box_iou(torch.tensor(gt), torch.tensor(pred_boxes))[0]
    best_iou = float(ious.max())
    assert 0.05 < best_iou < 0.95, (
        f"fixture is degenerate: the best IoU is {best_iou:.3f}. At 1.0 the "
        f"normalised target equals a hard 1.0 target and the rule cannot fire; "
        f"at 0 nothing is selected."
    )
    assert int(ious.argmax()) == 1, "fixture: anchor 1 should be the best boxed"

    fg_mask, _, _, aligned = _assign(
        model, torch, gt, [label], scores, pred_boxes, points
    )
    assert bool(fg_mask.all()), f"fixture: expected all three selected, {fg_mask}"
    top = float(aligned.max())
    assert int(aligned.argmax()) == 1, (
        f"{module.__name__}: the largest soft target is at anchor "
        f"{int(aligned.argmax())}, but anchor 1 has the best IoU "
        f"({ious.tolist()})"
    )
    assert top == pytest.approx(best_iou, abs=1e-4), (
        f"{module.__name__}: the best-aligned anchor's soft target is "
        f"{top:.4f}; normalised against its ground truth's best IoU it must be "
        f"{best_iou:.4f}. A hard target would give 1.0; the un-normalised "
        f"metric would give "
        f"{0.5 ** module.TAL_ALPHA * best_iou ** module.TAL_BETA:.4f}, which "
        f"rescales the entire classification loss."
    )


def guard_tal_breaks_ties_by_iou_not_by_alignment(module) -> None:
    """An anchor claimed by two ground truths goes to the one it overlaps best
    **by IoU** — the published tie-break, and not the same thing as the
    alignment metric.

    The fixture makes the two disagree: ground truth 0 has the higher alignment
    metric (it is far better classified) while ground truth 1 has the higher
    IoU. Both a by-alignment tie-break and NO tie-break at all award the anchor
    to ground truth 0, so this single assertion catches both.
    """
    import torch
    from torchvision.ops import box_iou

    model = _build(module, 3)
    gt = [[0.0, 0.0, 100.0, 100.0], [50.0, 0.0, 150.0, 100.0]]
    labels = [1, 2]
    pred_boxes = [[30.0, 0.0, 130.0, 100.0]]
    scores = [{1: 0.95, 2: 0.05}]
    points = [[75.0, 50.0]]  # inside BOTH ground truths

    ious = box_iou(torch.tensor(gt), torch.tensor(pred_boxes)).flatten()
    alignment = [
        float(s) ** module.TAL_ALPHA * float(i) ** module.TAL_BETA
        for s, i in ((0.95, ious[0]), (0.05, ious[1]))
    ]
    assert float(ious[1]) > float(ious[0]), (
        f"fixture is degenerate: ground truth 1 must have the better IoU — "
        f"{ious.tolist()}"
    )
    assert alignment[0] > alignment[1], (
        f"fixture is degenerate: ground truth 0 must have the better ALIGNMENT, "
        f"or a by-alignment tie-break would agree with a by-IoU one and the "
        f"rule cannot fire — {alignment}"
    )

    fg_mask, matched_labels, matched_boxes, _ = _assign(
        model, torch, gt, labels, scores, pred_boxes, points
    )
    assert bool(fg_mask.all()) and matched_labels.numel() == 1, (
        f"fixture: the single anchor must be selected exactly once, got "
        f"{fg_mask.tolist()} / {matched_labels.tolist()}"
    )
    assert int(matched_labels[0]) == labels[1], (
        f"{module.__name__}: the contested anchor was awarded to ground truth "
        f"with label {int(matched_labels[0])}; the published assigner resolves a "
        f"multi-claim by IoU, so it belongs to label {labels[1]} (IoU "
        f"{float(ious[1]):.3f} against {float(ious[0]):.3f}). Ground truth 0 "
        f"has the higher ALIGNMENT metric ({alignment[0]:.4f} against "
        f"{alignment[1]:.4f}), so both a by-alignment tie-break and no "
        f"tie-break at all land here."
    )
    assert matched_boxes[0].tolist() == gt[1], (
        f"{module.__name__}: the label and the box came from different ground "
        f"truths: {matched_boxes[0].tolist()} against {gt[1]}"
    )


# --------------------------------------------------------------------------
# end-to-end: does it train, and does it learn
# --------------------------------------------------------------------------


#: Attribute names for the head's per-level 1x1 predictors, split by what the
#: gradient below proves. A BOX predictor only receives gradient if something
#: was assigned as a POSITIVE; a CLASS predictor receives it from the negatives
#: too, so it cannot distinguish "assigned nothing" from "trained normally".
#: That asymmetry is the whole mechanism of the guard. BOTH branches are checked
#: — an assigner broken only at topk=1 would leave the one2many branch healthy
#: and the deployed head untrained.
_BOX_PREDICTOR_GROUPS = ("box_preds", "one2one_box_preds")
_CLASS_PREDICTOR_GROUPS = ("cls_preds", "one2one_cls_preds")


def guard_positives_reach_both_box_regression_branches(module) -> None:
    """One train step must leave BOTH branches' box predictors with a real
    gradient — which happens only if each assigner matched something.

    This is the assign-nothing guard, and it is ``requires_grad``-aware in the
    direction that matters: a bare ``p.grad is None`` sweep false-flags a
    deliberately frozen parameter, while the real defect is a **trainable**
    parameter the loss never reaches. Three assertions, failing for different
    reasons:

    * no trainable parameter may have a ``None`` gradient at all — that is a
      branch detached from the loss entirely;
    * each box group must have a non-zero gradient somewhere. The template falls
      back to ``prediction.sum() * 0.0`` when there are no positives (so the
      loss dict keeps its shape and no gradient is ``None``), which means an
      all-negative assignment shows up here as an exactly-zero box gradient and
      **nowhere else**;
    * each class group must too, as a sanity check on the fixture.

    Covering the one2one group separately matters here: an assigner that breaks
    only at ``topk=1`` leaves the one2many branch — and therefore the loss
    curve, and the backbone — perfectly healthy while the DEPLOYED head is
    never trained on a single positive.

    Deliberately NOT asserted: that all three levels receive positives. At
    random initialisation the alignment landscape is dominated by centre jitter
    rather than by scale, so which level wins a given ground truth is not
    deterministic, and a fixture pretending otherwise would pass for the wrong
    reason. Per-level structure is pinned by the deterministic geometry guards
    instead.
    """
    import torch

    edge = 320
    model = _build(module, 3, edge)
    model.train()

    # Object sizes matched to the three strides (8 / 16 / 32) and centred on a
    # grid point of each, so every level has a well-placed candidate.
    targets = [
        {
            "boxes": torch.tensor(
                [
                    [24.0, 24.0, 32.0, 32.0],
                    [72.0, 72.0, 88.0, 88.0],
                    [144.0, 144.0, 176.0, 176.0],
                ]
            ),
            "labels": torch.tensor([1, 2, 3], dtype=torch.int64),
        }
    ]
    losses = model([torch.rand(3, edge, edge)], targets)
    assert isinstance(losses, dict) and losses, (
        f"{module.__name__}: train mode returned {type(losses).__name__}, not a "
        f"non-empty loss dict"
    )
    total = sum(losses.values())
    assert torch.isfinite(total), f"{module.__name__}: total loss is {total!r}"
    total.backward()

    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing, (
        f"{module.__name__}: {len(missing)} TRAINABLE parameter(s) got no "
        f"gradient at all, so the loss never reaches them: {missing[:6]}"
    )

    def alive(group):
        predictors = getattr(model.head, group)
        assert len(predictors) == len(model.head.strides), (
            f"{group} has {len(predictors)} entries for "
            f"{len(model.head.strides)} levels — a rename must fail the lookup "
            f"rather than quietly narrowing this probe to nothing"
        )
        return [
            level
            for level, predictor in enumerate(predictors)
            if predictor.weight.grad is not None
            and float(predictor.weight.grad.abs().sum()) > 0.0
        ]

    for group in _BOX_PREDICTOR_GROUPS:
        assert alive(group), (
            f"{module.__name__}: every level of {group} received an exactly "
            f"zero gradient, so NO ground truth was assigned a positive anchor "
            f"in that branch. Three objects were supplied, one at each stride's "
            f"scale. An all-negative image still yields a finite, small loss "
            f"and a clean train step, which is why nothing else in this suite "
            f"sees it — and if this is the one2one group, the loss curve and "
            f"the backbone stay perfectly healthy while the DEPLOYED head "
            f"never trains."
        )
    for group in _CLASS_PREDICTOR_GROUPS:
        assert alive(group), (
            f"{module.__name__}: every level of {group} received a zero "
            f"gradient — that classification branch is detached from the loss"
        )


def guard_constructs_with_no_network(module) -> None:
    """The architecture must build with the network genuinely unavailable.

    ``tests/test_model_contract.py`` covers two thirds of this already: it greps
    the source for hub-fetch patterns and runs the whole session with
    ``HF_HUB_OFFLINE``. Neither closes the socket, so on a warm torch cache a
    template that fetches indirectly still passes — and this template is
    hand-written precisely so that nothing is fetched. So: point ``TORCH_HOME``
    and the hub caches at an empty directory and make DNS and socket creation
    raise, then build and run a forward pass.
    """
    import os
    import socket

    import torch

    original = (socket.socket, socket.getaddrinfo, socket.create_connection)

    def refuse(*_args, **_kwargs):
        raise OSError("network access is blocked by test_yolov10_s")

    with tempfile.TemporaryDirectory(prefix="tb-nonet-") as cache:
        keys = ("TORCH_HOME", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME")
        saved = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                os.environ[key] = cache
            socket.socket = refuse
            socket.getaddrinfo = refuse
            socket.create_connection = refuse
            model = _build(module, 3, 64)
            model.eval()
            with torch.no_grad():
                model([torch.rand(3, 64, 64)])
        except OSError as error:
            raise AssertionError(
                f"{module.__name__}: construction or a forward pass tried to "
                f"reach the network — {error}. This template is written from "
                f"scratch so that nothing is fetched; the #199 egress lockdown "
                f"means a fetch is an edge-only failure, invisible on a warm "
                f"local cache."
            ) from error
        finally:
            socket.socket, socket.getaddrinfo, socket.create_connection = original
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


#: Overfit-probe settings. 128 px rather than the declared 640 because this runs
#: many forward+backward passes and is a guard, not a benchmark; the entry point
#: takes the edge as its second argument precisely so the transform can be built
#: smaller. The thresholds below carry several times the measured margin, to
#: absorb BLAS variation across platforms.
_OVERFIT_STEPS = 200
_OVERFIT_LR = 4e-3
_OVERFIT_EDGE = 128
_OVERFIT_BOX = [40.0, 40.0, 88.0, 88.0]
_OVERFIT_LABEL = 1


def guard_overfits_a_single_object(module) -> None:
    """The template must actually LEARN — and then detect what it learned
    THROUGH THE ONE2ONE HEAD.

    Everything else in this file is a single step or a synthetic call. This is
    the end-to-end claim: ``_OVERFIT_STEPS`` Adam steps on one image with one
    object, then an eval pass that has to find it. It is the only guard that
    closes the loop from both assigners through all six losses to the NMS-free
    decode — and because eval reads the one2one branch, it is also the only
    guard that proves the DEPLOYED head learns anything at all from a head whose
    gradient is cut off from the backbone.

    ⚠️ AND HERE IS WHAT IT DOES **NOT** COVER. It does not see the missing
    ``detach``: the one2one head still learns, the losses still fall, and this
    guard stays green. The gradient graph is invisible from here, which is the
    entire reason ``guard_one2one_head_is_detached_from_the_backbone`` measures
    gradients directly rather than outcomes. Nor does it see a wrong assigner
    RANKING — on a sibling template an inverted IoU term in the alignment metric
    overfit a one-object fixture past every threshold like this one, because the
    metric only chooses *which* anchors are positive and the box loss then
    regresses whichever were chosen towards the true box. Both properties have
    deterministic guards of their own; the IoU assertion below is NOT evidence
    for either.
    """
    import torch
    from torchvision.ops import box_iou

    torch.manual_seed(0)
    model = _build(module, 2, _OVERFIT_EDGE)
    assert model.input_size == _OVERFIT_EDGE, (
        f"{module.__name__}: the entry point did not take the edge — got "
        f"input_size={model.input_size}. This probe needs the small build; fix "
        f"the call rather than dropping the check."
    )

    image = torch.rand(3, _OVERFIT_EDGE, _OVERFIT_EDGE)
    boxes = torch.tensor([_OVERFIT_BOX])
    targets = [{"boxes": boxes, "labels": torch.tensor([_OVERFIT_LABEL])}]

    optimizer = torch.optim.Adam(model.parameters(), lr=_OVERFIT_LR)
    model.train()
    first = None
    last = None
    for _ in range(_OVERFIT_STEPS):
        optimizer.zero_grad()
        total = sum(model([image], targets).values())
        last = float(total.detach())
        if first is None:
            first = last
        total.backward()
        optimizer.step()

    assert last < 0.5 * first, (
        f"{module.__name__}: {_OVERFIT_STEPS} steps on ONE image moved the loss "
        f"only {first:.3f} -> {last:.3f}. A detector that cannot overfit a "
        f"single object is not learning from its own assignments."
    )

    model.eval()
    with torch.no_grad():
        prediction = model([image])[0]

    assert prediction["boxes"].numel(), f"{module.__name__}: no predictions at all"
    best = int(prediction["scores"].argmax())
    score = float(prediction["scores"][best])
    label = int(prediction["labels"][best])
    iou = float(box_iou(prediction["boxes"][best : best + 1], boxes))

    assert score >= 0.10, (
        f"{module.__name__}: after overfitting one object the best score is "
        f"only {score:.3f}. The loss came down, so the model is training — but "
        f"the ONE2ONE head never learned to be CONFIDENT about anything, which "
        f"is what a broken soft classification target looks like. There is no "
        f"objectness branch here to carry that quality signal, so the "
        f"classifier is the only place it can live. Note this reads the "
        f"one2one branch: a healthy loss curve driven entirely by the one2many "
        f"branch would still fail here."
    )
    assert label == _OVERFIT_LABEL, (
        f"{module.__name__}: the best-scoring detection is class {label}, but "
        f"the single object it was trained on is class {_OVERFIT_LABEL}"
    )
    assert iou >= 0.35, (
        f"{module.__name__}: the best-scoring detection has IoU {iou:.3f} with "
        f"the one box it was trained on. The classifier learned the object and "
        f"the regressor did not follow, so the assigner is rewarding anchors "
        f"that cannot localise it, or the DFL target and the decode disagree. "
        f"NOTE this is a weaker signal than it looks — see the docstring."
    )


# --------------------------------------------------------------------------
# the guard table, and the mutations that prove each can go red
# --------------------------------------------------------------------------

GUARDS = {
    "published_architecture": guard_matches_the_published_architecture,
    "arch_table_is_live": guard_architecture_table_is_a_live_knob,
    "class_count_slope": guard_head_scales_linearly_with_the_class_count,
    "module_tree_size": guard_module_tree_size_is_pinned,
    "no_stateful_norm": guard_no_stateful_normalisation,
    "derived_norm_groups": guard_norm_groups_are_derived_from_the_channel_count,
    "scdown_depthwise_stride": guard_scdown_downsamples_with_the_depthwise_conv,
    "repvggdw_sums_first": guard_repvggdw_sums_its_branches_before_one_activation,
    "cib_inverted": guard_cib_is_inverted_and_mostly_depthwise,
    "c2f_fuses_intermediates": guard_c2f_fuses_every_intermediate_block,
    "sppf_series": guard_sppf_pools_in_series,
    "psa_partial": guard_psa_attends_to_only_half_the_channels,
    "attention_matches_manual": guard_attention_matches_the_manual_formula,
    "deepest_stage_modules_are_applied": (
        guard_sppf_and_psa_are_applied_to_the_deepest_map
    ),
    "attention_heads_derived": (
        guard_attention_head_count_is_derived_and_reaches_the_attention
    ),
    "one2one_detached": guard_one2one_head_is_detached_from_the_backbone,
    "dual_head_independent": guard_dual_head_branches_are_independent,
    "consistent_dual_assignment": guard_dual_assignment_is_consistent,
    "eval_uses_one2one": guard_eval_decodes_the_one2one_branch,
    "light_class_tower": guard_class_tower_is_depthwise_separable,
    "decoupled_head": guard_head_is_decoupled,
    "reg_max_is_live": guard_reg_max_reaches_the_head_and_the_decode,
    "seed_excluded_prefixes": (
        guard_seed_excluded_prefixes_are_exactly_the_class_shaped_keys
    ),
    "head_flatten_order": guard_head_flatten_order_matches_the_anchor_table,
    "decode_per_level_stride": guard_decode_scales_by_each_levels_stride,
    "dfl_decode_expectation": guard_dfl_decode_is_the_softmax_expectation,
    "dfl_target_cell_units": guard_dfl_target_is_in_cell_units_of_its_own_level,
    "dfl_loss_interpolates": (
        guard_dfl_loss_interpolates_between_the_two_bracketing_bins
    ),
    "decode_per_image": guard_decode_is_per_image_and_aligned,
    "decode_is_nms_free": guard_decode_is_nms_free,
    "original_coordinates": guard_predictions_are_in_original_image_coordinates,
    "declared_size_measured": guard_declared_image_size_is_the_measured_edge,
    "tal_metric_exponents": guard_tal_metric_weights_localisation_over_classification,
    "tal_inside_the_box": guard_tal_requires_the_anchor_point_inside_the_box,
    "tal_topk_ranking": guard_tal_selects_the_topk_best_ranked_candidates,
    "tal_normalised_target": guard_tal_target_is_the_normalised_alignment_metric,
    "tal_tie_break_by_iou": guard_tal_breaks_ties_by_iou_not_by_alignment,
    "positives_reach_box_branch": guard_positives_reach_both_box_regression_branches,
    "no_network": guard_constructs_with_no_network,
    "overfits_one_object": guard_overfits_a_single_object,
}

#: ``(name, anchor, replacement, guard)``. The anchor must be unique in the
#: file — ``_mutate`` refuses otherwise, so a drifted anchor is a RED rather
#: than a patch that silently applies to nothing and reports "passed".
MUTATIONS = [
    # ---- THE HEADLINE ONE -------------------------------------------------
    # No parameter, no shape, no loss key, no published figure moves. The
    # losses fall and the template still overfits one object. Only the
    # gradient graph differs.
    (
        "one2one_head_sees_undetached_features",
        "            [feature.detach() for feature in features],\n"
        "            self.one2one_box_convs,",
        "            [feature for feature in features],\n"
        "            self.one2one_box_convs,",
        "one2one_detached",
    ),
    # ---- architecture -----------------------------------------------------
    (
        "cib_expansion_halves_the_inner_mixer",
        "        return CIB(channels, channels, 1.0, shortcut)",
        "        return CIB(channels, channels, 0.5, shortcut)",
        "published_architecture",
    ),
    (
        "bottleneck_squeezes_the_branch_again",
        "        return Bottleneck(channels, channels, 1.0, shortcut)",
        "        return Bottleneck(channels, channels, 0.5, shortcut)",
        "published_architecture",
    ),
    (
        "scdown_used_at_every_transition",
        "            if index >= len(stages) - 2:",
        "            if index >= 0:",
        "published_architecture",
    ),
    (
        "neck_stride32_fusion_is_a_plain_c2f",
        "        self.bu_p5 = C2fCIB(c4 + c5, c5, n=blocks, shortcut=True)",
        "        self.bu_p5 = C2f(c4 + c5, c5, n=blocks, shortcut=True)",
        "published_architecture",
    ),
    # ⚠️ BOTH OF THESE ARE INVISIBLE TO THE PARAMETER COUNT. The module stays
    # CONSTRUCTED and is merely not applied, so every parameter, every buffer
    # and every state_dict key is unchanged and `published_architecture` cannot
    # see it — measured: it DID NOT RAISE. A whole module that reaches nothing
    # is the module-scale version of "a constant that reaches nothing".
    (
        "psa_dropped_from_the_backbone_forward",
        "        return outputs[1], outputs[2], self.psa(self.sppf(outputs[3]))",
        "        return outputs[1], outputs[2], self.sppf(outputs[3])",
        "deepest_stage_modules_are_applied",
    ),
    (
        "sppf_dropped_from_the_backbone_forward",
        "        return outputs[1], outputs[2], self.psa(self.sppf(outputs[3]))",
        "        return outputs[1], outputs[2], self.psa(outputs[3])",
        "deepest_stage_modules_are_applied",
    ),
    (
        "class_tower_copied_from_yolov8",
        """        return nn.Sequential(
            nn.Sequential(
                ConvNormAct(channels, channels, 3, stride=1, groups=channels),
                ConvNormAct(channels, self.cls_hidden, 1, stride=1),
            ),""",
        """        return nn.Sequential(
            nn.Sequential(
                ConvNormAct(channels, self.cls_hidden, 3, stride=1),
                ConvNormAct(self.cls_hidden, self.cls_hidden, 3, stride=1),
            ),""",
        "light_class_tower",
    ),
    (
        "p5_block_kind_hardcoded_at_the_shipped_scale",
        "        p5_kind = BACKBONE_P5_BLOCK",
        '        p5_kind = "c2f_cib"',
        "arch_table_is_live",
    ),
    (
        "stem_width_hardcoded_at_the_shipped_scale",
        "        stem_ch = _round_channels(STEM_CHANNELS)",
        "        stem_ch = 32",
        "arch_table_is_live",
    ),
    (
        "one2one_branch_shares_the_one2many_towers",
        "        self.one2one_box_convs = copy.deepcopy(self.box_convs)",
        "        self.one2one_box_convs = self.box_convs",
        "dual_head_independent",
    ),
    (
        "one2one_predictors_built_fresh_instead_of_copied",
        "        self.one2one_cls_preds = copy.deepcopy(self.cls_preds)",
        "        self.one2one_cls_preds = nn.ModuleList(\n"
        "            nn.Conv2d(self.cls_hidden, num_classes, 1) for _ in in_channels\n"
        "        )",
        "dual_head_independent",
    ),
    (
        # Shares the CLASS predictors specifically, so the slope halves. Kept
        # distinct from `one2one_branch_shares_the_one2many_towers` (which
        # shares the box towers) because only the class predictors are sized
        # from the class count, and only they move this guard.
        "one2one_class_predictors_shared_so_the_slope_halves",
        "        self.one2one_cls_preds = copy.deepcopy(self.cls_preds)",
        "        self.one2one_cls_preds = self.cls_preds",
        "class_count_slope",
    ),
    (
        "extra_block_per_stage",
        "        blocks = _round_depth(NECK_BLOCKS)",
        "        blocks = _round_depth(NECK_BLOCKS) + 1",
        "module_tree_size",
    ),
    (
        "batch_norm_comes_back",
        "        self.norm = nn.GroupNorm(_norm_groups(out_ch), out_ch, eps=1e-3)",
        "        self.norm = nn.BatchNorm2d(out_ch, eps=1e-3)",
        "no_stateful_norm",
    ),
    (
        "hardcoded_32_groups",
        "        self.norm = nn.GroupNorm(_norm_groups(out_ch), out_ch, eps=1e-3)\n"
        "        self.act = nn.SiLU(inplace=True) if act else nn.Identity()",
        "        self.norm = nn.GroupNorm(32, out_ch, eps=1e-3)\n"
        "        self.act = nn.SiLU(inplace=True) if act else nn.Identity()",
        "derived_norm_groups",
    ),
    # ---- the v10 blocks ---------------------------------------------------
    (
        "scdown_strides_on_the_pointwise_conv",
        "        self.cv1 = ConvNormAct(in_ch, out_ch, 1, stride=1)\n"
        "        self.cv2 = ConvNormAct(\n"
        "            out_ch, out_ch, ksize, stride=stride, groups=out_ch, act=False\n"
        "        )",
        "        self.cv1 = ConvNormAct(in_ch, out_ch, 1, stride=stride)\n"
        "        self.cv2 = ConvNormAct(\n"
        "            out_ch, out_ch, ksize, stride=1, groups=out_ch, act=False\n"
        "        )",
        "scdown_depthwise_stride",
    ),
    (
        "repvggdw_activates_each_branch",
        "        return self.act(self.conv(x) + self.conv1(x))",
        "        return self.act(self.conv(x)) + self.act(self.conv1(x))",
        "repvggdw_sums_first",
    ),
    (
        "cib_inner_mixer_is_not_the_large_kernel",
        """            (
                RepVGGDW(expanded)
                if large_kernel
                else ConvNormAct(expanded, expanded, 3, stride=1, groups=expanded)
            ),""",
        "            ConvNormAct(expanded, expanded, 3, stride=1, groups=expanded),",
        "cib_inverted",
    ),
    (
        "c2f_keeps_only_the_last_block_output",
        """        branches = list(self.cv1(x).chunk(2, dim=1))
        for block in self.m:
            branches.append(block(branches[-1]))
        return self.cv2(torch.cat(branches, dim=1))""",
        """        branches = list(self.cv1(x).chunk(2, dim=1))
        tail = branches[-1]
        for block in self.m:
            tail = block(tail)
        branches = branches + [tail] * len(self.m)
        return self.cv2(torch.cat(branches, dim=1))""",
        "c2f_fuses_intermediates",
    ),
    (
        "sppf_pools_in_parallel",
        "            outputs.append(self.pool(outputs[-1]))",
        "            outputs.append(self.pool(outputs[0]))",
        "sppf_series",
    ),
    (
        "psa_attends_to_both_halves",
        "        attending = attending + self.attn(attending)",
        "        bypass = bypass + self.attn(bypass)\n"
        "        attending = attending + self.attn(attending)",
        "psa_partial",
    ),
    (
        "psa_drops_the_attention_residual",
        "        attending = attending + self.attn(attending)\n"
        "        attending = attending + self.ffn(attending)",
        "        attending = self.attn(attending)\n"
        "        attending = attending + self.ffn(attending)",
        "psa_partial",
    ),
    # The layout bug the SDPA conversion makes available: shape-legal here
    # (tokens 25 vs key_dim 32 both feed a legal matmul), finite losses,
    # attends over CHANNELS instead of spatial positions.
    (
        "attention_sdpa_gets_the_channels_first_layout",
        """        attended = F.scaled_dot_product_attention(
            query.transpose(-2, -1),
            key.transpose(-2, -1),
            value.transpose(-2, -1),
            scale=self.scale,
        ).transpose(-2, -1)""",
        """        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.scale,
        )""",
        "attention_matches_manual",
    ),
    (
        "attention_scale_left_to_the_sdpa_default",
        "            scale=self.scale,\n        ).transpose(-2, -1)",
        "            scale=self.scale * 2.0,\n        ).transpose(-2, -1)",
        "attention_matches_manual",
    ),
    (
        "attention_head_count_hardcoded",
        "        self.num_heads = max(1, dim // head_dim)",
        "        self.num_heads = 8",
        "attention_heads_derived",
    ),
    # ---- dual assignment and eval ----------------------------------------
    (
        "dual_assignment_collapses_to_one2many",
        '            ("one2one_", one2one, ONE2ONE_TOPK),',
        '            ("one2one_", one2one, TAL_TOPK),',
        "consistent_dual_assignment",
    ),
    (
        "one2one_topk_above_one",
        "TAL_TOPK = 10\nONE2ONE_TOPK = 1",
        "TAL_TOPK = 10\nONE2ONE_TOPK = 3",
        "consistent_dual_assignment",
    ),
    (
        "eval_decodes_the_one2many_branch",
        "        detections = self._predictions(*one2one, image_list.image_sizes)",
        "        detections = self._predictions(*one2many, image_list.image_sizes)",
        "eval_uses_one2one",
    ),
    (
        "coupled_head",
        "            self.cls_convs.append(self._class_tower(channels))",
        "            self.cls_convs.append(self.box_convs[-1])",
        "decoupled_head",
    ),
    (
        "hardcoded_box_channel_width",
        "            self.box_preds.append(nn.Conv2d(self.box_hidden, 4 * reg_max, 1))",
        "            self.box_preds.append(nn.Conv2d(self.box_hidden, 64, 1))",
        "reg_max_is_live",
    ),
    (
        "head_reg_max_from_its_own_default",
        "        self.head = YOLOv10Head(\n"
        "            self.num_classes, self.neck.out_channels, reg_max=self.reg_max\n"
        "        )",
        "        self.head = YOLOv10Head(self.num_classes, self.neck.out_channels)",
        "reg_max_is_live",
    ),
    (
        "class_tower_width_tracks_the_class_count",
        "        self.cls_hidden = max(in_channels[0], min(num_classes, 100))",
        "        self.cls_hidden = max(16, min(num_classes, 100))",
        "seed_excluded_prefixes",
    ),
    # The one a prefix list copied from a single-head sibling produces.
    (
        "seed_prefixes_cover_only_the_one2many_head",
        """SEED_EXCLUDED_PREFIXES = (
    "head.cls_preds.0.",
    "head.cls_preds.1.",
    "head.cls_preds.2.",
    "head.one2one_cls_preds.0.",
    "head.one2one_cls_preds.1.",
    "head.one2one_cls_preds.2.",
)""",
        """SEED_EXCLUDED_PREFIXES = (
    "head.cls_preds.0.",
    "head.cls_preds.1.",
    "head.cls_preds.2.",
)""",
        "seed_excluded_prefixes",
    ),
    # ---- geometry and decode ---------------------------------------------
    (
        "transposed_head_flatten",
        "                cls_output.permute(0, 2, 3, 1).reshape("
        "batch, height * width, -1)",
        "                cls_output.permute(0, 3, 2, 1).reshape("
        "batch, height * width, -1)",
        "head_flatten_order",
    ),
    (
        "single_stride_decode",
        "    anchor_x, anchor_y, stride = anchors[:, 0], anchors[:, 1], anchors[:, 2]",
        "    anchor_x, anchor_y, stride = anchors[:, 0], anchors[:, 1], "
        "torch.full_like(anchors[:, 2], 8.0)",
        "decode_per_level_stride",
    ),
    (
        "dfl_decode_takes_an_argmax",
        "    return (dist_logits.softmax(dim=-1) * bins).sum(dim=-1)",
        "    return dist_logits.argmax(dim=-1).to(dist_logits.dtype)",
        "dfl_decode_expectation",
    ),
    (
        "dfl_decode_skips_the_softmax",
        "    return (dist_logits.softmax(dim=-1) * bins).sum(dim=-1)\n",
        "    return (dist_logits * bins).sum(dim=-1)\n",
        "dfl_decode_expectation",
    ),
    (
        "dfl_target_uses_one_stride",
        "    scaled = boxes_xyxy / stride.unsqueeze(-1)",
        "    scaled = boxes_xyxy / 8.0",
        "dfl_target_cell_units",
    ),
    (
        "dfl_loss_collapses_the_two_bins",
        "    upper = lower + 1",
        "    upper = lower",
        "dfl_loss_interpolates",
    ),
    (
        "dfl_loss_drops_the_interpolation_weights",
        "    return (loss_lower * weight_lower + loss_upper * weight_upper)"
        ".mean(dim=-1)",
        "    return ((loss_lower + loss_upper) * 0.5).mean(dim=-1)",
        "dfl_loss_interpolates",
    ),
    (
        "decode_truncates_the_batch",
        "        for boxes, class_scores, (height, width) in zip("
        "decoded, scores, image_sizes):",
        "        for boxes, class_scores, (height, width) in zip("
        "decoded[:1], scores[:1], image_sizes):",
        "decode_per_image",
    ),
    (
        "decode_misaligns_boxes",
        '                    "boxes": anchor_boxes[box_index][keep],',
        '                    "boxes": anchor_boxes[box_index.flip(0)][keep],',
        "decode_per_image",
    ),
    (
        "background_channel_kept",
        "            class_scores = class_scores[:, 1:]\n"
        "            num_anchors, num_classes = class_scores.shape",
        "            num_anchors, num_classes = class_scores.shape",
        "decode_per_image",
    ),
    # The one the label assertion alone CANNOT see: every real class still
    # lands on its own label, so no label 0 is ever emitted -- the background
    # channel is simply renamed and spends a slot from the top of the ranking.
    (
        "background_channel_rotated_to_the_last_label",
        "            class_scores = class_scores[:, 1:]",
        "            class_scores = torch.cat(\n"
        "                (class_scores[:, 1:], class_scores[:, :1]), dim=1\n"
        "            )",
        "decode_per_image",
    ),
    (
        "nms_reintroduced",
        "            keep = flat_scores > self.score_thresh",
        "            from torchvision.ops import batched_nms as _nms\n"
        "            keep = _nms(\n"
        "                anchor_boxes[box_index], flat_scores, labels, 0.7\n"
        "            )",
        "decode_is_nms_free",
    ),
    (
        "no_postprocess",
        """        return self.transform.postprocess(
            detections, image_list.image_sizes, original_image_sizes
        )""",
        "        return detections",
        "original_coordinates",
    ),
    (
        "transform_resizes_past_the_declared_edge",
        "            min_size=self.input_size,\n            max_size=self.input_size,",
        "            min_size=self.input_size * 2,\n"
        "            max_size=self.input_size * 2,",
        "declared_size_measured",
    ),
    # ---- the assigner ----------------------------------------------------
    (
        "swapped_alignment_exponents",
        "        alignment = scores.pow(TAL_ALPHA) * ious.pow(TAL_BETA)",
        "        alignment = scores.pow(TAL_BETA) * ious.pow(TAL_ALPHA)",
        "tal_metric_exponents",
    ),
    (
        "alignment_prefers_the_worst_iou",
        "        alignment = scores.pow(TAL_ALPHA) * ious.pow(TAL_BETA)\n",
        "        alignment = scores.pow(TAL_ALPHA) * (1.0 - ious).pow(TAL_BETA)\n",
        "tal_metric_exponents",
    ),
    (
        "no_inside_the_box_rule",
        "        inside = self._anchors_inside(gt_boxes, anchor_points)\n"
        "        candidate = alignment * inside.to(alignment.dtype)",
        "        inside = torch.ones_like(alignment, dtype=torch.bool)\n"
        "        candidate = alignment * inside.to(alignment.dtype)",
        "tal_inside_the_box",
    ),
    (
        "topk_bound_removed",
        "        selection = min(topk, num_anchors)",
        "        selection = num_anchors",
        "tal_topk_ranking",
    ),
    (
        "topk_takes_the_worst",
        "        _, positions = torch.topk(candidate, selection, dim=1)",
        "        _, positions = torch.topk(candidate, selection, dim=1, largest=False)",
        "tal_topk_ranking",
    ),
    (
        "target_not_normalised",
        "        normalised = (assigned * best_iou / (best_alignment + _EPS))"
        ".amax(dim=0)",
        "        normalised = assigned.amax(dim=0)",
        "tal_normalised_target",
    ),
    (
        "hard_class_target",
        "        best_alignment = assigned.amax(dim=1, keepdim=True)",
        "        assigned = matching.to(alignment.dtype)\n"
        "        best_alignment = assigned.amax(dim=1, keepdim=True)",
        "tal_normalised_target",
    ),
    (
        "tie_break_by_alignment",
        "            best = (ious * matching.to(ious.dtype)).argmax(dim=0)",
        "            best = (alignment * matching.to(alignment.dtype)).argmax(dim=0)",
        "tal_tie_break_by_iou",
    ),
    (
        "no_tie_break",
        """        claimed_by = matching.sum(dim=0)
        contested = claimed_by > 1
        if bool(contested.any()):""",
        """        claimed_by = matching.sum(dim=0)
        contested = claimed_by > 1
        if False:""",
        "tal_tie_break_by_iou",
    ),
    (
        "assign_nothing",
        "        fg_mask = matching.any(dim=0)",
        "        matching = torch.zeros_like(matching)\n"
        "        fg_mask = matching.any(dim=0)",
        "positives_reach_box_branch",
    ),
    (
        "fetches_at_construction",
        "        self.backbone = YOLOv10Backbone()",
        '        __import__("socket").getaddrinfo("download.pytorch.org", 443)\n'
        "        self.backbone = YOLOv10Backbone()",
        "no_network",
    ),
    (
        "soft_class_target_zeroed",
        "                cls_targets[index, fg_mask, labels] = aligned",
        "                cls_targets[index, fg_mask, labels] = aligned * 0.0",
        "overfits_one_object",
    ),
]

#: Mutations whose whole point is that they are SILENT elsewhere: each one
#: leaves ``model(images, targets)`` returning a finite loss dict and
#: ``model(images)`` returning well-formed predictions, so
#: ``tests/test_od_torchvision_family_train_step.py`` stays green against every
#: one of them. Asserted in ``test_silent_mutations_still_train`` so nobody
#: concludes the family train-step test already covers this file.
_SILENT_MUTATIONS = frozenset(
    {
        # ⚠️ THE ONE THIS TEMPLATE EXISTS TO GUARD. Not only does training stay
        # green -- the parameter count, every shape and every published figure
        # are unchanged too, so the architecture guards cannot see it either.
        "one2one_head_sees_undetached_features",
        # v10-specific blocks: none of these changes a shape or a loss key, and
        # the SCDown one does not change the parameter count either.
        "scdown_strides_on_the_pointwise_conv",
        "repvggdw_activates_each_branch",
        "psa_attends_to_both_halves",
        "psa_drops_the_attention_residual",
        "attention_head_count_hardcoded",
        "sppf_pools_in_parallel",
        "psa_dropped_from_the_backbone_forward",
        "sppf_dropped_from_the_backbone_forward",
        # the dual assignment, and the deployed head
        "dual_assignment_collapses_to_one2many",
        "one2one_topk_above_one",
        "eval_decodes_the_one2many_branch",
        "nms_reintroduced",
        # assigner and decode, shared with the sibling DFL templates
        "swapped_alignment_exponents",
        "alignment_prefers_the_worst_iou",
        "no_inside_the_box_rule",
        "topk_bound_removed",
        "topk_takes_the_worst",
        "target_not_normalised",
        "hard_class_target",
        "tie_break_by_alignment",
        "assign_nothing",
        "soft_class_target_zeroed",
        "dfl_decode_takes_an_argmax",
        "dfl_loss_collapses_the_two_bins",
        "dfl_target_uses_one_stride",
        "background_channel_kept",
        "background_channel_rotated_to_the_last_label",
    }
)


def test_the_template_exists() -> None:
    """Guard the guard: every table in this file is keyed on one file."""
    assert TEMPLATE.is_file(), f"{TEMPLATE} is missing — this whole file is dead"
    source = TEMPLATE.read_text(encoding="utf-8")
    assert re.search(
        r'^\s*model_type\s*=\s*"torchvision_detection"', source, re.MULTILINE
    ), (
        f"{TEMPLATE.name} must declare model_type = 'torchvision_detection'. The "
        f"legacy 'yolo' family is a fixed 7x7 grid at 448px with one object per "
        f"cell and an external loss.py, and it is frozen (backend#2982)."
    )
    assert "yolo" != (
        re.search(r'^\s*model_type\s*=\s*"(\w*)"', source, re.MULTILINE).group(1)
    )


def test_no_mutation_baseline() -> None:
    """The zero row of the sweep.

    A mutation sweep without a no-mutation baseline cannot distinguish "the
    guard caught the mutation" from "the guard fails on everything", and a
    ``_mutate`` anchor that no longer applies would otherwise report a pristine
    template as a caught mutation. This runs the whole table against the
    unmutated file in ONE module load, so the baseline is cheap and always
    present.
    """
    module = _load()
    failures = []
    for name, guard in GUARDS.items():
        try:
            guard(module)
        except AssertionError as error:  # pragma: no cover — reported, not hit
            failures.append(f"{name}: {error}")
    assert not failures, (
        "guard(s) fail on the UNMUTATED template, so every 'mutation caught' "
        "result below is meaningless:\n" + "\n\n".join(failures)
    )


@pytest.mark.parametrize("guard_name", sorted(GUARDS))
def test_guard_passes_on_the_shipped_template(guard_name: str) -> None:
    """The per-guard positive control, so a failure names one guard."""
    GUARDS[guard_name](_load())


@pytest.mark.parametrize(
    "name,anchor,replacement,target",
    MUTATIONS,
    ids=[entry[0] for entry in MUTATIONS],
)
def test_mutation_is_caught_by_its_guard(
    name: str, anchor: str, replacement: str, target: str
) -> None:
    """Point the guard at a template edited to break exactly what it checks.

    A guard that cannot be made to fail proves nothing about the code it covers,
    and for a hand-written detector "it returned a loss dict" is our own code
    answering its own question. Keeping the mutation in the suite is what stops
    a guard rotting into a tautology as the template changes.
    """
    mutated = _mutate(anchor, replacement)
    guard = GUARDS[target]

    with pytest.raises(AssertionError) as excinfo:
        guard(mutated)

    message = str(excinfo.value)
    assert message.strip(), f"{name}: the guard failed with an empty message"
    # The failure must be the guard's own, not a fixture-degeneracy assertion:
    # several guards carry those, and a mutation that trips one has told us the
    # fixture stopped exercising the rule, not that the rule is guarded.
    assert "fixture is degenerate" not in message and "fixture:" not in message, (
        f"{name}: the mutation tripped a fixture-degeneracy assertion rather "
        f"than the guard itself — the fixture no longer exercises the rule "
        f"under test:\n{message}"
    )


@pytest.mark.parametrize("guard_name", sorted(GUARDS))
def test_every_guard_has_a_mutation(guard_name: str) -> None:
    """No guard may be un-proven.

    Without this, adding a guard and forgetting its mutation leaves an assertion
    nobody has ever seen fail — which is how a wrong assigner ships green.
    """
    covered = {target for *_, target in MUTATIONS}
    assert guard_name in covered, (
        f"{guard_name} has no entry in MUTATIONS. Every guard here must be shown "
        f"able to go red; add the textual edit that breaks it."
    )


def test_every_mutation_targets_a_real_guard() -> None:
    """The other direction: a mutation naming a renamed guard would be skipped
    silently by the parametrized test's ``KeyError``-free lookup."""
    unknown = sorted({target for *_, target in MUTATIONS} - set(GUARDS))
    assert not unknown, f"MUTATIONS name guards that do not exist: {unknown}"


def test_silent_mutations_still_train() -> None:
    """The point of this whole file, stated as a test: the mutations do NOT
    break training.

    Each mutation named in ``_SILENT_MUTATIONS`` leaves
    ``model(images, targets)`` returning a finite loss dict and
    ``model(images)`` returning well-formed predictions, so the family
    train-step test stays green against every one of them. That is the reason
    the guards above exist, and asserting it here stops someone concluding the
    family test already covers this template.
    """
    import torch

    selected = [entry for entry in MUTATIONS if entry[0] in _SILENT_MUTATIONS]
    assert len(selected) == len(_SILENT_MUTATIONS), (
        f"_SILENT_MUTATIONS names "
        f"{sorted(_SILENT_MUTATIONS - {e[0] for e in MUTATIONS})} which are not "
        f"in MUTATIONS"
    )

    targets = [
        {
            "boxes": torch.tensor(
                [[10.0, 10.0, 60.0, 60.0], [70.0, 70.0, 110.0, 120.0]]
            ),
            "labels": torch.tensor([1, 3], dtype=torch.int64),
        },
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        },
    ]
    images = [torch.rand(3, 128, 160), torch.rand(3, 144, 128)]

    for name, anchor, replacement, _ in selected:
        module = _mutate(anchor, replacement)
        model = _build(module, 3, 128)
        model.train()
        losses = model(images, targets)
        assert isinstance(losses, dict) and losses, f"{name}: no loss dict"
        for key, value in losses.items():
            assert torch.isfinite(value).all(), f"{name}: loss {key} is {value!r}"
        model.eval()
        with torch.no_grad():
            predictions = model(images)
        assert isinstance(predictions, list) and len(predictions) == len(images), (
            f"{name}: eval returned {predictions!r}"
        )
        for prediction in predictions:
            assert {"boxes", "scores", "labels"} <= set(prediction), (
                f"{name}: eval prediction is missing keys"
            )
        del model, module


def test_the_detach_mutation_is_invisible_to_the_architecture_guards() -> None:
    """The claim that justifies ``one2one_detached`` existing at all.

    A reader can reasonably ask why the published-count guard is not enough. It
    is not, and this measures rather than asserts it: with the ``detach``
    removed the parameter count, the buffer count, the state_dict key set and
    the loss keys are all IDENTICAL. Every structural guard in this file is
    blind to it by construction.
    """
    import torch

    name, anchor, replacement, _ = next(
        entry
        for entry in MUTATIONS
        if entry[0] == "one2one_head_sees_undetached_features"
    )
    pristine = _build(_load(), 7)
    mutated = _build(_mutate(anchor, replacement), 7)

    assert sum(p.numel() for p in pristine.parameters()) == sum(
        p.numel() for p in mutated.parameters()
    ), f"{name}: the mutation DID change the parameter count"
    assert len(pristine.state_dict()) == len(mutated.state_dict())
    assert set(pristine.state_dict()) == set(mutated.state_dict())
    assert sum(b.numel() for b in pristine.buffers()) == sum(
        b.numel() for b in mutated.buffers()
    )

    edge = 128
    torch.manual_seed(0)
    image = torch.rand(3, edge, edge)
    targets = _overfit_targets(torch, edge)
    for model in (pristine, mutated):
        model.train()
    assert sorted(pristine([image], targets)) == sorted(mutated([image], targets)), (
        f"{name}: the mutation changed the loss KEYS, so it is not silent and "
        f"this test's premise is wrong"
    )
