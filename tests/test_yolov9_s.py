"""Guards for ``object_detection/pytorch/yolov9_s.py``, each proven able to go
red by a mutation that is kept in the suite.

Why this file exists
--------------------
``tests/test_od_torchvision_family_train_step.py`` proves a template returns a
loss dict and a ``List[Dict]`` of xyxy predictions. For a template that wraps a
torchvision builder that is a real assertion: the loss is the library's. For
``yolov9_s.py`` the backbone, neck, head, assigner and all three losses are
**our own code**, so "returns a loss dict" proves only that our code returns a
dict. Every interesting way a hand-written detector is wrong is silent:

* a GELAN stage that fans its two branches out in PARALLEL instead of chaining
  them. At this scale ``mid // 2 == inner`` for the fine stages, so the shapes
  match, the parameter count is unchanged and the model trains — it is simply a
  shallower block than the architecture it claims to be;
* ``AConv`` without its average pool. The strided conv that follows produces
  the same output size either way on an even edge, so nothing about the shape,
  the parameter count or the loss notices;
* ``SPPELAN``'s three pools applied in parallel rather than in series, which
  costs the block two thirds of its receptive field and changes no shape;
* a ``RepConv`` that activates each branch before summing, which is identical
  in count and shape and is no longer re-parameterisable;
* the assigner matching **nothing** — BCE over an all-negative image is finite
  and small, so the train step passes and the model learns no objects;
* the assigner matching the wrong anchors — a swapped alignment exponent picks
  the best-*classified* candidate instead of the best-*localised* one, changes
  no cardinality whatsoever, and leaves every loss finite;
* the assigner losing its per-level structure — one stride used for every
  level, or a DFL target computed in pixels instead of cells;
* the DFL decode taking an argmax, or forgetting its softmax, so boxes quantise
  or explode while the loss (computed on the logits) never notices;
* predictions never mapped back to the original image coordinates, so mAP is
  computed against boxes in the resized frame.

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
built model's real detections are indistinguishable from noise and any
assertion taken from ``model(images)`` at initialisation is checking almost
nothing. ``guard_decode_is_per_image_and_aligned`` therefore drives
``_predictions`` **directly**, with synthetic head outputs that clear the score
threshold by a wide margin, **at batch two** — a per-image bug is invisible at
batch one by construction — plus a second fixture where the *background*
channel is the strongest, which the first cannot see.

**Cardinality is invariant to cost.** Asserting *how many* anchors an assigner
selects proves nothing about the metric that ranks them: a swapped focal
``alpha`` hid in a sibling template through a full mutation sweep because every
assertion counted proposals. So the assigner guards here assert **which**
anchor or ground truth is selected, and with what soft target — and each one
first asserts that its fixture can distinguish the two answers (a fixture where
the correct and the mutated rule agree is worse than no fixture, because it
reads as coverage).

**A self-measured number is not evidence.** ``_PINNED_TOTALS`` is a tripwire,
labelled as one. The parameter count is asserted against
``_reference_parameters``, which is derived from the published architecture
tables with nothing from ``model_zoo/`` imported, and anchored to **three**
figures from outside this repo entirely — see ``_PUBLISHED``. Two of those
three are then re-checked against the BUILT model by rebuilding this module
with the other scales' tables, so the table is a proven-live knob rather than a
constant that reaches nothing.
"""

import importlib.util
import pathlib
import re
import tempfile

import pytest

ROOT = pathlib.Path(__file__).parent.parent
OD_PYTORCH = ROOT / "model_zoo" / "object_detection" / "pytorch"
TEMPLATE = OD_PYTORCH / "yolov9_s.py"

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


def _apply_arch(module, arch) -> None:
    """Point a loaded template at another published scale's tables.

    The mechanism the ``arch_table_is_live`` guard uses. YOLOv9 has no width
    multiplier — each published scale is its own channel table AND its own
    block/downsampler choice — so the table IS the scale selector, and setting
    it is the only way to ask this module for another scale.
    """
    module.STEM_CHANNELS = arch["stem"]
    module.BACKBONE_STAGES = arch["stages"]
    module.SPPELAN_HIDDEN = arch["sppelan_hidden"]
    module.NECK_TOP_DOWN = arch["top_down"]
    module.NECK_BOTTOM_UP = arch["bottom_up"]
    module.DOWNSAMPLE = arch["downsample"]


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
# So the reference below is arithmetic on (in, out, kernel), transcribed from
# Ultralytics' `cfg/models/v9/yolov9{t,s,c}.yaml` and
# `ultralytics/nn/modules/{block,conv,head}.py`, with NOTHING under `model_zoo/`
# imported. And the transcription is itself anchored to figures from outside
# this repo — the parameter counts those yaml files carry in their own header
# comments — so it cannot drift into agreeing with a wrong template.

#: Published parameter totals, from the header comment of each published yaml.
#: THREE scales, not one, so the anchor pins the channel tables, the block
#: counts, the two aggregation kinds AND the two downsampler kinds rather than
#: one arithmetic total:
#:
#:   yolov9t.yaml: 544 layers, 2128720 parameters,  8.5 GFLOPs
#:   yolov9s.yaml: 544 layers, 7318368 parameters, 27.6 GFLOPs
#:   yolov9c.yaml: 358 layers, 25590912 parameters, 104.0 GFLOPs
#:
#: Each of those totals includes ONE tensor this template does not store: the
#: 16-element DFL projection vector, which upstream keeps as a frozen
#: ``Conv2d`` weight (``requires_grad=False``) and this template builds with
#: ``torch.arange`` inside the decode. So THIS TEMPLATE'S PARAMETER COUNT IS
#: THE PUBLISHED TOTAL MINUS EXACTLY 16, and that gap is stated here rather
#: than absorbed into a tolerance.
#:
#: ⚠️ ``yolov9m.yaml`` IS DELIBERATELY ABSENT, and the reason is recorded
#: instead of hidden. Its header declares 20216160 parameters; the arithmetic
#: below — the same arithmetic that reproduces t, s and c to the parameter —
#: derives 20206056 from its table, a gap of 10088 (0.05%). Building this
#: module with the m table reproduces the derivation, not the header, and no
#: single-argument perturbation of the m table closes the gap. The file's
#: header count has never been edited since the commit that introduced it (when
#: the header still read ``# YOLOv9t``), and a later upstream commit fixed the
#: *layer* counts in these headers without recomputing the parameter counts. So
#: the m figure is treated as unreproduced rather than as an oracle: an anchor
#: nobody can derive is not an anchor. Three that reproduce exactly are.
_PUBLISHED = {
    "yolov9t": 2_128_720,
    "yolov9s": 7_318_368,
    "yolov9c": 25_590_912,
}
#: Parameters upstream stores for the DFL bin vector and this template does not.
_DFL_PROJECTION_CONSTANTS = 16
#: The class count the published figures are quoted at.
_PUBLISHED_CLASSES = 80
#: The scale this template ships.
_SHIPPED_SCALE = "yolov9s"

#: The published architecture tables, transcribed from the yaml files.
#: ``stages`` rows are ``(kind, downsample_out, out, mid, inner, blocks)`` and
#: ``downsample_out`` is ``None`` for the stage the stem already reached.
#:
#: These are the TEST's OWN copies. They are never read off the template — that
#: is what lets a guard catch a template whose table has been edited, and what
#: makes the cross-scale rebuild a real second measurement rather than the same
#: numbers compared with themselves.
_ARCH = {
    "yolov9t": {
        "stem": (16, 32),
        "stages": (
            ("elan1", None, 32, 32, 16, 0),
            ("csp_elan", 64, 64, 64, 32, 3),
            ("csp_elan", 96, 96, 96, 48, 3),
            ("csp_elan", 128, 128, 128, 64, 3),
        ),
        "sppelan_hidden": 64,
        "top_down": ((96, 96, 48, 3), (64, 64, 32, 3)),
        "bottom_up": ((48, 96, 96, 48, 3), (64, 128, 128, 64, 3)),
        "downsample": "aconv",
    },
    "yolov9s": {
        "stem": (32, 64),
        "stages": (
            ("elan1", None, 64, 64, 32, 0),
            ("csp_elan", 128, 128, 128, 64, 3),
            ("csp_elan", 192, 192, 192, 96, 3),
            ("csp_elan", 256, 256, 256, 128, 3),
        ),
        "sppelan_hidden": 128,
        "top_down": ((192, 192, 96, 3), (128, 128, 64, 3)),
        "bottom_up": ((96, 192, 192, 96, 3), (128, 256, 256, 128, 3)),
        "downsample": "aconv",
    },
    "yolov9c": {
        "stem": (64, 128),
        "stages": (
            ("csp_elan", None, 256, 128, 64, 1),
            ("csp_elan", 256, 512, 256, 128, 1),
            ("csp_elan", 512, 512, 512, 256, 1),
            ("csp_elan", 512, 512, 512, 256, 1),
        ),
        "sppelan_hidden": 256,
        "top_down": ((512, 512, 256, 1), (256, 256, 128, 1)),
        "bottom_up": ((256, 512, 512, 256, 1), (512, 512, 512, 256, 1)),
        "downsample": "adown",
    },
}


def _conv(in_ch, out_ch, kernel, bias=False):
    return in_ch * out_ch * kernel * kernel + (out_ch if bias else 0)


def _norm(channels):
    """An affine normalisation layer: one scale and one shift per channel.

    GroupNorm and BatchNorm are IDENTICAL here, which is what makes comparing a
    GroupNorm build against a published BatchNorm count legitimate — see the
    federated note in the template. What differs is the BUFFERS, and those are
    pinned separately in ``guard_no_stateful_normalisation``.
    """
    return 2 * channels


def _cna(in_ch, out_ch, kernel):
    """conv -> norm (affine); the activation has no parameters."""
    return _conv(in_ch, out_ch, kernel) + _norm(out_ch)


def _rep_conv(in_ch, out_ch):
    """RepConv: a 3x3 branch and a 1x1 branch, each conv-plus-norm.

    No identity-normalisation third branch — upstream's ``RepConv`` defaults to
    ``bn=False`` and every YOLOv9 config takes that default.
    """
    return _cna(in_ch, out_ch, 3) + _cna(in_ch, out_ch, 1)


def _rep_bottleneck(channels):
    """RepConv 3x3 at FULL branch width, then a plain 3x3 (upstream e=1.0)."""
    return _rep_conv(channels, channels) + _cna(channels, channels, 3)


def _rep_csp(in_ch, out_ch, blocks):
    half = int(out_ch * 0.5)
    return (
        _cna(in_ch, half, 1)
        + _cna(in_ch, half, 1)
        + _cna(2 * half, out_ch, 1)
        + blocks * _rep_bottleneck(half)
    )


def _stage(kind, in_ch, out_ch, mid, inner, blocks):
    """A GELAN aggregation stage: widen, split, chain two branches, fuse.

    ``elan1`` swaps the computational block for a single 3x3 — and takes no
    bottlenecks, which is why a row declaring some is an error rather than a
    number that gets dropped.
    """
    if kind == "elan1":
        assert not blocks, f"an elan1 stage cannot carry {blocks} bottlenecks"
        branch = _cna(mid // 2, inner, 3) + _cna(inner, inner, 3)
    elif kind == "csp_elan":
        branch = (
            _rep_csp(mid // 2, inner, blocks)
            + _cna(inner, inner, 3)
            + _rep_csp(inner, inner, blocks)
            + _cna(inner, inner, 3)
        )
    else:  # pragma: no cover — a typo in _ARCH, not a template defect
        raise AssertionError(f"unknown stage kind {kind!r}")
    return _cna(in_ch, mid, 1) + branch + _cna(mid + 2 * inner, out_ch, 1)


def _downsample(kind, in_ch, out_ch):
    if kind == "aconv":
        return _cna(in_ch, out_ch, 3)
    if kind == "adown":
        half = out_ch // 2
        return _cna(in_ch // 2, half, 3) + _cna(in_ch // 2, half, 1)
    raise AssertionError(f"unknown downsampler {kind!r}")  # pragma: no cover


def _sppelan(in_ch, out_ch, hidden, repeats=3):
    """One shared max-pool, so only the two 1x1s carry parameters."""
    return _cna(in_ch, hidden, 1) + _cna(hidden * (repeats + 1), out_ch, 1)


def _detect(head_channels, class_channels, reg_max=16):
    box_hidden = max(16, head_channels[0] // 4, reg_max * 4)
    cls_hidden = max(head_channels[0], min(class_channels, 100))
    total = 0
    for channels in head_channels:
        total += _cna(channels, box_hidden, 3) + _cna(box_hidden, box_hidden, 3)
        total += _conv(box_hidden, 4 * reg_max, 1, bias=True)
        total += _cna(channels, cls_hidden, 3) + _cna(cls_hidden, cls_hidden, 3)
        total += _conv(cls_hidden, class_channels, 1, bias=True)
    return total


def _head_channels(arch):
    """The three widths the head sees, DERIVED from the neck rows.

    Not a fourth literal: a wrong neck output width would otherwise be able to
    hide behind a head table that still agreed with it.
    """
    return (arch["top_down"][1][0], arch["bottom_up"][0][1], arch["bottom_up"][1][1])


def _reference_parameters(class_channels, arch, reg_max=16):
    """YOLOv9 parameter count, derived from a published table alone."""
    stem_first, stem_second = arch["stem"]
    down = arch["downsample"]

    total = _cna(3, stem_first, 3) + _cna(stem_first, stem_second, 3)
    in_ch = stem_second
    widths = []
    for kind, down_out, out_ch, mid, inner, blocks in arch["stages"]:
        if down_out is not None:
            total += _downsample(down, in_ch, down_out)
            in_ch = down_out
        total += _stage(kind, in_ch, out_ch, mid, inner, blocks)
        widths.append(out_ch)
        in_ch = out_ch
    total += _sppelan(widths[-1], widths[-1], arch["sppelan_hidden"])

    c3, c4, c5 = widths[1], widths[2], widths[3]
    (p4_out, p4_mid, p4_inner, p4_n), (p3_out, p3_mid, p3_inner, p3_n) = arch[
        "top_down"
    ]
    total += _stage("csp_elan", c5 + c4, p4_out, p4_mid, p4_inner, p4_n)
    total += _stage("csp_elan", p4_out + c3, p3_out, p3_mid, p3_inner, p3_n)

    (d4, o4, m4, i4, n4), (d5, o5, m5, i5, n5) = arch["bottom_up"]
    total += _downsample(down, p3_out, d4)
    total += _stage("csp_elan", d4 + p4_out, o4, m4, i4, n4)
    total += _downsample(down, o4, d5)
    total += _stage("csp_elan", d5 + c5, o5, m5, i5, n5)

    total += _detect(_head_channels(arch), class_channels, reg_max)
    return total


#: Published per-stage structure at the shipped scale, independent of the total:
#: it says WHAT drifted when the count disagrees.
_REFERENCE_STRUCTURE = {
    "backbone_out": (128, 192, 256),
    # YOLOv9 keeps a per-level width at the head — unlike RTMDet, which projects
    # all three levels to a common width. Getting this wrong is a plausible
    # cross-template copy and changes the count too.
    "neck_out": (128, 192, 256),
    "backbone_stage_kinds": ("ELAN1", "CSPELAN", "CSPELAN", "CSPELAN"),
    "backbone_blocks": (0, 3, 3, 3),
    "neck_blocks": (3, 3, 3, 3),
    # ⚠️ NOT the level widths. YOLOv9's bottom-up pass downsamples to 96 before
    # a 192-wide stage and to 128 before a 256-wide one; YOLOv8's neck
    # downsamples to the level width, and a copy from there lines up shape-wise
    # because the fusion conv's input is a sum.
    "neck_downsample_out": (96, 128),
    "box_hidden": 64,
    "cls_hidden": 128,
    "strides": (8, 16, 32),
    "reg_max": 16,
    "downsample": "AConv",
}


def test_the_reference_derivation_matches_the_published_figures() -> None:
    """The transcription, checked against the numbers it is transcribed from.

    Runs before anything is built, and needs no torch: if this fails, the
    reference is wrong and every comparison against it is worthless. THREE
    scales are pinned, so the check covers both aggregation kinds (``ELAN1`` and
    ``CSPELAN``), both downsamplers (``AConv`` and ``ADown``), two block counts
    and three unrelated channel tables rather than one arithmetic accident. The
    DFL gap is asserted as an exact constant rather than hidden in a tolerance.
    """
    assert _SHIPPED_SCALE in _PUBLISHED, "the shipped scale must be anchored"
    assert len(_PUBLISHED) >= 2, (
        "one published figure cannot distinguish a wrong table from a wrong "
        "transcription of it — that is the whole reason this dict has more than "
        "one row"
    )
    for scale, published in sorted(_PUBLISHED.items()):
        derived = _reference_parameters(_PUBLISHED_CLASSES, _ARCH[scale])
        assert derived == published - _DFL_PROJECTION_CONSTANTS, (
            f"the spec transcription derives {derived:,} parameters for {scale} "
            f"at {_PUBLISHED_CLASSES} classes, but {scale}.yaml's header reports "
            f"{published:,} — of which exactly "
            f"{_DFL_PROJECTION_CONSTANTS} are the frozen DFL projection this "
            f"template does not store, so {published - _DFL_PROJECTION_CONSTANTS:,} "
            f"is expected (off by "
            f"{derived - (published - _DFL_PROJECTION_CONSTANTS):+,}). Fix the "
            f"transcription against the yaml before trusting any comparison "
            f"that uses it."
        )


def test_the_three_scales_are_actually_different() -> None:
    """Guard the guard above: three rows that happened to describe the same
    architecture would read as a three-way anchor and be a one-way one."""
    totals = {
        scale: _reference_parameters(_PUBLISHED_CLASSES, arch)
        for scale, arch in _ARCH.items()
    }
    assert len(set(totals.values())) == len(totals), (
        f"two anchored scales derive the same parameter count: {totals}"
    )
    kinds = {arch["downsample"] for arch in _ARCH.values()}
    assert kinds == {"aconv", "adown"}, (
        f"the anchored scales must cover BOTH published downsamplers, got "
        f"{sorted(kinds)} — otherwise the ADown arithmetic is never checked "
        f"against a published figure"
    )
    stage_kinds = {row[0] for arch in _ARCH.values() for row in arch["stages"]}
    assert stage_kinds == {"elan1", "csp_elan"}, (
        f"the anchored scales must cover BOTH published aggregation kinds, got "
        f"{sorted(stage_kinds)}"
    )


# --------------------------------------------------------------------------
# structure guards
# --------------------------------------------------------------------------


def _backbone_blocks(model):
    """Bottleneck count per backbone stage, read off the BUILT modules."""
    counts = []
    for stage in model.backbone.stages:
        branch = stage.cv2
        counts.append(len(branch[0].m) if hasattr(branch, "__getitem__") else 0)
    return tuple(counts)


def _neck_blocks(model):
    neck = model.neck
    return tuple(
        len(getattr(neck, name).cv2[0].m)
        for name in ("td_p4", "td_p3", "bu_p4", "bu_p5")
    )


def guard_matches_the_published_architecture(module) -> None:
    """The built module tree must match the PUBLISHED architecture, re-derived.

    The independent half of the evidence. ``module_tree_size`` pins the totals
    this repo measured, so it can only catch a regression away from whatever was
    shipped; this one re-computes the count from the published table and
    compares, so it catches shipping the wrong architecture in the first place —
    which is what happened on a sibling template.
    """
    class_channels = module.output_classes + 1  # the deliberate label-space +1
    expected = _reference_parameters(class_channels, _ARCH[_SHIPPED_SCALE])
    model = _build(module, module.output_classes)
    actual = sum(p.numel() for p in model.parameters())

    assert actual == expected, (
        f"{module.__name__}: built model has {actual:,} parameters; YOLOv9-S "
        f"re-derived from its published table has {expected:,} at the same "
        f"{class_channels} class channels — a difference of "
        f"{actual - expected:+,}. Something in the channel table, the block "
        f"counts, the kernel sizes, the aggregation shape or the head does not "
        f"match the design this template claims to implement. This is the check "
        f"a parameter count measured off the model itself CANNOT make."
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
    kinds = tuple(type(stage).__name__ for stage in model.backbone.stages)
    assert kinds == reference["backbone_stage_kinds"], (
        f"{module.__name__}: backbone stages are {kinds}, published design has "
        f"{reference['backbone_stage_kinds']} — the stride-4 stage is an ELAN1 "
        f"at the t/s scales and a full CSPELAN at c/e"
    )
    assert _backbone_blocks(model) == reference["backbone_blocks"], (
        f"{module.__name__}: backbone stages hold {_backbone_blocks(model)} "
        f"RepCSP bottlenecks, published design has "
        f"{reference['backbone_blocks']}"
    )
    assert _neck_blocks(model) == reference["neck_blocks"], (
        f"{module.__name__}: neck stages hold {_neck_blocks(model)} bottlenecks, "
        f"published design has {reference['neck_blocks']}"
    )
    downsamples = (model.neck.bu_down3, model.neck.bu_down4)
    assert tuple(d.conv.conv.out_channels for d in downsamples) == (
        reference["neck_downsample_out"]
    ), (
        f"{module.__name__}: the bottom-up downsamplers emit "
        f"{tuple(d.conv.conv.out_channels for d in downsamples)} channels; "
        f"YOLOv9-S publishes {reference['neck_downsample_out']}, which is "
        f"NOT the level width — a copy of YOLOv8's neck lines up shape-wise "
        f"because the fusion conv's input is a sum"
    )
    assert type(model.neck.bu_down3).__name__ == reference["downsample"], (
        f"{module.__name__}: downsampler is "
        f"{type(model.neck.bu_down3).__name__}, the t/s/m scales publish "
        f"{reference['downsample']}"
    )
    assert model.head.box_hidden == reference["box_hidden"]
    assert model.head.cls_hidden == reference["cls_hidden"]
    assert tuple(model.head.strides) == reference["strides"]
    assert model.head.reg_max == reference["reg_max"]


def guard_architecture_table_is_a_live_knob(module) -> None:
    """The architecture tables must REACH the built model — and the proof is a
    second and third published parameter count, measured on the rebuild.

    The failure mode this is written against is a declared table that is read
    once and then contradicted by a hardcoded literal deeper in the builder:
    the shipped scale still comes out right, the constant reads as
    configuration, and it is decoration. YOLOv9 makes that especially easy to
    get wrong because it has NO width multiplier — every scale is its own table,
    its own aggregation kind for the stride-4 stage and its own downsampler — so
    "the table is the scale" has to be true rather than merely intended.

    So this rebuilds with the published YOLOv9-C table (``ADown``, a CSPELAN at
    stride 4, one bottleneck per stage) and the published YOLOv9-T table (the
    narrowest), and asserts each rebuild carries that scale's published
    parameter count. Neither number is derived from this repo.
    """
    import torch

    shipped = _build(module, 3)
    assert not any(
        type(sub).__name__ == "ADown" for sub in shipped.modules()
    ), (
        f"{module.__name__}: the SHIPPED build contains an ADown, but the t/s/m "
        f"scales publish AConv — so the rebuild below would not be exercising a "
        f"different code path and this guard would be checking nothing"
    )

    for scale in ("yolov9c", "yolov9t"):
        rebuilt_module = _reload(module)
        _apply_arch(rebuilt_module, _ARCH[scale])
        try:
            model = _build(rebuilt_module, _PUBLISHED_CLASSES - 1)
        except Exception as error:  # noqa: BLE001 — any build failure is the bug
            raise AssertionError(
                f"{module.__name__}: rebuilding with the published {scale} "
                f"table failed with {type(error).__name__}: {error}. The "
                f"architecture table is meant to be the scale selector; a "
                f"builder that only works for the shipped table has hardcoded "
                f"something it declares."
            ) from error

        actual = sum(p.numel() for p in model.parameters())
        expected = _PUBLISHED[scale] - _DFL_PROJECTION_CONSTANTS
        assert actual == expected, (
            f"{module.__name__}: rebuilt with the published {scale} table the "
            f"model has {actual:,} parameters, but {scale}.yaml's header "
            f"reports {_PUBLISHED[scale]:,} — i.e. {expected:,} once the "
            f"{_DFL_PROJECTION_CONSTANTS} frozen DFL projection constants this "
            f"template does not store are removed (off by "
            f"{actual - expected:+,}). Either the table does not reach the "
            f"builder, or a literal deeper in it overrides the table for every "
            f"scale but the shipped one."
        )
        assert actual != sum(p.numel() for p in shipped.parameters()), (
            f"fixture is degenerate: the {scale} rebuild has the same parameter "
            f"count as the shipped build"
        )

        model.eval()
        with torch.no_grad():
            model([torch.rand(3, 96, 96)])


#: Buffer and tensor totals measured off this repo's own build, as a cheap
#: regression tripwire.
#:
#: ⚠️ SELF-MEASURED. They prove the code is consistent with itself and nothing
#: more — see the block comment above ``_reference_parameters`` for the sibling
#: template where exactly such a number was cited as evidence and was wrong.
#: Parameters are asserted against the re-derived published tables in
#: ``guard_matches_the_published_architecture`` and
#: ``guard_architecture_table_is_a_live_knob``; what lives here is only what
#: those derivations do not cover.
#:
#: Updating these is legitimate when the architecture changes on purpose; state
#: the intended change in the commit message.
_PINNED_TOTALS = {"buffers": 0, "tensors": 675}


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
        f"checked against the re-derived published tables. If the change was "
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

    Unlike on ``yolov8_s.py``, this is load-bearing in the SHIPPED build: the
    stride-16 stage's ``RepCSP`` bottlenecks run at 48 channels (``inner = 96``,
    halved by the CSP split) and ``GroupNorm(32, 48)`` raises outright, so a
    hardcoded 32 does not even construct. That is asserted first, because "the
    shipped build needs it" is a stronger statement than "another scale would".

    Then it rebuilds at the published YOLOv9-T table — the narrowest scale — and
    asserts a sub-32 group count is genuinely produced there too, so the
    derivation is shown to cover the whole published range rather than to
    happen to work once.
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

    # ⚠️ WRAPPED, and this is the point of the guard rather than defensive
    # tidiness: on THIS scale a hardcoded 32 raises ``GroupNorm: num_channels
    # must be divisible by num_groups`` while the model is being constructed,
    # so the failure arrives as a ValueError from torch rather than as a failed
    # assertion. Letting it propagate would make the mutation sweep record an
    # ERROR instead of a caught mutation.
    try:
        shipped = _build(module, 3)
    except Exception as error:  # noqa: BLE001 — any build failure is the bug
        raise AssertionError(
            f"{module.__name__}: the SHIPPED build failed with "
            f"{type(error).__name__}: {error}. YOLOv9-S puts 48 channels inside "
            f"its stride-16 RepCSP bottlenecks, so a hardcoded GroupNorm group "
            f"count of 32 does not construct at all — which is why the count is "
            f"derived from the channel count."
        ) from error

    shipped_pairs = {
        (sub.num_groups, sub.num_channels)
        for sub in shipped.modules()
        if isinstance(sub, nn.GroupNorm)
    }
    assert shipped_pairs, "no GroupNorm in the shipped build — nothing checked"
    indivisible = sorted(
        (groups, channels)
        for groups, channels in shipped_pairs
        if channels % 32 and groups != 32
    )
    assert indivisible, (
        f"the SHIPPED build no longer contains a channel count that 32 groups "
        f"cannot divide ({sorted(shipped_pairs)}), so a hardcoded 32 would "
        f"construct and this guard has lost its strongest case. YOLOv9-S's "
        f"stride-16 stage is meant to put 48 channels inside its RepCSP."
    )
    for groups, channels in sorted(shipped_pairs):
        assert channels % groups == 0, (
            f"{module.__name__}: GroupNorm({groups}, {channels}) does not divide"
        )

    narrow = _reload(module)
    _apply_arch(narrow, _ARCH["yolov9t"])
    try:
        model = _build(narrow, 3)
    except Exception as error:  # noqa: BLE001 — any build failure is the bug
        raise AssertionError(
            f"{module.__name__}: rebuilding at the published YOLOv9-T table "
            f"failed with {type(error).__name__}: {error}. A hardcoded "
            f"GroupNorm group count crashes here — the stem is 16 channels at "
            f"that scale — which is why the count is derived."
        ) from error

    pairs = {
        (sub.num_groups, sub.num_channels)
        for sub in model.modules()
        if isinstance(sub, nn.GroupNorm)
    }
    assert pairs, "no GroupNorm at the narrower scale — nothing was checked"
    assert any(groups < 32 for groups, _ in pairs), (
        f"fixture is degenerate: every GroupNorm at the YOLOv9-T table still "
        f"takes 32 groups ({sorted(pairs)}), so a hardcoded 32 would pass this "
        f"guard too and the derivation is not being exercised"
    )
    for groups, channels in sorted(pairs):
        assert channels % groups == 0, (
            f"{module.__name__}: GroupNorm({groups}, {channels}) at the "
            f"YOLOv9-T table does not divide"
        )

    model.eval()
    with torch.no_grad():
        model([torch.rand(3, 96, 96)])


def guard_cspelan_chains_its_two_branches(module) -> None:
    """A GELAN stage must feed ``cv3`` **``cv2``'s output**, not the raw split
    half — that chain is what makes the block four tensors deep.

    ⚠️ THIS IS SILENT AT EVERY SHIPPED WIDTH. ``cv2`` and ``cv3`` both emit
    ``inner`` channels and ``cv3`` consumes ``inner``, and the fine stages have
    ``mid // 2 == inner``, so a parallel fan-out from the split half type-checks,
    keeps the parameter count identical and trains. The stage is then a
    two-branch ELAN rather than the chained GELAN the architecture is named for.

    Checked functionally: the expected branch list is reconstructed from ``cv1``,
    ``cv2`` and ``cv3`` and compared against the tensor ``cv4`` is actually
    handed.
    """
    import torch

    inner = 8
    stage = module.CSPELAN(16, 16, mid=2 * inner, inner=inner, blocks=1)
    stage.eval()
    assert stage.mid // 2 == stage.inner, (
        f"fixture is degenerate: mid // 2 is {stage.mid // 2} and inner is "
        f"{stage.inner}. They must MATCH, because that is the condition under "
        f"which a parallel fan-out is shape-legal and therefore silent — a "
        f"fixture where they differ would be caught by a shape error instead "
        f"of by this guard"
    )
    assert len(stage.cv2[0].m) >= 1, (
        "fixture is degenerate: with no RepCSP bottleneck cv2 is close to a "
        "1x1 pair and cv2(h) can be near-identity, weakening the comparison"
    )

    probe = torch.rand(1, 16, 6, 6) + 0.5
    captured = []
    handle = stage.cv4.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        with torch.no_grad():
            stage(probe)
    finally:
        handle.remove()
    assert len(captured) == 1, "the fusion conv's pre-hook did not fire once"

    with torch.no_grad():
        first, second = stage.cv1(probe).chunk(2, dim=1)
        chained = stage.cv2(second)
        expected = torch.cat([first, second, chained, stage.cv3(chained)], dim=1)
        parallel = torch.cat([first, second, chained, stage.cv3(second)], dim=1)

    assert not torch.allclose(expected, parallel, atol=1e-4), (
        "fixture is degenerate: chaining and fanning out produce the same "
        "tensor, so the rule cannot fire"
    )
    assert captured[0].shape == expected.shape, (
        f"CSPELAN hands its fusion conv {tuple(captured[0].shape)}; the GELAN "
        f"branch list is {tuple(expected.shape)}"
    )
    assert torch.allclose(captured[0], expected, atol=1e-5), (
        "CSPELAN's fusion conv is not receiving [split_0, split_1, cv2(split_1), "
        "cv3(cv2(split_1))]. The channel count is right and the model trains, so "
        "the likely shape is a PARALLEL fan-out — cv3 is being fed the raw split "
        "half, which halves the block's depth and is exactly what GELAN's "
        "aggregation exists to avoid."
    )


def guard_rep_conv_sums_its_branches_before_one_activation(module) -> None:
    """``RepConvNormAct`` must sum its 3x3 and 1x1 branches and THEN activate.

    That single activation over the sum is what makes the block
    re-parameterisable: the two conv-plus-norm branches are affine, so their sum
    collapses into one 3x3 kernel at deployment. Activating each branch first
    keeps the parameter count, the shapes and the loss keys identical and
    destroys that property — the sum of two SiLU outputs is not affine in the
    input.
    """
    import torch
    import torch.nn.functional as F

    block = module.RepConvNormAct(4, 6)
    block.eval()
    probe = torch.rand(1, 4, 5, 5) * 4.0 - 2.0

    with torch.no_grad():
        actual = block(probe)
        branch3 = block.conv3(probe)
        branch1 = block.conv1(probe)
        summed_then_activated = F.silu(branch3 + branch1)
        activated_then_summed = F.silu(branch3) + F.silu(branch1)

    assert not torch.allclose(
        summed_then_activated, activated_then_summed, atol=1e-4
    ), (
        "fixture is degenerate: the two orders agree on this input, so the rule "
        "cannot fire — the probe needs values where SiLU is genuinely nonlinear"
    )
    assert torch.allclose(actual, summed_then_activated, atol=1e-5), (
        f"{module.__name__}: RepConvNormAct is not summing its branches before "
        f"the activation. Max deviation from sum-then-activate is "
        f"{float((actual - summed_then_activated).abs().max()):.4g}, against "
        f"{float((actual - activated_then_summed).abs().max()):.4g} from "
        f"activate-then-sum. The parameter count, the shapes and every loss are "
        f"unchanged either way; what is lost is the re-parameterisation the "
        f"'Rep' in RepNCSP names."
    )


def guard_aconv_pools_before_the_strided_conv(module) -> None:
    """``AConv`` must average-pool before its stride-2 conv.

    ⚠️ THE POOL IS SHAPE-SILENT. ``avg_pool2d(x, 2, stride=1)`` shrinks each
    edge by exactly one, and the stride-2 conv then emits the SAME spatial size
    it would have emitted from the unpooled map on an even edge. So deleting it
    changes no shape, no parameter, no loss key — only the anti-aliasing the
    block exists for. This guard therefore measures the tensor the conv
    receives, and asserts the shape-silence explicitly so nobody assumes a
    shape check would have caught it.
    """
    import torch
    import torch.nn.functional as F

    block = module.AConv(8, 16)
    block.eval()
    probe = torch.rand(1, 8, 10, 10)

    with torch.no_grad():
        pooled_output = block(probe)
        unpooled_output = block.conv(probe)
    assert pooled_output.shape == unpooled_output.shape, (
        f"fixture is degenerate: with the pool the output is "
        f"{tuple(pooled_output.shape)} and without it "
        f"{tuple(unpooled_output.shape)}. They must MATCH — the whole point is "
        f"that a missing pool is invisible to a shape check"
    )

    seen = []
    handle = block.conv.register_forward_pre_hook(
        lambda _module, inputs: seen.append(inputs[0].detach().clone())
    )
    try:
        with torch.no_grad():
            block(probe)
    finally:
        handle.remove()

    assert len(seen) == 1, "the conv's pre-hook did not fire once"
    expected = F.avg_pool2d(probe, 2, stride=1, padding=0, count_include_pad=True)
    assert seen[0].shape == expected.shape, (
        f"{module.__name__}: AConv's conv received {tuple(seen[0].shape)} for a "
        f"{tuple(probe.shape)} input; the 2x2 stride-1 average pool makes it "
        f"{tuple(expected.shape)}. The pool is missing, and no shape "
        f"downstream can tell."
    )
    assert torch.allclose(seen[0], expected, atol=1e-6), (
        f"{module.__name__}: AConv's conv receives a tensor of the right shape "
        f"that is not the 2x2 stride-1 average of its input"
    )


def guard_sppelan_pools_in_series(module) -> None:
    """``SPPELAN``'s max-pools must be applied **in series**, each on the
    previous output.

    Series is what gives the 5/9/13 effective receptive field from one 5x5
    kernel — the whole reason the block is "fast". Applying all three to
    ``cv1``'s output instead produces identical shapes, identical parameters and
    three identical branches, so the block widens the deepest stage's receptive
    field by 5 rather than 13 and nothing else notices.
    """
    import torch

    block = module.SPPELAN(4, 8, 4)
    block.eval()
    assert block.repeats >= 2, (
        f"fixture is degenerate: {block.repeats} repeat(s). At one repeat the "
        f"series and parallel arrangements contain the same tensors"
    )

    probe = torch.rand(1, 4, 9, 9)
    captured = []
    handle = block.cv5.register_forward_pre_hook(
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
        f"SPPELAN hands its fusion conv {tuple(captured[0].shape)}, expected "
        f"{tuple(expected.shape)}"
    )
    assert torch.allclose(captured[0], expected, atol=1e-5), (
        "SPPELAN's pools are not applied in series. The three branches are the "
        "same 5x5 pooling of cv1's output rather than 5 / 9 / 13, so the "
        "deepest stage's receptive field is a third of the design's — and every "
        "shape, parameter and loss is unchanged."
    )


def guard_head_is_decoupled(module) -> None:
    """The classification and box towers must share no parameters.

    Checked by parameter identity rather than by reading the constructor, and
    rather than by relying on a crash. On this head the two towers happen to be
    DIFFERENT widths — 64 channels for the box branch against 128 for the class
    branch — so the crudest coupling raises a shape error on the first forward.
    That is luck: YOLOX's two towers are the same width, and there the identical
    edit trains happily and reports the same loss keys. A width-matched coupling
    here would be equally silent, and identity catches it either way.
    """
    head = _build(module, 3).head
    cls_ids = {id(p) for p in head.cls_convs.parameters()}
    box_ids = {id(p) for p in head.box_convs.parameters()}

    assert cls_ids and box_ids, (
        f"expected both conv towers to hold parameters, got {len(cls_ids)} cls "
        f"/ {len(box_ids)} box"
    )
    shared = cls_ids & box_ids
    assert not shared, (
        f"{module.__name__}: the classification and box towers share "
        f"{len(shared)} parameter tensor(s) — the head is COUPLED, not "
        f"decoupled. It would train and log identical loss keys either way."
    )


def guard_reg_max_reaches_the_head_and_the_decode(module) -> None:
    """``REG_MAX`` must be a live knob, not a constant that reaches nothing.

    The failure mode this is written against is a declared parameter that is
    passed to a stage and never used: changing it alters neither the model nor
    any test, so it reads as configuration and is decoration. Here it decides
    the box branch's channel count, the head's reshape, the decode's bin vector
    and the DFL target's clamp — so the guard rebuilds at a DIFFERENT value and
    asserts all four moved with it.
    """
    import torch

    assert module.REG_MAX == 16, (
        f"{module.__name__}: REG_MAX is {module.REG_MAX}; YOLOv9 publishes 16, "
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
        for level, predictor in enumerate(model.head.box_preds):
            assert predictor.out_channels == 4 * probe, (
                f"{module.__name__}: box predictor {level} emits "
                f"{predictor.out_channels} channels at REG_MAX = {probe}; the "
                f"DFL head's width is 4 * reg_max = {4 * probe}. A hardcoded "
                f"channel count leaves the knob decorative and the head's "
                f"reshape is the first thing to notice."
            )

        model.eval()
        with torch.no_grad():
            _, dist_logits, anchors = model.head(
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


def guard_seed_excluded_prefixes_are_exactly_the_class_shaped_keys(module) -> None:
    """``SEED_EXCLUDED_PREFIXES`` must name every class-count-dependent tensor
    and nothing else.

    Re-derives the declaration the way ``tools/derive_seed_excluded.py`` does —
    build twice at different class counts and diff the state_dict shapes — so a
    head that grows a second class-shaped tensor, or a declared prefix that has
    gone stale, is red here rather than an edge-only strict-load failure
    (backend#2642).

    It also pins the property the declaration DEPENDS on: the class tower's
    width, ``max(in_channels[0], min(num_classes, 100))``, is 128 for every
    class count at this scale because ``in_channels[0]`` is 128 and the class
    term is capped at 100. That is a property of THIS scale, not of the formula
    — rebuild with the YOLOv9-T table (64 channels at P3) and ``cls_hidden``
    becomes 80 at 80 classes, i.e. class-count dependent, at which point the
    seed would carry two more tensors per level than the declaration admits.
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
        f"shape mismatch backend#2642 exists to remove. Re-run "
        f"tools/derive_seed_excluded.py and tools/seed_contract.py apply; do "
        f"not hand-edit the constant."
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
    """The head's flattened predictions and the anchor table it returns
    alongside them must agree, cell for cell.

    Rewires level 0 so its class channel 0 is literally its input feature's
    channel 0, feeds a map coding ``y * 100 + x``, and reads the codes back in
    the order the head emitted them — then compares against the cell
    coordinates the head returned in the same call. If the two are flattened in
    different orders, every anchor is matched against another cell's prediction:
    the assigner's geometry and the classifier's evidence come from different
    places, every loss stays finite, and the model simply cannot learn to
    localise.
    """
    import torch
    from torch import nn

    model = _build(module, 2)
    head = model.head
    channels_per_level = [tower[0].conv.in_channels for tower in head.cls_convs]

    features = []
    for level, (height, width) in enumerate(_ORDERING_SHAPES):
        channels = channels_per_level[level]
        head.cls_convs[level] = nn.Identity()
        head.box_convs[level] = nn.Identity()
        head.cls_preds[level] = _select_channel_zero(nn, channels, model.num_classes)
        head.box_preds[level] = _zero_conv(nn, channels, 4 * head.reg_max)
        features.append(_positional_feature(torch, channels, height, width))

    with torch.no_grad():
        cls_logits, _, anchors = head(tuple(features))

    height, width = _ORDERING_SHAPES[0]
    assert height != width, (
        "fixture is degenerate: level 0's feature map is square, so the two "
        "flattening orders coincide and a transposed reshape cannot be seen"
    )
    count = height * width
    # The anchor table holds cell centres, so subtract the half-cell offset to
    # recover the integer cell the code was written into.
    cells = [(float(a[0]) - 0.5, float(a[1]) - 0.5) for a in anchors[:count]]
    expected = [y * 100.0 + x for x, y in cells]
    actual = [float(v) for v in cls_logits[0, :count, 0]]
    assert actual == expected, (
        f"{module.__name__}: the head's flattened predictions do not line up "
        f"with the anchor table it returns alongside them. Cell (y, x) codes "
        f"read back as {actual}, expected {expected}. The predictions and the "
        f"anchor coordinates are flattened in DIFFERENT orders, which is "
        f"invisible on the square feature maps the template actually builds."
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
    ``-10``, i.e. ``sigmoid(-10) = 4.5e-5``, two orders of magnitude below
    ``SCORE_THRESH`` — so exactly the named entries survive the threshold and
    every assertion downstream is about them.
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
    """Decoding is driven directly, with scores actually above threshold, at
    batch size two.

    A freshly built DFL head predicts around ``sigmoid(-9)`` on every class, so
    a forward pass at initialisation returns nothing but noise and any check
    downstream of it is vacuous — the decode has nothing to get wrong. That is
    how a real defect shipped through every guard on a sibling template: its
    post-processing iterated the wrong axis and ``zip`` truncated silently
    instead of raising, so it processed level 0 of image 0 only and was broken
    outright at batch > 1. Both halves are addressed here: synthetic head
    outputs with one confident detection per image, and **two** images.
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
            f"{module.__name__}: image {index} produced no detection although "
            f"its class logit was set to +10, which is far above SCORE_THRESH "
            f"({model.score_thresh}) — the fixture is not exercising the decode"
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
    # have had, and yields dataset label -1 once the handler shifts back.
    #
    # This needs its own fixture because the one above deliberately puts its
    # confident scores on REAL classes, so it cannot see a channel-0 leak. And a
    # freshly built model cannot see it either: the prior sits below
    # score_thresh, so `model(images)` returns nothing at initialisation and any
    # assertion over an empty label tensor is vacuous.
    background_logit, real_logit = 10.0, 8.0
    bg_cls, bg_dist, bg_anchors = _synthetic_head_output(
        torch, model, cells, 8, {(0, 2): 0}
    )
    bg_cls[0, 2, classes - 1] = real_logit  # a real class too, but weaker
    bg_results = model._predictions(bg_cls, bg_dist, bg_anchors, [(64, 64)])
    bg_labels = bg_results[0]["labels"]
    assert bg_labels.numel(), (
        f"{module.__name__}: the background-channel fixture decoded to nothing, "
        f"so the assertions below are vacuous — check the scores clear "
        f"score_thresh"
    )
    assert not bool((bg_labels == 0).any()), (
        f"{module.__name__}: decode returned label 0, the background channel: "
        f"{sorted(set(bg_labels.tolist()))}. It is trained only as a negative "
        f"and must be dropped BEFORE the score threshold and the top-k, not "
        f"left to the engine — the detection budget is spent here."
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
    # Rank everything, so the fixture cannot pass by returning nothing.
    model.score_thresh = 0.0
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


# --------------------------------------------------------------------------
# task-aligned assignment
#
# EVERY GUARD BELOW ASSERTS *WHICH* ANCHOR OR GROUND TRUTH IS SELECTED, AND
# WITH WHAT SOFT TARGET — never how many. Cardinality is invariant to any
# reweighting of the alignment metric: swap the two exponents and exactly the
# same number of anchors is chosen, from the wrong end of the ranking. That gap
# hid a swapped focal alpha in `sparse_rcnn` through a full mutation sweep and
# two review passes.
# --------------------------------------------------------------------------


def _assign(model, torch, gt_boxes, gt_labels, scores_by_anchor, pred_boxes, points):
    """Call ``assign`` with explicit per-anchor class probabilities.

    ``scores_by_anchor`` is ``[{channel: probability}, ...]``, one dict per
    anchor; unset channels are 0. Building the score matrix by hand is what lets
    a fixture separate "best classified" from "best localised".
    """
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
    )


def guard_tal_metric_weights_localisation_over_classification(module) -> None:
    """The alignment metric must be ``score ** TAL_ALPHA * iou ** TAL_BETA``,
    with the published 0.5 and 6.0 — **in that order**.

    The two exponents are wildly asymmetric, so IoU dominates: a well-localised
    but poorly-classified anchor beats a confidently-classified but badly-boxed
    one. Swapping them reverses the ranking, selects exactly the same NUMBER of
    anchors, and leaves every loss finite — which is why this asserts which
    anchor receives the larger soft target rather than counting anything.

    The fixture states its own discriminating property: under the published
    exponents anchor B wins clearly, under the swap anchor A does. If either
    stops holding, the fixture — not the rule — has broken.
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
    """Exactly the ``TAL_TOPK`` **highest-metric** candidates per ground truth.

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

    fg_mask, _, _, _ = _assign(model, torch, gt, [label], scores, pred_boxes, points)
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

    This head has **no objectness branch**: the score it ranks by at inference
    is the classifier's, so the classifier is what has to carry localisation
    quality. A hard 1.0 target trains happily and simply removes the model's
    ability to say "this is a car, but I have it badly boxed"; dropping the
    normalisation leaves the raw metric, which at ``iou ** 6`` is a small
    number and quietly rescales the whole classification loss.

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
#: That asymmetry is the whole mechanism of the guard.
_BOX_PREDICTOR_GROUP = "box_preds"
_CLASS_PREDICTOR_GROUP = "cls_preds"


def guard_positives_reach_the_box_regression_branch(module) -> None:
    """One train step must leave the box predictors with a real gradient —
    which happens only if the assigner matched something.

    This is the assign-nothing guard, and it is ``requires_grad``-aware in the
    direction that matters: a bare ``p.grad is None`` sweep false-flags a
    deliberately frozen parameter, while the real defect is a **trainable**
    parameter the loss never reaches. Three assertions, failing for different
    reasons:

    * no trainable parameter may have a ``None`` gradient at all — that is a
      branch detached from the loss entirely;
    * the box group must have a non-zero gradient somewhere. The template falls
      back to ``prediction.sum() * 0.0`` when there are no positives (so the
      loss dict keeps its shape and no gradient is ``None``), which means an
      all-negative assignment shows up here as an exactly-zero box gradient and
      **nowhere else**;
    * the class group must too, as a sanity check on the fixture.

    Deliberately NOT asserted: that all three levels receive positives. At
    random initialisation the alignment landscape is dominated by centre jitter
    rather than by scale, so which level wins a given ground truth is not
    deterministic, and a fixture pretending otherwise would pass for the wrong
    reason. Per-level structure is pinned by the deterministic geometry guards
    instead (``decode_per_level_stride``, ``dfl_target_cell_units``).
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

    assert alive(_BOX_PREDICTOR_GROUP), (
        f"{module.__name__}: every level of {_BOX_PREDICTOR_GROUP} received an "
        f"exactly zero gradient, so NO ground truth was assigned a positive "
        f"anchor. Three objects were supplied, one at each stride's scale. An "
        f"all-negative image still yields a finite, small loss and a clean "
        f"train step, which is why nothing else in this suite sees it."
    )
    assert alive(_CLASS_PREDICTOR_GROUP), (
        f"{module.__name__}: every level of {_CLASS_PREDICTOR_GROUP} received a "
        f"zero gradient — the classification branch is detached from the loss"
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
        raise OSError("network access is blocked by test_yolov9_s")

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
#: 200 forward+backward passes and is a guard, not a benchmark; the entry point
#: takes the edge as its second argument precisely so the transform can be built
#: smaller. Measured on the shipped template under the engine pin: the loss
#: falls 13.254 -> 1.491 and the top detection reaches score 0.992 at IoU 0.996,
#: so the thresholds below carry several times the needed margin and absorb BLAS
#: variation across platforms.
_OVERFIT_STEPS = 200
_OVERFIT_LR = 4e-3
_OVERFIT_EDGE = 128
_OVERFIT_BOX = [40.0, 40.0, 88.0, 88.0]
_OVERFIT_LABEL = 1


def guard_overfits_a_single_object(module) -> None:
    """The template must actually LEARN — and then detect what it learned.

    Everything else in this file is a single step or a synthetic call. This is
    the end-to-end claim: ``_OVERFIT_STEPS`` Adam steps on one image with one
    object, then an eval pass that has to find it. It is the only guard that
    closes the loop from the assigner through the three losses to the DFL
    decode, and it is what makes "trains" and "evaluates" claims about this
    template rather than about its return types.

    It also covers what nothing else can: zeroing the soft classification
    target leaves every structural guard green, every loss finite and every
    fixture satisfied, and the model simply never learns to fire.

    ⚠️ AND HERE IS WHAT IT DOES **NOT** COVER, measured rather than assumed. An
    assigner that prefers the WORST-localised candidate — ``ious`` replaced by
    ``1 - ious`` in the alignment metric — still overfits this fixture past
    every threshold below: measured loss 13.260 -> 3.702 (the bar is
    ``< 0.5 * first``, i.e. 6.63), best score 0.141 (bar 0.10), correct class,
    IoU 0.940 (bar 0.35). The reason is that the metric only chooses *which*
    anchors are positive; the box and DFL losses then regress whichever anchors
    were chosen towards the true box, and the inside-the-box prefilter keeps
    them all on the object. So a wrong ranking is not visible end-to-end on one
    object, and the IoU assertion below is NOT evidence that the assigner ranks
    correctly. That property is pinned by ``tal_metric_exponents``, which is
    where that mutation is registered after surviving here.

    Note the score margin is thin — 0.141 against a 0.10 bar — so this guard
    ALMOST catches it. Raising the bar to catch it would be the wrong fix: the
    threshold would then be tuned to one mutation on one seed rather than to
    "the model learned to be confident", and the ranking property has a
    deterministic guard already.
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
    model.score_thresh = 0.0  # rank everything; the assertions are about the top
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
        f"it never learned to be CONFIDENT about anything, which is what a "
        f"broken soft classification target looks like. There is no objectness "
        f"branch here to carry that quality signal, so the classifier is the "
        f"only place it can live, and every structural guard stays green "
        f"through this."
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
        f"NOTE this is a weaker signal than it looks: an inverted IoU term in "
        f"the alignment metric passes here, because the box loss regresses "
        f"whichever anchors were chosen towards the true box. See "
        f"tal_metric_exponents."
    )


# --------------------------------------------------------------------------
# the guard table, and the mutations that prove each can go red
# --------------------------------------------------------------------------

GUARDS = {
    "published_architecture": guard_matches_the_published_architecture,
    "arch_table_is_live": guard_architecture_table_is_a_live_knob,
    "module_tree_size": guard_module_tree_size_is_pinned,
    "no_stateful_norm": guard_no_stateful_normalisation,
    "derived_norm_groups": guard_norm_groups_are_derived_from_the_channel_count,
    "cspelan_chains_branches": guard_cspelan_chains_its_two_branches,
    "rep_conv_sums_first": guard_rep_conv_sums_its_branches_before_one_activation,
    "aconv_pools_first": guard_aconv_pools_before_the_strided_conv,
    "sppelan_series": guard_sppelan_pools_in_series,
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
    "original_coordinates": guard_predictions_are_in_original_image_coordinates,
    "declared_size_measured": guard_declared_image_size_is_the_measured_edge,
    "tal_metric_exponents": guard_tal_metric_weights_localisation_over_classification,
    "tal_inside_the_box": guard_tal_requires_the_anchor_point_inside_the_box,
    "tal_topk_ranking": guard_tal_selects_the_topk_best_ranked_candidates,
    "tal_normalised_target": guard_tal_target_is_the_normalised_alignment_metric,
    "tal_tie_break_by_iou": guard_tal_breaks_ties_by_iou_not_by_alignment,
    "positives_reach_box_branch": guard_positives_reach_the_box_regression_branch,
    "no_network": guard_constructs_with_no_network,
    "overfits_one_object": guard_overfits_a_single_object,
}

#: ``(name, anchor, replacement, guard)``. The anchor must be unique in the
#: file — ``_mutate`` refuses otherwise, so a drifted anchor is a RED rather
#: than a patch that silently applies to nothing and reports "passed".
MUTATIONS = [
    (
        "rep_bottleneck_squeezes_the_branch_again",
        "            *(RepBottleneck(hidden, hidden, 1.0, shortcut) "
        "for _ in range(blocks))",
        "            *(RepBottleneck(hidden, hidden, 0.5, shortcut) "
        "for _ in range(blocks))",
        "published_architecture",
    ),
    (
        "rep_conv_loses_its_1x1_branch",
        "        self.conv1 = ConvNormAct(in_ch, out_ch, 1, stride=1, act=False)",
        "        self.conv1 = ConvNormAct(in_ch, out_ch, 3, stride=1, act=False)",
        "published_architecture",
    ),
    (
        "neck_downsamples_to_the_level_width",
        "        self.bu_down3 = _build_downsample(downsample, p3_out, bu4_down)",
        "        self.bu_down3 = _build_downsample(downsample, p3_out, bu4_out)",
        "published_architecture",
    ),
    (
        "stem_widths_hardcoded_at_the_shipped_scale",
        "        first, second = stem_channels",
        "        first, second = (32, 64)",
        "arch_table_is_live",
    ),
    (
        "downsampler_kind_hardcoded",
        "                self.downsamples.append(_build_downsample(downsample, "
        "in_ch, down_out))",
        '                self.downsamples.append(_build_downsample("aconv", '
        "in_ch, down_out))",
        "arch_table_is_live",
    ),
    (
        "stage_kind_ignored_by_the_builder",
        "        return ELAN1(in_ch, out_ch, mid, inner)",
        "        return CSPELAN(in_ch, out_ch, mid, inner, 1)",
        "published_architecture",
    ),
    (
        "extra_bottleneck_per_stage",
        "for _ in range(blocks))",
        "for _ in range(blocks + 1))",
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
    (
        "cspelan_fans_out_in_parallel",
        """        branches = list(self.cv1(x).chunk(2, dim=1))
        branches.append(self.cv2(branches[-1]))
        branches.append(self.cv3(branches[-1]))
        return self.cv4(torch.cat(branches, dim=1))""",
        """        branches = list(self.cv1(x).chunk(2, dim=1))
        split = branches[-1]
        branches.append(self.cv2(split))
        branches.append(self.cv3(split))
        return self.cv4(torch.cat(branches, dim=1))""",
        "cspelan_chains_branches",
    ),
    (
        "rep_conv_activates_each_branch",
        "        return self.act(self.conv3(x) + self.conv1(x))",
        "        return self.act(self.conv3(x)) + self.act(self.conv1(x))",
        "rep_conv_sums_first",
    ),
    (
        "aconv_drops_the_average_pool",
        "        return self.conv(F.avg_pool2d(x, 2, stride=1, padding=0, "
        "count_include_pad=True))",
        "        return self.conv(x)",
        "aconv_pools_first",
    ),
    (
        "sppelan_pools_in_parallel",
        "            outputs.append(self.pool(outputs[-1]))",
        "            outputs.append(self.pool(outputs[0]))",
        "sppelan_series",
    ),
    (
        "coupled_head",
        """            self.cls_convs.append(
                nn.Sequential(
                    ConvNormAct(channels, self.cls_hidden, 3, stride=1),
                    ConvNormAct(self.cls_hidden, self.cls_hidden, 3, stride=1),
                )
            )""",
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
        "        self.head = YOLOv9Head(\n"
        "            self.num_classes, self.neck.out_channels, reg_max=self.reg_max\n"
        "        )",
        "        self.head = YOLOv9Head(self.num_classes, self.neck.out_channels)",
        "reg_max_is_live",
    ),
    (
        "class_tower_width_tracks_the_class_count",
        "        self.cls_hidden = max(in_channels[0], min(num_classes, 100))",
        "        self.cls_hidden = max(16, min(num_classes, 100))",
        "seed_excluded_prefixes",
    ),
    (
        "seed_prefix_dropped",
        'SEED_EXCLUDED_PREFIXES = ("head.cls_preds.0.", "head.cls_preds.1.", '
        '"head.cls_preds.2.")',
        'SEED_EXCLUDED_PREFIXES = ("head.cls_preds.0.", "head.cls_preds.1.")',
        "seed_excluded_prefixes",
    ),
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
        "            candidate_boxes = boxes[box_index]",
        "            candidate_boxes = boxes[: box_index.shape[0]]",
        "decode_per_image",
    ),
    (
        "background_channel_kept",
        "            class_scores = class_scores[:, 1:]\n"
        "            num_anchors, num_classes = class_scores.shape",
        "            num_anchors, num_classes = class_scores.shape",
        "decode_per_image",
    ),
    # The one the label assertion alone CANNOT see, and the reason the
    # background fixture also asserts a score: every real class still lands on
    # its own label, so the first fixture is green and no label 0 is ever
    # emitted -- the background channel is simply renamed to the last label and
    # spends a detection slot from the top of the ranking.
    (
        "background_channel_rotated_to_the_last_label",
        "            class_scores = class_scores[:, 1:]",
        "            class_scores = torch.cat(\n"
        "                (class_scores[:, 1:], class_scores[:, :1]), dim=1\n"
        "            )",
        "decode_per_image",
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
    (
        "swapped_alignment_exponents",
        "        alignment = scores.pow(TAL_ALPHA) * ious.pow(TAL_BETA)",
        "        alignment = scores.pow(TAL_BETA) * ious.pow(TAL_ALPHA)",
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
        "        topk = min(TAL_TOPK, num_anchors)",
        "        topk = num_anchors",
        "tal_topk_ranking",
    ),
    (
        "topk_takes_the_worst",
        "        _, positions = torch.topk(candidate, topk, dim=1)",
        "        _, positions = torch.topk(candidate, topk, dim=1, largest=False)",
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
        "        self.backbone = GELANBackbone()",
        '        __import__("socket").getaddrinfo("download.pytorch.org", 443)\n'
        "        self.backbone = GELANBackbone()",
        "no_network",
    ),
    (
        "soft_class_target_zeroed",
        "                cls_targets[index, fg_mask, labels] = aligned",
        "                cls_targets[index, fg_mask, labels] = aligned * 0.0",
        "overfits_one_object",
    ),
    # RETARGETED, and the reason is worth keeping. This was written against
    # `overfits_one_object`, on the assumption that an assigner preferring the
    # WORST-localised candidate could not learn to box an object. MEASURED on
    # THIS template, not inherited from the sibling's note: it can. Loss
    # 13.260 -> 3.702, best score 0.141, correct class, IoU 0.940 -- past all
    # three of that guard's bars (6.63 / 0.10 / 0.35), so the overfit guard
    # stays green and the mutation SURVIVED it. With the inside-the-box
    # prefilter still in place the box and DFL losses regress whichever anchors
    # were chosen towards the true box. The metric's direction is a ranking
    # property, so it is pinned by the ranking guard, which catches it
    # outright.
    (
        "alignment_prefers_the_worst_iou",
        "        alignment = scores.pow(TAL_ALPHA) * ious.pow(TAL_BETA)\n",
        "        alignment = scores.pow(TAL_ALPHA) * (1.0 - ious).pow(TAL_BETA)\n",
        "tal_metric_exponents",
    ),
]

#: Mutations whose whole point is that they are SILENT elsewhere: each one
#: leaves ``model(images, targets)`` returning a finite loss dict, so
#: ``tests/test_od_torchvision_family_train_step.py`` stays green against every
#: one of them. Asserted in ``test_silent_mutations_still_train`` so nobody
#: concludes the family train-step test already covers this file.
_SILENT_MUTATIONS = frozenset(
    {
        # GELAN-specific, and the reason this template needed its own suite:
        # none of these changes a shape, a parameter count or a loss key.
        "cspelan_fans_out_in_parallel",
        "rep_conv_activates_each_branch",
        "aconv_drops_the_average_pool",
        "sppelan_pools_in_parallel",
        # assigner and decode, shared with the sibling DFL templates
        "swapped_alignment_exponents",
        "no_inside_the_box_rule",
        "topk_bound_removed",
        "topk_takes_the_worst",
        "target_not_normalised",
        "hard_class_target",
        "tie_break_by_alignment",
        "assign_nothing",
        "soft_class_target_zeroed",
        "alignment_prefers_the_worst_iou",
        "dfl_decode_takes_an_argmax",
        "dfl_loss_collapses_the_two_bins",
        "dfl_target_uses_one_stride",
        # NOT "coupled_head". Measured: on THIS head coupling the towers is not
        # silent, it raises. The box tower is 64 channels wide and the class
        # tower 128 (`max(16, ch[0] // 4, 4 * reg_max)` against
        # `max(ch[0], min(nc, 100))`), so feeding the box tower's output to the
        # class predictor is a shape error on the first forward. That is luck,
        # not design -- YOLOX's two towers are the same width, where the same
        # edit trains happily -- so `decoupled_head` still checks parameter
        # identity rather than relying on a crash.
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
