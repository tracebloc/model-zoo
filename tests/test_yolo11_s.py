"""Guards for ``object_detection/pytorch/yolo11_s.py``, each proven able to go
red by a mutation that is kept in the suite.

Why this file exists
--------------------
``tests/test_od_torchvision_family_train_step.py`` proves a template returns a
loss dict and a ``List[Dict]`` of xyxy predictions. For a template that wraps a
torchvision builder that is a real assertion: the loss is the library's. For
``yolo11_s.py`` the backbone, neck, head, assigner and all three losses are
**our own code**, so "returns a loss dict" proves only that our code returns a
dict. Every interesting way a hand-written detector is wrong is silent:

* the assigner matches **nothing** — BCE over an all-negative image is finite
  and small, so the train step passes and the model learns no objects;
* the assigner matches the wrong anchors — a swapped alignment exponent picks
  the best-*classified* candidate instead of the best-*localised* one, changes
  no cardinality whatsoever, and leaves every loss finite;
* a ``C3k2`` is secretly a ``C3``, or its inner bottleneck runs at the wrong
  width, or a stage's block kind is wrong;
* a module is **constructed and never called** — see the second trap below;
* the DFL decode takes an argmax, or forgets its softmax, and boxes quantise or
  explode while the loss (computed on the logits) never notices;
* predictions are never mapped back to the original image coordinates, so mAP
  is computed against boxes in the resized frame;
* NMS is dropped, and every object is reported ten times.

None of those fail a train step. So each is a named guard here, and each guard
is paired with a **mutation** — an exact textual edit to the shipped template
that the guard must catch. ``_mutate`` asserts its anchor appears exactly once,
so a mutation that no longer applies is a RED, not a survivor reported as
"passed"; ``test_no_mutation_baseline`` runs the whole guard table against the
unmutated file so a sweep always carries its own zero row; and
``test_mutation_is_caught_by_its_guard`` runs **one named guard** per mutation
rather than the whole file, so a red is attributed to the guard under test and
not to a neighbour that happens to notice.

Four traps this file is shaped around
--------------------------------------
**A BYPASSED MODULE IS INVISIBLE TO EVERY PARAMETER COUNT.** This is measured,
not theoretical. Deleting ``self.c2psa(...)`` from ``YOLO11Backbone.forward``
leaves C2PSA *constructed*, so the parameter total, the buffer total and the
``state_dict`` key set are all unchanged, the loss keys are unchanged, and the
model trains — ~991,000 parameters shipped and averaged every federated round
for nothing. The published-count guard **DID NOT RAISE** against it. The same
holds for the SPPF and for the attention's positional encoding. Those need
functional "was it applied" guards, which is what
``deepest_stage_modules_are_applied`` and ``attention_reference_operator`` are.

**The eval path is nearly vacuous on a fresh model.** YOLO11's class prior is
``log(5 / nc / (640 / stride) ** 2)``, which at stride 8 is far below
``SCORE_THRESH`` — so a freshly built model's detections are a handful of
stride-32 anchors firing on noise, and any assertion taken from
``model(images)`` at initialisation is checking almost nothing.
``guard_decode_is_per_image_and_aligned`` therefore drives ``_predictions``
**directly**, with synthetic head outputs that clear the score threshold by a
wide margin, **at batch two** — a per-image bug is invisible at batch one by
construction.

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
``_reference_yolo11_layers``, which is derived from the published arch table
with nothing from ``model_zoo/`` imported, and anchored to figures from outside
this repo entirely — see ``_PUBLISHED`` and ``_PUBLISHED_LAYERS``. And it is
compared **per layer**, not only in total: two compensating errors survive a
total and cannot survive eighteen rows.
"""

import contextlib
import importlib.util
import pathlib
import tempfile

import pytest

ROOT = pathlib.Path(__file__).parent.parent
OD_PYTORCH = ROOT / "model_zoo" / "object_detection" / "pytorch"
TEMPLATE = OD_PYTORCH / "yolo11_s.py"

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
        return module


def _load():
    return _exec_source(TEMPLATE.read_text(encoding="utf-8"), TEMPLATE.stem)


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
# from `ultralytics/cfg/models/11/yolo11.yaml` and `nn/modules/{block,conv,head}.py`,
# with NOTHING under `model_zoo/` imported. And the transcription is itself
# anchored to figures from outside this repo — the yaml's own per-scale model
# summaries — so it cannot drift into agreeing with a wrong template.
#
# ⚠️ AND IT IS CHECKED PER LAYER, NOT ONLY IN TOTAL. On the sibling `yolov10_s`
# the total matched while individual layers were wrong; only a per-layer
# comparison found it. `_PUBLISHED_LAYERS` holds the upstream per-layer table so
# the same class of compensating error cannot survive here either.

#: Published parameter counts, quoted verbatim from ``yolo11.yaml``'s own
#: ``scales:`` comments (identical in ultralytics ``v8.3.0``, the YOLO11 release
#: tag, and on ``main`` — two years apart, so the figures are stable and not a
#: transient of one release)::
#:
#:   n: [0.50, 0.25, 1024] # summary: 2624080 parameters, 2624064 gradients
#:   s: [0.50, 0.50, 1024] # summary: 9458752 parameters, 9458736 gradients
#:   m: [0.50, 1.00,  512] # summary: 20114688 parameters, 20114672 gradients
#:   l: [1.00, 1.00,  512] # summary: 25372160 parameters, 25372144 gradients
#:   x: [1.00, 1.50,  512] # summary: 56966176 parameters, 56966160 gradients
#:
#: FIVE scales, not one, so the anchor pins the width handling, the depth
#: handling AND the per-scale block-kind override rather than one arithmetic
#: total. The depth multiplier in particular is 0.50 at BOTH n and s, so it
#: separates only at l/x — an n-and-s-only anchor cannot see it at all.
#:
#: The 16-parameter gap in each is the DFL projection vector, which upstream
#: stores as a frozen ``Conv2d`` weight (``requires_grad=False``, hence
#: "gradients") and this template does not store at all — it builds the bin
#: indices with ``torch.arange`` inside the decode. So THIS TEMPLATE'S TOTAL
#: PARAMETER COUNT IS THE PUBLISHED *GRADIENT* COUNT, exactly, and the
#: derivation below is checked against both figures with the gap stated
#: explicitly rather than absorbed into a tolerance.
#:
#: ⚠️ THERE IS NO FUSED/UNFUSED AMBIGUITY HERE, unlike on `yolov10_s`, where
#: three different published figures per scale exist and the docs quote the
#: narrowest. Measured against ultralytics 8.3.0: YOLO11-S is 9,458,752
#: unfused and 9,443,760 after `model.fuse()`, a 0.16% difference, and the docs
#: table's one-decimal "9.4M" cannot distinguish them. The row above is the
#: unfused graph, which is what any GroupNorm build must be compared against
#: anyway — GroupNorm does not fold into a convolution the way BatchNorm does.
_PUBLISHED = {
    # scale: (width, depth, max_channels, c3k forced on every C3k2,
    #         published total, published gradients)
    "n": (0.25, 0.50, 1024, False, 2_624_080, 2_624_064),
    "s": (0.50, 0.50, 1024, False, 9_458_752, 9_458_736),
    "m": (1.00, 0.50, 512, True, 20_114_688, 20_114_672),
    "l": (1.00, 1.00, 512, True, 25_372_160, 25_372_144),
    "x": (1.50, 1.00, 512, True, 56_966_176, 56_966_160),
}

#: The scale this template ships at.
_SHIPPED_SCALE = "s"

#: Parameters upstream stores for the DFL bin vector and this template does not.
_DFL_PROJECTION_CONSTANTS = 16

#: The class count the published figures are quoted at.
_PUBLISHED_CLASSES = 80

#: ``parse_model``'s per-scale override, transcribed::
#:
#:     if m is C3k2 and scale in "mlx":  # for M/L/X sizes
#:         args[3] = True
#:
#: i.e. at m/l/x every ``C3k2`` uses a nested ``C3k`` regardless of what the
#: yaml row says. Recorded here because it is the reason an m/l/x total cannot
#: be reproduced from the yaml text alone, and because a "scale" in YOLO11 is
#: therefore a multiplier triple PLUS a block kind.
_C3K_FORCED_SCALES = ("m", "l", "x")

#: Per-layer parameter counts of the upstream model, measured with
#: ``ultralytics==8.3.0`` (the YOLO11 release tag) as::
#:
#:     from ultralytics.nn.tasks import DetectionModel, yaml_model_load
#:     m = DetectionModel(yaml_model_load("yolo11s.yaml"), nc=80, verbose=False)
#:     [sum(p.numel() for p in layer.parameters()) for layer in m.model]
#:
#: Keyed by the yaml's own layer index. Layers 11/12/14/15/18/21 are
#: ``nn.Upsample`` and ``Concat`` and hold no parameters, so they are omitted
#: rather than listed as zeros. Layer 23 (``Detect``) is quoted MINUS the 16 DFL
#: projection constants, so it is directly comparable with this template's head.
#:
#: ⚠️ THIS IS THE PER-LAYER ANCHOR, and it is the point of the whole block. A
#: total can be hit by two errors that cancel; on `yolov10_s` exactly that
#: happened. Eighteen rows at two scales cannot be.
#:
#: Only n and s are listed because those are the two scales whose block-kind
#: table this template ships (the m/l/x rows need the override above, and are
#: covered in TOTAL by ``_PUBLISHED`` and by ``guard_arch_table_is_live``).
_PUBLISHED_LAYERS = {
    "n": {
        0: 464,
        1: 4_672,
        2: 6_640,
        3: 36_992,
        4: 26_080,
        5: 147_712,
        6: 87_040,
        7: 295_424,
        8: 346_112,
        9: 164_608,
        10: 249_728,
        13: 111_296,
        16: 32_096,
        17: 36_992,
        19: 86_720,
        20: 147_712,
        22: 378_880,
        23: 464_896,
    },
    "s": {
        0: 928,
        1: 18_560,
        2: 26_080,
        3: 147_712,
        4: 103_360,
        5: 590_336,
        6: 346_112,
        7: 1_180_672,
        8: 1_380_352,
        9: 656_896,
        10: 990_976,
        13: 443_776,
        16: 127_680,
        17: 147_712,
        19: 345_472,
        20: 590_336,
        22: 1_511_424,
        23: 850_352,
    },
}

#: What each yaml layer is, for readable failure messages.
_LAYER_LABELS = {
    0: "stem Conv[64,3,2]",
    1: "downsample Conv[128,3,2]",
    2: "C3k2[256,c3k=F,e=.25]",
    3: "downsample Conv[256,3,2]",
    4: "C3k2[512,c3k=F,e=.25] -> P3",
    5: "downsample Conv[512,3,2]",
    6: "C3k2[512,c3k=T] -> P4",
    7: "downsample Conv[1024,3,2]",
    8: "C3k2[1024,c3k=T]",
    9: "SPPF[1024,5]",
    10: "C2PSA[1024] -> P5",
    13: "neck C3k2[512,c3k=F] top-down P4",
    16: "neck C3k2[256,c3k=F] top-down P3",
    17: "neck Conv[256,3,2]",
    19: "neck C3k2[512,c3k=F] bottom-up P4",
    20: "neck Conv[512,3,2]",
    22: "neck C3k2[1024,c3k=T] bottom-up P5",
    23: "Detect head (DFL constants excluded)",
}


def _conv(in_ch, out_ch, kernel, groups=1, bias=False):
    return (in_ch // groups) * out_ch * kernel * kernel + (out_ch if bias else 0)


def _norm(channels):
    """An affine normalisation layer: one scale and one shift per channel.

    GroupNorm and BatchNorm are IDENTICAL here, which is what makes comparing a
    GroupNorm build against a published BatchNorm count legitimate — see the
    federated note in the template. What differs is the BUFFERS, and those are
    pinned separately in ``guard_no_stateful_normalisation``.
    """
    return 2 * channels


def _cna(in_ch, out_ch, kernel, groups=1):
    """conv -> norm (affine); the activation has no parameters."""
    return _conv(in_ch, out_ch, kernel, groups) + _norm(out_ch)


def _reference_yolo11_layers(
    class_channels,
    width=0.50,
    depth=0.50,
    max_channels=1024,
    reg_max=16,
    c3k_forced=False,
):
    """YOLO11 parameter count per yaml layer, derived from the published spec.

    Transcribed from ``yolo11.yaml`` plus the module definitions it names, with
    nothing imported from ``model_zoo/``. Returns ``{yaml layer index:
    parameters}`` for the parameterised layers only; layer 23 EXCLUDES the DFL
    projection constants, matching ``_PUBLISHED_LAYERS``.

    The four things that are easy to get wrong and are therefore spelled out:

    * a stage's output width is **not** its downsample's — ``Conv, [128, 3, 2]``
      is followed by ``C3k2, [256, ...]``;
    * ``C3k2``'s plain inner block runs at **half** the split branch
      (``Bottleneck``'s own ``e=0.5`` default), where ``C2f`` passes ``e=1.0``;
    * ``C3k`` is a ``C3``: THREE convs, both of the first two taking the full
      input, and its own bottlenecks at full branch width with ``kxk`` kernels;
    * the class tower is depthwise-separable, so its spatial convs are
      ``groups=channels``.
    """

    def scale_width(channels):
        scaled = min(channels, max_channels) * width
        return max(8, int(-(-scaled // 8)) * 8)

    def scale_depth(blocks):
        return max(int(round(blocks * depth)), 1)

    def bottleneck(in_ch, out_ch, expansion, kernel=3):
        hidden = max(1, int(out_ch * expansion))
        return _cna(in_ch, hidden, kernel) + _cna(hidden, out_ch, kernel)

    def c3k(channels, blocks=2, expansion=0.5, kernel=3):
        # A C3: cv1 and cv2 BOTH take the full input, cv3 fuses the two halves.
        hidden = int(channels * expansion)
        return (
            _cna(channels, hidden, 1)
            + _cna(channels, hidden, 1)
            + _cna(2 * hidden, channels, 1)
            # e=1.0 here -- FULL branch width, unlike C3k2's own plain path.
            + blocks * bottleneck(hidden, hidden, 1.0, kernel)
        )

    def c3k2(in_ch, out_ch, blocks, use_c3k, expansion):
        half = int(out_ch * expansion)
        total = (
            _cna(in_ch, 2 * half, 1)
            # (2 + n), not 2: the C2f skeleton fuses EVERY intermediate output.
            + _cna((2 + blocks) * half, out_ch, 1)
        )
        for _ in range(blocks):
            total += c3k(half) if use_c3k else bottleneck(half, half, 0.5)
        return total

    def sppf(in_ch, out_ch, repeats=3):
        half = in_ch // 2
        return _cna(in_ch, half, 1) + _cna(half * (repeats + 1), out_ch, 1)

    def attention(dim, head_dim=64, attn_ratio=0.5):
        heads = max(1, dim // head_dim)
        key_dim = int((dim // heads) * attn_ratio)
        return (
            _cna(dim, dim + 2 * key_dim * heads, 1)
            + _cna(dim, dim, 1)
            + _cna(dim, dim, 3, groups=dim)  # depthwise positional encoding
        )

    def psablock(channels):
        return (
            attention(channels)
            + _cna(channels, 2 * channels, 1)
            + _cna(2 * channels, channels, 1)
        )

    def c2psa(channels, blocks, ratio=0.5):
        half = int(channels * ratio)
        return (
            _cna(channels, 2 * half, 1)
            + _cna(2 * half, channels, 1)
            + blocks * psablock(half)
        )

    layers = {}

    # -- backbone, yaml 0-10 ------------------------------------------------
    stem = scale_width(64)
    layers[0] = _cna(3, stem, 3)

    # (downsample width, stage width, blocks, c3k, expansion) at FULL width.
    stage_spec = (
        (128, 256, 2, False, 0.25),
        (256, 512, 2, False, 0.25),
        (512, 512, 2, True, 0.50),
        (1024, 1024, 2, True, 0.50),
    )
    in_ch = stem
    widths = []
    index = 1
    for down_full, out_full, blocks_full, use_c3k, expansion in stage_spec:
        down = scale_width(down_full)
        out = scale_width(out_full)
        layers[index] = _cna(in_ch, down, 3)
        layers[index + 1] = c3k2(
            down,
            out,
            scale_depth(blocks_full),
            use_c3k or c3k_forced,
            expansion,
        )
        widths.append(out)
        in_ch = out
        index += 2

    deepest = scale_width(1024)
    layers[9] = sppf(widths[-1], deepest)
    layers[10] = c2psa(deepest, scale_depth(2))

    # -- neck, yaml 11-22 ---------------------------------------------------
    c3, c4, c5 = widths[1], widths[2], deepest
    # The yaml's OWN fusion widths, not the backbone's.
    td_p4 = scale_width(512)
    td_p3 = scale_width(256)
    bu_p4 = scale_width(512)
    bu_p5 = scale_width(1024)
    down3 = scale_width(256)
    down4 = scale_width(512)
    blocks = scale_depth(2)

    layers[13] = c3k2(c5 + c4, td_p4, blocks, c3k_forced, 0.50)
    layers[16] = c3k2(td_p4 + c3, td_p3, blocks, c3k_forced, 0.50)
    layers[17] = _cna(td_p3, down3, 3)
    layers[19] = c3k2(down3 + td_p4, bu_p4, blocks, c3k_forced, 0.50)
    layers[20] = _cna(bu_p4, down4, 3)
    # The only neck fusion the yaml marks c3k=True at n/s.
    layers[22] = c3k2(down4 + c5, bu_p5, blocks, True, 0.50)

    # -- head, yaml 23 ------------------------------------------------------
    head_channels = (td_p3, bu_p4, bu_p5)
    box_hidden = max(16, head_channels[0] // 4, reg_max * 4)
    cls_hidden = max(head_channels[0], min(class_channels, 100))
    head = 0
    for channels in head_channels:
        head += _cna(channels, box_hidden, 3) + _cna(box_hidden, box_hidden, 3)
        head += _conv(box_hidden, 4 * reg_max, 1, bias=True)
        # The depthwise-separable class tower: DW 3x3 then 1x1, twice.
        head += _cna(channels, channels, 3, groups=channels)
        head += _cna(channels, cls_hidden, 1)
        head += _cna(cls_hidden, cls_hidden, 3, groups=cls_hidden)
        head += _cna(cls_hidden, cls_hidden, 1)
        head += _conv(cls_hidden, class_channels, 1, bias=True)
    layers[23] = head
    return layers


def _reference_yolo11_parameters(class_channels, **kwargs):
    return sum(_reference_yolo11_layers(class_channels, **kwargs).values())


#: Published per-stage structure at the shipped scale, independent of the total:
#: they say WHAT drifted when the count disagrees.
_REFERENCE_STRUCTURE = {
    # The stride-8/16/32 maps LEAVING the backbone. Note stride 8 and 16 are the
    # same width at this scale -- yaml layers 4 and 6 both return [512].
    "backbone_out": (256, 256, 512),
    # ⚠️ NOT the backbone's widths, unlike yolov8_s. The stride-8 fusion returns
    # yaml [256] -> 128 channels, half the backbone's stride-8 map. Reusing the
    # backbone widths here is the single most likely cross-template copy error
    # and is worth about a million parameters.
    "neck_out": (128, 256, 512),
    "backbone_stage_blocks": (1, 1, 1, 1),
    "neck_stage_blocks": (1, 1, 1, 1),
    # c3k=True on the two deep backbone stages and the bottom-up P5 fusion only.
    "backbone_c3k": (False, False, True, True),
    "neck_c3k": (False, False, False, True),
    "c2psa_blocks": 1,
    "box_hidden": 64,
    "cls_hidden": 128,
    "strides": (8, 16, 32),
    "reg_max": 16,
}


def test_the_reference_derivation_matches_the_published_figures() -> None:
    """The transcription, checked against the numbers it is transcribed from.

    Runs before anything is built and needs no torch: if this fails, the
    reference is wrong and every comparison against it is worthless. FIVE
    scales are pinned, so the check covers the width multiplier, the depth
    multiplier and the m/l/x block-kind override rather than one arithmetic
    accident — and the DFL gap is asserted explicitly rather than hidden in a
    tolerance.
    """
    for scale, (width, depth, max_channels, forced, total, gradients) in sorted(
        _PUBLISHED.items()
    ):
        assert forced == (scale in _C3K_FORCED_SCALES), (
            f"yolo11{scale}: the c3k-forced flag in _PUBLISHED disagrees with "
            f"_C3K_FORCED_SCALES {_C3K_FORCED_SCALES}"
        )
        assert total - gradients == _DFL_PROJECTION_CONSTANTS, (
            f"published YOLO11-{scale.upper()}: total {total:,} minus gradients "
            f"{gradients:,} is {total - gradients}, not the "
            f"{_DFL_PROJECTION_CONSTANTS} DFL projection constants. One of the "
            f"two figures is mis-transcribed."
        )
        derived = _reference_yolo11_parameters(
            _PUBLISHED_CLASSES,
            width=width,
            depth=depth,
            max_channels=max_channels,
            c3k_forced=forced,
        )
        assert derived == gradients, (
            f"the spec transcription derives {derived:,} parameters at YOLO11-"
            f"{scale.upper()} (width {width} / depth {depth} / max_channels "
            f"{max_channels} / c3k_forced {forced}) at {_PUBLISHED_CLASSES} "
            f"classes, but the published summary reports {gradients:,} "
            f"gradients ({total:,} total, of which "
            f"{_DFL_PROJECTION_CONSTANTS} are the frozen DFL projection this "
            f"template does not store) — off by {derived - gradients:+,}. Fix "
            f"the transcription against yolo11.yaml before trusting any "
            f"comparison that uses it."
        )


def test_the_reference_derivation_matches_the_published_figures_per_layer() -> None:
    """The same transcription, checked ROW BY ROW rather than in total.

    This is the check the sibling ``yolov10_s`` needed and initially lacked:
    its total matched the published figure while individual layers did not, and
    only a per-layer comparison found it. Two errors that cancel survive a sum;
    they do not survive eighteen independent rows at two scales.

    ``_PUBLISHED_LAYERS`` is measured off upstream ``ultralytics==8.3.0`` — see
    its docstring for the exact three lines — so both sides of this comparison
    come from outside this repo.
    """
    for scale, expected in sorted(_PUBLISHED_LAYERS.items()):
        width, depth, max_channels, forced, _, gradients = _PUBLISHED[scale]
        derived = _reference_yolo11_layers(
            _PUBLISHED_CLASSES,
            width=width,
            depth=depth,
            max_channels=max_channels,
            c3k_forced=forced,
        )
        assert set(derived) == set(expected), (
            f"yolo11{scale}: the derivation covers yaml layers "
            f"{sorted(derived)} but the upstream table covers "
            f"{sorted(expected)}"
        )
        mismatched = {
            index: (derived[index], expected[index])
            for index in sorted(expected)
            if derived[index] != expected[index]
        }
        assert not mismatched, (
            "the spec transcription disagrees with the upstream per-layer table "
            f"at YOLO11-{scale.upper()} in {len(mismatched)} of "
            f"{len(expected)} layers:\n"
            + "\n".join(
                f"  yaml {index:>2} {_LAYER_LABELS[index]:38} derived "
                f"{got:>10,}  upstream {want:>10,}  ({got - want:+,})"
                for index, (got, want) in mismatched.items()
            )
        )
        assert sum(expected.values()) == gradients, (
            f"yolo11{scale}: the upstream per-layer table sums to "
            f"{sum(expected.values()):,}, not the published {gradients:,} "
            f"gradients — the table itself is mis-transcribed"
        )


# --------------------------------------------------------------------------
# mapping the built tree onto the yaml's layer numbering
# --------------------------------------------------------------------------
#
# The template groups the yaml's flat layer list into a backbone, a neck and a
# head, so a per-layer comparison needs the mapping written down once. Keyed by
# yaml index; each value is an attribute path from the model.

_LAYER_PATHS = {
    0: ("backbone", "stem"),
    1: ("backbone", "downsamples", 0),
    2: ("backbone", "stages", 0),
    3: ("backbone", "downsamples", 1),
    4: ("backbone", "stages", 1),
    5: ("backbone", "downsamples", 2),
    6: ("backbone", "stages", 2),
    7: ("backbone", "downsamples", 3),
    8: ("backbone", "stages", 3),
    9: ("backbone", "sppf"),
    10: ("backbone", "c2psa"),
    13: ("neck", "td_p4"),
    16: ("neck", "td_p3"),
    17: ("neck", "bu_conv3"),
    19: ("neck", "bu_p4"),
    20: ("neck", "bu_conv4"),
    22: ("neck", "bu_p5"),
    23: ("head",),
}


def _resolve(model, path):
    node = model
    for part in path:
        node = node[part] if isinstance(part, int) else getattr(node, part)
    return node


def _built_layers(model):
    """``{yaml layer index: parameters}`` for the built model."""
    return {
        index: sum(p.numel() for p in _resolve(model, path).parameters())
        for index, path in _LAYER_PATHS.items()
    }


@contextlib.contextmanager
def _at_published_scale(module, scale):
    """Rebuild-scope: point the template's live knobs at another published scale.

    Sets the two multipliers, the channel cap and — for m/l/x — the block-kind
    override ``parse_model`` applies, then restores every one of them. The
    knobs are module globals and class attributes read at CONSTRUCTION time,
    which is exactly what makes them knobs rather than decoration.
    """
    width, depth, max_channels, forced, _, _ = _PUBLISHED[scale]
    backbone = module.YOLO11Backbone
    neck = module.YOLO11PAFPN
    saved = (
        module.WIDTH_MULT,
        module.DEPTH_MULT,
        module.MAX_CHANNELS,
        backbone.STAGES,
        neck.NECK_C3K,
    )
    try:
        module.WIDTH_MULT = width
        module.DEPTH_MULT = depth
        module.MAX_CHANNELS = max_channels
        if forced:
            backbone.STAGES = tuple(
                (down, out, blocks, True, expansion)
                for down, out, blocks, _, expansion in saved[3]
            )
            neck.NECK_C3K = (True,) * len(saved[4])
        yield
    finally:
        (
            module.WIDTH_MULT,
            module.DEPTH_MULT,
            module.MAX_CHANNELS,
            backbone.STAGES,
            neck.NECK_C3K,
        ) = saved


# --------------------------------------------------------------------------
# structure guards
# --------------------------------------------------------------------------


def _backbone_stage_blocks(model):
    return tuple(len(stage.m) for stage in model.backbone.stages)


def _backbone_c3k(model):
    return tuple(bool(stage.c3k) for stage in model.backbone.stages)


_NECK_STAGE_NAMES = ("td_p4", "td_p3", "bu_p4", "bu_p5")


def _neck_stage_blocks(model):
    return tuple(len(getattr(model.neck, name).m) for name in _NECK_STAGE_NAMES)


def _neck_c3k(model):
    return tuple(bool(getattr(model.neck, name).c3k) for name in _NECK_STAGE_NAMES)


def guard_matches_the_published_architecture(module) -> None:
    """The built module tree must match the PUBLISHED architecture, re-derived —
    **per yaml layer**, not only in total.

    The independent half of the evidence. ``module_tree_size`` pins the totals
    this repo measured, so it can only catch a regression away from whatever was
    shipped; this one re-computes the count from the published spec and
    compares, so it catches shipping the wrong architecture in the first place —
    which is what happened on a sibling template.

    ⚠️ AND THE PER-LAYER COMPARISON IS THE POINT. On ``yolov10_s`` the total
    matched the published figure while individual layers did not; two errors
    that cancel are invisible to a sum. So the eighteen parameterised yaml
    layers are compared one by one and the total is asserted afterwards as a
    cheap cross-check, with the failure message naming the layers that moved.
    """
    class_channels = module.output_classes + 1  # the deliberate label-space +1
    expected = _reference_yolo11_layers(class_channels)
    model = _build(module, module.output_classes)
    actual = _built_layers(model)

    assert set(actual) == set(expected), (
        f"{module.__name__}: the built tree maps yaml layers {sorted(actual)} "
        f"but the reference derives {sorted(expected)}. _LAYER_PATHS has "
        f"drifted from the module structure — fix the mapping rather than the "
        f"comparison, or this guard silently narrows to fewer layers."
    )
    mismatched = {
        index: (actual[index], expected[index])
        for index in sorted(expected)
        if actual[index] != expected[index]
    }
    assert not mismatched, (
        f"{module.__name__}: the built model disagrees with YOLO11-S re-derived "
        f"from its published spec in {len(mismatched)} of {len(expected)} yaml "
        f"layers, at the same {class_channels} class channels:\n"
        + "\n".join(
            f"  yaml {index:>2} {_LAYER_LABELS[index]:38} built {got:>10,}  "
            f"published {want:>10,}  ({got - want:+,})"
            for index, (got, want) in mismatched.items()
        )
        + "\n\nSomething in the width, depth, kernel sizes, block counts, block "
        "KINDS or head shape does not match the design this template claims to "
        "implement. This is the check a parameter count measured off the model "
        "itself CANNOT make."
    )

    total = sum(p.numel() for p in model.parameters())
    assert total == sum(expected.values()), (
        f"{module.__name__}: every mapped layer matches but the model's total "
        f"is {total:,} against {sum(expected.values()):,} — there are "
        f"parameters OUTSIDE the eighteen mapped layers, i.e. a module the yaml "
        f"has no row for."
    )

    reference = _REFERENCE_STRUCTURE
    assert tuple(model.backbone.out_channels) == reference["backbone_out"], (
        f"{module.__name__}: backbone emits {tuple(model.backbone.out_channels)} "
        f"channels, published design has {reference['backbone_out']}"
    )
    assert tuple(model.neck.out_channels) == reference["neck_out"], (
        f"{module.__name__}: neck emits {tuple(model.neck.out_channels)} "
        f"channels, published design has {reference['neck_out']}. YOLO11's neck "
        f"fusion widths are the yaml's OWN, not the backbone's — its stride-8 "
        f"fusion returns half the backbone's stride-8 width."
    )
    assert _backbone_stage_blocks(model) == reference["backbone_stage_blocks"], (
        f"{module.__name__}: backbone C3k2 stages hold "
        f"{_backbone_stage_blocks(model)} blocks, published design has "
        f"{reference['backbone_stage_blocks']} at this depth multiplier"
    )
    assert _neck_stage_blocks(model) == reference["neck_stage_blocks"], (
        f"{module.__name__}: neck C3k2 stages hold {_neck_stage_blocks(model)} "
        f"blocks, published design has {reference['neck_stage_blocks']}"
    )
    assert len(model.backbone.c2psa.m) == reference["c2psa_blocks"], (
        f"{module.__name__}: C2PSA holds {len(model.backbone.c2psa.m)} "
        f"PSABlock(s), published design has {reference['c2psa_blocks']} at this "
        f"depth multiplier"
    )
    assert model.head.box_hidden == reference["box_hidden"]
    assert model.head.cls_hidden == reference["cls_hidden"]
    assert tuple(model.head.strides) == reference["strides"]
    assert model.head.reg_max == reference["reg_max"]


def guard_c3k_block_kinds_match_the_yaml(module) -> None:
    """Each stage must hold the block KIND its yaml row asks for.

    ``C3k2``'s ``c3k`` flag selects between a plain bottleneck and a nested
    ``C3k``, and the yaml sets it per layer: ``True`` on the two deep backbone
    stages and on the bottom-up stride-32 fusion, ``False`` on the other five.
    Flipping one moves a few hundred thousand parameters, so
    ``published_architecture`` also catches it — this guard exists for the
    message, because "layer 6 holds a Bottleneck where the yaml says C3k" is a
    diagnosis and "layer 6 is 259,072 parameters light" is a symptom.

    It also asserts the flag reaches the constructed block rather than being
    stored and ignored: the KIND of the module in ``m`` is checked, not
    ``stage.c3k``.
    """
    model = _build(module, 3)

    backbone_kinds = tuple(
        type(stage.m[0]).__name__ for stage in model.backbone.stages
    )
    expected_backbone = tuple(
        "C3k" if flag else "Bottleneck"
        for flag in _REFERENCE_STRUCTURE["backbone_c3k"]
    )
    assert backbone_kinds == expected_backbone, (
        f"{module.__name__}: the backbone stages hold {backbone_kinds}, the "
        f"yaml asks for {expected_backbone} (c3k="
        f"{_REFERENCE_STRUCTURE['backbone_c3k']}). A C3k is a THREE-conv C3 "
        f"whose own bottlenecks run at full branch width; a plain Bottleneck "
        f"squeezes to half. Both train."
    )
    assert _backbone_c3k(model) == _REFERENCE_STRUCTURE["backbone_c3k"], (
        f"{module.__name__}: the stages report c3k={_backbone_c3k(model)} while "
        f"holding {backbone_kinds} — the flag is stored but not reaching the "
        f"block it selects"
    )

    neck_kinds = tuple(
        type(getattr(model.neck, name).m[0]).__name__ for name in _NECK_STAGE_NAMES
    )
    expected_neck = tuple(
        "C3k" if flag else "Bottleneck" for flag in _REFERENCE_STRUCTURE["neck_c3k"]
    )
    assert neck_kinds == expected_neck, (
        f"{module.__name__}: the neck fusions hold {neck_kinds}, the yaml asks "
        f"for {expected_neck} (c3k={_REFERENCE_STRUCTURE['neck_c3k']} for "
        f"{_NECK_STAGE_NAMES}). Only the bottom-up stride-32 fusion is c3k at "
        f"n/s."
    )
    assert _neck_c3k(model) == _REFERENCE_STRUCTURE["neck_c3k"]


def guard_arch_table_is_live(module) -> None:
    """The arch table and BOTH multipliers must be live knobs, not decoration.

    ``published_architecture`` pins the shipped scale. That is not enough on its
    own: a width hardcoded at the shipped value, a depth multiplier copied from
    ``yolov8.yaml``, or a head width formula simplified to what this build
    happens to need all satisfy it exactly. So this guard rebuilds at the
    **four other published scales** and asserts each total, which is four
    independent numbers from outside this repo.

    Each of the four is load-bearing and none is redundant:

    * **N** (width 0.25) separates a hardcoded channel width, and is the only
      scale where the class tower's ``max(ch[0], min(nc, 100))`` is won by the
      CLASS term rather than by ``ch[0]`` — so a formula flattened to
      ``in_channels[0]`` is right at s and wrong here.
    * **M** (width 1.00, max_channels 512) separates the channel CAP, which
      does not bind at n or s where it is 1024.
    * **L** (depth 1.00) is the ONLY scale that separates ``DEPTH_MULT``:
      YOLO11's is 0.50 at both n and s, and ``max(round(2 * 0.33), 1)`` equals
      ``max(round(2 * 0.50), 1)`` equals 1, so YOLOv8's 0.33 is INVISIBLE at
      every scale this template otherwise touches.
    * **X** (width 1.50) separates the rounding direction — ``make_divisible``
      is ``ceil``, and a nearest-rounding version agrees at n/s/m/l.
    """
    for scale in sorted(set(_PUBLISHED) - {_SHIPPED_SCALE}):
        _, _, _, _, _, gradients = _PUBLISHED[scale]
        with _at_published_scale(module, scale):
            try:
                model = _build(module, _PUBLISHED_CLASSES - 1)
            except Exception as error:  # noqa: BLE001 — any failure is the bug
                raise AssertionError(
                    f"{module.__name__}: rebuilding at the published YOLO11-"
                    f"{scale.upper()} scale failed with "
                    f"{type(error).__name__}: {error}. The multipliers and the "
                    f"arch table are live knobs — it is how a scale is "
                    f"selected — so every published scale must construct."
                ) from error
            total = sum(p.numel() for p in model.parameters())
        assert total == gradients, (
            f"{module.__name__}: rebuilt at the published YOLO11-"
            f"{scale.upper()} scale the model has {total:,} parameters, but "
            f"that scale's published summary reports {gradients:,} gradients "
            f"(off by {total - gradients:+,}). Something the shipped scale "
            f"does not exercise is hardcoded: at s the width multiplier is "
            f"0.50, the depth multiplier's 0.50 and YOLOv8's 0.33 give the "
            f"SAME block count, the 1024 channel cap never binds, and the "
            f"class tower's width is won by in_channels[0] rather than by the "
            f"class term. Each of the four other scales separates one of those."
        )


#: Buffer and tensor totals measured off this repo's own build, as a cheap
#: regression tripwire.
#:
#: ⚠️ SELF-MEASURED. They prove the code is consistent with itself and nothing
#: more — see the block comment above ``_reference_yolo11_layers`` for the
#: sibling template where exactly such a number was cited as evidence and was
#: wrong. Parameters are asserted per yaml layer against the re-derived
#: published spec in ``guard_matches_the_published_architecture``; what lives
#: here is only what that derivation does not cover.
#:
#: Updating these is legitimate when the architecture changes on purpose; state
#: the intended change in the commit message.
_PINNED_TOTALS = {"buffers": 0, "tensors": 255}


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
        f"checked per yaml layer against the re-derived published spec in "
        f"guard_matches_the_published_architecture. If the change was "
        f"deliberate, update the row and say so in the commit."
    )


def guard_no_stateful_normalisation(module) -> None:
    """No BatchNorm anywhere, and zero buffer elements.

    ``running_mean``/``running_var`` are buffers the averaging service ships
    and averages every federated round, and they average badly across non-IID
    clients. Asserted two ways because they fail differently: a module-type scan
    names the offending layer, while the buffer total also catches a stateful
    norm this scan has never heard of.

    ⚠️ THE PARAMETER COUNT CANNOT SEE THIS. BatchNorm and GroupNorm both carry
    exactly one scale and one shift per channel, so swapping GroupNorm for
    BatchNorm here leaves ``published_architecture`` completely green — the
    difference is entirely in the BUFFERS. That is why this is a separate guard
    rather than a comment.

    NOT satisfied by ``FrozenBatchNorm2d`` either, and that is the point of
    preferring GroupNorm: Frozen BN moves ``weight``/``bias`` into buffers,
    which WOULD change the parameter count and silently invalidate the
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
        f"service ships and averages every round, and the parameter count is "
        f"IDENTICAL either way — nothing else here can see this. Use GroupNorm."
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

    ⚠️ AND UNLIKE ``yolov8_s``, THIS ONE BITES AT THE SHIPPED SCALE. There the
    honest caveat was that every channel count happens to be a multiple of 32 at
    ``WIDTH_MULT = 0.50``, so the derivation only mattered at a narrower width.
    Here the shallowest ``C3k2`` runs at ``expansion = 0.25`` and its inner
    bottleneck squeezes again by half, so the shipped tree already contains a
    **16-channel** norm and ``GroupNorm(32, 16)`` raises outright. So the guard
    does two things: it asserts a sub-32 group count is genuinely produced AT
    THE SHIPPED SCALE, and it still rebuilds at YOLO11-N's 0.25 to keep the
    width multiplier proven live.

    Construction is wrapped, because the failure this guard is written against
    is a ``ValueError`` at build time rather than a wrong number.
    """
    import torch
    from torch import nn

    assert module._norm_groups(16) == 16, "16 channels cannot take 32 groups"
    assert module._norm_groups(48) == 24, "48 takes 24 groups, not 32"
    assert module._norm_groups(3) == 3
    assert module._norm_groups(1) == 1
    assert module._norm_groups(64) == 32, (
        "64 channels should take the full 32 groups — if this changed, the "
        "shipped build's norms changed with it"
    )

    def build_or_explain(label, num_classes, edge=None):
        try:
            return _build(module, num_classes, edge)
        except Exception as error:  # noqa: BLE001 — any build failure is the bug
            raise AssertionError(
                f"{module.__name__}: building at {label} failed with "
                f"{type(error).__name__}: {error}. A hardcoded GroupNorm group "
                f"count crashes here, which is why the count is derived: "
                f"YOLO11's shallow C3k2 stages run at expansion 0.25 and their "
                f"inner bottleneck halves that again, so this tree carries "
                f"16-channel norms at the SHIPPED width — never mind at a "
                f"narrower one."
            ) from error

    shipped = build_or_explain("the shipped scale", 3)
    shipped_pairs = {
        (sub.num_groups, sub.num_channels)
        for sub in shipped.modules()
        if isinstance(sub, nn.GroupNorm)
    }
    assert shipped_pairs, "no GroupNorm at the shipped width — nothing checked"
    assert any(groups < 32 for groups, _ in shipped_pairs), (
        f"fixture is degenerate: every GroupNorm at the shipped width takes 32 "
        f"groups ({sorted(shipped_pairs)}), so this guard has stopped "
        f"exercising the derivation at the scale that ships"
    )
    for groups, channels in sorted(shipped_pairs):
        assert channels % groups == 0, (
            f"{module.__name__}: GroupNorm({groups}, {channels}) does not divide"
        )

    with _at_published_scale(module, "n"):
        model = build_or_explain("the published YOLO11-N width", 3)
        pairs = {
            (sub.num_groups, sub.num_channels)
            for sub in model.modules()
            if isinstance(sub, nn.GroupNorm)
        }
        assert pairs, "no GroupNorm at the narrower width — nothing was checked"
        for groups, channels in sorted(pairs):
            assert channels % groups == 0, (
                f"{module.__name__}: GroupNorm({groups}, {channels}) at the "
                f"YOLO11-N width does not divide"
            )
        model.eval()
        with torch.no_grad():
            model([torch.rand(3, 96, 96)])


# --------------------------------------------------------------------------
# block topology — every guard here is FUNCTIONAL, because every mutation it
# catches leaves the parameter count, the tensor shapes and the state_dict key
# set completely unchanged.
# --------------------------------------------------------------------------


def _capture_input(module_under_test):
    """``(handle, captured)`` — a pre-hook recording the tensor a module is
    handed, so a guard can compare it against a hand-computed expectation."""
    captured = []

    def record(_module, inputs):
        captured.append(inputs[0].detach().clone())

    return module_under_test.register_forward_pre_hook(record), captured


def guard_deepest_stage_modules_are_applied(module) -> None:
    """``SPPF`` and ``C2PSA`` must actually be CALLED, not merely constructed.

    ⚠️ THIS IS THE GUARD THE PARAMETER COUNT CANNOT REPLACE, and that is
    measured rather than argued. Dropping ``self.c2psa(...)`` from the
    backbone's ``forward`` leaves the module in the tree, so:

    * the parameter total is unchanged (both are still registered submodules);
    * the buffer total is unchanged (zero either way);
    * the ``state_dict`` key set is unchanged;
    * the output shapes are unchanged (C2PSA is width-preserving by
      construction, and the SPPF maps 1024 -> 1024 at this scale);
    * the loss keys are unchanged and the model trains.

    ``guard_matches_the_published_architecture`` **DID NOT RAISE** against that
    mutation. What it costs is ~991,000 parameters shipped to every edge and
    averaged every federated round while contributing nothing, plus the
    receptive field and the attention the architecture exists for.

    So this hooks both modules and asserts each fired. It also asserts the
    ORDER — SPPF then C2PSA, yaml layers 9 then 10 — since swapping them is
    also shape-clean at this scale.
    """
    import torch

    model = _build(module, 3, 64)
    fired = []
    handles = [
        model.backbone.sppf.register_forward_hook(
            lambda *_: fired.append("SPPF")
        ),
        model.backbone.c2psa.register_forward_hook(
            lambda *_: fired.append("C2PSA")
        ),
    ]
    try:
        model.eval()
        with torch.no_grad():
            model.backbone(torch.rand(1, 3, 64, 64))
    finally:
        for handle in handles:
            handle.remove()

    for name, why in (
        (
            "SPPF",
            "the three-in-series 5x5 pooling that widens the deepest stage's "
            "receptive field without another stride",
        ),
        (
            "C2PSA",
            "YOLO11's partial self-attention, the block that most distinguishes "
            "it from YOLOv8 and the largest single layer in the backbone at "
            "~991,000 parameters",
        ),
    ):
        assert name in fired, (
            f"{module.__name__}: {name} was never called during a backbone "
            f"forward. It is still CONSTRUCTED, so the parameter count, the "
            f"buffer count, every state_dict key and every output shape are "
            f"UNCHANGED — measured: the published-architecture guard does not "
            f"raise on this. {name} is {why}. Its parameters are still shipped "
            f"to every edge and averaged every federated round, for nothing."
        )
    assert fired == ["SPPF", "C2PSA"], (
        f"{module.__name__}: the deepest stage ran {fired}; yaml layers 9 and "
        f"10 are SPPF then C2PSA, in that order. Both are width-preserving at "
        f"this scale, so swapping them is shape-clean and silent."
    )


def guard_sppf_pools_in_series(module) -> None:
    """The SPPF's three pools must be applied IN SERIES, not in parallel.

    One 5x5 max-pool three times over has the receptive field of 5/9/13
    pooling; three parallel 5x5 pools produce a tensor of exactly the same
    shape from exactly the same parameters and a third of the receptive field.
    The difference is visible only in what the fusion conv is handed, so that
    is what is compared.
    """
    import torch

    block = module.SPPF(8, 8)
    block.eval()
    handle, captured = _capture_input(block.cv2)
    probe = torch.rand(1, 8, 9, 9)
    try:
        with torch.no_grad():
            block(probe)
            branch = block.cv1(probe)
            once = block.pool(branch)
            twice = block.pool(once)
            thrice = block.pool(twice)
            expected = torch.cat([branch, once, twice, thrice], dim=1)
            parallel = torch.cat([branch, once, once, once], dim=1)
    finally:
        handle.remove()

    assert len(captured) == 1, "the fusion conv's pre-hook did not fire once"
    assert not torch.allclose(expected, parallel, atol=1e-5), (
        "fixture is degenerate: series and parallel pooling agree on this "
        "probe, so the rule cannot fire — the input is probably too smooth"
    )
    assert torch.allclose(captured[0], expected, atol=1e-5), (
        f"{module.__name__}: SPPF's pools are not applied in series. The three "
        f"branches are the same 5x5 pooling of cv1's output rather than the "
        f"5 / 9 / 13 effective windows, so the deepest stage's receptive field "
        f"is a third of the published one. Identical shape, identical "
        f"parameters, trains fine."
    )


def guard_c3k2_fuses_every_intermediate_block(module) -> None:
    """``C3k2`` must concatenate the two split halves **and every intermediate
    block output**, in that order — the ``C2f`` skeleton it inherits.

    Two things make this need a functional check rather than a channel count:

    * at the shipped depth every ``C3k2`` holds exactly ONE block, and at
      ``n == 1`` a C3-shaped fusion that keeps only the last output has a
      DIFFERENT channel count, so the constructor catches that particular
      slip — but a fusion fed ``[split_0, split_1, m0(split_0)]`` instead of
      ``m0(split_1)`` has the same count, the same shapes and a different
      graph;
    * the branch ORDER is likewise invisible: all branches are ``hidden``
      channels wide, so a reversed list is accepted by the fusion conv.

    So the expected branch list is reconstructed from ``cv1`` and ``m`` and
    compared tensor-for-tensor against what ``cv2`` is actually handed. Run at
    ``blocks=2`` as well, where the (2 + n) shape itself becomes checkable.
    """
    import torch

    for blocks in (1, 2):
        stage = module.C3k2(16, 16, blocks=blocks)
        stage.eval()
        assert len(stage.m) == blocks, (
            f"C3k2 built {len(stage.m)} block(s) for blocks={blocks}"
        )
        assert stage.cv2.conv.in_channels == (2 + blocks) * stage.hidden, (
            f"C3k2's fusion conv takes {stage.cv2.conv.in_channels} channels "
            f"for {blocks} block(s) at hidden width {stage.hidden}; the C2f "
            f"skeleton's shape is (2 + n) * hidden = "
            f"{(2 + blocks) * stage.hidden}. A (2 * hidden) input is a C3."
        )

        probe = torch.rand(1, 16, 6, 6) + 0.5
        handle, captured = _capture_input(stage.cv2)
        try:
            with torch.no_grad():
                stage(probe)
                first, second = stage.cv1(probe).chunk(2, dim=1)
                deep = [second]
                for block in stage.m:
                    deep.append(block(deep[-1]))
                expected = torch.cat([first, second] + deep[1:], dim=1)
                # The two plausible wrong graphs, both shape-clean.
                wrong_source = torch.cat(
                    [first, second] + [block(first) for block in stage.m], dim=1
                )
        finally:
            handle.remove()

        assert len(captured) == 1, "the fusion conv's pre-hook did not fire once"
        assert not torch.allclose(expected, wrong_source, atol=1e-5), (
            f"fixture is degenerate at blocks={blocks}: feeding the blocks the "
            f"OTHER split half gives the same tensor, so the rule cannot fire"
        )
        assert captured[0].shape == expected.shape, (
            f"C3k2 hands its fusion conv {tuple(captured[0].shape)} at "
            f"blocks={blocks}; the C2f branch list is {tuple(expected.shape)}"
        )
        assert torch.allclose(captured[0], expected, atol=1e-5), (
            f"{module.__name__}: C3k2's fusion conv is not receiving "
            f"[split_0, split_1, m0(split_1), ...] at blocks={blocks}. The "
            f"channel count is right and the model trains, so the likely shapes "
            f"are a chain rooted on the wrong split half, or a reversed branch "
            f"list — both leave every tensor the same size."
        )


def guard_c3k_routes_its_blocks_through_the_first_branch(module) -> None:
    """``C3k`` is a ``C3``: ``cv1``'s branch goes through the bottlenecks and
    ``cv2``'s is an untouched skip.

    Both convs take the FULL input and emit the same width, so swapping which
    branch is mixed leaves every shape and every parameter identical. The block
    still mixes — just not the half the design mixes, and the skip that is
    supposed to carry the unmodified signal no longer does.

    ``C3k`` is only reached on the two deep backbone stages and the bottom-up
    stride-32 fusion, which is exactly why it needs its own guard: it is a
    minority path and a wrong one is a small fraction of the total loss.
    """
    import torch

    block = module.C3k(12, 12)
    block.eval()
    handle, captured = _capture_input(block.cv3)
    probe = torch.rand(1, 12, 5, 5) + 0.5
    try:
        with torch.no_grad():
            block(probe)
            mixed = block.m(block.cv1(probe))
            skip = block.cv2(probe)
            expected = torch.cat((mixed, skip), dim=1)
            swapped = torch.cat((block.cv1(probe), block.m(skip)), dim=1)
    finally:
        handle.remove()

    assert len(captured) == 1, "C3k's fusion conv pre-hook did not fire once"
    assert not torch.allclose(expected, swapped, atol=1e-5), (
        "fixture is degenerate: routing the skip through the bottlenecks gives "
        "the same tensor, so the rule cannot fire"
    )
    assert torch.allclose(captured[0], expected, atol=1e-5), (
        f"{module.__name__}: C3k's fusion conv is not receiving "
        f"[m(cv1(x)), cv2(x)]. cv1 and cv2 both take the full input at the same "
        f"output width, so mixing the wrong branch — or mixing both — is "
        f"shape-identical, parameter-identical and trains."
    )


def guard_bottlenecks_keep_their_identity_branch(module) -> None:
    """Every equal-width ``Bottleneck`` in this tree must be RESIDUAL, in the
    neck as well as the backbone.

    ⚠️ THIS IS A REAL DIFFERENCE FROM ``yolov8.yaml`` AND IT IS EASY TO INVERT.
    YOLOv8's neck ``C2f``s are constructed ``shortcut=False``; every ``C3k2`` in
    ``yolo11.yaml`` is ``shortcut=True``, including all four neck fusions. The
    yaml's ``False`` literals — ``C3k2, [512, False]`` — are the **c3k** flag,
    not the shortcut, and reading them as shortcuts is the natural mistake
    because that is what the same position meant in the previous generation.

    It changes **no parameter and no shape**, only whether the block is
    residual, so nothing structural can see it. Checked functionally on a neck
    block and on a ``C3k``'s inner block, since they are constructed by
    different code paths.
    """
    import torch

    model = _build(module, 3)
    probes = {
        "neck top-down P3 (yaml 16)": model.neck.td_p3.m[0],
        "neck bottom-up P5 -> C3k inner (yaml 22)": model.neck.bu_p5.m[0].m[0],
        "backbone deep stage -> C3k inner (yaml 8)": model.backbone.stages[3].m[0],
    }
    for label, block in probes.items():
        if type(block).__name__ == "C3k":
            block = block.m[0]
        channels = block.conv1.conv.in_channels
        assert block.conv2.conv.out_channels == channels, (
            f"fixture: {label} is not equal-width ({channels} -> "
            f"{block.conv2.conv.out_channels}), so no shortcut is expected and "
            f"this probe proves nothing"
        )
        assert bool(block.use_add), (
            f"{module.__name__}: {label} is NOT residual. Every C3k2 in "
            f"yolo11.yaml is shortcut=True — the yaml's False literals are the "
            f"c3k flag, not the shortcut, unlike yolov8.yaml whose neck C2f "
            f"blocks genuinely are shortcut=False. This changes no parameter "
            f"and no shape."
        )
        block.eval()
        probe = torch.rand(1, channels, 5, 5) + 0.5
        with torch.no_grad():
            got = block(probe)
            inner = block.conv2(block.conv1(probe))
        assert not torch.allclose(inner, got, atol=1e-5), (
            f"fixture is degenerate: {label}'s residual adds nothing "
            f"measurable on this probe"
        )
        assert torch.allclose(got, probe + inner, atol=1e-5), (
            f"{module.__name__}: {label} does not compute x + conv2(conv1(x)). "
            f"It reports use_add=True, so the flag is set and the addition is "
            f"not happening — or is happening in the wrong place."
        )


def guard_c2psa_attends_to_only_one_half(module) -> None:
    """``C2PSA`` splits its stage in two and attends to **one half**; the other
    reaches the fusion UNCHANGED.

    That partial application is what the name means and what makes attention
    affordable at the stride-32 width — it halves both the token-mixing cost and
    the attention modules' parameters. Attending both halves, or dropping the
    bypass and duplicating the attended half, leaves every parameter, every
    shape and every key identical.
    """
    import torch

    block = module.C2PSA(16, 16)
    block.eval()
    handle, captured = _capture_input(block.cv2)
    probe = torch.rand(1, 16, 5, 5) + 0.5
    try:
        with torch.no_grad():
            block(probe)
            bypass, attend = block.cv1(probe).split(
                (block.hidden, block.hidden), dim=1
            )
            expected = torch.cat((bypass, block.m(attend)), dim=1)
            both = torch.cat((block.m(bypass), block.m(attend)), dim=1)
    finally:
        handle.remove()

    assert len(captured) == 1, "C2PSA's fusion conv pre-hook did not fire once"
    assert not torch.allclose(expected, both, atol=1e-5), (
        "fixture is degenerate: attending the bypass half changes nothing "
        "measurable, so the rule cannot fire"
    )
    assert torch.allclose(captured[0], expected, atol=1e-5), (
        f"{module.__name__}: C2PSA's fusion conv is not receiving "
        f"[bypass_half, attended_half]. The bypass half must reach the fusion "
        f"UNCHANGED — that partial application is the 'P' in PSA. Attending "
        f"both halves doubles the token mixing the design deliberately halves, "
        f"at identical parameter count and identical shapes."
    )


def guard_psablock_applies_both_residuals(module) -> None:
    """``PSABlock`` applies attention and the feed-forward each as a residual.

    Both are load-bearing and both are invisible to every structural check:
    dropping either leaves the parameter count, every tensor shape, every
    ``state_dict`` key and every loss key identical, and the model trains. The
    residuals are what make the block an identity at initialisation scale, so
    the stage does not need re-tuning to accept the attention.
    """
    import torch

    block = module.PSABlock(16)
    block.eval()
    probe = torch.rand(1, 16, 5, 5) + 0.5
    with torch.no_grad():
        got = block(probe)
        attended = probe + block.attn(probe)
        expected = attended + block.ffn(attended)
        no_attn_residual = block.attn(probe)
        no_attn_residual = no_attn_residual + block.ffn(no_attn_residual)
        no_ffn_residual = block.ffn(attended)

    assert not torch.allclose(expected, no_attn_residual, atol=1e-5), (
        "fixture is degenerate: dropping the attention residual changes nothing"
    )
    assert not torch.allclose(expected, no_ffn_residual, atol=1e-5), (
        "fixture is degenerate: dropping the ffn residual changes nothing"
    )
    assert torch.allclose(got, expected, atol=1e-5), (
        f"{module.__name__}: PSABlock does not compute "
        f"y = x + attn(x) then y + ffn(y). Max deviation from that "
        f"{float((got - expected).abs().max()):.4f}, against "
        f"{float((got - no_attn_residual).abs().max()):.4f} from the "
        f"attention-residual-dropped form and "
        f"{float((got - no_ffn_residual).abs().max()):.4f} from the "
        f"ffn-residual-dropped one. Every parameter, shape and key is identical "
        f"in all three."
    )


def guard_attention_head_count_is_derived(module) -> None:
    """The attention's head count must be DERIVED from ``ATTENTION_HEAD_DIM``.

    ⚠️ THE HEAD COUNT IS PARAMETER-INVARIANT, which makes it the textbook
    "constant that reaches nothing". ``qkv``'s output width is
    ``dim + 2 * num_heads * int((dim / num_heads) * ATTN_RATIO)``, and for any
    head count that divides ``dim`` that is exactly ``dim * (1 + ATTN_RATIO)``.
    So a hardcoded ``num_heads = 8`` changes no parameter, no shape, no
    ``state_dict`` key, no loss key and no published figure — it only
    re-factorises the attention into eight narrower heads, which is a different
    operator with identical bookkeeping.

    Asserted BOTH ways: the derivation is pinned on the built module, and the
    invariance is asserted explicitly so nobody later "simplifies" this guard
    into a parameter comparison that cannot work.
    """
    dim = 256
    reference = module.Attention(dim)
    expected_heads = max(1, dim // module.ATTENTION_HEAD_DIM)
    assert reference.num_heads == expected_heads, (
        f"{module.__name__}: Attention({dim}) runs {reference.num_heads} heads; "
        f"derived from ATTENTION_HEAD_DIM={module.ATTENTION_HEAD_DIM} it is "
        f"{expected_heads}. A hardcoded count changes NO parameter and NO "
        f"shape, so nothing else in this file can see it."
    )
    assert reference.head_dim == module.ATTENTION_HEAD_DIM, (
        f"{module.__name__}: Attention({dim}) has head_dim "
        f"{reference.head_dim}, expected {module.ATTENTION_HEAD_DIM}"
    )

    hardcoded = module.Attention(dim, head_dim=dim // 8)
    assert hardcoded.num_heads == 8, "fixture: the probe did not build 8 heads"
    assert sum(p.numel() for p in hardcoded.parameters()) == sum(
        p.numel() for p in reference.parameters()
    ), (
        "fixture is degenerate — and if this ever fires, DELETE this guard "
        "rather than weakening it: the head count has become visible to the "
        "parameter count, which means the published-architecture comparison "
        "covers it and this guard is redundant. Today it is not."
    )

    built = _build(module, 3).backbone.c2psa
    attention = built.m[0].attn
    assert attention.num_heads == max(1, built.hidden // module.ATTENTION_HEAD_DIM), (
        f"{module.__name__}: the SHIPPED C2PSA's attention runs "
        f"{attention.num_heads} heads at {built.hidden} channels; derived it is "
        f"{max(1, built.hidden // module.ATTENTION_HEAD_DIM)}"
    )


def guard_attention_matches_the_reference_operator(module) -> None:
    """The attention must compute the published operator, not merely a tensor of
    the right shape.

    Two silent failures live here and neither is visible to any count:

    * **the SDPA layout.** Upstream holds q/k/v channels-first,
      ``(batch, heads, width, tokens)``, while SDPA's contract is
      tokens-second-to-last. The transposes on the way in and out are part of
      the call: without the outgoing one the result has the SAME NUMBER OF
      ELEMENTS, so the ``reshape`` back to ``(B, C, H, W)`` succeeds and
      silently scrambles which token each channel came from. Losses stay
      finite; the operator is not attention over space any more.
    * **the ``pe`` residual.** ``pe`` is a depthwise positional encoding added
      to the attended values. Dropping the addition leaves ``pe``
      **constructed** — so the parameter count, buffer count and key set are
      unchanged, exactly like the bypassed-module trap in the file docstring —
      and removes the only thing giving this attention positional information.

    So the whole operator is recomputed here from the module's own weights, in
    the explicit matmul-softmax-matmul form, and compared.
    """
    import torch

    dim = 64
    attention = module.Attention(dim)
    attention.eval()
    probe = torch.rand(1, dim, 4, 3) + 0.5

    with torch.no_grad():
        got = attention(probe)

        batch, channels, height, width = probe.shape
        tokens = height * width
        qkv = attention.qkv(probe).view(
            batch, attention.num_heads, 2 * attention.key_dim + attention.head_dim,
            tokens,
        )
        query, key, value = qkv.split(
            [attention.key_dim, attention.key_dim, attention.head_dim], dim=2
        )
        # The published form, channels-first: (q * scale)^T @ k, softmax over
        # the last axis, then v @ attn^T.
        weights = ((query * attention.scale).transpose(-2, -1) @ key).softmax(dim=-1)
        attended = (value @ weights.transpose(-2, -1)).view(
            batch, channels, height, width
        )
        positional = attention.pe(value.reshape(batch, channels, height, width))
        expected = attention.proj(attended + positional)
        without_pe = attention.proj(attended)

    assert not torch.allclose(expected, without_pe, atol=1e-5), (
        "fixture is degenerate: the positional encoding contributes nothing "
        "measurable, so dropping its residual could not be seen"
    )
    assert got.shape == expected.shape, (
        f"{module.__name__}: attention returned {tuple(got.shape)}, the "
        f"reference operator gives {tuple(expected.shape)}"
    )
    assert torch.allclose(got, expected, atol=1e-4), (
        f"{module.__name__}: the attention does not compute the published "
        f"operator. Max deviation {float((got - expected).abs().max()):.4f} "
        f"from proj(attn(q,k,v) + pe(v)), against "
        f"{float((got - without_pe).abs().max()):.4f} from the same thing with "
        f"the pe residual dropped. Both candidate failures are invisible "
        f"elsewhere: a missing SDPA transpose keeps the element count (so the "
        f"reshape succeeds and scrambles the layout), and a dropped pe "
        f"residual leaves pe constructed so no parameter count moves."
    )


# --------------------------------------------------------------------------
# head shape
# --------------------------------------------------------------------------


def guard_head_is_decoupled(module) -> None:
    """The classification and box towers must share no parameters.

    Checked by parameter identity rather than by reading the constructor, and
    rather than by relying on a crash. On this head the two towers are
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


def guard_class_tower_is_depthwise_separable(module) -> None:
    """YOLO11's class tower is two depthwise-separable pairs, not two dense 3x3
    convolutions.

    Each pair is a depthwise ``kxk`` spatial mixer (``groups == channels``)
    followed by a pointwise ``1x1``. YOLOv8 spends two DENSE 3x3 convolutions
    here, which type-checks, trains identically and is about 1.5M parameters
    heavier at this scale — so ``published_architecture`` catches it too. This
    guard exists for the DIAGNOSIS: "the first spatial conv is dense" names the
    edit, where "the head is 1,548,288 parameters heavy" does not.

    ⚠️ Stated plainly because a redundant guard that reads as coverage is worse
    than none: this is not independent evidence. It is a better error message
    for a failure the per-layer count already sees.
    """
    head = _build(module, 3).head
    for level, tower in enumerate(head.cls_convs):
        spatial = [
            sub
            for sub in tower.modules()
            if isinstance(sub, module.ConvNormAct)
            and sub.conv.kernel_size == (3, 3)
        ]
        pointwise = [
            sub
            for sub in tower.modules()
            if isinstance(sub, module.ConvNormAct)
            and sub.conv.kernel_size == (1, 1)
        ]
        assert len(spatial) == 2 and len(pointwise) == 2, (
            f"{module.__name__}: cls_convs[{level}] holds {len(spatial)} 3x3 "
            f"and {len(pointwise)} 1x1 conv(s); YOLO11's tower is two "
            f"depthwise-separable PAIRS — 3x3 depthwise then 1x1 pointwise, "
            f"twice."
        )
        depthwise = [
            sub
            for sub in spatial
            if sub.conv.groups == sub.conv.in_channels == sub.conv.out_channels
        ]
        assert len(depthwise) == 2, (
            f"{module.__name__}: cls_convs[{level}] has {len(depthwise)} "
            f"depthwise 3x3 convolution(s), expected 2. A DENSE 3x3 in either "
            f"spatial position is YOLOv8's tower — it trains identically and is "
            f"~1.5M parameters heavier at this scale."
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
        f"{module.__name__}: REG_MAX is {module.REG_MAX}; YOLO11 publishes 16, "
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
    term is capped at 100. That cap is doing real work — at YOLO11-N's 64 the
    class term WINS at the published 80 classes — so the invariance is checked
    at the shipped scale rather than assumed from the formula.

    ⚠️ THREE prefixes, not six. YOLO11 has ONE detection branch; the
    ``yolov10_s.py`` next door needs six because its NMS-free design duplicates
    the whole head. A list copied from there names three keys this tree does not
    have, and the ``dead`` assertion below is what catches that direction.
    """
    low, high = 7, 200
    a = _build(module, low)
    b = _build(module, high)

    assert a.head.cls_hidden == b.head.cls_hidden, (
        f"{module.__name__}: the class tower is {a.head.cls_hidden} channels "
        f"wide at {low} classes and {b.head.cls_hidden} at {high}. The seed "
        f"contract assumes only the 1x1 predictors depend on the class count; a "
        f"class-count-dependent tower belongs in SEED_EXCLUDED_PREFIXES too — "
        f"and there are TWO norm layers and TWO convs per level behind it, so "
        f"the declaration would need twelve more prefixes, not three."
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
        f"missing from a seed. Re-derive it. If these look like one2one/one2many "
        f"names, the list was copied from yolov10_s.py, which has a second head "
        f"this template does not."
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
    # The class tower's first module is a depthwise-separable PAIR, so its
    # input width is read off that pair's first conv rather than off the tower.
    channels_per_level = [tower[0][0].conv.in_channels for tower in head.cls_convs]

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
        f"3.0 * sum(range({reg_max})) = {3.0 * sum(range(reg_max))} — finite, "
        f"trainable, and nonsense."
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

    A freshly built YOLO11 head predicts around ``sigmoid(-9)`` on every class
    at the fine levels, so a forward pass at initialisation returns a handful of
    coarse-level anchors firing on noise and any check downstream of it is
    nearly vacuous — the decode has almost nothing to get wrong. That is how a
    real defect shipped through every guard on a sibling template: its
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
    # confident scores on REAL classes, so it cannot see a channel-0 leak.
    bg_cls, bg_dist, bg_anchors = _synthetic_head_output(
        torch, model, cells, 8, {(0, 2): 0}
    )
    bg_cls[0, 2, classes - 1] = 8.0  # a real class too, but weaker
    bg_results = model._predictions(bg_cls, bg_dist, bg_anchors, [(64, 64)])
    bg_labels = bg_results[0]["labels"]
    assert bg_labels.numel(), (
        f"{module.__name__}: the background-channel fixture decoded to nothing, "
        f"so the assertion below is vacuous — check the scores clear "
        f"score_thresh"
    )
    assert not bool((bg_labels == 0).any()), (
        f"{module.__name__}: decode returned label 0, the background channel: "
        f"{sorted(set(bg_labels.tolist()))}. It is trained only as a negative "
        f"and must be dropped BEFORE the score threshold and the NMS budget, "
        f"not left to the engine — the detection budget is spent here."
    )


def guard_decode_suppresses_duplicates(module) -> None:
    """YOLO11 IS NMS-BASED, and the decode must suppress duplicates.

    ⚠️ THIS IS THE GUARD THAT IS THE MIRROR IMAGE OF ITS NEIGHBOUR. The sibling
    ``yolov10_s.py`` is NMS-FREE and has a guard —
    ``guard_decode_is_nms_free`` — asserting that five anchors decoding to the
    same box at the same class come back as FIVE detections. YOLO11 assigns
    one-to-many, so several anchors are deliberately trained to fire on one
    object and duplicates are the design's expected raw output: here the same
    fixture must come back as ONE.

    So the two templates' decode guards are contradictory by design, and a
    reviewer reading them side by side should see that they are supposed to be.
    Dropping the ``batched_nms`` call leaves the loss, the shapes, the keys and
    the parameter count untouched and reports every object several times, which
    tanks mAP through false positives while the training loss falls normally.

    Also asserts NMS is CLASS-AWARE: the same box under two different labels
    must survive as two detections, because ``batched_nms`` suppresses within a
    class and a class-agnostic call would delete the second object.
    """
    import torch

    model = _build(module, 3)
    duplicates = 5
    # Every anchor identical, so every decoded box is identical too.
    anchors = torch.tensor([[0.5, 0.5, 8.0]] * duplicates)
    cls_logits = torch.full((1, duplicates, model.num_classes), -10.0)
    dist_logits = torch.full((1, duplicates, 4, model.head.reg_max), -50.0)
    dist_logits[..., 1] = 50.0
    cls_logits[0, :, 2] = 10.0

    decoded = module._decode_boxes(dist_logits, anchors)
    spread = float((decoded[0] - decoded[0, 0:1]).abs().max())
    assert spread < 1e-4, (
        f"fixture is degenerate: the {duplicates} anchors decode to boxes that "
        f"differ by {spread:.4f}px, so NMS has nothing to suppress"
    )

    results = model._predictions(cls_logits, dist_logits, anchors, [(64, 64)])
    kept = int(results[0]["boxes"].shape[0])
    assert kept == 1, (
        f"{module.__name__}: {duplicates} anchors decoding to the SAME box at "
        f"the same class and score returned {kept} detection(s), expected 1. "
        f"YOLO11 is NMS-BASED — its head is assigned one-to-MANY, so duplicate "
        f"boxes are the expected raw output and suppressing them is this "
        f"function's job. (The `yolov10_s` sibling asserts the OPPOSITE on this "
        f"fixture, deliberately: it is NMS-free.) Dropping the suppression "
        f"leaves every loss, shape, key and parameter identical and reports "
        f"every object {duplicates} times."
    )

    # Class-aware, not class-agnostic: same box, two labels, both must survive.
    two_classes = torch.full((1, 2, model.num_classes), -10.0)
    two_classes[0, 0, 1] = 10.0
    two_classes[0, 1, 2] = 10.0
    pair_dist = torch.full((1, 2, 4, model.head.reg_max), -50.0)
    pair_dist[..., 1] = 50.0
    pair_anchors = torch.tensor([[0.5, 0.5, 8.0]] * 2)
    pair = model._predictions(two_classes, pair_dist, pair_anchors, [(64, 64)])
    assert int(pair[0]["boxes"].shape[0]) == 2, (
        f"{module.__name__}: one box under two DIFFERENT labels came back as "
        f"{int(pair[0]['boxes'].shape[0])} detection(s), expected 2. NMS must "
        f"be class-aware (batched_nms with the label as the batch index); a "
        f"class-agnostic call deletes genuinely co-located objects of different "
        f"classes, which is common and which mAP punishes as a miss."
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
    effective resolution off the transform's configured ``min_size`` and
    compares it against the ``MODERN_YOLO`` family anchor. What this adds, for
    this template only, is a stronger measurement: a forward hook reports the
    spatial size of the tensor the backbone is **actually handed**, after the
    pad to ``size_divisible=32``, and asserts it is **square** — which the
    family guard does not, since it compares a single edge.
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


def guard_head_emits_every_level(module) -> None:
    """The head must emit anchors for ALL THREE levels, at the right strides.

    A head that loops over two of the three feature maps — a stray slice, or a
    ``zip`` against a mismatched list — returns a shorter anchor table, a
    shorter prediction tensor, a finite loss, well-formed predictions and a
    clean train step. Nothing else in this file notices: the parameter count is
    unchanged (the third level's towers are still constructed, the
    bypassed-module trap again), and the assigner is happy to work over fewer
    anchors.

    What it costs is the entire coarse level, i.e. every large object.

    This is asserted by CELL COUNT PER STRIDE rather than by gradient flow, on
    purpose. ``positives_reach_the_box_regression_branch`` explains why a
    per-level gradient assertion is NOT sound here: at random initialisation
    which level wins a given ground truth is not deterministic, so a guard
    demanding a positive at every level would be flaky. The anchor table's
    shape is deterministic.
    """
    import torch

    edge = 64
    model = _build(module, 3, edge)
    model.eval()
    with torch.no_grad():
        cls_logits, dist_logits, anchors = model.head(
            model.neck(model.backbone(torch.rand(1, 3, edge, edge)))
        )

    strides = tuple(int(s) for s in model.head.strides)
    expected = {stride: (edge // stride) ** 2 for stride in strides}
    actual = {
        stride: int((anchors[:, 2] == float(stride)).sum()) for stride in strides
    }
    assert actual == expected, (
        f"{module.__name__}: at a {edge}x{edge} input the head emitted "
        f"{actual} anchors per stride, expected {expected} — one cell per "
        f"position at each of the three levels. A level missing from the head's "
        f"loop leaves the parameter count untouched (its towers are still "
        f"constructed), the loss finite and the predictions well formed, and "
        f"silently discards every object that level is responsible for."
    )
    total = sum(expected.values())
    assert int(anchors.shape[0]) == total, (
        f"{module.__name__}: the anchor table holds {int(anchors.shape[0])} "
        f"rows for {total} expected cells"
    )
    assert int(cls_logits.shape[1]) == total and int(dist_logits.shape[1]) == total, (
        f"{module.__name__}: the head returned {int(cls_logits.shape[1])} class "
        f"rows and {int(dist_logits.shape[1])} distribution rows against "
        f"{total} anchors — the three tensors are not the same length, so every "
        f"downstream index is against a different table"
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
#
# The assigner is UNCHANGED from YOLOv8 at this generation — same four rules,
# same exponents. It is re-guarded here rather than assumed because the code is
# duplicated (zero relative imports repo-wide), and a duplicated assigner that
# leaves its guards behind is exactly how one of the four rules goes missing.
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

    YOLO11 has **no objectness branch**: the score it ranks by at inference is
    the classifier's, so the classifier is what has to carry localisation
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
        f"with label {int(matched_labels[0])}; the published assigner resolves "
        f"a multi-claim by IoU, so it belongs to label {labels[1]} (IoU "
        f"{float(ious[1]):.3f} against {float(ious[0]):.3f}). Ground truth 0 "
        f"has the higher ALIGNMENT metric ({alignment[0]:.4f} against "
        f"{alignment[1]:.4f}), so both a by-alignment tie-break and no "
        f"tie-break at all land here."
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
    reason. That is why ``guard_head_emits_every_level`` pins the per-level
    structure by the anchor table's deterministic shape instead, and why the
    per-level geometry lives in ``decode_per_level_stride`` and
    ``dfl_target_cell_units``.
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


def guard_the_whole_trunk_is_trained(module) -> None:
    """One train step must put a non-zero gradient on EVERY backbone and neck
    parameter, including C2PSA's.

    ⚠️ WHAT THIS IS AND IS NOT. YOLO11 has **no gradient isolation to protect**:
    the ``detach`` that is the single most silent thing in ``yolov10_s.py``
    exists only because that architecture has a second head that must not train
    the trunk. YOLO11 has one head, so there is no ``detach`` here and nothing
    for a gradient-isolation guard to assert. Saying so explicitly, because the
    obvious move when porting from the sibling is to carry its detach guard
    across, and the assertion it makes would be **vacuous here** — a template
    with no detach trivially satisfies "nothing is detached".

    What IS worth pinning is the opposite direction, and it is not vacuous: a
    module that is constructed, called, and yet sits OUTSIDE the loss graph
    would show up here and nowhere else. A ``torch.no_grad()`` accidentally
    scoping a block, a ``.detach()`` added for a shape fix, or a residual
    written as ``x + y.detach()`` all leave the parameter count, the shapes,
    the keys and the loss values untouched, and quietly freeze a chunk of the
    trunk while the rest keeps learning.
    """
    import torch

    edge = 128
    model = _build(module, 3, edge)
    model.train()
    targets = [
        {
            "boxes": torch.tensor([[20.0, 20.0, 100.0, 100.0]]),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
    ]
    total = sum(model([torch.rand(3, edge, edge)], targets).values())
    assert torch.isfinite(total), f"{module.__name__}: total loss is {total!r}"
    total.backward()

    trunk = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(("backbone.", "neck."))
    ]
    assert len(trunk) > 100, (
        f"fixture is degenerate: only {len(trunk)} trunk parameters found — the "
        f"name prefixes have moved and this probe is checking almost nothing"
    )
    dead = sorted(
        name
        for name, parameter in trunk
        if parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
    )
    assert not dead, (
        f"{module.__name__}: {len(dead)} of {len(trunk)} backbone/neck "
        f"parameter(s) received a zero or absent gradient from one train step, "
        f"so they are OUTSIDE the loss graph and will never move: "
        f"{dead[:8]}. YOLO11 has one head and no deliberate gradient "
        f"isolation, so every trunk parameter must train. A stray detach or an "
        f"over-scoped no_grad leaves the parameter count, the shapes, the keys "
        f"and the loss values all unchanged."
    )
    attention = [name for name, _ in trunk if ".c2psa." in name]
    assert attention, (
        "fixture is degenerate: no C2PSA parameters matched, so the block this "
        "guard most cares about is not being checked"
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
        raise OSError("network access is blocked by test_yolo11_s")

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
#: smaller. Measured on the shipped template — the thresholds below carry
#: several times the needed margin and absorb BLAS variation across platforms.
_OVERFIT_STEPS = 200
_OVERFIT_LR = 4e-3
_OVERFIT_EDGE = 128
_OVERFIT_BOX = [40.0, 40.0, 88.0, 88.0]
_OVERFIT_LABEL = 1


def guard_overfits_a_single_object(module) -> None:
    """The template must actually LEARN — and then detect what it learned.

    Everything else in this file is a single step or a synthetic call. This is
    the end-to-end claim: 200 Adam steps on one image with one object, then an
    eval pass that has to find it. It is the only guard that closes the loop
    from the assigner through the three losses to the DFL decode and the NMS,
    and it is what makes "trains" and "evaluates" claims about this template
    rather than about its return types.

    It also covers what nothing else can: zeroing the soft classification
    target leaves every structural guard green, every loss finite and every
    fixture satisfied, and the model simply never learns to fire.

    ⚠️ AND HERE IS WHAT IT DOES **NOT** COVER, inherited from the sibling's
    measurement rather than re-measured here. An assigner that prefers the
    WORST-localised candidate — ``ious`` replaced by ``1 - ious`` in the
    alignment metric — still overfits a fixture like this one, because the
    metric only chooses *which* anchors are positive; the box and DFL losses
    then regress whichever anchors were chosen towards the true box, and the
    inside-the-box prefilter keeps them all on the object. So the IoU assertion
    below is NOT evidence that the assigner ranks correctly. That property is
    pinned by ``tal_metric_exponents``.
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
        f"NOTE this is a weaker signal than it looks — see the docstring."
    )


# --------------------------------------------------------------------------
# the guard table, and the mutations that prove each can go red
# --------------------------------------------------------------------------

GUARDS = {
    "published_architecture": guard_matches_the_published_architecture,
    "c3k_block_kinds": guard_c3k_block_kinds_match_the_yaml,
    "arch_table_is_live": guard_arch_table_is_live,
    "module_tree_size": guard_module_tree_size_is_pinned,
    "no_stateful_norm": guard_no_stateful_normalisation,
    "derived_norm_groups": guard_norm_groups_are_derived_from_the_channel_count,
    "deepest_stage_modules_are_applied": guard_deepest_stage_modules_are_applied,
    "sppf_series": guard_sppf_pools_in_series,
    "c3k2_fuses_all_blocks": guard_c3k2_fuses_every_intermediate_block,
    "c3k_branch_routing": guard_c3k_routes_its_blocks_through_the_first_branch,
    "bottleneck_residual": guard_bottlenecks_keep_their_identity_branch,
    "c2psa_partial": guard_c2psa_attends_to_only_one_half,
    "psablock_residuals": guard_psablock_applies_both_residuals,
    "attention_heads_derived": guard_attention_head_count_is_derived,
    "attention_reference_operator": guard_attention_matches_the_reference_operator,
    "decoupled_head": guard_head_is_decoupled,
    "light_class_tower": guard_class_tower_is_depthwise_separable,
    "reg_max_is_live": guard_reg_max_reaches_the_head_and_the_decode,
    "seed_excluded_prefixes": (
        guard_seed_excluded_prefixes_are_exactly_the_class_shaped_keys
    ),
    "head_flatten_order": guard_head_flatten_order_matches_the_anchor_table,
    "head_emits_every_level": guard_head_emits_every_level,
    "decode_per_level_stride": guard_decode_scales_by_each_levels_stride,
    "dfl_decode_expectation": guard_dfl_decode_is_the_softmax_expectation,
    "dfl_target_cell_units": guard_dfl_target_is_in_cell_units_of_its_own_level,
    "dfl_loss_interpolates": (
        guard_dfl_loss_interpolates_between_the_two_bracketing_bins
    ),
    "decode_per_image": guard_decode_is_per_image_and_aligned,
    "decode_suppresses_duplicates": guard_decode_suppresses_duplicates,
    "original_coordinates": guard_predictions_are_in_original_image_coordinates,
    "declared_size_measured": guard_declared_image_size_is_the_measured_edge,
    "tal_metric_exponents": guard_tal_metric_weights_localisation_over_classification,
    "tal_inside_the_box": guard_tal_requires_the_anchor_point_inside_the_box,
    "tal_topk_ranking": guard_tal_selects_the_topk_best_ranked_candidates,
    "tal_normalised_target": guard_tal_target_is_the_normalised_alignment_metric,
    "tal_tie_break_by_iou": guard_tal_breaks_ties_by_iou_not_by_alignment,
    "positives_reach_box_branch": guard_positives_reach_the_box_regression_branch,
    "whole_trunk_is_trained": guard_the_whole_trunk_is_trained,
    "no_network": guard_constructs_with_no_network,
    "overfits_one_object": guard_overfits_a_single_object,
}

#: ``(name, anchor, replacement, guard)``. The anchor must be unique in the
#: file — ``_mutate`` refuses otherwise, so a drifted anchor is a RED rather
#: than a patch that silently applies to nothing and reports "passed".
MUTATIONS = [
    # -- the published architecture, per layer ------------------------------
    (
        "c3k2_inner_bottleneck_runs_at_full_width",
        "C3K2_INNER_EXPANSION = 0.50",
        "C3K2_INNER_EXPANSION = 1.0",
        "published_architecture",
    ),
    (
        "neck_reuses_the_backbone_stride8_width",
        "    NECK_WIDTHS = (512, 256, 512, 1024)",
        "    NECK_WIDTHS = (512, 512, 512, 1024)",
        "published_architecture",
    ),
    (
        "sppf_pools_four_times",
        "    def __init__(self, in_ch, out_ch, ksize=5, repeats=3):",
        "    def __init__(self, in_ch, out_ch, ksize=5, repeats=4):",
        "published_architecture",
    ),
    (
        "c3k_at_the_shallowest_stage",
        "        (128, 256, 2, False, C3K2_SHALLOW_EXPANSION),",
        "        (128, 256, 2, True, C3K2_SHALLOW_EXPANSION),",
        "c3k_block_kinds",
    ),
    (
        "neck_stride32_fusion_is_not_c3k",
        "    NECK_C3K = (False, False, False, True)",
        "    NECK_C3K = (False, False, False, False)",
        "c3k_block_kinds",
    ),
    # -- the knobs the shipped scale does not exercise ----------------------
    (
        "stem_width_hardcoded_at_the_shipped_scale",
        "        stem_ch = _round_channels(self.STEM_CHANNELS)",
        "        stem_ch = 32",
        "arch_table_is_live",
    ),
    (
        "class_tower_width_hardcoded_to_the_feature_width",
        "        self.cls_hidden = max(in_channels[0], min(num_classes, 100))",
        "        self.cls_hidden = in_channels[0]",
        "arch_table_is_live",
    ),
    (
        "depth_multiplier_taken_from_yolov8",
        "    return max(int(round(blocks * DEPTH_MULT)), 1)",
        "    return max(int(round(blocks * 0.33)), 1)",
        "arch_table_is_live",
    ),
    (
        "c2psa_depth_hardcoded_at_one_block",
        "        self.c2psa = C2PSA(deepest, deepest, blocks=_round_depth(C2PSA_BLOCKS))",
        "        self.c2psa = C2PSA(deepest, deepest, blocks=1)",
        "arch_table_is_live",
    ),
    (
        "channel_cap_hardcoded_at_the_shipped_value",
        "    scaled = min(channels, MAX_CHANNELS) * WIDTH_MULT",
        "    scaled = min(channels, 1024) * WIDTH_MULT",
        "arch_table_is_live",
    ),
    (
        "extra_block_per_backbone_stage",
        "                    blocks=_round_depth(blocks_full),",
        "                    blocks=_round_depth(blocks_full) + 1,",
        "module_tree_size",
    ),
    # -- normalisation ------------------------------------------------------
    (
        "batch_norm_comes_back",
        "        self.norm = nn.GroupNorm(_norm_groups(out_ch), out_ch, eps=1e-3)",
        "        self.norm = nn.BatchNorm2d(out_ch, eps=1e-3)",
        "no_stateful_norm",
    ),
    (
        "hardcoded_32_groups",
        "        self.norm = nn.GroupNorm(_norm_groups(out_ch), out_ch, eps=1e-3)",
        "        self.norm = nn.GroupNorm(32, out_ch, eps=1e-3)",
        "derived_norm_groups",
    ),
    # -- constructed but never called --------------------------------------
    (
        "c2psa_dropped_from_the_backbone_forward",
        "        return outputs[1], outputs[2], self.c2psa(self.sppf(outputs[3]))",
        "        return outputs[1], outputs[2], self.sppf(outputs[3])",
        "deepest_stage_modules_are_applied",
    ),
    (
        "sppf_dropped_from_the_backbone_forward",
        "        return outputs[1], outputs[2], self.c2psa(self.sppf(outputs[3]))",
        "        return outputs[1], outputs[2], self.c2psa(outputs[3])",
        "deepest_stage_modules_are_applied",
    ),
    (
        "sppf_and_c2psa_swapped",
        "        return outputs[1], outputs[2], self.c2psa(self.sppf(outputs[3]))",
        "        return outputs[1], outputs[2], self.sppf(self.c2psa(outputs[3]))",
        "deepest_stage_modules_are_applied",
    ),
    (
        "sppf_pools_in_parallel",
        "            outputs.append(self.pool(outputs[-1]))",
        "            outputs.append(self.pool(outputs[0]))",
        "sppf_series",
    ),
    # -- block topology, all shape-clean ----------------------------------
    (
        "c3k2_blocks_chain_from_the_wrong_split_half",
        "            branches.append(block(branches[-1]))",
        "            branches.append(block(branches[0]))",
        "c3k2_fuses_all_blocks",
    ),
    (
        "c3k2_fusion_branch_order_reversed",
        "        return self.cv2(torch.cat(branches, dim=1))",
        "        return self.cv2(torch.cat(branches[::-1], dim=1))",
        "c3k2_fuses_all_blocks",
    ),
    (
        "c3k_mixes_the_skip_instead_of_its_own_branch",
        "        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))",
        "        return self.cv3(torch.cat((self.cv1(x), self.m(self.cv2(x))), dim=1))",
        "c3k_branch_routing",
    ),
    (
        "neck_blocks_lose_their_shortcut_as_in_yolov8",
        "        self.use_add = shortcut and in_ch == out_ch",
        "        self.use_add = False",
        "bottleneck_residual",
    ),
    (
        "c2psa_attends_to_both_halves",
        "        return self.cv2(torch.cat((bypass, self.m(attend)), dim=1))",
        "        return self.cv2(torch.cat((self.m(bypass), self.m(attend)), dim=1))",
        "c2psa_partial",
    ),
    (
        "c2psa_drops_the_bypass_half",
        "        return self.cv2(torch.cat((bypass, self.m(attend)), dim=1))",
        "        return self.cv2(torch.cat((self.m(attend), self.m(attend)), dim=1))",
        "c2psa_partial",
    ),
    (
        "psablock_drops_the_attention_residual",
        "        x = x + self.attn(x) if self.add else self.attn(x)",
        "        x = self.attn(x)",
        "psablock_residuals",
    ),
    (
        "psablock_drops_the_ffn_residual",
        "        return x + self.ffn(x) if self.add else self.ffn(x)",
        "        return self.ffn(x)",
        "psablock_residuals",
    ),
    # -- attention ---------------------------------------------------------
    (
        "attention_head_count_hardcoded",
        "        self.num_heads = max(1, dim // head_dim)",
        "        self.num_heads = 8",
        "attention_heads_derived",
    ),
    (
        "sdpa_output_layout_not_transposed_back",
        "        ).transpose(-2, -1)\n        attended = attended.reshape(",
        "        )\n        attended = attended.reshape(",
        "attention_reference_operator",
    ),
    (
        "attention_drops_the_positional_encoding_residual",
        "        return self.proj(\n"
        "            attended + self.pe(value.reshape(batch, channels, height, width))\n"
        "        )",
        "        return self.proj(attended)",
        "attention_reference_operator",
    ),
    # -- head --------------------------------------------------------------
    (
        "coupled_head",
        "            self.cls_convs.append(self._class_tower(channels))",
        "            self.cls_convs.append(self.box_convs[-1])",
        "decoupled_head",
    ),
    (
        "class_tower_copied_from_yolov8",
        "                ConvNormAct(channels, channels, 3, stride=1, groups=channels),",
        "                ConvNormAct(channels, channels, 3, stride=1),",
        "light_class_tower",
    ),
    (
        "head_reg_max_from_its_own_default",
        "            self.num_classes, self.neck.out_channels, reg_max=self.reg_max",
        "            self.num_classes, self.neck.out_channels",
        "reg_max_is_live",
    ),
    (
        "hardcoded_box_channel_width",
        "            self.box_preds.append(nn.Conv2d(self.box_hidden, 4 * reg_max, 1))",
        "            self.box_preds.append(nn.Conv2d(self.box_hidden, 64, 1))",
        "reg_max_is_live",
    ),
    (
        "class_tower_width_tracks_the_class_count",
        "        self.cls_hidden = max(in_channels[0], min(num_classes, 100))",
        "        self.cls_hidden = max(in_channels[0], num_classes)",
        "seed_excluded_prefixes",
    ),
    # -- geometry and decode ----------------------------------------------
    (
        "transposed_head_flatten",
        "                cls_output.permute(0, 2, 3, 1).reshape(batch, height * width, -1)",
        "                cls_output.permute(0, 3, 2, 1).reshape(batch, height * width, -1)",
        "head_flatten_order",
    ),
    (
        "head_drops_the_coarsest_level",
        "        for level, (feature, stride) in enumerate(zip(features, self.strides)):",
        "        for level, (feature, stride) in enumerate(zip(features[:2], self.strides)):",
        "head_emits_every_level",
    ),
    (
        "single_stride_decode",
        "    anchor_x, anchor_y, stride = anchors[:, 0], anchors[:, 1], anchors[:, 2]",
        "    anchor_x, anchor_y, stride = anchors[:, 0], anchors[:, 1], anchors[:, 2] * 0.0 + 8.0",
        "decode_per_level_stride",
    ),
    (
        "dfl_decode_takes_an_argmax",
        "    return (dist_logits.softmax(dim=-1) * bins).sum(dim=-1)",
        "    return (dist_logits.argmax(dim=-1) * 1.0).to(dist_logits.dtype)",
        "dfl_decode_expectation",
    ),
    (
        "dfl_decode_skips_the_softmax",
        "    return (dist_logits.softmax(dim=-1) * bins).sum(dim=-1)",
        "    return (dist_logits * bins).sum(dim=-1)",
        "dfl_decode_expectation",
    ),
    (
        "dfl_target_uses_one_stride",
        "    scaled = boxes_xyxy / stride.unsqueeze(-1)",
        "    scaled = boxes_xyxy / 8.0",
        "dfl_target_cell_units",
    ),
    (
        "dfl_loss_drops_the_interpolation_weights",
        "    return (loss_lower * weight_lower + loss_upper * weight_upper).mean(dim=-1)",
        "    return (loss_lower * 0.5 + loss_upper * 0.5).mean(dim=-1)",
        "dfl_loss_interpolates",
    ),
    (
        "dfl_loss_collapses_the_two_bins",
        "    return (loss_lower * weight_lower + loss_upper * weight_upper).mean(dim=-1)",
        "    return loss_lower.mean(dim=-1)",
        "dfl_loss_interpolates",
    ),
    (
        "background_channel_kept",
        "            class_scores = class_scores[:, 1:]",
        "            class_scores = class_scores[:, 0:]",
        "decode_per_image",
    ),
    (
        "decode_truncates_the_batch",
        "        for boxes, class_scores, (height, width) in zip(decoded, scores, image_sizes):",
        "        for boxes, class_scores, (height, width) in zip(decoded[:1], scores, image_sizes):",
        "decode_per_image",
    ),
    (
        "nms_removed",
        "            keep = batched_nms(candidate_boxes, flat_scores, labels, self.nms_thresh)",
        "            keep = flat_scores.argsort(descending=True)",
        "decode_suppresses_duplicates",
    ),
    (
        "nms_is_class_agnostic",
        "            keep = batched_nms(candidate_boxes, flat_scores, labels, self.nms_thresh)",
        "            keep = batched_nms(candidate_boxes, flat_scores, torch.zeros_like(labels), self.nms_thresh)",
        "decode_suppresses_duplicates",
    ),
    (
        "no_postprocess",
        "        return self.transform.postprocess(\n"
        "            detections, image_list.image_sizes, original_image_sizes\n"
        "        )",
        "        return detections",
        "original_coordinates",
    ),
    (
        "transform_resizes_past_the_declared_edge",
        "            min_size=self.input_size,\n            max_size=self.input_size,",
        "            min_size=self.input_size * 2,\n            max_size=self.input_size * 2,",
        "declared_size_measured",
    ),
    # -- task-aligned assignment ------------------------------------------
    (
        "swapped_alignment_exponents",
        "        alignment = scores.pow(TAL_ALPHA) * ious.pow(TAL_BETA)",
        "        alignment = scores.pow(TAL_BETA) * ious.pow(TAL_ALPHA)",
        "tal_metric_exponents",
    ),
    (
        "alignment_prefers_the_worst_iou",
        "        alignment = scores.pow(TAL_ALPHA) * ious.pow(TAL_BETA)",
        "        alignment = scores.pow(TAL_ALPHA) * (1.0 - ious).pow(TAL_BETA)",
        "tal_metric_exponents",
    ),
    (
        "no_inside_the_box_rule",
        "        return (x > left) & (x < right) & (y > top) & (y < bottom)",
        "        return ((x + left) * 0 == 0)",
        "tal_inside_the_box",
    ),
    (
        "topk_takes_the_worst",
        "        _, positions = torch.topk(candidate, topk, dim=1)",
        "        _, positions = torch.topk(-candidate, topk, dim=1)",
        "tal_topk_ranking",
    ),
    (
        "topk_bound_removed",
        "        selected = torch.zeros_like(candidate, dtype=torch.bool)",
        "        selected = torch.ones_like(candidate, dtype=torch.bool)",
        "tal_topk_ranking",
    ),
    (
        "hard_class_target",
        "        normalised = (assigned * best_iou / (best_alignment + _EPS)).amax(dim=0)",
        "        normalised = matching.to(alignment.dtype).amax(dim=0)",
        "tal_normalised_target",
    ),
    (
        "target_not_normalised",
        "        normalised = (assigned * best_iou / (best_alignment + _EPS)).amax(dim=0)",
        "        normalised = assigned.amax(dim=0)",
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
        "        if bool(contested.any()):",
        "        if False:",
        "tal_tie_break_by_iou",
    ),
    (
        "assign_nothing",
        "        matching = selected & inside & (candidate > 0.0)",
        "        matching = selected & inside & (candidate > 1e9)",
        "positives_reach_box_branch",
    ),
    (
        "neck_detaches_the_backbone",
        "        c3, c4, c5 = features",
        "        c3, c4, c5 = (feature.detach() for feature in features)",
        "whole_trunk_is_trained",
    ),
    (
        "fetches_at_construction",
        "        self.num_classes = int(num_classes) + 1",
        "        import socket\n\n"
        "        socket.getaddrinfo(\"download.pytorch.org\", 443)\n"
        "        self.num_classes = int(num_classes) + 1",
        "no_network",
    ),
    (
        "soft_class_target_zeroed",
        "                cls_targets[index, fg_mask, labels] = aligned",
        "                cls_targets[index, fg_mask, labels] = aligned * 0.0",
        "overfits_one_object",
    ),
]


#: Mutations that leave the model TRAINING — a finite loss dict and well-formed
#: predictions — so the family train-step test stays green against every one of
#: them. That is the reason the guards above exist, and asserting it here stops
#: someone concluding the family test already covers this template.
#:
#: Deliberately a subset: some mutations in the table above are caught by a
#: shape or a construction error as well, and listing those here would overstate
#: the claim.
_SILENT_MUTATIONS = frozenset(
    {
        "c2psa_dropped_from_the_backbone_forward",
        "sppf_dropped_from_the_backbone_forward",
        "sppf_and_c2psa_swapped",
        "sppf_pools_in_parallel",
        "c3k2_blocks_chain_from_the_wrong_split_half",
        "c3k2_fusion_branch_order_reversed",
        "c3k_mixes_the_skip_instead_of_its_own_branch",
        "neck_blocks_lose_their_shortcut_as_in_yolov8",
        "c2psa_attends_to_both_halves",
        "c2psa_drops_the_bypass_half",
        "psablock_drops_the_attention_residual",
        "psablock_drops_the_ffn_residual",
        "attention_head_count_hardcoded",
        "sdpa_output_layout_not_transposed_back",
        "attention_drops_the_positional_encoding_residual",
        "batch_norm_comes_back",
        "swapped_alignment_exponents",
        "alignment_prefers_the_worst_iou",
        "no_inside_the_box_rule",
        "topk_takes_the_worst",
        "topk_bound_removed",
        "hard_class_target",
        "target_not_normalised",
        "tie_break_by_alignment",
        "no_tie_break",
        "assign_nothing",
        "neck_detaches_the_backbone",
        "soft_class_target_zeroed",
        "nms_removed",
        "nms_is_class_agnostic",
        "background_channel_kept",
        "dfl_decode_takes_an_argmax",
        "dfl_decode_skips_the_softmax",
        "dfl_target_uses_one_stride",
        "dfl_loss_drops_the_interpolation_weights",
        "dfl_loss_collapses_the_two_bins",
        "single_stride_decode",
        "transposed_head_flatten",
        "head_drops_the_coarsest_level",
        "no_postprocess",
        "c3k2_inner_bottleneck_runs_at_full_width",
        "c3k_at_the_shallowest_stage",
        "neck_stride32_fusion_is_not_c3k",
        "class_tower_copied_from_yolov8",
        "class_tower_width_tracks_the_class_count",
    }
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
    """Point ONE named guard at a template edited to break exactly what it
    checks.

    ⚠️ THE ATTRIBUTION IS THE POINT, and it is why this runs a single guard
    rather than the whole table. A sweep that runs every guard against every
    mutation proves that *something* reddened, not that the guard under test
    did — so a guard that is defined and never actually exercised looks covered.
    Here the mutation names its guard and only that guard runs, so a red is
    attributable and a green is a genuine survivor.

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


def test_mutation_names_are_unique() -> None:
    """Two mutations sharing a name make the sweep's report ambiguous — the
    pytest id collides and one of them is indistinguishable from the other in
    any table published from this suite."""
    names = [entry[0] for entry in MUTATIONS]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"duplicate mutation names: {duplicates}"


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
