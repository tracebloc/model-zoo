"""Guards for ``object_detection/pytorch/yolov12_s.py``, each proven able to go
red by a mutation that is kept in the suite.

Why this file exists
--------------------
``tests/test_od_torchvision_family_train_step.py`` proves a template returns a
loss dict and a ``List[Dict]`` of xyxy predictions. For a template that wraps a
torchvision builder that is a real assertion: the loss is the library's. For
``yolov12_s.py`` the backbone, neck, head, assigner and all three losses are
**our own code**, so "returns a loss dict" proves only that our code returns a
dict. Every interesting way a hand-written detector is wrong is silent:

* the assigner matches **nothing** — BCE over an all-negative image is finite
  and small, so the train step passes and the model learns no objects;
* the assigner matches the wrong anchors — a swapped alignment exponent picks
  the best-*classified* candidate instead of the best-*localised* one, changes
  no cardinality whatsoever, and leaves every loss finite;
* **Area Attention attends globally.** Skip the band reshape and the module
  computes ordinary self-attention: identical parameters, identical shapes,
  identical keys, finite losses — and the paper's entire contribution gone.
* an ``A2C2f`` is built like a ``C2f`` (splitting, ``2 + n`` fusion branches)
  rather than like R-ELAN (no split, ``1 + n``);
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

Five traps this file is shaped around
--------------------------------------
**A BYPASSED MODULE IS INVISIBLE TO EVERY PARAMETER COUNT.** Measured on the
siblings and re-measured here. ``YOLOv12Backbone.forward``'s deepest output can
be replaced by ``self.downsamples[3](outputs[2])``, which has the same shape and
the same stride and leaves the whole 2.69M-parameter stride-32 ``A2C2f``
constructed, exported in the ``state_dict``, shipped to every edge and averaged
every federated round while contributing nothing. The published-count guard
**DID NOT RAISE** against it. Same for the Area-Attention positional encoding
and for ``A2C2f``'s ``gamma``. Those need functional "was it applied" guards.

**A REMOVAL IS THE THING A DIFF-DRIVEN PORT NEVER PERFORMS.** YOLOv12 deletes
YOLO11's ``SPPF`` and ``C2PSA`` outright. Nothing about a parameter total taken
in isolation says "and there should be nothing here" — the count only disagrees
once you have the right expectation. So the per-layer table is the anchor, and
``guard_no_pooling_or_partial_attention_in_the_tree`` states the removal
positively rather than leaving it implied by arithmetic.

**The eval path is nearly vacuous on a fresh model.** The class prior is
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
``_reference_yolov12_layers``, which is derived from the published arch table
with nothing from ``model_zoo/`` imported, and anchored to figures from outside
this repo entirely — see ``_PUBLISHED`` and ``_PUBLISHED_LAYERS``. And it is
compared **per layer at all five published scales**, not only in total: two
compensating errors survive a total and cannot survive eighty rows.
"""

import contextlib
import copy
import importlib.util
import pathlib
import tempfile

import pytest

ROOT = pathlib.Path(__file__).parent.parent
OD_PYTORCH = ROOT / "model_zoo" / "object_detection" / "pytorch"
TEMPLATE = OD_PYTORCH / "yolov12_s.py"

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

    ⚠️ THE ANCHOR MUST BE THE CODE, NOT A DOCSTRING MENTION OF IT. A harness on
    a sibling PR patched a constant's appearance inside prose and produced a
    false survivor; the uniqueness check below is what catches that, because a
    constant that is also discussed in a comment occurs more than once. Anchor
    on the assignment line, not on the bare name.
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
# from `ultralytics/cfg/models/12/yolo12.yaml` and
# `nn/modules/{block,conv,head}.py` plus `nn/tasks.py`'s `parse_model`, with
# NOTHING under `model_zoo/` imported. And the transcription is itself anchored
# to figures from outside this repo — the yaml's own per-scale model summaries —
# so it cannot drift into agreeing with a wrong template.
#
# ⚠️ AND IT IS CHECKED PER LAYER, NOT ONLY IN TOTAL. On the sibling `yolov10_s`
# the total matched while individual layers were wrong; only a per-layer
# comparison found it. `_PUBLISHED_LAYERS` holds the upstream per-layer table at
# ALL FIVE scales, so the same class of compensating error cannot survive here.

#: Published parameter counts, quoted verbatim from Ultralytics
#: ``cfg/models/12/yolo12.yaml``'s own ``scales:`` comments::
#:
#:   n: [0.50, 0.25, 1024] # summary: 272 layers, 2,602,288 parameters, 2,602,272 gradients, 6.7 GFLOPs
#:   s: [0.50, 0.50, 1024] # summary: 272 layers, 9,284,096 parameters, 9,284,080 gradients, 21.7 GFLOPs
#:   m: [0.50, 1.00, 512] # summary: 292 layers, 20,199,168 parameters, 20,199,152 gradients, 68.1 GFLOPs
#:   l: [1.00, 1.00, 512] # summary: 488 layers, 26,450,784 parameters, 26,450,768 gradients, 89.7 GFLOPs
#:   x: [1.00, 1.50, 512] # summary: 488 layers, 59,210,784 parameters, 59,210,768 gradients, 200.3 GFLOPs
#:
#: **Identical in ultralytics 8.3.78 — the release that first shipped
#: ``cfg/models/12/`` — and on ``main`` as of 2026-09, layer counts included.**
#: Eighteen months apart, so the figures are not a transient of one release.
#:
#: FIVE scales, not one, so the anchor pins the width handling, the depth
#: handling AND BOTH per-scale block overrides rather than one arithmetic total.
#:
#: The 16-parameter gap in each is the DFL projection vector, which upstream
#: stores as a frozen ``Conv2d`` weight (``requires_grad=False``, hence
#: "gradients") and this template does not store at all — it builds the bin
#: indices with ``torch.arange`` inside the decode. So THIS TEMPLATE'S TOTAL
#: PARAMETER COUNT IS THE PUBLISHED *GRADIENT* COUNT, exactly, and the
#: derivation below is checked against both figures with the gap stated
#: explicitly rather than absorbed into a tolerance.
#:
#: ⚠️ THERE ARE THREE PUBLISHED FIGURES FOR YOLOv12-S AND THIS IS ONE OF THEM.
#: The authors' own ``v1.0`` tag publishes 9,285,632 / 9,285,616 for the SAME
#: topology (its ``Conv`` signature takes ``bias`` in the position ultralytics
#: uses for padding, so every Area-Attention positional encoding gains a bias:
#: ``+1,536 == 4 * 128 + 4 * 256``, the eight ABlock widths here). The authors'
#: current ``main`` is YOLOv12-turbo, 9,127,424, and is a DIFFERENT
#: architecture — grouped downsample convs at yaml layers 1 and 3. Do not
#: "correct" the rows below towards either; the template is anchored to the
#: ultralytics yaml because that one is installable and therefore checkable.
_PUBLISHED = {
    # scale: (width, depth, max_channels, published total, published gradients)
    "n": (0.25, 0.50, 1024, 2_602_288, 2_602_272),
    "s": (0.50, 0.50, 1024, 9_284_096, 9_284_080),
    "m": (1.00, 0.50, 512, 20_199_168, 20_199_152),
    "l": (1.00, 1.00, 512, 26_450_784, 26_450_768),
    "x": (1.50, 1.00, 512, 59_210_784, 59_210_768),
}

#: The scale this template ships at.
_SHIPPED_SCALE = "s"

#: Parameters upstream stores for the DFL bin vector and this template does not.
_DFL_PROJECTION_CONSTANTS = 16

#: The class count the published figures are quoted at.
_PUBLISHED_CLASSES = 80

#: ``parse_model``'s TWO per-scale overrides, transcribed. They use DIFFERENT
#: scale sets, which is the thing to notice::
#:
#:     if m is C3k2:
#:         if scale in "mlx":            # every C3k2's c3k flag -> True
#:             args[3] = True
#:     if m is A2C2f:
#:         if scale in {"l", "x"}:       # residual=True, mlp_ratio=1.2
#:             args.extend((True, 1.2))
#:
#: Recorded here because they are the reason an m/l/x total cannot be reproduced
#: from the yaml text alone, and because a "scale" in YOLOv12 is therefore a
#: multiplier triple PLUS two block properties.
_C3K_FORCED_SCALES = ("m", "l", "x")
_A2C2F_RESIDUAL_SCALES = ("l", "x")
#: The MLP expansion each of those two regimes uses.
_MLP_RATIO_DEFAULT = 2.0
_MLP_RATIO_RESIDUAL = 1.2

#: Per-layer parameter counts of the upstream model, measured with
#: ``ultralytics==8.3.78`` as::
#:
#:     from ultralytics.nn.tasks import DetectionModel, yaml_model_load
#:     m = DetectionModel(yaml_model_load("yolo12s.yaml"), nc=80, verbose=False)
#:     [sum(p.numel() for p in layer.parameters()) for layer in m.model]
#:
#: Keyed by the yaml's own layer index. Layers 9/10/12/13/16/19 are
#: ``nn.Upsample`` and ``Concat`` and hold no parameters, so they are omitted
#: rather than listed as zeros. Layer 21 (``Detect``) is quoted MINUS the 16 DFL
#: projection constants, so it is directly comparable with this template's head.
#:
#: ⚠️ THIS IS THE PER-LAYER ANCHOR, and it is the point of the whole block. A
#: total can be hit by two errors that cancel; on `yolov10_s` exactly that
#: happened. Sixteen rows at five scales cannot be.
_PUBLISHED_LAYERS = {
    "n": {
        0: 464,
        1: 4_672,
        2: 6_640,
        3: 36_992,
        4: 26_080,
        5: 147_712,
        6: 180_864,
        7: 295_424,
        8: 689_408,
        11: 86_912,
        14: 24_000,
        15: 36_992,
        17: 74_624,
        18: 147_712,
        20: 378_880,
        21: 464_896,
    },
    "s": {
        0: 928,
        1: 18_560,
        2: 26_080,
        3: 147_712,
        4: 103_360,
        5: 590_336,
        6: 689_408,
        7: 1_180_672,
        8: 2_689_536,
        11: 345_856,
        14: 95_104,
        15: 147_712,
        17: 296_704,
        18: 590_336,
        20: 1_511_424,
        21: 850_352,
    },
    "m": {
        0: 1_856,
        1: 73_984,
        2: 111_872,
        3: 590_336,
        4: 444_928,
        5: 2_360_320,
        6: 2_689_536,
        7: 2_360_320,
        8: 2_689_536,
        11: 1_248_768,
        14: 378_624,
        15: 590_336,
        17: 1_183_232,
        18: 2_360_320,
        20: 1_642_496,
        21: 1_472_688,
    },
    "l": {
        0: 1_856,
        1: 73_984,
        2: 173_824,
        3: 590_336,
        4: 691_712,
        5: 2_360_320,
        6: 4_272_944,
        7: 2_360_320,
        8: 4_272_944,
        11: 2_102_784,
        14: 592_640,
        15: 590_336,
        17: 2_037_248,
        18: 2_360_320,
        20: 2_496_512,
        21: 1_472_688,
    },
    "x": {
        0: 2_784,
        1: 166_272,
        2: 389_760,
        3: 1_327_872,
        4: 1_553_664,
        5: 5_309_952,
        6: 9_512_128,
        7: 5_309_952,
        8: 9_512_128,
        11: 4_727_040,
        14: 1_331_328,
        15: 1_327_872,
        17: 4_579_584,
        18: 5_309_952,
        20: 5_612_544,
        21: 3_237_936,
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
    6: "A2C2f[512,a2=T,area=4] -> P4",
    7: "downsample Conv[1024,3,2]",
    8: "A2C2f[1024,a2=T,area=1] -> P5",
    11: "neck A2C2f[512,a2=F] top-down P4",
    14: "neck A2C2f[256,a2=F] top-down P3",
    15: "neck Conv[256,3,2]",
    17: "neck A2C2f[512,a2=F] bottom-up P4",
    18: "neck Conv[512,3,2]",
    20: "neck C3k2[1024,c3k=T] bottom-up P5",
    21: "Detect head (DFL constants excluded)",
}

#: Each layer this template maps a yaml row onto. The mapping is the thing that
#: makes a per-layer comparison possible at all, and it is deliberately explicit
#: rather than derived from iteration order — an iteration-order mapping shifts
#: silently when a module is added.
_LAYER_PATHS = {
    0: "backbone.stem",
    1: "backbone.downsamples.0",
    2: "backbone.stages.0",
    3: "backbone.downsamples.1",
    4: "backbone.stages.1",
    5: "backbone.downsamples.2",
    6: "backbone.stages.2",
    7: "backbone.downsamples.3",
    8: "backbone.stages.3",
    11: "neck.td_p4",
    14: "neck.td_p3",
    15: "neck.bu_conv3",
    17: "neck.bu_p4",
    18: "neck.bu_conv4",
    20: "neck.bu_p5",
    21: "head",
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


def _reference_yolov12_layers(
    class_channels,
    width=0.50,
    depth=0.50,
    max_channels=1024,
    reg_max=16,
    c3k_forced=False,
    a2c2f_residual=False,
):
    """YOLOv12 parameter count per yaml layer, derived from the published spec.

    Transcribed from ``yolo12.yaml`` plus the module definitions it names, with
    nothing imported from ``model_zoo/``. Returns ``{yaml layer index:
    parameters}`` for the parameterised layers only; layer 21 EXCLUDES the DFL
    projection constants, matching ``_PUBLISHED_LAYERS``.

    The things that are easy to get wrong and are therefore spelled out:

    * **``A2C2f`` does not split.** ``cv1`` reduces to ``c_``; ``cv2`` fuses
      ``1 + n`` branches. ``C3k2``/``C2f`` widen to ``2 * c_`` and fuse
      ``2 + n``. Two convolutions, both different.
    * an attention entry is a **pair** of ``ABlock``s, and the pair size is a
      literal upstream — the depth-scaled ``n`` counts entries, not blocks;
    * the Area-Attention positional encoding is a **7x7** depthwise conv;
    * ``a2c2f_residual`` adds a ``gamma`` vector of ``c2`` numbers AND narrows
      the MLP to 1.2 — both, together, only at l/x;
    * ``C3k2``'s plain inner block runs at **half** the split branch
      (``Bottleneck``'s own ``e=0.5`` default), where ``C2f`` passes ``e=1.0``;
    * ``C3k`` is a ``C3``: THREE convs, both of the first two taking the full
      input, and its own bottlenecks at full branch width with ``kxk`` kernels;
    * the class tower is depthwise-separable, so its spatial convs are
      ``groups=channels``;
    * **there is no SPPF and no C2PSA.** Nothing to derive; noted because their
      absence is the change this generation makes to the backbone.
    """
    mlp_ratio = _MLP_RATIO_RESIDUAL if a2c2f_residual else _MLP_RATIO_DEFAULT

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

    def area_attention(dim, head_dim=32):
        heads = max(1, dim // head_dim)
        all_head_dim = (dim // heads) * heads
        return (
            _cna(dim, 3 * all_head_dim, 1)
            + _cna(all_head_dim, dim, 1)
            # SEVEN, not three: YOLO11's Attention.pe is 3x3.
            + _cna(all_head_dim, dim, 7, groups=dim)
        )

    def ablock(dim):
        hidden = int(dim * mlp_ratio)
        return area_attention(dim) + _cna(dim, hidden, 1) + _cna(hidden, dim, 1)

    def a2c2f(in_ch, out_ch, entries, a2, expansion=0.5):
        hidden = int(out_ch * expansion)
        total = (
            # ONE branch out of cv1, NOT two.
            _cna(in_ch, hidden, 1)
            # (1 + n), NOT (2 + n): there is no split branch to fuse.
            + _cna((1 + entries) * hidden, out_ch, 1)
        )
        if a2 and a2c2f_residual:
            total += out_ch  # the gamma layer-scale vector
        for _ in range(entries):
            # TWO ABlocks per entry, a literal upstream and not depth-scaled.
            total += 2 * ablock(hidden) if a2 else c3k(hidden)
        return total

    layers = {}

    # -- backbone, yaml 0-8 -------------------------------------------------
    stem = scale_width(64)
    layers[0] = _cna(3, stem, 3)

    # (downsample width, stage width, yaml repeats, kind, expansion) at FULL
    # width. The two shallow stages DOUBLE their downsample's width; the two
    # deep ones keep it.
    stage_spec = (
        (128, 256, 2, "c3k2", 0.25),
        (256, 512, 2, "c3k2", 0.25),
        (512, 512, 4, "a2c2f", 0.50),
        (1024, 1024, 4, "a2c2f", 0.50),
    )
    in_ch = stem
    widths = []
    index = 1
    for down_full, out_full, repeats_full, kind, expansion in stage_spec:
        down = scale_width(down_full)
        out = scale_width(out_full)
        layers[index] = _cna(in_ch, down, 3)
        if kind == "a2c2f":
            layers[index + 1] = a2c2f(down, out, scale_depth(repeats_full), True)
        else:
            layers[index + 1] = c3k2(
                down, out, scale_depth(repeats_full), c3k_forced, expansion
            )
        widths.append(out)
        in_ch = out
        index += 2

    # ⚠️ NOTHING BETWEEN LAYER 8 AND THE NECK. YOLO11 has SPPF (yaml 9) and
    # C2PSA (yaml 10) here; YOLOv12 has neither, and the neck's stride-32 input
    # is layer 8's own output.

    # -- neck, yaml 9-20 ----------------------------------------------------
    c3, c4, c5 = widths[1], widths[2], widths[3]
    # The yaml's OWN fusion widths, not the backbone's.
    td_p4 = scale_width(512)
    td_p3 = scale_width(256)
    bu_p4 = scale_width(512)
    bu_p5 = scale_width(1024)
    down3 = scale_width(256)
    down4 = scale_width(512)
    entries = scale_depth(2)

    layers[11] = a2c2f(c5 + c4, td_p4, entries, False)
    layers[14] = a2c2f(td_p4 + c3, td_p3, entries, False)
    layers[15] = _cna(td_p3, down3, 3)
    layers[17] = a2c2f(down3 + td_p4, bu_p4, entries, False)
    layers[18] = _cna(bu_p4, down4, 3)
    # The ONE neck fusion that is a C3k2 rather than an A2C2f, and the yaml
    # marks it c3k=True.
    layers[20] = c3k2(down4 + c5, bu_p5, entries, True, 0.50)

    # -- head, yaml 21 ------------------------------------------------------
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
    layers[21] = head
    return layers


def _reference_yolov12_parameters(class_channels, **kwargs):
    return sum(_reference_yolov12_layers(class_channels, **kwargs).values())


def _reference_kwargs(scale):
    """The reference derivation's knobs for one published scale."""
    width, depth, max_channels, _, _ = _PUBLISHED[scale]
    return {
        "width": width,
        "depth": depth,
        "max_channels": max_channels,
        "c3k_forced": scale in _C3K_FORCED_SCALES,
        "a2c2f_residual": scale in _A2C2F_RESIDUAL_SCALES,
    }


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
    # The two shallow C3k2 stages have yaml repeat 2 -> 1 entry; the two deep
    # A2C2f stages have yaml repeat 4 -> 2 entries. THE 4s ARE WHY THE DEPTH
    # MULTIPLIER IS VISIBLE AT THIS SCALE, unlike on YOLO11.
    "backbone_stage_entries": (1, 1, 2, 2),
    "neck_stage_entries": (1, 1, 1, 1),
    "backbone_kinds": ("c3k2", "c3k2", "a2c2f", "a2c2f"),
    "neck_kinds": ("a2c2f", "a2c2f", "a2c2f", "c3k2"),
    # area 4 on the stride-16 attention stage, 1 (global) on stride-32.
    "backbone_areas": (4, 1),
    # c3k is False on both shallow backbone C3k2s at n/s, and True on the one
    # neck C3k2.
    "backbone_c3k": (False, False),
    "neck_c3k": (True,),
    # Every neck A2C2f is a2=False: THERE IS NO ATTENTION IN THE NECK.
    "neck_a2": (False, False, False),
    "ablocks_per_entry": 2,
    "attention_head_dim": 32,
    "box_hidden": 64,
    "cls_hidden": 128,
    "strides": (8, 16, 32),
    "reg_max": 16,
}


def test_the_reference_derivation_matches_the_published_figures() -> None:
    """The transcription, checked against the numbers it is transcribed from.

    Runs before anything is built and needs no torch: if this fails, the
    reference is wrong and every comparison against it is worthless. FIVE
    independent totals from outside this repo, so the derivation has to be right
    about the width handling, the depth handling, the channel cap, the rounding
    direction and BOTH per-scale block overrides — not just arithmetically
    self-consistent.
    """
    for scale, (_, _, _, total, gradients) in sorted(_PUBLISHED.items()):
        derived = _reference_yolov12_parameters(
            _PUBLISHED_CLASSES, **_reference_kwargs(scale)
        )
        assert derived == gradients, (
            f"the reference derivation gives {derived:,} at the published "
            f"YOLOv12-{scale.upper()} scale, but yolo12.yaml's own summary says "
            f"{gradients:,} gradients (off by {derived - gradients:+,}). The "
            f"transcription is wrong, so every guard that compares against it "
            f"is meaningless."
        )
        assert total - gradients == _DFL_PROJECTION_CONSTANTS, (
            f"YOLOv12-{scale.upper()}: the published parameter/gradient gap is "
            f"{total - gradients}, not the {_DFL_PROJECTION_CONSTANTS} DFL "
            f"projection constants this file assumes it to be. That assumption "
            f"is what makes the gradient figure the right comparison for a "
            f"template that stores no bin vector."
        )


def test_the_reference_derivation_matches_the_published_figures_per_layer() -> None:
    """The same check, PER YAML LAYER, at all five scales.

    ⚠️ THIS IS THE ONE THAT MATTERS. A total can be hit by two errors that
    cancel: on `yolov10_s` the total matched while individual layers did not,
    and it was found only by comparing every layer. Sixteen rows at five scales
    is 80 independent comparisons against figures measured off upstream.

    The n and s rows also separate a mistake the totals cannot: at those two
    scales neither per-scale override is active, so a derivation that got an
    override's SCALE SET wrong would still be right here and wrong at m/l/x —
    and the m/l/x rows catch that in the other direction.
    """
    for scale in sorted(_PUBLISHED_LAYERS):
        expected = _PUBLISHED_LAYERS[scale]
        derived = _reference_yolov12_layers(
            _PUBLISHED_CLASSES, **_reference_kwargs(scale)
        )
        assert set(derived) == set(expected), (
            f"YOLOv12-{scale.upper()}: the reference derives layers "
            f"{sorted(derived)} but the upstream table lists {sorted(expected)}"
        )
        mismatches = {
            index: (derived[index], expected[index])
            for index in sorted(expected)
            if derived[index] != expected[index]
        }
        assert not mismatches, (
            f"YOLOv12-{scale.upper()}: the reference disagrees with the "
            f"upstream per-layer measurement:\n"
            + "\n".join(
                f"  yaml {index:2d} {_LAYER_LABELS[index]:34s} derived "
                f"{got:>10,} upstream {want:>10,} ({got - want:+,})"
                for index, (got, want) in mismatches.items()
            )
        )
        assert sum(expected.values()) == _PUBLISHED[scale][4], (
            f"YOLOv12-{scale.upper()}: the upstream per-layer table sums to "
            f"{sum(expected.values()):,}, not the published "
            f"{_PUBLISHED[scale][4]:,} gradients — the table itself is wrong"
        )


# --------------------------------------------------------------------------
# structure guards
# --------------------------------------------------------------------------


@contextlib.contextmanager
def _at_published_scale(module, scale):
    """Rebuild-scope: point the template's live knobs at another published scale.

    Sets the two multipliers, the channel cap and BOTH per-scale block
    overrides ``parse_model`` applies, then restores every one of them. The
    knobs are module globals read at CONSTRUCTION time, which is exactly what
    makes them knobs rather than decoration.

    ⚠️ TWO OVERRIDES WITH DIFFERENT SCALE SETS, which is the YOLOv12-specific
    part: ``C3K2_FORCE_C3K`` at m/l/x, ``A2C2F_RESIDUAL`` (plus the narrower
    ``MLP_RATIO``) at l/x only. Getting m wrong in either direction changes the
    total, so the m rebuild is what separates them.
    """
    width, depth, max_channels, _, _ = _PUBLISHED[scale]
    saved = (
        module.WIDTH_MULT,
        module.DEPTH_MULT,
        module.MAX_CHANNELS,
        module.C3K2_FORCE_C3K,
        module.A2C2F_RESIDUAL,
        module.MLP_RATIO,
    )
    try:
        module.WIDTH_MULT = width
        module.DEPTH_MULT = depth
        module.MAX_CHANNELS = max_channels
        module.C3K2_FORCE_C3K = scale in _C3K_FORCED_SCALES
        module.A2C2F_RESIDUAL = scale in _A2C2F_RESIDUAL_SCALES
        module.MLP_RATIO = (
            _MLP_RATIO_RESIDUAL
            if scale in _A2C2F_RESIDUAL_SCALES
            else _MLP_RATIO_DEFAULT
        )
        yield
    finally:
        (
            module.WIDTH_MULT,
            module.DEPTH_MULT,
            module.MAX_CHANNELS,
            module.C3K2_FORCE_C3K,
            module.A2C2F_RESIDUAL,
            module.MLP_RATIO,
        ) = saved


_NECK_STAGE_NAMES = ("td_p4", "td_p3", "bu_p4", "bu_p5")


def _kind_of(stage) -> str:
    """``"a2c2f"`` or ``"c3k2"`` for a built stage, by class."""
    return {"A2C2f": "a2c2f", "C3k2": "c3k2"}.get(
        type(stage).__name__, type(stage).__name__
    )


def _backbone_kinds(model):
    return tuple(_kind_of(stage) for stage in model.backbone.stages)


def _neck_stages(model):
    return tuple(getattr(model.neck, name) for name in _NECK_STAGE_NAMES)


def _neck_kinds(model):
    return tuple(_kind_of(stage) for stage in _neck_stages(model))


def _backbone_stage_entries(model):
    return tuple(len(stage.m) for stage in model.backbone.stages)


def _neck_stage_entries(model):
    return tuple(len(stage.m) for stage in _neck_stages(model))


def _attention_stages(model):
    """The backbone stages that are ``A2C2f`` with ``a2=True``, in order."""
    return tuple(
        stage
        for stage in model.backbone.stages
        if _kind_of(stage) == "a2c2f" and bool(stage.a2)
    )


def _ablocks(model):
    return tuple(
        sub for sub in model.modules() if type(sub).__name__ == "ABlock"
    )


def _area_attentions(model):
    return tuple(
        sub for sub in model.modules() if type(sub).__name__ == "AreaAttention"
    )


def guard_matches_the_published_architecture(module) -> None:
    """Every yaml layer's parameter count equals the re-derived published one.

    Compared PER LAYER, not only in total, because a total can be hit by two
    errors that cancel — measured on `yolov10_s`, where exactly that happened.
    Sixteen rows, at the template's own class count and at the published 80, and
    the reference is arithmetic on the yaml with nothing from ``model_zoo/``
    imported (see ``_reference_yolov12_layers``).

    It is also where the ABSENCE of YOLO11's SPPF and C2PSA is priced in: the
    stride-32 rows are ``A2C2f`` alone, so carrying either module across shows
    up here as an unmapped module in the tree rather than as a total that
    happens to be large.
    """
    model = _build(module, _PUBLISHED_CLASSES - 1)
    named = dict(model.named_modules())

    missing = [path for path in _LAYER_PATHS.values() if path not in named]
    assert not missing, (
        f"{module.__name__}: the per-layer map names module path(s) {missing} "
        f"that do not exist in the built model. The map is how a yaml row is "
        f"compared against this template at all; fix it before trusting any "
        f"count below."
    )

    expected = _reference_yolov12_layers(_PUBLISHED_CLASSES)
    assert set(expected) == set(_LAYER_PATHS), (
        f"the reference derives layers {sorted(expected)} but the module map "
        f"covers {sorted(_LAYER_PATHS)}"
    )

    mismatches = {}
    for index, path in sorted(_LAYER_PATHS.items()):
        built = sum(p.numel() for p in named[path].parameters())
        if built != expected[index]:
            mismatches[index] = (path, built, expected[index])
    assert not mismatches, (
        f"{module.__name__}: per-layer parameter counts disagree with the "
        f"published YOLOv12-S spec:\n"
        + "\n".join(
            f"  yaml {index:2d} {_LAYER_LABELS[index]:34s} {path:26s} "
            f"built {built:>10,} published {want:>10,} ({built - want:+,})"
            for index, (path, built, want) in sorted(mismatches.items())
        )
    )

    total = sum(p.numel() for p in model.parameters())
    assert total == _PUBLISHED[_SHIPPED_SCALE][4], (
        f"{module.__name__}: {total:,} parameters at {_PUBLISHED_CLASSES} "
        f"classes, but yolo12.yaml's s row publishes "
        f"{_PUBLISHED[_SHIPPED_SCALE][4]:,} gradients "
        f"(off by {total - _PUBLISHED[_SHIPPED_SCALE][4]:+,}). Every layer "
        f"matched, so the difference is a module this map does not cover — "
        f"most likely something carried across from yolo11_s.py that YOLOv12 "
        f"does not have."
    )

    # Structure, at the template's own class count: what drifted, not by how
    # much.
    structure = {
        "backbone_out": tuple(int(c) for c in model.backbone.out_channels),
        "neck_out": tuple(int(c) for c in model.neck.out_channels),
        "backbone_stage_entries": _backbone_stage_entries(model),
        "neck_stage_entries": _neck_stage_entries(model),
        "backbone_kinds": _backbone_kinds(model),
        "neck_kinds": _neck_kinds(model),
        "backbone_areas": tuple(int(s.area) for s in _attention_stages(model)),
        "backbone_c3k": tuple(
            bool(s.c3k) for s in model.backbone.stages if _kind_of(s) == "c3k2"
        ),
        "neck_c3k": tuple(
            bool(s.c3k) for s in _neck_stages(model) if _kind_of(s) == "c3k2"
        ),
        "neck_a2": tuple(
            bool(s.a2) for s in _neck_stages(model) if _kind_of(s) == "a2c2f"
        ),
        "ablocks_per_entry": len(_attention_stages(model)[0].m[0]),
        "attention_head_dim": _area_attentions(model)[0].head_dim,
        "box_hidden": int(model.head.box_hidden),
        "cls_hidden": int(model.head.cls_hidden),
        "strides": tuple(int(s) for s in model.head.strides),
        "reg_max": int(model.head.reg_max),
    }
    differences = {
        key: (value, _REFERENCE_STRUCTURE[key])
        for key, value in structure.items()
        if value != _REFERENCE_STRUCTURE[key]
    }
    assert not differences, (
        f"{module.__name__}: structure disagrees with the published spec:\n"
        + "\n".join(
            f"  {key}: built {got!r}, published {want!r}"
            for key, (got, want) in sorted(differences.items())
        )
    )


def guard_block_kinds_match_the_yaml(module) -> None:
    """The kind of block at each stage is the yaml's, and it is not a pattern.

    Four independent facts, none derivable from the others:

    * the backbone is ``C3k2, C3k2, A2C2f, A2C2f`` — the substitution that IS
      YOLOv12;
    * the neck is ``A2C2f, A2C2f, A2C2f, C3k2`` — three of one and one of the
      other, with no rule behind it;
    * every neck ``A2C2f`` is ``a2=False``, so **the neck holds no attention**;
    * ``area`` is 4 on the stride-16 attention stage and 1 on stride-32.

    ``guard_matches_the_published_architecture`` sees most of this by arithmetic,
    but not all of it: an ``A2C2f`` in the neck built ``a2=False`` with the
    right widths is the same count whichever ``area`` it is handed, and the
    ``area`` values on the backbone stages are parameter-invariant outright. So
    this guard reads them off the built modules.
    """
    model = _build(module, 4)

    assert _backbone_kinds(model) == _REFERENCE_STRUCTURE["backbone_kinds"], (
        f"{module.__name__}: backbone stage kinds are "
        f"{_backbone_kinds(model)}, but yolo12.yaml is "
        f"{_REFERENCE_STRUCTURE['backbone_kinds']} — C3k2 on the two shallow "
        f"stages and A2C2f on the two deep ones. Substituting A2C2f for C3k2 on "
        f"the deep stages is the entire architectural change at this "
        f"generation."
    )
    assert _neck_kinds(model) == _REFERENCE_STRUCTURE["neck_kinds"], (
        f"{module.__name__}: neck fusion kinds are {_neck_kinds(model)}, but "
        f"yolo12.yaml is {_REFERENCE_STRUCTURE['neck_kinds']}: three A2C2f and "
        f"then a C3k2 on the bottom-up stride-32 fusion. There is no pattern "
        f"behind that — it is transcription."
    )

    neck_a2 = tuple(
        bool(s.a2) for s in _neck_stages(model) if _kind_of(s) == "a2c2f"
    )
    assert neck_a2 == _REFERENCE_STRUCTURE["neck_a2"], (
        f"{module.__name__}: the neck's A2C2f fusions report a2={neck_a2}, "
        f"expected {_REFERENCE_STRUCTURE['neck_a2']}. YOLOv12 is "
        f"attention-centric in its BACKBONE; every neck row is "
        f"`A2C2f, [w, False, -1]` and the False is a2. An attention neck is a "
        f"different model that trains."
    )

    areas = tuple(int(s.area) for s in _attention_stages(model))
    assert areas == _REFERENCE_STRUCTURE["backbone_areas"], (
        f"{module.__name__}: the attention stages use areas {areas}, but "
        f"yolo12.yaml sets 4 on the stride-16 stage and 1 on stride-32. `area` "
        f"changes NO parameter and NO shape — it decides how much of the map "
        f"each token may attend to, which is the paper's contribution."
    )

    c3k_flags = tuple(
        bool(s.c3k) for s in model.backbone.stages if _kind_of(s) == "c3k2"
    ) + tuple(bool(s.c3k) for s in _neck_stages(model) if _kind_of(s) == "c3k2")
    expected_c3k = (
        _REFERENCE_STRUCTURE["backbone_c3k"] + _REFERENCE_STRUCTURE["neck_c3k"]
    )
    assert c3k_flags == expected_c3k, (
        f"{module.__name__}: the C3k2 c3k flags are {c3k_flags}, expected "
        f"{expected_c3k} at this scale. The yaml's False literals are the c3k "
        f"flag, and only the neck's stride-32 fusion is True at n/s."
    )


def guard_no_pooling_or_partial_attention_in_the_tree(module) -> None:
    """YOLOv12 has NO ``SPPF`` and NO ``C2PSA``. Stated positively.

    ⚠️ A REMOVAL IS THE THING A PORT NEVER PERFORMS. ``yolo11_s.py`` is the
    closest sibling and its backbone ends ``C3k2 -> SPPF -> C2PSA``;
    ``yolo12.yaml`` contains neither module anywhere. A parameter total only
    disagrees once the expectation is already right, so the removal is asserted
    here rather than left implicit in arithmetic — and it is asserted on the
    MODULE TREE, so a pooling layer that is constructed and never called is
    caught too (that is exactly the shape a half-finished port leaves behind).

    What this does NOT prove: that nothing else was carried across. It names the
    two modules YOLOv12 deletes, by type; the per-layer count is what covers
    everything else.
    """
    from torch import nn

    model = _build(module, 3)
    pooling = sorted(
        f"{name or '<root>'}: {type(sub).__name__}"
        for name, sub in model.named_modules()
        if isinstance(
            sub,
            (
                nn.modules.pooling._MaxPoolNd,
                nn.modules.pooling._AvgPoolNd,
                nn.modules.pooling._AdaptiveAvgPoolNd,
                nn.modules.pooling._AdaptiveMaxPoolNd,
            ),
        )
    )
    assert not pooling, (
        f"{module.__name__}: the tree contains pooling layer(s) {pooling}. "
        f"YOLOv12 has no SPPF — yolo12.yaml's backbone ends at layer 8's "
        f"A2C2f, with no spatial-pyramid pooling anywhere. A MaxPool here is "
        f"YOLO11's SPPF carried across, or the remains of one."
    )

    forbidden = {"SPPF", "SPP", "C2PSA", "C2fPSA", "PSABlock", "PSA"}
    present = sorted(
        f"{name or '<root>'}: {type(sub).__name__}"
        for name, sub in model.named_modules()
        if type(sub).__name__ in forbidden
    )
    assert not present, (
        f"{module.__name__}: the tree contains {present}. YOLOv12's attention "
        f"is Area Attention inside the backbone's A2C2f stages, not a C2PSA "
        f"block bolted onto the end of it, and there is no SPPF. Both are "
        f"YOLO11 modules."
    )
    assert not any(
        name in ("SPPF", "C2PSA", "PSABlock") for name in vars(module)
    ), (
        f"{module.__name__}: the template still DEFINES one of SPPF / C2PSA / "
        f"PSABlock. Even unused, a defined-and-unbuilt YOLO11 block is a port "
        f"that stopped halfway, and the next edit reconnects it."
    )


def guard_arch_table_is_live(module) -> None:
    """The arch table, both multipliers and BOTH per-scale overrides must be
    live knobs, not decoration.

    ``published_architecture`` pins the shipped scale. That is not enough on its
    own: a width hardcoded at the shipped value, or a head width formula
    simplified to what this build happens to need, satisfies it exactly. So this
    guard rebuilds at the **four other published scales** and asserts each
    total, which is four independent numbers from outside this repo.

    Each of the four is load-bearing:

    * **N** (width 0.25) separates a hardcoded channel width, and is the only
      scale where the class tower's ``max(ch[0], min(nc, 100))`` is won by the
      CLASS term rather than by ``ch[0]`` — so a formula flattened to
      ``in_channels[0]`` is right at s and wrong here.
    * **M** (width 1.00, max_channels 512) separates the channel CAP, which
      does not bind at n or s where it is 1024 — and separates the TWO
      overrides from each other, because m takes the ``C3k2`` one and not the
      ``A2C2f`` one.
    * **L** (depth 1.00) doubles the ``A2C2f`` entry count and turns the
      residual/1.2-MLP regime on.
    * **X** (width 1.50) separates a width multiplier hardcoded at 1.00, which
      m and l both satisfy.

    ⚠️ AND HERE IS WHAT NONE OF THE FIVE SCALES SEPARATES, stated because a
    sibling asserts the opposite and is wrong: **the rounding DIRECTION in
    ``_round_channels`` is unobservable from the published figures.** Every
    full-width figure ``yolo12.yaml`` uses is 64/128/256/512/1024, and every one
    of them times every published width multiplier (0.25 / 0.50 / 1.00 / 1.50,
    under the 1024 and 512 caps) is already an exact multiple of 8 — so
    ``math.ceil`` and ``round`` agree at all five scales, on every row.
    Measured: replacing the ``ceil`` with ``round`` survives this guard AND the
    per-layer count. ``yolo11_s.py``'s ``guard_arch_table_is_live`` docstring
    claims its X rebuild separates exactly this, and the same mutation survives
    there too — measured against that file, not inferred. The ``ceil`` here is
    faithful transcription of ``make_divisible``; it is NOT a checked property,
    and there is deliberately no mutation for it rather than a mutation that
    would report a false survivor.

    It also gives both override constants their only reader. The template ships
    one scale, so nothing in it consumes ``C3K_AT_SCALE_OVERRIDE_SCALES`` or
    ``A2C2F_RESIDUAL_SCALES``; the scales this guard has to switch each override
    on for, to reach the published totals, ARE those sets, so comparing the two
    makes the declarations falsifiable rather than documentary.
    """
    assert tuple(module.C3K_AT_SCALE_OVERRIDE_SCALES) == _C3K_FORCED_SCALES, (
        f"{module.__name__}: C3K_AT_SCALE_OVERRIDE_SCALES declares "
        f"{tuple(module.C3K_AT_SCALE_OVERRIDE_SCALES)}, but the scales this "
        f"guard must force every C3k2's c3k flag on for, to reach their "
        f"published totals, are {_C3K_FORCED_SCALES}. That constant transcribes "
        f"`parse_model`'s `if m is C3k2 and scale in \"mlx\"`, and this is the "
        f"only thing that reads it — a wrong list is a wrong claim about "
        f"upstream sitting in a shipped template."
    )
    assert tuple(module.A2C2F_RESIDUAL_SCALES) == _A2C2F_RESIDUAL_SCALES, (
        f"{module.__name__}: A2C2F_RESIDUAL_SCALES declares "
        f"{tuple(module.A2C2F_RESIDUAL_SCALES)}, but the scales that need "
        f"`residual=True` and the 1.2 MLP to reach their published totals are "
        f"{_A2C2F_RESIDUAL_SCALES}. NOTE THE DIFFERENCE FROM THE OTHER "
        f"OVERRIDE: this one is {{l, x}} and the C3k2 one is mlx, so 'm' is the "
        f"scale that tells them apart."
    )
    assert module.MLP_RATIO_AT_RESIDUAL_SCALES == _MLP_RATIO_RESIDUAL, (
        f"{module.__name__}: MLP_RATIO_AT_RESIDUAL_SCALES is "
        f"{module.MLP_RATIO_AT_RESIDUAL_SCALES}, not the {_MLP_RATIO_RESIDUAL} "
        f"`parse_model` extends A2C2f's args with at l/x"
    )
    assert module.MLP_RATIO == _MLP_RATIO_DEFAULT, (
        f"{module.__name__}: the shipped MLP_RATIO is {module.MLP_RATIO}, not "
        f"A2C2f's own {_MLP_RATIO_DEFAULT} default — s takes neither override"
    )
    assert module.A2C2F_RESIDUAL is False and module.C3K2_FORCE_C3K is False, (
        f"{module.__name__}: the shipped scale is s, which takes NEITHER "
        f"override, but A2C2F_RESIDUAL={module.A2C2F_RESIDUAL} and "
        f"C3K2_FORCE_C3K={module.C3K2_FORCE_C3K}"
    )

    for scale in sorted(set(_PUBLISHED) - {_SHIPPED_SCALE}):
        gradients = _PUBLISHED[scale][4]
        with _at_published_scale(module, scale):
            try:
                model = _build(module, _PUBLISHED_CLASSES - 1)
            except Exception as error:  # noqa: BLE001 — any failure is the bug
                raise AssertionError(
                    f"{module.__name__}: rebuilding at the published YOLOv12-"
                    f"{scale.upper()} scale failed with "
                    f"{type(error).__name__}: {error}. The multipliers and the "
                    f"arch table are live knobs — it is how a scale is "
                    f"selected — so every published scale must construct."
                ) from error
            total = sum(p.numel() for p in model.parameters())
            named = dict(model.named_modules())
            per_layer = {
                index: sum(p.numel() for p in named[path].parameters())
                for index, path in _LAYER_PATHS.items()
            }
        assert total == gradients, (
            f"{module.__name__}: rebuilt at the published YOLOv12-"
            f"{scale.upper()} scale the model has {total:,} parameters, but "
            f"that scale's published summary reports {gradients:,} gradients "
            f"(off by {total - gradients:+,}). Something the shipped scale "
            f"does not exercise is hardcoded."
        )
        expected = _PUBLISHED_LAYERS[scale]
        wrong = {
            index: (per_layer[index], expected[index])
            for index in sorted(expected)
            if per_layer[index] != expected[index]
        }
        assert not wrong, (
            f"{module.__name__}: at YOLOv12-{scale.upper()} the total matches "
            f"but individual layers do not — two errors cancelling, which is "
            f"the exact failure a total cannot see:\n"
            + "\n".join(
                f"  yaml {index:2d} {_LAYER_LABELS[index]:34s} built "
                f"{got:>10,} published {want:>10,} ({got - want:+,})"
                for index, (got, want) in wrong.items()
            )
        )


#: Buffer and tensor totals measured off this repo's own build, as a cheap
#: regression tripwire.
#:
#: ⚠️ SELF-MEASURED. They prove the code is consistent with itself and nothing
#: more — see the block comment above ``_reference_yolov12_layers`` for the
#: sibling template where exactly such a number was cited as evidence and was
#: wrong. Parameters are checked per yaml layer against the re-derived published
#: spec at all five scales; this row only covers what that cannot.
#:
#: Updating these is legitimate when the architecture changes on purpose; state
#: the intended change in the commit message.
_PINNED_TOTALS = {"buffers": 0, "tensors": 351}


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
    BatchNorm here leaves ``published_architecture`` completely green — measured
    on ``yolo11_s`` and re-measured here. The difference is entirely in the
    BUFFERS, which is why this is a separate guard rather than a comment.

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
        f"{name}: {type(sub).__name__}"
        for name, sub in model.named_modules()
        if isinstance(sub, _BatchNorm)
    )
    assert not stateful, (
        f"{module.__name__}: BatchNorm-family layer(s) {stateful}. Their "
        f"running statistics are buffers the averaging service ships and "
        f"averages every round, and they average badly across non-IID clients. "
        f"Use GroupNorm (see CLAUDE.md). Note the PARAMETER COUNT DOES NOT "
        f"MOVE when you do this, so the published-architecture guard stays "
        f"green — this is the only thing that catches it."
    )
    buffers = sum(b.numel() for b in model.buffers())
    assert buffers == 0, (
        f"{module.__name__}: the model carries {buffers} buffer elements. "
        f"Every one is shipped and averaged each federated round; this "
        f"architecture needs none, including the DFL bin vector, which is "
        f"built with torch.arange inside the decode."
    )
    norms = [sub for sub in model.modules() if isinstance(sub, nn.GroupNorm)]
    assert norms, (
        f"{module.__name__}: no GroupNorm at all. A trunk with no "
        f"normalisation trains to finite loss with activations an order of "
        f"magnitude out (backend#3093), so 'no BatchNorm' has to be paired "
        f"with 'and something normalises'."
    )


def guard_norm_groups_are_derived_from_the_channel_count(module) -> None:
    """``_norm_groups`` must derive the group count, not assume 32.

    ⚠️ AND IT BITES AT THE SHIPPED SCALE. The shallowest ``C3k2`` runs at
    ``expansion = 0.25`` and its inner bottleneck squeezes again by half, so the
    shipped tree already contains a **16-channel** norm and ``GroupNorm(32, 16)``
    raises outright.

    It bites a second, independent way at l/x, and that one is specific to
    YOLOv12: the Area-Attention MLP width there is ``int(256 * 1.2) == 307``, a
    PRIME, so no group count above 1 divides it. A derivation that reached for
    "the largest power of two" rather than "the largest divisor" would build at
    every other scale and fail there.

    Construction is wrapped, because the failure this guard is written against
    is a ``ValueError`` at build time rather than a wrong number.
    """
    import torch
    from torch import nn

    assert module._norm_groups(16) == 16, "16 channels cannot take 32 groups"
    assert module._norm_groups(48) == 24, "48 takes 24 groups, not 32"
    assert module._norm_groups(307) == 1, (
        "307 is prime, so the only legal group count is 1 — this is the width "
        "an Area-Attention MLP takes at the l/x scales"
    )
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
                f"count crashes here, which is why the count is derived: the "
                f"shallow C3k2 stages run at expansion 0.25 and their inner "
                f"bottleneck halves that again, so this tree carries "
                f"16-channel norms at the SHIPPED width — and 307-channel ones "
                f"at l/x."
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

    for scale in ("n", "l"):
        with _at_published_scale(module, scale):
            model = build_or_explain(f"the published YOLOv12-{scale.upper()}", 3)
            pairs = {
                (sub.num_groups, sub.num_channels)
                for sub in model.modules()
                if isinstance(sub, nn.GroupNorm)
            }
            assert pairs, f"no GroupNorm at the {scale} scale — nothing checked"
            for groups, channels in sorted(pairs):
                assert channels % groups == 0, (
                    f"{module.__name__}: GroupNorm({groups}, {channels}) at the "
                    f"YOLOv12-{scale.upper()} scale does not divide"
                )
        if scale == "n":
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


def _capture_calls(module_under_test):
    """``(handle, calls)`` — a forward hook recording that a module RAN, with
    the tensors it saw. The only way to prove a constructed module is applied."""
    calls = []

    def record(_module, inputs, output):
        calls.append((inputs[0].detach().clone(), output.detach().clone()))

    return module_under_test.register_forward_hook(record), calls


def guard_the_deepest_output_is_the_last_stages_own_output(module) -> None:
    """The stride-32 map the neck receives must be stage 3's OUTPUT.

    ⚠️ THIS IS THE GUARD THE PARAMETER COUNT CANNOT REPLACE, and it is measured
    rather than argued. The stride-32 ``A2C2f`` maps 512 -> 512 at this scale
    and its downsample maps 256 -> 512, so returning
    ``self.downsamples[3](outputs[2])`` instead of ``outputs[3]`` is
    shape-identical, stride-identical and leaves the 2.69M-parameter stage
    constructed: the parameter total, the buffer total, the ``state_dict`` keys
    and the loss keys are all unchanged and the model trains. That is a third of
    the network shipped to every edge and averaged every round for nothing.

    The neck uses the stride-32 map TWICE (the yaml's layer 19 concatenates with
    layer 8 again), so the check follows both consumers: the top-down upsample
    and the bottom-up fusion must both see the same tensor the stage produced.
    """
    import torch

    model = _build(module, 3)
    model.eval()
    stage = model.backbone.stages[3]
    handle, calls = _capture_calls(stage)
    try:
        with torch.no_grad():
            features = model.backbone(torch.rand(1, 3, 128, 128))
    finally:
        handle.remove()

    assert len(calls) == 1, (
        f"{module.__name__}: the deepest backbone stage ran {len(calls)} times "
        f"in one backbone forward, expected exactly 1"
    )
    produced = calls[0][1]
    assert features[2].shape == produced.shape, (
        f"{module.__name__}: the backbone's stride-32 output is "
        f"{tuple(features[2].shape)} but its deepest stage produced "
        f"{tuple(produced.shape)}"
    )
    assert torch.equal(features[2], produced), (
        f"{module.__name__}: the backbone's stride-32 output is NOT the "
        f"deepest stage's output. The stage is still constructed, so the "
        f"parameter count, the state_dict keys and the losses are all "
        f"unchanged and the model trains — the whole attention stage is simply "
        f"bypassed. Its downsample emits the same width at the same stride, "
        f"which is what makes the substitution shape-clean."
    )


def guard_a2c2f_is_relan_not_a_c2f(module) -> None:
    """``A2C2f``'s convolutions must be R-ELAN's, not ``C2f``'s.

    ⚠️ THE SINGLE MOST EXPENSIVE THING TO GET WRONG IN THIS FILE, and the
    YOLOv12 analogue of "``C3k2`` is not ``C2f`` renamed". Two facts:

    * ``cv1`` emits ``hidden``, NOT ``2 * hidden``. There is no split.
    * ``cv2`` takes ``(1 + n) * hidden``, NOT ``(2 + n) * hidden``. The stem
      output plus one branch per entry — there is no split half to fuse.

    Both are asserted on the constructed convolutions BEFORE any forward,
    deliberately: a wrong fusion width raises a ``RuntimeError`` deep inside a
    ``cat``, which is a worse failure report than naming the rule.

    Then the branch list ``cv2`` is actually handed is compared tensor-for-
    tensor against the hand-computed one, at ``blocks=2`` as well as 1, because
    the branch ORDER is invisible to any channel count — every branch is the
    same width.
    """
    import torch

    for blocks in (1, 2):
        for a2 in (True, False):
            stage = module.A2C2f(64, 64, blocks=blocks, a2=a2, area=1)
            stage.eval()
            hidden = stage.hidden
            assert stage.cv1.conv.out_channels == hidden, (
                f"A2C2f(a2={a2}, blocks={blocks}): cv1 emits "
                f"{stage.cv1.conv.out_channels} channels at hidden width "
                f"{hidden}. R-ELAN reduces to hidden and does NOT split; "
                f"2 * hidden is C2f/C3k2's skeleton, which is a different "
                f"block with a different parameter count."
            )
            assert stage.cv2.conv.in_channels == (1 + blocks) * hidden, (
                f"A2C2f(a2={a2}, blocks={blocks}): the fusion conv takes "
                f"{stage.cv2.conv.in_channels} channels; R-ELAN fuses "
                f"(1 + n) * hidden = {(1 + blocks) * hidden} — the cv1 output "
                f"plus one branch per entry. (2 + n) is C2f, which has a split "
                f"half to contribute."
            )
            assert len(stage.m) == blocks, (
                f"A2C2f built {len(stage.m)} entries for blocks={blocks}"
            )

            probe = torch.rand(1, 64, 8, 8) + 0.5
            handle, captured = _capture_input(stage.cv2)
            try:
                with torch.no_grad():
                    stage(probe)
                    branches = [stage.cv1(probe)]
                    for entry in stage.m:
                        branches.append(entry(branches[-1]))
                    expected = torch.cat(branches, dim=1)
                    reversed_order = torch.cat(list(reversed(branches)), dim=1)
            finally:
                handle.remove()

            assert len(captured) == 1, "the fusion conv's pre-hook did not fire"
            assert not torch.allclose(expected, reversed_order, atol=1e-5), (
                f"fixture is degenerate at blocks={blocks}, a2={a2}: the branch "
                f"list reads the same reversed, so its order cannot be checked"
            )
            assert captured[0].shape == expected.shape, (
                f"A2C2f(a2={a2}) hands its fusion conv "
                f"{tuple(captured[0].shape)}, expected "
                f"{tuple(expected.shape)}"
            )
            assert torch.allclose(captured[0], expected, atol=1e-5), (
                f"{module.__name__}: A2C2f(a2={a2}, blocks={blocks}) is not "
                f"handing its fusion conv [cv1(x), m0(cv1(x)), ...]. The "
                f"channel count is right and the model trains, so the likely "
                f"shape is a reversed branch list or entries reading the wrong "
                f"predecessor."
            )


def guard_a2c2f_chains_its_entries(module) -> None:
    """Each ``A2C2f`` entry must take the PREVIOUS entry's output.

    ELAN's depth comes from the chain: entry ``i`` sees everything entries
    ``0..i-1`` did, and the fusion sees every stage of that chain. Feeding every
    entry ``cv1``'s output instead — a one-character edit — gives a wide,
    shallow block with **identical shapes, identical parameters, identical keys
    and a finite loss**, so nothing structural can see it.

    Checked at ``blocks=2``, which is the shipped depth for the attention
    stages and the smallest depth at which chaining and fanning-out differ at
    all: at ``blocks=1`` they are the same graph, which is why the neck's
    fusions cannot carry this guard.
    """
    import torch

    stage = module.A2C2f(64, 64, blocks=2, a2=False)
    stage.eval()
    assert len(stage.m) == 2, "fixture: this guard needs two entries to differ"

    probe = torch.rand(1, 64, 8, 8) + 0.5
    handle, captured = _capture_input(stage.m[1])
    try:
        with torch.no_grad():
            stage(probe)
            stem = stage.cv1(probe)
            chained = stage.m[0](stem)
    finally:
        handle.remove()

    assert len(captured) == 1, "the second entry's pre-hook did not fire once"
    assert not torch.allclose(stem, chained, atol=1e-5), (
        "fixture is degenerate: the first entry is the identity on this probe, "
        "so chaining and fanning-out agree and the rule cannot fire"
    )
    assert torch.allclose(captured[0], chained, atol=1e-5), (
        f"{module.__name__}: A2C2f's second entry is not receiving the first "
        f"entry's output. Every entry reading cv1's output instead is a wide "
        f"block where the design is a deep one, with every shape, every "
        f"parameter and every state_dict key identical."
    )


def guard_a2c2f_entry_is_a_fixed_pair_of_ablocks(module) -> None:
    """An attention entry holds exactly ``A2C2F_BLOCKS_PER_ENTRY`` ``ABlock``s,
    and that count is NOT depth-scaled.

    Upstream writes the pair as a literal — ``for _ in range(2)`` — while the
    yaml's repeat count (already depth-scaled) decides how many entries there
    are. Confusing the two halves the attention in the backbone and takes a
    branch off the fusion, so it is expensive; but the pair size itself is the
    part that looks like a depth and is not, which is why it is asserted
    against a rebuild at a DIFFERENT depth.
    """
    assert module.A2C2F_BLOCKS_PER_ENTRY == 2, (
        f"{module.__name__}: A2C2F_BLOCKS_PER_ENTRY is "
        f"{module.A2C2F_BLOCKS_PER_ENTRY}; upstream's literal is 2"
    )

    model = _build(module, 3)
    stages = _attention_stages(model)
    assert stages, "no attention stages built — nothing checked"
    for index, stage in enumerate(stages):
        for entry_index, entry in enumerate(stage.m):
            names = [type(sub).__name__ for sub in entry]
            assert names == ["ABlock"] * module.A2C2F_BLOCKS_PER_ENTRY, (
                f"{module.__name__}: attention stage {index} entry "
                f"{entry_index} holds {names}, expected "
                f"{module.A2C2F_BLOCKS_PER_ENTRY} ABlocks. The yaml's repeat "
                f"count is the number of ENTRIES; each entry is a fixed pair."
            )

    shipped_entries = tuple(len(stage.m) for stage in stages)
    with _at_published_scale(module, "l"):
        deep = _build(module, 3)
        deep_stages = _attention_stages(deep)
        deep_entries = tuple(len(stage.m) for stage in deep_stages)
        pair_sizes = {len(entry) for stage in deep_stages for entry in stage.m}
    assert deep_entries != shipped_entries, (
        f"fixture is degenerate: the l scale builds the same entry counts "
        f"{shipped_entries} as s, so this rebuild cannot separate the entry "
        f"count from the pair size"
    )
    assert pair_sizes == {module.A2C2F_BLOCKS_PER_ENTRY}, (
        f"{module.__name__}: at the l scale the entries hold {pair_sizes} "
        f"ABlocks each. The pair size is a literal and must NOT move with the "
        f"depth multiplier — only the number of entries does "
        f"({shipped_entries} -> {deep_entries})."
    )


def guard_a2c2f_layer_scale_is_off_here_and_applied_when_on(module) -> None:
    """``gamma`` must be absent at the shipped scale and APPLIED when present.

    Two halves, and the second is the one a count cannot reach:

    * at s there is no ``gamma`` at all, because ``parse_model`` only extends
      ``residual=True`` at l/x. A ``gamma`` here would be 768 extra parameters
      and the published count would catch it — so that half is cheap.
    * with ``residual`` on, ``gamma`` is 256 or 512 numbers out of 26 million.
      Constructing it and never applying it changes no shape, no key set and no
      loss, and moves the total by 0.002%. So the residual is checked
      FUNCTIONALLY against a hand-computed ``x + gamma * cv2(cat(...))``.
    """
    import torch

    shipped = _build(module, 3)
    for index, stage in enumerate(_attention_stages(shipped)):
        assert stage.gamma is None, (
            f"{module.__name__}: attention stage {index} carries a gamma at the "
            f"shipped s scale. `parse_model` extends A2C2f's args with "
            f"(True, 1.2) only for l/x; at s the block is not residual."
        )

    stage = module.A2C2f(64, 64, blocks=2, a2=True, area=1, residual=True)
    stage.eval()
    assert stage.gamma is not None, (
        "A2C2f(residual=True) built no gamma, so the l/x regime is unreachable"
    )
    assert stage.gamma.requires_grad, "gamma must be a trained parameter"
    assert tuple(stage.gamma.shape) == (64,), (
        f"gamma is {tuple(stage.gamma.shape)}, expected one number per output "
        f"channel"
    )
    assert float(stage.gamma.detach()[0]) == pytest.approx(
        module.LAYER_SCALE_INIT
    ), (
        f"gamma initialises at {float(stage.gamma.detach()[0])}, not the published "
        f"{module.LAYER_SCALE_INIT}"
    )

    probe = torch.rand(1, 64, 8, 8) + 0.5
    with torch.no_grad():
        got = stage(probe)
        branches = [stage.cv1(probe)]
        for entry in stage.m:
            branches.append(entry(branches[-1]))
        fused = stage.cv2(torch.cat(branches, dim=1))
        expected = probe + stage.gamma.view(1, -1, 1, 1) * fused
    assert not torch.allclose(fused, expected, atol=1e-5), (
        "fixture is degenerate: the residual add changes nothing measurable"
    )
    assert torch.allclose(got, expected, atol=1e-5), (
        f"{module.__name__}: a residual A2C2f does not compute "
        f"x + gamma * cv2(cat(branches)). gamma is 256 or 512 numbers out of "
        f"26 million at the l scale, so constructing it and dropping it from "
        f"the forward is invisible to the parameter count, the shapes, the "
        f"keys and the loss."
    )


def guard_the_neck_holds_no_attention(module) -> None:
    """No ``AreaAttention`` and no ``ABlock`` anywhere in the neck.

    "Attention-centric" invites the opposite assumption, and the yaml is
    explicit: every neck ``A2C2f`` row is ``[w, False, -1]``, the ``False``
    being ``a2``. Attention lives in the two deep BACKBONE stages only.

    The parameter count does see this — an attention neck is a different total —
    but only once the expectation is right; asserted directly so a reviewer
    reading the neck does not have to derive it from arithmetic.
    """
    model = _build(module, 3)
    stray = sorted(
        f"{name}: {type(sub).__name__}"
        for name, sub in model.neck.named_modules()
        if type(sub).__name__ in {"AreaAttention", "ABlock"}
    )
    assert not stray, (
        f"{module.__name__}: the neck contains attention module(s) {stray}. "
        f"Every neck A2C2f in yolo12.yaml is a2=False — R-ELAN's plumbing "
        f"around a C3k, no attention. The area argument on those rows is -1, "
        f"which is upstream saying 'unused'."
    )
    attention_stages = _attention_stages(model)
    assert len(attention_stages) == 2, (
        f"{module.__name__}: {len(attention_stages)} backbone stages carry "
        f"attention, expected 2 (yaml layers 6 and 8). If this is 0 the neck "
        f"check above is vacuous."
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
            f"{(2 + blocks) * stage.hidden}. A (1 + n) input is an A2C2f, and "
            f"a (2 * hidden) one is a C3."
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
                # The plausible wrong graph, shape-clean.
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

    ``C3k`` is reached by TWO different routes in this template — inside the
    neck's stride-32 ``C3k2`` (``c3k=True``) and inside every ``a2=False``
    ``A2C2f`` — which is more coverage than it had on YOLO11 and worth keeping
    separate, because the two constructions pass different arguments.
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

    # Both construction routes must actually produce a C3k, or the guard above
    # is exercising a class the shipped tree never builds.
    model = _build(module, 3)
    routes = {
        "neck stride-32 C3k2 (yaml 20)": model.neck.bu_p5.m[0],
        "neck top-down P3 A2C2f (yaml 14)": model.neck.td_p3.m[0],
    }
    for label, inner in routes.items():
        assert type(inner).__name__ == "C3k", (
            f"{module.__name__}: {label} holds a {type(inner).__name__}, not a "
            f"C3k, so the C3k guard covers a class the shipped tree does not "
            f"build by that route"
        )


def guard_bottlenecks_keep_their_identity_branch(module) -> None:
    """Every equal-width ``Bottleneck`` in this tree must be RESIDUAL.

    ⚠️ THIS IS A REAL DIFFERENCE FROM ``yolov8.yaml`` AND IT IS EASY TO INVERT.
    YOLOv8's neck ``C2f``s are constructed ``shortcut=False``; every ``C3k2`` in
    ``yolo12.yaml`` is ``shortcut=True``, and so is every ``C3k`` an ``A2C2f``
    builds. The yaml's ``False`` literals are the **c3k** / **a2** flags, not the
    shortcut, and reading them as shortcuts is the natural mistake because that
    is what the same position meant two generations earlier.

    It changes **no parameter and no shape**, only whether the block is
    residual, so nothing structural can see it. Checked on a bottleneck reached
    through ``C3k2``'s plain path and on ones reached through both ``C3k``
    routes, since all three are constructed by different code paths.
    """
    import torch

    model = _build(module, 3)
    probes = {
        "backbone shallow C3k2 plain path (yaml 2)": model.backbone.stages[0].m[0],
        "neck stride-32 C3k2 -> C3k inner (yaml 20)": model.neck.bu_p5.m[0].m[0],
        "neck A2C2f -> C3k inner (yaml 14)": model.neck.td_p3.m[0].m[0],
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
            f"{module.__name__}: {label} is NOT residual. Every C3k2 and every "
            f"C3k in yolo12.yaml is shortcut=True — the yaml's False literals "
            f"are the c3k and a2 flags. This changes no parameter and no shape."
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


# --------------------------------------------------------------------------
# Area Attention — the paper's contribution, and the part of this template
# that no parameter count reaches at all
# --------------------------------------------------------------------------


def _isolated_attention(module, dim, area):
    """An ``AreaAttention`` with every non-attention cross-token path removed.

    ⚠️ WHY THIS FIXTURE HAS TO EXIST, and what it costs. The band-isolation
    property — a token in band 0 cannot see a token in band 3 — is TRUE of the
    attention and FALSE of the module as shipped, because two other things in it
    mix across tokens:

    * ``qkv``'s and ``proj``'s **GroupNorm** normalise over the whole spatial
      extent, so a perturbation anywhere moves every normalised value;
    * ``pe`` is a **7x7** depthwise convolution, which reaches across a band
      boundary by design.

    Neither is a defect and neither is what this guard is about. So the fixture
    replaces the two norms with identities and zeroes ``pe``, leaving the
    attention as the ONLY path from one token to another. Measured, so the claim
    is not hopeful: with these three neutralised, perturbing band 3 moves band 0
    by exactly 0.0 at ``area = 4`` and by ~2.3 at ``area = 1``.

    What it does NOT prove: that the shipped module isolates bands end to end.
    It cannot, and the design does not intend to — ``pe`` deliberately carries
    positional signal across boundaries. The narrower claim is the true one.
    """
    import torch
    from torch import nn

    attention = copy.deepcopy(module.AreaAttention(dim, area=area))
    attention.eval()
    attention.qkv.norm = nn.Identity()
    attention.proj.norm = nn.Identity()
    with torch.no_grad():
        attention.pe.conv.weight.zero_()
        attention.pe.norm.weight.fill_(1.0)
        attention.pe.norm.bias.zero_()
    return attention


def guard_area_attention_is_banded(module) -> None:
    """A token must not attend outside its own area.

    ⚠️ THIS IS THE WHOLE PAPER, AND IT IS INVISIBLE TO EVERYTHING ELSE.
    ``area`` changes no parameter, no shape, no ``state_dict`` key, no loss key
    and no loss value's finiteness. A module that ignores it computes ordinary
    global self-attention and trains perfectly well — YOLOv12 without its
    contribution, at YOLOv12's cost.

    So: perturb the last row of the map, which at ``area = 4`` on an 8x8 map
    lies wholly inside the last band, and require the FIRST band's output to be
    bit-identical. Then do the same at ``area = 1`` and require it to change —
    otherwise the fixture cannot tell the two apart and the guard is theatre.

    Also asserts the bands are CONTIGUOUS, which is the other way to get this
    wrong: an interleaved split (every ``area``-th token) is shape-identical and
    silent, and would leave band 0 holding tokens from every row.
    """
    import torch

    banded = _isolated_attention(module, 64, area=4)
    probe = torch.rand(2, 64, 8, 8) + 0.5
    perturbed = probe.clone()
    perturbed[:, :, 7, :] += 5.0

    with torch.no_grad():
        base = banded(probe)
        moved = banded(perturbed)
    first_band = (base[:, :, 0:2, :] - moved[:, :, 0:2, :]).abs().max()
    last_band = (base[:, :, 6:8, :] - moved[:, :, 6:8, :]).abs().max()

    assert float(last_band) > 1e-3, (
        f"fixture is degenerate: perturbing the last row moved the last band by "
        f"{float(last_band)}, so the probe is not exercising anything"
    )

    global_attention = _isolated_attention(module, 64, area=1)
    global_attention.load_state_dict(banded.state_dict())
    with torch.no_grad():
        global_base = global_attention(probe)
        global_moved = global_attention(perturbed)
    global_first_band = (
        (global_base[:, :, 0:2, :] - global_moved[:, :, 0:2, :]).abs().max()
    )
    assert float(global_first_band) > 1e-3, (
        f"fixture is degenerate: even with area=1 — global attention — the "
        f"first band did not move ({float(global_first_band)}), so this probe "
        f"cannot distinguish banded attention from global attention and the "
        f"assertion below would pass on either"
    )

    assert float(first_band) == 0.0, (
        f"{module.__name__}: with area=4, perturbing a token in the LAST band "
        f"moved the FIRST band's output by {float(first_band)}. Area Attention "
        f"restricts every token to its own band — that is the paper's entire "
        f"contribution and it changes no parameter, no shape and no "
        f"state_dict key, so nothing else in this file can see it. The same "
        f"probe moves the first band by {float(global_first_band)} at area=1, "
        f"which is what a module ignoring `area` computes."
    )

    # Contiguity: band 0 must be the FIRST N/area tokens in row-major order.
    # An interleaved split is shape-identical and would put row 7 in band 0.
    interleaved_probe = probe.clone()
    interleaved_probe[:, :, :, 7] += 5.0  # the last COLUMN touches every band
    with torch.no_grad():
        column_moved = banded(interleaved_probe)
    column_first_band = (
        (base[:, :, 0:2, :] - column_moved[:, :, 0:2, :]).abs().max()
    )
    assert float(column_first_band) > 1e-3, (
        f"{module.__name__}: perturbing the last COLUMN — which crosses every "
        f"band under a contiguous row-major split — did not move the first "
        f"band ({float(column_first_band)}). The bands are not the contiguous "
        f"row-major segments upstream's reshape produces."
    )


def guard_area_divides_the_token_count(module) -> None:
    """Every attention stage's ``area`` must divide its feature map's tokens.

    The band reshape needs ``H * W % area == 0``. It holds here by
    construction — the transform pads to ``size_divisible = 32``, so at stride
    16 both edges are even and the token count is a multiple of 4 — and the
    guard asserts the arithmetic rather than trusting the sentence, at the
    declared ``image_size`` and at the smallest padded edge the transform can
    produce.

    It also pins that the module REFUSES a bad combination with a readable
    error rather than letting a bare ``reshape`` fail somewhere inside the
    attention path.
    """
    import torch

    model = _build(module, 3)
    stages = _attention_stages(model)
    assert stages, "no attention stages built — nothing checked"
    # Attention stages are the stride-16 and stride-32 backbone levels.
    strides = (16, 32)
    assert len(stages) == len(strides), (
        f"fixture: {len(stages)} attention stages against {len(strides)} known "
        f"strides"
    )
    for edge in (module.image_size, 32):
        for stride, stage in zip(strides, stages):
            side = edge // stride
            tokens = side * side
            assert tokens % int(stage.area) == 0, (
                f"{module.__name__}: at a {edge}px input the stride-{stride} "
                f"map is {side}x{side} = {tokens} tokens, which area="
                f"{stage.area} does not divide. The band reshape needs it to. "
                f"The transform pads to a multiple of 32, so this is a claim "
                f"about the AREA values, not about the input."
            )

    attention = module.AreaAttention(64, area=3)
    with pytest.raises(ValueError) as excinfo:
        attention(torch.rand(1, 64, 4, 4))
    assert "area" in str(excinfo.value), (
        f"{module.__name__}: AreaAttention accepted an area that does not "
        f"divide its token count, or failed with an unreadable error: "
        f"{excinfo.value}"
    )


def guard_attention_head_count_is_derived(module) -> None:
    """The head count must be derived from ``ATTENTION_HEAD_DIM``, not fixed.

    ⚠️ AND IT IS PARAMETER-INVARIANT, which is the whole reason this guard
    exists. ``all_head_dim = num_heads * (dim // num_heads)`` equals ``dim`` for
    ANY count that divides ``dim``, so ``qkv``'s output width — and therefore
    every parameter in the module — is identical whatever the head count is. A
    hardcoded ``num_heads = 8`` changes no parameter, no shape, no loss key and
    no published figure. It changes the attention's factorisation and its
    temperature, and nothing that can be counted.

    So the derivation is asserted on the built modules, at two widths, and the
    invariance itself is asserted too — if the invariance ever stops holding,
    delete this guard rather than weakening it, because the published count
    would then cover the head count on its own.
    """
    model = _build(module, 3)
    attentions = _area_attentions(model)
    assert attentions, "no AreaAttention built — nothing checked"

    widths = set()
    for attention in attentions:
        dim = int(attention.dim)
        widths.add(dim)
        expected = max(1, dim // module.ATTENTION_HEAD_DIM)
        assert int(attention.num_heads) == expected, (
            f"{module.__name__}: an AreaAttention at dim={dim} reports "
            f"num_heads={attention.num_heads}; upstream derives "
            f"dim // {module.ATTENTION_HEAD_DIM} = {expected}. A hardcoded "
            f"count changes NO parameter and NO shape here, which is exactly "
            f"why it needs asserting."
        )
        assert int(attention.head_dim) == module.ATTENTION_HEAD_DIM, (
            f"{module.__name__}: head_dim is {attention.head_dim}, not the "
            f"{module.ATTENTION_HEAD_DIM} the derivation implies"
        )
        assert int(attention.all_head_dim) == dim, (
            f"{module.__name__}: all_head_dim is {attention.all_head_dim} at "
            f"dim={dim}; the head count must divide the width exactly"
        )
    assert len(widths) >= 2, (
        f"fixture is degenerate: every AreaAttention in the tree is the same "
        f"width ({widths}), so a hardcoded count could coincide with the "
        f"derivation everywhere"
    )

    # The invariance this guard's existence rests on, asserted rather than
    # assumed. If it fails, the published parameter count now covers the head
    # count and this guard should be DELETED, not loosened.
    for count in (1, 2, 4, 8):
        probe = module.AreaAttention(64, num_heads=count, area=1)
        total = sum(p.numel() for p in probe.parameters())
        reference = module.AreaAttention(64, num_heads=1, area=1)
        assert total == sum(p.numel() for p in reference.parameters()), (
            f"{module.__name__}: AreaAttention's parameter count DOES depend "
            f"on num_heads ({count} gives {total}). This guard's docstring "
            f"says it does not, and the argument for the guard existing at all "
            f"is that invariance. Re-derive both."
        )


def guard_attention_matches_the_reference_operator(module) -> None:
    """SDPA must compute the same thing the published matmul-softmax-matmul does.

    Upstream writes the attention explicitly and holds q/k/v CHANNELS-FIRST;
    this template routes it through ``scaled_dot_product_attention``, which
    needs tokens second-to-last. Feeding the channels-first tensors straight in
    attends over CHANNELS instead of over spatial positions: same output shape,
    finite losses, completely different operator.

    So the reference is the upstream formula — including the band split, the
    ``pe(v)`` addition and the projection — evaluated on the module's own
    weights, and compared tensor-for-tensor. Run at ``area = 4`` and ``area =
    1``, because the band reshape is part of what the operator IS.

    It also covers the ``pe`` residual, which is the second bypassable module
    in this block: ``pe`` is 6,528 parameters at this width, and dropping the
    ``+ self.pe(value)`` leaves it constructed, keyed and averaged.
    """
    import torch

    for area, shape in ((4, (2, 64, 8, 8)), (1, (2, 64, 5, 5))):
        attention = module.AreaAttention(64, area=area)
        attention.eval()
        probe = torch.rand(*shape) + 0.5

        with torch.no_grad():
            got = attention(probe)
            reference = _reference_area_attention(torch, attention, probe)
            # The two wrong operators, both shape-clean.
            no_pe = _reference_area_attention(
                torch, attention, probe, with_pe=False
            )
            channels_first = _reference_area_attention(
                torch, attention, probe, attend_over_channels=True
            )

        assert got.shape == reference.shape
        assert not torch.allclose(reference, no_pe, atol=1e-5), (
            f"fixture is degenerate at area={area}: dropping the positional "
            f"encoding changes nothing measurable"
        )
        assert not torch.allclose(reference, channels_first, atol=1e-5), (
            f"fixture is degenerate at area={area}: attending over channels "
            f"gives the same tensor as attending over positions"
        )
        assert torch.allclose(got, reference, atol=1e-4), (
            f"{module.__name__}: at area={area} the SDPA path does not match "
            f"the published operator (max deviation "
            f"{float((got - reference).abs().max())}). The likely causes are a "
            f"missing layout transpose — which attends over channels — or a "
            f"dropped pe(v) residual. Both keep the output shape and the loss "
            f"finite."
        )


def _reference_area_attention(
    torch, attention, x, with_pe=True, attend_over_channels=False
):
    """Upstream ``AAttn.forward``, transcribed, on ``attention``'s own weights.

    Explicit ``(q^T k) * head_dim ** -0.5``, softmax, ``v @ attn^T`` — no SDPA —
    plus the band reshape and its inverse, ``pe(v)`` and the projection. Nothing
    from ``model_zoo/`` decides any of this except the weights.
    """
    batch, channels, height, width = x.shape
    tokens = height * width
    qkv = attention.qkv(x).flatten(2).transpose(1, 2)
    groups = batch
    if attention.area > 1:
        qkv = qkv.reshape(batch * attention.area, tokens // attention.area, 3 * channels)
        groups = batch * attention.area
    seq = qkv.shape[1]
    stacked = qkv.view(groups, seq, attention.num_heads, 3 * attention.head_dim)
    stacked = stacked.permute(0, 2, 3, 1)
    query, key, value = stacked.split([attention.head_dim] * 3, dim=2)
    if attend_over_channels:
        # The bug: treat the CHANNEL axis as the sequence.
        scores = (query @ key.transpose(-2, -1)) * (attention.head_dim**-0.5)
        scores = scores.softmax(dim=-1)
        attended = scores @ value
    else:
        scores = (query.transpose(-2, -1) @ key) * (attention.head_dim**-0.5)
        scores = scores.softmax(dim=-1)
        attended = value @ scores.transpose(-2, -1)
    attended = attended.permute(0, 3, 1, 2)
    values = value.permute(0, 3, 1, 2)
    if attention.area > 1:
        attended = attended.reshape(batch, tokens, channels)
        values = values.reshape(batch, tokens, channels)
    attended = attended.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
    values = values.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
    if with_pe:
        attended = attended + attention.pe(values)
    return attention.proj(attended)


def guard_attention_weights_use_the_published_initialisation(module) -> None:
    """``ABlock``'s convolutions must be ``trunc_normal_(std=0.02)``.

    Upstream's ``ABlock._init_weights`` is the only non-default initialisation
    in this architecture, and it is invisible to every count, every shape and
    every key — an attention block initialised Kaiming-uniform trains, just
    worse and less predictably. Measured: the shipped ``qkv`` conv has std
    ~0.020 against ~0.051 for the same shape at PyTorch's default, so the two
    are separable by a wide margin.

    Asserted on EVERY conv in the block (attention and MLP alike, which is what
    ``self.apply`` reaches) and paired with a same-shape control built outside
    an ``ABlock``, so the guard is comparing against the default rather than
    against a remembered number.
    """
    import torch

    torch.manual_seed(0)
    block = module.ABlock(128, area=1)
    convs = {
        name: sub
        for name, sub in block.named_modules()
        if isinstance(sub, torch.nn.Conv2d)
    }
    assert len(convs) >= 5, (
        f"fixture: only {len(convs)} convs in an ABlock, expected qkv/proj/pe "
        f"plus two MLP layers"
    )
    control = module.ConvNormAct(128, 3 * 128, 1)
    control_std = float(control.conv.weight.detach().std())
    assert abs(control_std - module.ATTENTION_INIT_STD) > 0.01, (
        f"fixture is degenerate: PyTorch's default init for this shape already "
        f"has std {control_std:.4f}, indistinguishable from the published "
        f"{module.ATTENTION_INIT_STD}"
    )
    for name, conv in sorted(convs.items()):
        std = float(conv.weight.detach().std())
        assert abs(std - module.ATTENTION_INIT_STD) < 0.005, (
            f"{module.__name__}: ABlock's {name} weight has std {std:.4f}, not "
            f"the published {module.ATTENTION_INIT_STD}. `self.apply("
            f"self._init_weights)` reaches every Conv2d in the block; without "
            f"it they keep PyTorch's Kaiming-uniform default (std ~"
            f"{control_std:.4f} for this shape), which changes no parameter "
            f"count and no shape."
        )
        # ⚠️ NOT asserted: that the values are truncated at two standard
        # deviations. `nn.init.trunc_normal_(t, std=0.02)` leaves `a`/`b` at
        # their ±2 defaults, which are ABSOLUTE bounds and not multiples of
        # `std` — so at std 0.02 the truncation is 100 sigma away and does
        # nothing at all. Measured: the shipped pe conv reaches 0.078, ~3.9
        # sigma. The "trunc" in the name is inert at this scale, upstream
        # included, and an assertion at 2 sigma would fail on a correct build.
        assert float(conv.weight.detach().abs().max()) < 10 * module.ATTENTION_INIT_STD, (
            f"{module.__name__}: ABlock's {name} has a weight of magnitude "
            f"{float(conv.weight.detach().abs().max()):.4f}, which is more than 10 "
            f"standard deviations out — the std matched but the distribution "
            f"is not remotely normal"
        )


def guard_ablock_applies_both_residuals(module) -> None:
    """``ABlock`` is ``x + attn(x)`` then ``y + mlp(y)``. Both, and in that order.

    Both residuals are invisible to every structural check: dropping either
    leaves the parameter count, every shape, every ``state_dict`` key and every
    loss key identical, and the model trains. An attention block without its
    residual is also the classic way to make a deep stack untrainable while the
    first epoch still looks fine.

    Compared against a hand-computed expectation rather than read off the
    constructor, and each half is checked against the value the OTHER mistake
    would produce, so a guard that only saw one of them cannot pass.
    """
    import torch

    block = module.ABlock(64, area=1)
    block.eval()
    probe = torch.rand(1, 64, 8, 8) + 0.5
    with torch.no_grad():
        got = block(probe)
        after_attention = probe + block.attn(probe)
        expected = after_attention + block.mlp(after_attention)
        no_attention_residual = block.attn(probe)
        no_attention_residual = no_attention_residual + block.mlp(
            no_attention_residual
        )
        no_mlp_residual = block.mlp(after_attention)

    assert not torch.allclose(expected, no_attention_residual, atol=1e-5), (
        "fixture is degenerate: dropping the attention residual changes nothing"
    )
    assert not torch.allclose(expected, no_mlp_residual, atol=1e-5), (
        "fixture is degenerate: dropping the MLP residual changes nothing"
    )
    assert torch.allclose(got, expected, atol=1e-5), (
        f"{module.__name__}: ABlock does not compute "
        f"y = x + attn(x); y + mlp(y). Deviation from the attention-residual-"
        f"dropped variant is "
        f"{float((got - no_attention_residual).abs().max()):.4f} and from the "
        f"MLP-residual-dropped variant "
        f"{float((got - no_mlp_residual).abs().max()):.4f}; both are "
        f"shape-identical, parameter-identical and train."
    )


# --------------------------------------------------------------------------
# the head, and the decode
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
    """The class tower is two depthwise-separable pairs, not two dense 3x3
    convolutions.

    Each pair is a depthwise 3x3 spatial mixer (``groups == channels``) followed
    by a pointwise 1x1. YOLOv8 spends two DENSE 3x3 convolutions here, which
    type-checks, trains identically and is about 1.5M parameters heavier at this
    scale — so ``published_architecture`` catches it too. This guard exists for
    the DIAGNOSIS: "the first spatial conv is dense" names the edit, where "the
    head is 1,548,288 parameters heavy" does not.

    ⚠️ Stated plainly because a redundant guard that reads as coverage is worse
    than none: this is not independent evidence. It is a better error message
    for a failure the per-layer count already sees.

    Nothing about the head changed between YOLO11 and YOLOv12 —
    ``parse_model`` sets ``Detect.legacy = False`` for both — so this is the
    same tower as the sibling's, asserted here because the code is duplicated.
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
            f"and {len(pointwise)} 1x1 conv(s); the tower is two "
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
        f"{module.__name__}: REG_MAX is {module.REG_MAX}; YOLOv12 publishes 16, "
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
    term is capped at 100. That cap is doing real work — at the n width's 64 the
    class term WINS at the published 80 classes — so the invariance is checked
    at the shipped scale rather than assumed from the formula.

    ⚠️ THREE prefixes, not six. YOLOv12 has ONE detection branch; the
    ``yolov10_s.py`` two files away needs six because its NMS-free design
    duplicates the whole head. A list copied from there names three keys this
    tree does not have, and the ``dead`` assertion below is what catches that
    direction.
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

    A freshly built head predicts around ``sigmoid(-9)`` on every class at the
    fine levels, so a forward pass at initialisation returns a handful of
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
    """YOLOv12 IS NMS-BASED, and the decode must suppress duplicates.

    ⚠️ THIS GUARD IS THE MIRROR IMAGE OF ONE TWO FILES AWAY. ``yolov10_s.py`` is
    NMS-FREE and has ``guard_decode_is_nms_free``, asserting that five anchors
    decoding to the same box at the same class come back as FIVE detections.
    YOLOv12 assigns one-to-many, so several anchors are deliberately trained to
    fire on one object and duplicates are the design's expected raw output:
    here the same fixture must come back as ONE.

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
        f"YOLOv12 is NMS-BASED — its head is assigned one-to-MANY, so duplicate "
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
        f"frame. The metrics read them as dataset pixels, so mAP would be near "
        f"zero while the loss fell normally. Call transform.postprocess()."
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
# same exponents, and upstream's `DetectionModel` builds the same
# `v8DetectionLoss` for `yolo12.yaml` as for `yolo11.yaml`. It is re-guarded
# here rather than assumed because the code is duplicated (zero relative
# imports repo-wide), and a duplicated assigner that leaves its guards behind is
# exactly how one of the four rules goes missing.
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

    YOLOv12 has **no objectness branch**: the score it ranks by at inference is
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
    assert matched_boxes.shape == (1, 4), (
        f"{module.__name__}: the matched box table is "
        f"{tuple(matched_boxes.shape)} for one positive anchor"
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
    parameter, the Area-Attention blocks' included.

    ⚠️ WHAT THIS IS AND IS NOT. YOLOv12 has **no gradient isolation to
    protect**: the ``detach`` that is the single most silent thing in
    ``yolov10_s.py`` exists only because that architecture has a second head
    that must not train the trunk. YOLOv12 has one head, so there is no
    ``detach`` here and nothing for a gradient-isolation guard to assert. Saying
    so explicitly, because the obvious move when porting from a sibling is to
    carry its detach guard across, and the assertion it makes would be
    **vacuous here** — a template with no detach trivially satisfies "nothing is
    detached".

    What IS worth pinning is the opposite direction, and it is not vacuous: a
    module that is constructed, called, and yet sits OUTSIDE the loss graph
    would show up here and nowhere else. A ``torch.no_grad()`` accidentally
    scoping a block, a ``.detach()`` added for a shape fix, or a residual
    written as ``x + y.detach()`` all leave the parameter count, the shapes,
    the keys and the loss values untouched, and quietly freeze a chunk of the
    trunk while the rest keeps learning.

    ⚠️ NON-ZERO IS ASSERTED FOR THE TRUNK ONLY, and that is deliberate. Across
    the OD roster ``grad is not None`` holds for every trainable parameter on
    every template, but a non-zero gradient does NOT: a small batch assigns no
    target to some FPN levels, so parts of the HEAD legitimately sit at exactly
    zero. Measured here: with one object, six of the head's box-tower tensors
    do. The trunk is upstream of every level's loss and has no such excuse.
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
        f"{dead[:8]}. YOLOv12 has one head and no deliberate gradient "
        f"isolation, so every trunk parameter must train. A stray detach or an "
        f"over-scoped no_grad leaves the parameter count, the shapes, the keys "
        f"and the loss values all unchanged."
    )
    attention = [name for name, _ in trunk if ".attn." in name]
    assert attention, (
        "fixture is degenerate: no Area-Attention parameters matched, so the "
        "blocks this guard most cares about are not being checked"
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
        raise OSError("network access is blocked by test_yolov12_s")

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
#:
#: ⚠️ 128 px also keeps the stride-16 token count at 64, divisible by the
#: stride-16 stage's area of 4 — see ``guard_area_divides_the_token_count``.
#: A probe edge that broke that would fail loudly rather than silently, but it
#: would fail in this guard rather than where the rule lives.
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
    "block_kinds": guard_block_kinds_match_the_yaml,
    "no_pooling_or_partial_attention": (
        guard_no_pooling_or_partial_attention_in_the_tree
    ),
    "arch_table_is_live": guard_arch_table_is_live,
    "module_tree_size": guard_module_tree_size_is_pinned,
    "no_stateful_norm": guard_no_stateful_normalisation,
    "derived_norm_groups": guard_norm_groups_are_derived_from_the_channel_count,
    "deepest_output_is_the_last_stage": (
        guard_the_deepest_output_is_the_last_stages_own_output
    ),
    "a2c2f_relan_shape": guard_a2c2f_is_relan_not_a_c2f,
    "a2c2f_chains_entries": guard_a2c2f_chains_its_entries,
    "a2c2f_entry_is_a_block_pair": guard_a2c2f_entry_is_a_fixed_pair_of_ablocks,
    "a2c2f_layer_scale": guard_a2c2f_layer_scale_is_off_here_and_applied_when_on,
    "neck_holds_no_attention": guard_the_neck_holds_no_attention,
    "area_attention_is_banded": guard_area_attention_is_banded,
    "area_divides_the_tokens": guard_area_divides_the_token_count,
    "attention_heads_derived": guard_attention_head_count_is_derived,
    "attention_reference_operator": guard_attention_matches_the_reference_operator,
    "attention_init": guard_attention_weights_use_the_published_initialisation,
    "ablock_residuals": guard_ablock_applies_both_residuals,
    "c3k2_fuses_all_blocks": guard_c3k2_fuses_every_intermediate_block,
    "c3k_branch_routing": guard_c3k_routes_its_blocks_through_the_first_branch,
    "bottleneck_residual": guard_bottlenecks_keep_their_identity_branch,
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
        "positional_encoding_kernel_from_yolo11",
        "ATTENTION_PE_KERNEL = 7",
        "ATTENTION_PE_KERNEL = 3",
        "published_architecture",
    ),
    (
        "mlp_ratio_taken_from_the_lx_regime",
        "MLP_RATIO = 2.0",
        "MLP_RATIO = 1.2",
        "published_architecture",
    ),
    (
        "depth_multiplier_taken_from_yolov8",
        "    return max(int(round(blocks * DEPTH_MULT)), 1)",
        "    return max(int(round(blocks * 0.33)), 1)",
        "published_architecture",
    ),
    # -- which block is at which stage --------------------------------------
    (
        "deep_backbone_stage_is_a_c3k2_as_in_yolo11",
        '        # yaml 5-6   P4/16  -> head level 1.  area 4: four horizontal bands.\n'
        '        (512, 512, 4, "a2c2f", 4),',
        '        # yaml 5-6   P4/16\n'
        '        (512, 512, 4, "c3k2", C3K2_EXPANSION),',
        "block_kinds",
    ),
    (
        "neck_stride32_fusion_is_an_a2c2f",
        '    NECK_KINDS = ("a2c2f", "a2c2f", "a2c2f", "c3k2")',
        '    NECK_KINDS = ("a2c2f", "a2c2f", "a2c2f", "a2c2f")',
        "block_kinds",
    ),
    (
        "stride16_stage_attends_globally",
        '        (512, 512, 4, "a2c2f", 4),',
        '        (512, 512, 4, "a2c2f", 1),',
        "block_kinds",
    ),
    (
        "c3k_forced_at_the_shipped_scale",
        "C3K2_FORCE_C3K = False",
        "C3K2_FORCE_C3K = True",
        "block_kinds",
    ),
    (
        "neck_c3k2_fusion_loses_its_c3k_flag",
        "            return C3k2(in_ch, out_ch, blocks=blocks, c3k=True)",
        "            return C3k2(in_ch, out_ch, blocks=blocks, c3k=False)",
        "block_kinds",
    ),
    (
        "neck_fusions_use_area_attention",
        "            return A2C2f(in_ch, out_ch, blocks=blocks, a2=False, residual=False)",
        "            return A2C2f(in_ch, out_ch, blocks=blocks, a2=True, residual=False)",
        "neck_holds_no_attention",
    ),
    # -- YOLO11 modules that must NOT be here -------------------------------
    (
        "pooling_module_reintroduced",
        '        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")',
        "        self.upsample = nn.Sequential(\n"
        "            nn.MaxPool2d(1), nn.Upsample(scale_factor=2, mode=\"nearest\")\n"
        "        )",
        "no_pooling_or_partial_attention",
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
        "depth_multiplier_hardcoded_at_the_shipped_value",
        "    return max(int(round(blocks * DEPTH_MULT)), 1)",
        "    return max(int(round(blocks * 0.50)), 1)",
        "arch_table_is_live",
    ),
    (
        "channel_cap_hardcoded_at_the_shipped_value",
        "    scaled = min(channels, MAX_CHANNELS) * WIDTH_MULT",
        "    scaled = min(channels, 1024) * WIDTH_MULT",
        "arch_table_is_live",
    ),
    (
        "width_multiplier_hardcoded_in_the_rounder",
        "    scaled = min(channels, MAX_CHANNELS) * WIDTH_MULT",
        "    scaled = min(channels, MAX_CHANNELS) * 0.50",
        "arch_table_is_live",
    ),
    (
        "c3k_override_scales_misstate_upstream",
        'C3K_AT_SCALE_OVERRIDE_SCALES = ("m", "l", "x")',
        'C3K_AT_SCALE_OVERRIDE_SCALES = ("l", "x")',
        "arch_table_is_live",
    ),
    (
        "a2c2f_residual_scales_misstate_upstream",
        'A2C2F_RESIDUAL_SCALES = ("l", "x")',
        'A2C2F_RESIDUAL_SCALES = ("m", "l", "x")',
        "arch_table_is_live",
    ),
    (
        "residual_regime_mlp_ratio_misstated",
        "MLP_RATIO_AT_RESIDUAL_SCALES = 1.2",
        "MLP_RATIO_AT_RESIDUAL_SCALES = 2.0",
        "arch_table_is_live",
    ),
    (
        "extra_entry_per_backbone_stage",
        "            blocks = _round_depth(blocks_full)",
        "            blocks = _round_depth(blocks_full) + 1",
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
    (
        "norm_groups_take_the_largest_power_of_two",
        "    for groups in range(min(maximum, channels), 0, -1):",
        "    for groups in (32, 16, 8, 4, 2, 1):",
        "derived_norm_groups",
    ),
    # -- constructed but never called --------------------------------------
    (
        "deepest_stage_bypassed_in_the_forward",
        "        return outputs[1], outputs[2], outputs[3]",
        "        return outputs[1], outputs[2], self.downsamples[3](outputs[2])",
        "deepest_output_is_the_last_stage",
    ),
    (
        "layer_scale_constructed_but_not_applied",
        "        return x + self.gamma.view(1, -1, 1, 1) * fused",
        "        return fused",
        "a2c2f_layer_scale",
    ),
    (
        "layer_scale_initialised_at_one",
        "            self.gamma = nn.Parameter(LAYER_SCALE_INIT * torch.ones(out_ch))",
        "            self.gamma = nn.Parameter(torch.ones(out_ch))",
        "a2c2f_layer_scale",
    ),
    # -- R-ELAN's shape -----------------------------------------------------
    (
        "a2c2f_fuses_a_split_half_it_does_not_have",
        "        self.cv2 = ConvNormAct((1 + blocks) * hidden, out_ch, 1, stride=1)",
        "        self.cv2 = ConvNormAct((2 + blocks) * hidden, out_ch, 1, stride=1)",
        "a2c2f_relan_shape",
    ),
    (
        "a2c2f_stem_widens_like_a_c2f",
        "        self.cv1 = ConvNormAct(in_ch, hidden, 1, stride=1)\n"
        "        self.cv2 = ConvNormAct((1 + blocks) * hidden, out_ch, 1, stride=1)",
        "        self.cv1 = ConvNormAct(in_ch, 2 * hidden, 1, stride=1)\n"
        "        self.cv2 = ConvNormAct((1 + blocks) * hidden, out_ch, 1, stride=1)",
        "a2c2f_relan_shape",
    ),
    (
        "a2c2f_entries_all_read_the_stem",
        "        branches = [self.cv1(x)]\n"
        "        for entry in self.m:\n"
        "            branches.append(entry(branches[-1]))",
        "        branches = [self.cv1(x)]\n"
        "        for entry in self.m:\n"
        "            branches.append(entry(branches[0]))",
        "a2c2f_chains_entries",
    ),
    (
        "ablock_pair_becomes_a_single_block",
        "                    for _ in range(A2C2F_BLOCKS_PER_ENTRY)",
        "                    for _ in range(1)",
        "a2c2f_entry_is_a_block_pair",
    ),
    (
        "ablock_pair_size_tracks_the_depth",
        "                    for _ in range(A2C2F_BLOCKS_PER_ENTRY)",
        "                    for _ in range(_round_depth(2 * A2C2F_BLOCKS_PER_ENTRY))",
        "a2c2f_entry_is_a_block_pair",
    ),
    # -- Area Attention -----------------------------------------------------
    (
        "area_attention_attends_globally",
        "        banded = self.area > 1",
        "        banded = False",
        "area_attention_is_banded",
    ),
    (
        "area_bands_are_interleaved_not_contiguous",
        "            qkv = qkv.reshape(batch * self.area, tokens // self.area, 3 * channels)",
        "            qkv = (\n"
        "                qkv.reshape(batch, tokens // self.area, self.area, 3 * channels)\n"
        "                .transpose(1, 2)\n"
        "                .contiguous()\n"
        "                .reshape(batch * self.area, tokens // self.area, 3 * channels)\n"
        "            )",
        "area_attention_is_banded",
    ),
    (
        "stride16_area_does_not_divide_the_map",
        '        (512, 512, 4, "a2c2f", 4),',
        '        (512, 512, 4, "a2c2f", 3),',
        "area_divides_the_tokens",
    ),
    (
        "attention_head_count_hardcoded",
        "            num_heads = max(1, dim // ATTENTION_HEAD_DIM)",
        "            num_heads = 8",
        "attention_heads_derived",
    ),
    (
        "sdpa_gets_the_channels_first_layout",
        "        attended = F.scaled_dot_product_attention(query, key, value, scale=self.scale)",
        "        attended = F.scaled_dot_product_attention(\n"
        "            query.transpose(-2, -1),\n"
        "            key.transpose(-2, -1),\n"
        "            value.transpose(-2, -1),\n"
        "            scale=self.scale,\n"
        "        ).transpose(-2, -1)",
        "attention_reference_operator",
    ),
    (
        "positional_encoding_residual_dropped",
        "        return self.proj(attended + self.pe(value))",
        "        return self.proj(attended)",
        "attention_reference_operator",
    ),
    (
        "ablock_init_left_at_the_pytorch_default",
        "        self.apply(self._init_weights)",
        "        self.apply(lambda _module: None)",
        "attention_init",
    ),
    (
        "ablock_drops_the_attention_residual",
        "        x = x + self.attn(x)",
        "        x = self.attn(x)",
        "ablock_residuals",
    ),
    (
        "ablock_drops_the_mlp_residual",
        "        return x + self.mlp(x)",
        "        return self.mlp(x)",
        "ablock_residuals",
    ),
    # -- the blocks YOLOv12 keeps from YOLO11 -------------------------------
    (
        "c3k2_blocks_chain_from_the_wrong_split_half",
        "        branches = list(self.cv1(x).chunk(2, dim=1))\n"
        "        for block in self.m:\n"
        "            branches.append(block(branches[-1]))",
        "        branches = list(self.cv1(x).chunk(2, dim=1))\n"
        "        for block in self.m:\n"
        "            branches.append(block(branches[0]))",
        "c3k2_fuses_all_blocks",
    ),
    (
        "c3k2_fusion_branch_order_reversed",
        "        return self.cv2(torch.cat(branches, dim=1))",
        "        return self.cv2(torch.cat(list(reversed(branches)), dim=1))",
        "c3k2_fuses_all_blocks",
    ),
    (
        "c3k_mixes_the_skip_instead_of_its_own_branch",
        "        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))",
        "        return self.cv3(torch.cat((self.cv1(x), self.m(self.cv2(x))), dim=1))",
        "c3k_branch_routing",
    ),
    (
        "c3k2_plain_blocks_lose_their_shortcut_as_in_yolov8",
        "    def __init__(self, in_ch, out_ch, expansion, shortcut=True, kernel=3):",
        "    def __init__(self, in_ch, out_ch, expansion, shortcut=False, kernel=3):",
        "bottleneck_residual",
    ),
    (
        "c3k_passes_shortcut_false_down",
        "    def __init__(self, in_ch, out_ch, blocks=C3K_BLOCKS, shortcut=True):",
        "    def __init__(self, in_ch, out_ch, blocks=C3K_BLOCKS, shortcut=False):",
        "bottleneck_residual",
    ),
    # -- the head -----------------------------------------------------------
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
        "        self.head = YOLOv12Head(\n"
        "            self.num_classes, self.neck.out_channels, reg_max=self.reg_max\n"
        "        )",
        "        self.head = YOLOv12Head(self.num_classes, self.neck.out_channels)",
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
    (
        "transposed_head_flatten",
        "                cls_output.permute(0, 2, 3, 1).reshape(batch, height * width, -1)",
        "                cls_output.permute(0, 3, 2, 1).reshape(batch, height * width, -1)",
        "head_flatten_order",
    ),
    (
        "head_drops_the_coarsest_level",
        "        for level, (feature, stride) in enumerate(zip(features, self.strides)):",
        "        for level, (feature, stride) in enumerate(\n"
        "            zip(features[:-1], self.strides)\n"
        "        ):",
        "head_emits_every_level",
    ),
    # -- decode and DFL -----------------------------------------------------
    (
        "single_stride_decode",
        "    anchor_x, anchor_y, stride = anchors[:, 0], anchors[:, 1], anchors[:, 2]",
        "    anchor_x, anchor_y, stride = anchors[:, 0], anchors[:, 1], anchors[0, 2]",
        "decode_per_level_stride",
    ),
    (
        "dfl_decode_takes_an_argmax",
        "    return (dist_logits.softmax(dim=-1) * bins).sum(dim=-1)",
        "    return dist_logits.argmax(dim=-1).to(dist_logits.dtype) + 0.0 * bins.sum()",
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
        "    scaled = boxes_xyxy / stride[0]",
        "dfl_target_cell_units",
    ),
    (
        "dfl_loss_drops_the_interpolation_weights",
        "    return (loss_lower * weight_lower + loss_upper * weight_upper).mean(dim=-1)",
        "    return (0.5 * loss_lower + 0.5 * loss_upper).mean(dim=-1)",
        "dfl_loss_interpolates",
    ),
    (
        "dfl_loss_collapses_the_two_bins",
        "    upper = lower + 1",
        "    upper = lower",
        "dfl_loss_interpolates",
    ),
    (
        "background_channel_kept",
        "            class_scores = class_scores[:, 1:]\n"
        "            num_anchors, num_classes = class_scores.shape\n"
        "            flat_scores = class_scores.reshape(-1)\n"
        "            labels = (\n"
        "                torch.arange(1, num_classes + 1, device=boxes.device)",
        "            num_anchors, num_classes = class_scores.shape\n"
        "            flat_scores = class_scores.reshape(-1)\n"
        "            labels = (\n"
        "                torch.arange(0, num_classes, device=boxes.device)",
        "decode_per_image",
    ),
    (
        "decode_truncates_the_batch",
        "        for boxes, class_scores, (height, width) in zip(decoded, scores, image_sizes):",
        "        for boxes, class_scores, (height, width) in zip(\n"
        "            decoded[:1], scores, image_sizes\n"
        "        ):",
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
        "            keep = batched_nms(\n"
        "                candidate_boxes,\n"
        "                flat_scores,\n"
        "                torch.zeros_like(labels),\n"
        "                self.nms_thresh,\n"
        "            )",
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
        # ⚠️ BOTH edges, and that is not padding: `min_size` alone changes
        # nothing, because GeneralizedRCNNTransform takes
        # `min(min_size / short_side, max_size / long_side)` and `max_size`
        # still caps a square input at the declared edge. A mutation that moved
        # only `min_size` SURVIVED this guard on the first sweep, and it was the
        # mutation that was wrong, not the guard.
        "transform_resizes_past_the_declared_edge",
        "            min_size=self.input_size,\n            max_size=self.input_size,",
        "            min_size=self.input_size + 32,\n"
        "            max_size=self.input_size + 32,",
        "declared_size_measured",
    ),
    # -- task-aligned assignment -------------------------------------------
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
        "        inside = self._anchors_inside(gt_boxes, anchor_points)\n"
        "        candidate = alignment * inside.to(alignment.dtype)",
        "        inside = torch.ones_like(alignment, dtype=torch.bool)\n"
        "        candidate = alignment",
        "tal_inside_the_box",
    ),
    (
        "topk_takes_the_worst",
        "        _, positions = torch.topk(candidate, topk, dim=1)",
        "        _, positions = torch.topk(candidate, topk, dim=1, largest=False)",
        "tal_topk_ranking",
    ),
    (
        "topk_bound_removed",
        "        topk = min(TAL_TOPK, num_anchors)",
        "        topk = num_anchors",
        "tal_topk_ranking",
    ),
    (
        "hard_class_target",
        "            normalised[fg_mask],",
        "            torch.ones_like(normalised[fg_mask]),",
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
        "        if False and bool(contested.any()):",
        "tal_tie_break_by_iou",
    ),
    (
        "assign_nothing",
        "        matching = selected & inside & (candidate > 0.0)",
        "        matching = torch.zeros_like(selected)",
        "positives_reach_box_branch",
    ),
    # -- gradient flow, network, end to end --------------------------------
    (
        "neck_detaches_the_backbone",
        "        c3, c4, c5 = features",
        "        c3, c4, c5 = (feature.detach() for feature in features)",
        "whole_trunk_is_trained",
    ),
    (
        "fetches_at_construction",
        "        self.backbone = YOLOv12Backbone()",
        "        import urllib.request\n"
        "\n"
        '        urllib.request.urlopen("https://download.pytorch.org/models/x.pth")\n'
        "        self.backbone = YOLOv12Backbone()",
        "no_network",
    ),
    (
        "soft_class_target_zeroed",
        "                cls_targets[index, fg_mask, labels] = aligned",
        "                cls_targets[index, fg_mask, labels] = 0.0",
        "overfits_one_object",
    ),
]

#: The mutations asserted to leave TRAINING GREEN — a finite loss dict and
#: well-formed predictions — so ``test_od_torchvision_family_train_step.py``
#: stays green against every one of them. That is the reason this file exists,
#: stated as a test.
#:
#: It is EVERY mutation except the six below, each excluded for a stated reason
#: rather than by omission — a silent-set built by leaving things out is how a
#: mutation that actually breaks training gets counted as proof that nothing
#: does:
#:
#: * ``hardcoded_32_groups`` and ``norm_groups_take_the_largest_power_of_two`` —
#:   ``GroupNorm`` refuses to construct at a width the tree actually contains;
#: * ``stride16_area_does_not_divide_the_map`` — ``AreaAttention`` raises on the
#:   first forward, which is the point of that check;
#: * ``coupled_head`` — the two towers are different widths, so the coupling is
#:   a shape error;
#: * ``a2c2f_fuses_a_split_half_it_does_not_have`` and
#:   ``a2c2f_stem_widens_like_a_c2f`` — the fusion's channel arithmetic no
#:   longer closes;
#: * ``fetches_at_construction`` — it reaches for the network, which is the
#:   whole point of it, so it is a construction failure by design.
#:
#: ⚠️ AND ONE THAT IS EXCLUDED FOR A DIFFERENT REASON, worth naming because it
#: qualifies this file's own argument: ``decode_truncates_the_batch`` trains
#: perfectly happily but returns ONE prediction dict for two images, and
#: ``test_od_torchvision_family_train_step.py`` DOES assert
#: ``len(preds) == len(images)``. So it is the single mutation in this table
#: that the family test would catch on its own. Measured, not assumed — it was
#: in this set on the first sweep and failed here.
#:
#: ⚠️ SEVERAL OF THE SILENT ONES ARE NO-OPS AT THE SHIPPED SCALE, which is a
#: stronger statement than "they train". ``hardcoded_box_channel_width`` writes
#: 64 where ``4 * REG_MAX`` is already 64; ``head_reg_max_from_its_own_default``
#: falls back to the same 16; ``layer_scale_constructed_but_not_applied`` edits
#: a line s never reaches; and the three ``*_misstate_upstream`` mutations
#: change only a declared constant. Each builds a BIT-IDENTICAL model, so no
#: amount of training or evaluating could ever see them — only the rebuild-at-
#: another-scale guards can, which is what those guards are for.
_NOT_SILENT = frozenset(
    {
        "hardcoded_32_groups",
        "norm_groups_take_the_largest_power_of_two",
        "stride16_area_does_not_divide_the_map",
        "coupled_head",
        "a2c2f_fuses_a_split_half_it_does_not_have",
        "a2c2f_stem_widens_like_a_c2f",
        "decode_truncates_the_batch",
        "fetches_at_construction",
    }
)

_SILENT_MUTATIONS = frozenset(
    entry[0] for entry in MUTATIONS if entry[0] not in _NOT_SILENT
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
    stale = sorted(_NOT_SILENT - {entry[0] for entry in MUTATIONS})
    assert not stale, (
        f"_NOT_SILENT excludes {stale}, which are not in MUTATIONS — an "
        f"exclusion that matches nothing quietly shrinks this sweep"
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
