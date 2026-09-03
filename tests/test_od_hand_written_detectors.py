"""Guards for the hand-written ``torchvision_detection`` detectors, each proven
able to go red by a mutation that is kept in the suite.

Why this file exists
--------------------
``tests/test_od_torchvision_family_train_step.py`` proves a template returns a
loss dict and a ``List[Dict]`` of xyxy predictions. For a template that wraps a
torchvision builder that is a real assertion: the loss is the library's. For
``yolox_s.py`` and ``rtmdet_s.py`` the loss is **our own code**, so "returns a
loss dict" proves only that our code returns a dict. Every interesting way a
hand-written detector is wrong is silent:

* the assigner matches **nothing** — focal/BCE over an all-negative image is
  finite and small, so the train step passes and the model learns no objects;
* the assigner matches **everything**;
* the assigner loses its per-level structure — one stride used for every
  level, or a centre radius that no longer scales — and the losses stay finite;
* a "decoupled" head is secretly coupled, or a "shared-conv" head secretly
  is not;
* predictions are never mapped back to the original image coordinates, so mAP
  is computed against boxes in the resized frame.

None of those fail a train step. So each is a named guard here, and each guard
is paired with a **mutation** — an exact textual edit to the shipped template
that the guard must catch. The mutations run in CI, permanently: a guard that
stops being able to go red fails this file rather than passing forever. That is
the same discipline ``tests/test_check_dump_coverage.py`` applies to the dump
gate and ``test_guard_rejects_a_mask_headed_model`` applies to the train step.

Fixture degeneracy is the trap, not the guard
---------------------------------------------
A sibling session's ATSS work registered nine mutations and one survived:
dropping the centre-inside rule entirely, because the fixture used exactly two
anchors and with two candidates ``mean + population_std`` equals the larger IoU
identically — so the threshold *was* the better candidate and the rule was
never consulted. The analogue here is a fixture where ``dynamic_k`` collapses
to 1 for every ground truth: the dynamic part then does nothing and any
k-selection bug hides. Both ``dynamic_k`` guards therefore give one ground
truth four strong candidates (``k = 3``) and another exactly one (``k = 1``),
and assert the two counts **differ** — which a fixed k cannot produce. Several
guards additionally carry an explicit "fixture is degenerate" assertion, and
``test_mutation_is_caught_by_its_guard`` refuses to accept a mutation that
trips one of those instead of the guard proper.
"""

import importlib.util
import pathlib
import re
import tempfile

import pytest

ROOT = pathlib.Path(__file__).parent.parent
OD_PYTORCH = ROOT / "model_zoo" / "object_detection" / "pytorch"
YOLOX_PATH = OD_PYTORCH / "yolox_s.py"
RTMDET_PATH = OD_PYTORCH / "rtmdet_s.py"

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


def _load(path: pathlib.Path):
    return _exec_source(path.read_text(encoding="utf-8"), path.stem)


def _mutate(path: pathlib.Path, anchor: str, replacement: str):
    """Load ``path`` with ``anchor`` replaced by ``replacement``.

    The anchor must appear **exactly once**. A mutation whose anchor has drifted
    would otherwise silently patch nothing and the guard would "catch" a
    pristine template — a green that means the opposite of what it looks like.
    """
    source = path.read_text(encoding="utf-8")
    occurrences = source.count(anchor)
    assert occurrences == 1, (
        f"{path.name}: mutation anchor occurs {occurrences} times, expected 1. "
        f"The template moved under the mutation; re-anchor it rather than "
        f"loosening this check.\nanchor:\n{anchor}"
    )
    return _exec_source(source.replace(anchor, replacement), f"mutated_{path.stem}")


def _build(module, num_classes: int):
    entry_name = getattr(module, "main_class", None) or getattr(
        module, "main_method", None
    )
    assert entry_name, f"{module.__name__}: no main_class / main_method"
    return getattr(module, entry_name)(num_classes)


# --------------------------------------------------------------------------
# shared probes — the guards that read the same way for both detectors
# --------------------------------------------------------------------------


#: Attribute names the two heads use for their per-level 1x1 predictors, split
#: by what the gradient below proves. A box predictor only receives gradient if
#: something was assigned as a POSITIVE; a class predictor receives it from the
#: negatives too, so it cannot distinguish "assigned nothing" from "trained
#: normally". That asymmetry is the whole mechanism of the guard.
_BOX_PREDICTOR_GROUPS = ("reg_preds", "rtm_reg")
_CLASS_PREDICTOR_GROUPS = ("cls_preds", "obj_preds", "rtm_cls")


def _predictor_groups(model, names):
    """``{group: [(name, weight), ...]}`` for the predictor groups the head has.

    Resolved from the built model rather than hard-coded per template, and
    asserted non-empty: a rename must fail the lookup rather than quietly
    narrowing the probe to nothing.
    """
    head = model.head
    found = {}
    for group in names:
        if not hasattr(head, group):
            continue
        modules = getattr(head, group)
        assert len(modules) == len(head.strides), (
            f"{group} has {len(modules)} entries for {len(head.strides)} levels"
        )
        found[group] = [
            (f"head.{group}.{level}.weight", module.weight)
            for level, module in enumerate(modules)
        ]
    assert found, (
        f"{type(model).__name__}: none of the predictor groups {names} exist on "
        f"the head. A rename must be reflected here or this probe checks nothing."
    )
    return found


def guard_positives_reach_the_box_regression_branch(module) -> None:
    """One train step must leave the box-regression predictors with a real
    gradient — which happens only if the assigner matched something.

    This is the assign-nothing guard, and it is ``requires_grad``-aware in the
    direction that matters: a bare ``p.grad is None`` sweep false-flags
    deliberately frozen parameters, while the real defect is a **trainable**
    parameter the loss never reaches. Three assertions, failing for different
    reasons:

    * no trainable parameter may have a ``None`` gradient at all — that is a
      branch detached from the loss entirely;
    * the box-regression group must have a non-zero gradient somewhere. Both
      templates fall back to ``prediction.sum() * 0.0`` when there are no
      positives (so the loss dict keeps its shape and no gradient is ``None``),
      which means an all-negative assignment shows up here as an exactly-zero
      regression gradient and **nowhere else**;
    * the class group must too, as a sanity check on the fixture.

    Deliberately NOT asserted here: that all three levels receive positives.
    At random initialisation the IoU landscape is dominated by centre jitter
    rather than by scale — a stride-32 anchor's box is stride-sized but is not
    centred on the object — so which level wins a given ground truth is not
    deterministic, and a fixture pretending otherwise is exactly the kind that
    passes for the wrong reason. Per-level structure is pinned instead by two
    deterministic unit guards on the geometry itself
    (``decode_per_level_stride`` and ``centre_radius_stride`` /
    ``centre_prior_stride``).
    """
    import torch

    edge = module.image_size
    model = _build(module, 3)
    model.train()

    # Object sizes matched to the three strides (8 / 16 / 32) and centred on a
    # grid point of each, so every level has a well-placed candidate.
    targets = [
        {
            "boxes": torch.tensor(
                [
                    [48.0, 48.0, 56.0, 56.0],
                    [144.0, 144.0, 160.0, 160.0],
                    [288.0, 288.0, 320.0, 320.0],
                ]
            ),
            "labels": torch.tensor([0, 1, 2], dtype=torch.int64),
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

    def alive(groups):
        return {
            group: [
                name
                for name, parameter in entries
                if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
            ]
            for group, entries in _predictor_groups(model, groups).items()
        }

    box_alive = alive(_BOX_PREDICTOR_GROUPS)
    for group, names in box_alive.items():
        assert names, (
            f"{module.__name__}: every level of {group} received an exactly "
            f"zero gradient, so NO ground truth was assigned a positive "
            f"anchor. Three objects were supplied, one at each stride's scale. "
            f"An all-negative image still yields a finite, small loss and a "
            f"clean train step, which is why nothing else in this suite sees "
            f"it."
        )

    for group, names in alive(_CLASS_PREDICTOR_GROUPS).items():
        assert names, (
            f"{module.__name__}: every level of {group} received a zero "
            f"gradient — the classification branch is detached from the loss"
        )


def guard_constructs_with_no_network(module) -> None:
    """The architecture must build with the network genuinely unavailable.

    ``tests/test_model_contract.py`` covers two thirds of this already: it greps
    the source for hub-fetch patterns and runs the whole session with
    ``HF_HUB_OFFLINE``. Neither closes the socket, so on a warm torch cache a
    template that fetches indirectly still passes — and these two templates are
    hand-written precisely so that nothing is fetched. So: point ``TORCH_HOME``
    and the hub caches at an empty directory and make DNS and socket creation
    raise, then build.
    """
    import socket

    import torch

    original_socket = socket.socket
    original_getaddrinfo = socket.getaddrinfo
    original_create_connection = socket.create_connection

    def refuse(*_args, **_kwargs):
        raise OSError("network access is blocked by test_od_hand_written_detectors")

    with tempfile.TemporaryDirectory(prefix="tb-nonet-") as cache:
        keys = ("TORCH_HOME", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME")
        import os

        saved = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                os.environ[key] = cache
            socket.socket = refuse
            socket.getaddrinfo = refuse
            socket.create_connection = refuse
            model = _build(module, 3)
            model.eval()
            with torch.no_grad():
                model([torch.rand(3, 64, 64)])
        except OSError as error:
            raise AssertionError(
                f"{module.__name__}: construction or a forward pass tried to "
                f"reach the network — {error}. These templates are written from "
                f"scratch so that nothing is fetched; the #199 egress lockdown "
                f"means a fetch is an edge-only failure, invisible on a warm "
                f"local cache."
            ) from error
        finally:
            socket.socket = original_socket
            socket.getaddrinfo = original_getaddrinfo
            socket.create_connection = original_create_connection
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


#: Overfit-probe settings. 128 px rather than the declared 640 because this
#: runs 200 forward+backward passes and is a guard, not a benchmark; the entry
#: point takes the edge as its second argument precisely so the transform can
#: be built smaller. 200 steps is roughly 3x what either detector needs to
#: cross the thresholds below, so the margin absorbs BLAS variation across
#: platforms.
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
    from the assigner through the losses to the decode, and it is what makes
    "trains" and "evaluates" claims about these templates rather than about
    their return types.

    It also covers what nothing else can. Zeroing YOLOX's objectness target,
    or flipping the sign of either assigner's IoU cost, leaves every structural
    guard green, every loss finite and every fixture satisfied — the model
    simply never learns to fire. Measured: with the objectness target zeroed
    the loss still falls, and the top detection's score collapses from 0.94 to
    the sigmoid floor.
    """
    import torch
    from torchvision.ops import box_iou

    torch.manual_seed(0)
    entry_name = getattr(module, "main_class", None) or getattr(
        module, "main_method", None
    )
    entry = getattr(module, entry_name)
    model = entry(2, _OVERFIT_EDGE)
    assert model.input_size == _OVERFIT_EDGE, (
        f"{module.__name__}: {entry_name}(num_classes, input_size) did not take "
        f"the edge — got input_size={model.input_size}. This probe needs the "
        f"small build; fix the call rather than dropping the check."
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
        f"{module.__name__}: 200 steps on ONE image moved the loss only "
        f"{first:.3f} -> {last:.3f}. A detector that cannot overfit a single "
        f"object is not learning from its own assignments."
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
        f"broken objectness/quality target looks like. Every structural guard "
        f"stays green through this."
    )
    assert label == _OVERFIT_LABEL, (
        f"{module.__name__}: the best-scoring detection is class {label}, but "
        f"the single object it was trained on is class {_OVERFIT_LABEL}"
    )
    assert iou >= 0.35, (
        f"{module.__name__}: the best-scoring detection has IoU {iou:.3f} with "
        f"the one box it was trained on. The classifier learned the object and "
        f"the regressor did not follow, so the assigner is rewarding anchors "
        f"that cannot localise it — a sign-flipped IoU cost looks exactly like "
        f"this."
    )


def guard_declared_image_size_is_the_measured_edge(module) -> None:
    """The declared ``image_size`` must be the resolution the backbone receives.

    Scoped to these two templates on purpose. The **family-wide** version of
    this check is ``tests/test_od_declared_resolution.py`` (backend#3058,
    model-zoo#234), which reads the effective resolution off the transform's
    ``min_size`` / ``fixed_size`` and carries a shrink-only ratchet on its
    known-mismatch list. That file owns the family; a second scan over the same
    seventeen templates here would be duplication, not defence.

    What this adds for the two hand-written detectors, and only them, is a
    stronger measurement: a forward hook on the transform reports the spatial
    size of the tensor the backbone is **actually handed**, rather than the
    resize target the transform was configured with. The two can differ — the
    batch is padded to ``size_divisible=32`` after the resize — and these are
    the templates where that padding is load-bearing (YOLOX's ``Focus`` stem
    slices on even rows/columns; both heads sit at strides 8/16/32). It also
    asserts the result is **square**, which the family guard does not: that one
    compares a single edge, so a template resizing only its short side passes
    there.

    Measured off the built model, never read from the source text.
    """
    import torch

    declared = module.image_size
    assert isinstance(declared, int) and declared > 0, (
        f"{module.__name__}: image_size must be a positive int, got {declared!r}"
    )
    model = _build(module, 3)

    seen: list[tuple[int, int]] = []

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


#: Buffer and tensor totals measured off this repo's own build, as a cheap
#: regression tripwire.
#:
#: ⚠️ READ THIS BEFORE CITING THESE NUMBERS AS EVIDENCE. They are
#: SELF-MEASURED: taken from the model under test, so they can only ever prove
#: the code is consistent with itself. They cannot tell you the architecture is
#: the right one — and that is not a theoretical caveat. ``yolox_s`` shipped for
#: review with a self-measured 7,788,886-parameter count presented as proof
#: that "the design is real", while the real YOLOX-S is 8,942,326 at this class
#: count: its ``CSPLayer`` squeezed the already-halved CSP branch a second time
#: and the whole backbone and neck ran ~1.15M narrower. The count agreed with
#: the code perfectly, because it came from the code.
#:
#: ``guard_matches_the_published_architecture`` is the check that can actually
#: make that call: it re-derives the count from the published spec, with
#: nothing from ``model_zoo/`` imported. Parameters are therefore asserted
#: THERE, against the reference. What lives here is only what the reference
#: derivation does not cover — buffer elements and state_dict tensor count —
#: and its job is to notice an unintended change, not to certify a design.
#:
#: Updating these is legitimate when the architecture changes on purpose; state
#: the intended change in the commit message.
_PINNED_TOTALS = {
    # buffers are 0 by construction now: GroupNorm carries no running
    # statistics. That zero is the whole point of the norm choice -- BN's
    # running_mean/running_var are buffers the averaging service ships and
    # averages every round. A non-zero reading here means a BatchNorm (or
    # another stateful norm) came back.
    "YOLOXS": {"buffers": 0, "tensors": 240},
    "RTMDetS": {"buffers": 0, "tensors": 274},
}


def guard_module_tree_size_is_pinned(module) -> None:
    """The built model's buffer and tensor totals are exact.

    Measured at the template's declared ``output_classes`` so the number is
    reproducible from the file alone. Self-measured, so a tripwire rather than
    evidence — see the note on ``_PINNED_TOTALS``.
    """
    entry_name = getattr(module, "main_class", None) or getattr(
        module, "main_method", None
    )
    expected = _PINNED_TOTALS.get(entry_name)
    assert expected is not None, (
        f"{module.__name__}: entry point {entry_name!r} has no row in "
        f"_PINNED_TOTALS. A new hand-written detector must pin its totals, or "
        f"it ships at whatever width a typo leaves it."
    )

    model = _build(module, module.output_classes)
    actual = {
        "buffers": sum(b.numel() for b in model.buffers()),
        "tensors": len(model.state_dict()),
    }
    assert actual == expected, (
        f"{module.__name__}: module tree is {actual}, pinned at {expected} "
        f"(at output_classes={module.output_classes}). Some norm layer or head "
        f"shape moved. This is a SELF-MEASURED tripwire, not evidence the "
        f"architecture is right — parameters are checked against the "
        f"re-derived published spec in "
        f"guard_matches_the_published_architecture. If the change was "
        f"deliberate, update the row and say so in the commit."
    )


def guard_yolox_csp_bottleneck_runs_at_full_branch_width(module) -> None:
    """A ``CSPLayer``'s inner bottleneck must not squeeze the CSP branch again.

    The precise statement of the defect the totals table was added for. A
    ``CSPLayer`` splits its channels in half; the ``Bottleneck`` inside that
    half runs at ``expansion=1.0`` upstream, so its inner 1x1 is full branch
    width. Passing the bottleneck's own 0.5 there halves it a second time —
    every stage of the backbone and neck comes out narrower, the model trains
    perfectly well, and no official checkpoint will ever strict-load against
    it.

    Checked as a RATIO off a small hand-built stage, so it holds at any width
    multiplier and says what is wrong rather than just that a total moved.
    """
    layer = module.CSPLayer(64, 64, n=1)
    branch = layer.conv1.conv.out_channels
    inner = layer.m[0].conv1.conv.out_channels
    assert inner == branch, (
        f"yolox_s: CSPLayer's branch is {branch} channels but its inner "
        f"Bottleneck squeezes to {inner}. The CSP split already halved the "
        f"channels; upstream YOLOX (and YOLOv5's C3) pass expansion=1.0 to the "
        f"bottleneck so it runs at full branch width. Squeezing twice narrows "
        f"the entire backbone and neck — it trains fine and no COCO YOLOX-S "
        f"checkpoint will load against it. rtmdet_s.py's CSPNeXtBlock in this "
        f"same PR gets this right, which is the tell."
    )


def guard_rtmdet_csp_block_runs_at_full_branch_width(module) -> None:
    """The mirror of the YOLOX guard, on the duplicated CSP stage.

    ``rtmdet_s.py``'s ``CSPLayer`` is a separate implementation of the same
    idea, and mmdet likewise passes ``expansion=1.0`` to ``CSPNeXtBlock``. This
    file was the one that got it right, which is how the YOLOX slip was
    identified — so it gets the guard too, or the next edit can silently move
    the asymmetry to this side instead.
    """
    layer = module.CSPLayer(64, 64, n=1)
    branch = layer.main_conv.conv.out_channels
    inner = layer.blocks[0].conv1.conv.out_channels
    assert inner == branch, (
        f"rtmdet_s: CSPLayer's branch is {branch} channels but its inner "
        f"CSPNeXtBlock squeezes to {inner}. mmdet passes expansion=1.0 to the "
        f"block, so its 3x3 runs at full branch width. Squeezing again narrows "
        f"the whole backbone and neck, silently."
    )


# --------------------------------------------------------------------------
# the independent reference — arithmetic from the PUBLISHED specs
# --------------------------------------------------------------------------
#
# WHY THIS EXISTS, AND WHY IT IS NOT ANOTHER TOTAL.
#
# `yolox_s.py` shipped for review with its CSPLayer building the inner
# Bottleneck at the bottleneck's own `expansion=0.5` instead of the 1.0 upstream
# passes. Channels the CSP split had already halved were squeezed again, so the
# backbone and neck ran ~1.15M parameters narrower than YOLOX-S at every stage:
# 7,788,886 instead of 8,942,326. Thirty-six guards did not see it, and neither
# did the parameter count reported as evidence that the architecture was right —
# because that count was measured off the model under test. A number derived
# from the code can only ever confirm the code is self-consistent; it cannot
# tell you that you built the wrong model. Two reviewers caught it by counting
# the PUBLISHED architecture analytically and comparing.
#
# So the reference is re-derived here, layer by layer, from the published arch
# tables — and NOTHING under `model_zoo/` is imported to do it. The primitives
# below are plain arithmetic on (in, out, kernel, groups); the specs are
# transcribed from the papers and the reference configs. If a template's width,
# depth, kernel, block count or head shape drifts from the published design,
# the two numbers disagree and this file says so.
#
# The anchor matters as much as the arithmetic: `_PUBLISHED_YOLOX_S_PARAMETERS`
# ties the derivation to a figure from outside this repo entirely (YOLOX paper
# Table 2 / the official README, ~9.0M at 80 classes), so the transcription
# cannot drift into agreeing with a wrong template. Three independent numbers
# now agree for YOLOX-S: this derivation (8,968,255 at 80 class channels), the
# reviewer's own analytical count, and the published figure.


def _conv(in_ch, out_ch, kernel, groups=1, bias=False):
    return (in_ch // groups) * out_ch * kernel * kernel + (out_ch if bias else 0)


def _bn(channels):
    return 2 * channels


def _cba(in_ch, out_ch, kernel, groups=1):
    """conv -> BatchNorm (affine); the activation has no parameters."""
    return _conv(in_ch, out_ch, kernel, groups) + _bn(out_ch)


def _dws(in_ch, out_ch, kernel):
    """Depthwise kxk then pointwise 1x1, each with its own BatchNorm."""
    return _cba(in_ch, in_ch, kernel, groups=in_ch) + _cba(in_ch, out_ch, 1)


#: YOLOX (Ge et al. 2021) Table 2 / official README: YOLOX-S is ~9.0M
#: parameters at 80 classes. From outside this repo — it is what stops the
#: transcription below from drifting into agreement with a wrong template.
_PUBLISHED_YOLOX_S_PARAMETERS = 9_000_000
_PUBLISHED_TOLERANCE = 0.02


def _reference_yolox_s_parameters(class_channels):
    """YOLOX-S parameter count, derived from the published spec alone.

    width 0.5, depth 0.33, base 32 channels, base depth 1; CSP stages at
    depth/3·depth·3·depth blocks; decoupled head at 128 hidden channels.
    """
    base, depth = 32, 1

    def bottleneck(channels):  # upstream expansion is 1.0 — the defect above
        return _cba(channels, channels, 1) + _cba(channels, channels, 3)

    def csp(in_ch, out_ch, blocks):
        half = out_ch // 2
        return (
            _cba(in_ch, half, 1)
            + _cba(in_ch, half, 1)
            + _cba(2 * half, out_ch, 1)
            + blocks * bottleneck(half)
        )

    def spp(channels, kernels=3):
        half = channels // 2
        return _cba(channels, half, 1) + _cba(half * (kernels + 1), channels, 1)

    total = _cba(12, base, 3)  # Focus stem: 4 sliced phases of 3 channels
    total += _cba(base, base * 2, 3) + csp(base * 2, base * 2, depth)
    total += _cba(base * 2, base * 4, 3) + csp(base * 4, base * 4, depth * 3)
    total += _cba(base * 4, base * 8, 3) + csp(base * 8, base * 8, depth * 3)
    total += (
        _cba(base * 8, base * 16, 3)
        + spp(base * 16)
        + csp(base * 16, base * 16, depth)
    )

    c3, c4, c5 = base * 4, base * 8, base * 16
    total += _cba(c5, c4, 1) + csp(2 * c4, c4, depth)
    total += _cba(c4, c3, 1) + csp(2 * c3, c3, depth)
    total += _cba(c3, c3, 3) + csp(2 * c3, c4, depth)
    total += _cba(c4, c4, 3) + csp(2 * c4, c5, depth)

    hidden = 128
    for in_ch in (c3, c4, c5):
        total += _cba(in_ch, hidden, 1)                      # stem
        total += 4 * _cba(hidden, hidden, 3)                 # decoupled towers
        total += _conv(hidden, class_channels, 1, bias=True)
        total += _conv(hidden, 4, 1, bias=True)
        total += _conv(hidden, 1, 1, bias=True)
    return total


def _reference_rtmdet_s_parameters(class_channels):
    """RTMDet-S parameter count, derived from the published spec alone.

    CSPNeXt P5 arch table at widen 0.5 / deepen 0.33, CSPNeXt-PAFPN to 128
    channels, RTMDetSepBNHead with share_conv=True.

    NOTE — no published-total anchor for this one, deliberately. The YOLOX-S
    figure above is one I can point at; for RTMDet-S I could not confirm a
    published parameter count from a source independent of this session, so
    asserting one would be exactly the mistake this whole block exists to
    prevent — a number presented as evidence that is really just a guess. What
    IS pinned is that this transcription of the arch table and the built module
    tree agree to the parameter, plus the per-stage widths and block counts
    below. If someone can anchor the published figure, add it here.
    """
    def block(channels):  # CSPNeXtBlock, mmdet expansion 1.0
        return _cba(channels, channels, 3) + _dws(channels, channels, 5)

    def attention(channels):
        return _conv(channels, channels, 1, bias=True)

    def csp(in_ch, out_ch, blocks):
        mid = out_ch // 2
        return (
            _cba(in_ch, mid, 1)
            + _cba(in_ch, mid, 1)
            + blocks * block(mid)
            + attention(2 * mid)
            + _cba(2 * mid, out_ch, 1)
        )

    def sppf(channels, kernels=3):
        mid = channels // 2
        return _cba(channels, mid, 1) + _cba(mid * (kernels + 1), channels, 1)

    #: (in, out, blocks, use_spp) at widen 0.5 / deepen 0.33.
    stages = ((32, 64, 1, False), (64, 128, 2, False), (128, 256, 2, False), (256, 512, 1, True))
    total = _cba(3, 16, 3) + _cba(16, 16, 3) + _cba(16, 32, 3)  # stem
    for in_ch, out_ch, blocks, use_spp in stages:
        total += _cba(in_ch, out_ch, 3)
        if use_spp:
            total += sppf(out_ch)
        total += csp(out_ch, out_ch, blocks)

    c3, c4, c5, blocks, out = 128, 256, 512, 1, 128
    total += _cba(c5, c4, 1) + csp(2 * c4, c4, blocks)
    total += _cba(c4, c3, 1) + csp(2 * c3, c3, blocks)
    total += _cba(c3, c3, 3) + csp(2 * c3, c4, blocks)
    total += _cba(c4, c4, 3) + csp(2 * c4, c5, blocks)
    total += _cba(c3, out, 3) + _cba(c4, out, 3) + _cba(c5, out, 3)  # out_convs

    # share_conv=True: the tower convs are ONE set of weights for all three
    # levels (parameters() de-duplicates), while every level keeps its own BN.
    feat, levels, stacked = 128, 3, 2
    total += 2 * (_conv(out, feat, 3) + _conv(feat, feat, 3))
    total += levels * stacked * 2 * _bn(feat)
    total += levels * (
        _conv(feat, class_channels, 1, bias=True) + _conv(feat, 4, 1, bias=True)
    )
    return total


#: ``entry point -> (reference function, expected structure)``. The structure
#: rows are the published per-stage widths and block counts; they say WHAT
#: drifted when the total disagrees, and they hold independently of it.
_REFERENCE = {
    "YOLOXS": {
        "parameters": _reference_yolox_s_parameters,
        "backbone_out": (128, 256, 512),
        "neck_out": (128, 256, 512),
        "csp_blocks": (1, 3, 3, 1),
        "published": _PUBLISHED_YOLOX_S_PARAMETERS,
    },
    "RTMDetS": {
        "parameters": _reference_rtmdet_s_parameters,
        "backbone_out": (128, 256, 512),
        "neck_out": (128, 128, 128),
        "csp_blocks": (1, 2, 2, 1),
        "published": None,  # see the note in the reference function
    },
}


def _built_csp_block_counts(model):
    """Blocks per backbone CSP stage, read off the built model.

    Both backbones expose their stages differently — YOLOX names them
    ``dark2..dark5``, RTMDet holds a ``stages`` ModuleList — so this finds the
    CSP stage inside each and counts its blocks, rather than assuming a layout.
    """
    backbone = model.backbone
    if hasattr(backbone, "stages"):
        stages = list(backbone.stages)
    else:
        stages = [getattr(backbone, f"dark{i}") for i in range(2, 6)]
    counts = []
    for stage in stages:
        # Selected by CLASS NAME, not by "has an .m attribute": SPPBottleneck
        # also holds its max-pools in an `m` ModuleList, so an attribute probe
        # matches twice on the deepest stage and the count is meaningless.
        inner = [m for m in stage if type(m).__name__ == "CSPLayer"]
        assert len(inner) == 1, (
            f"expected exactly one CSPLayer per backbone stage, found "
            f"{len(inner)} in {[type(m).__name__ for m in stage]}"
        )
        blocks = getattr(inner[0], "m", None)
        if blocks is None:
            blocks = inner[0].blocks
        counts.append(len(blocks))
    return tuple(counts)


def guard_matches_the_published_architecture(module) -> None:
    """The built module tree must match the PUBLISHED architecture, re-derived.

    The independent half of the evidence. ``module_tree_size`` pins the totals
    this repo measured and so can only catch a regression away from whatever
    was shipped; this one re-computes the count from the published spec and
    compares, so it catches shipping the wrong architecture in the first place
    — which is what happened.
    """
    entry_name = getattr(module, "main_class", None) or getattr(
        module, "main_method", None
    )
    reference = _REFERENCE.get(entry_name)
    assert reference is not None, (
        f"{module.__name__}: entry point {entry_name!r} has no row in "
        f"_REFERENCE. A hand-written detector claiming to BE a published "
        f"architecture has to be checked against it, or the only evidence for "
        f"its shape is a number measured off itself."
    )

    class_channels = module.output_classes + 1  # the deliberate label-space +1
    expected = reference["parameters"](class_channels)
    model = _build(module, module.output_classes)
    actual = sum(p.numel() for p in model.parameters())

    assert actual == expected, (
        f"{module.__name__}: built model has {actual:,} parameters; the "
        f"published architecture re-derived from its spec has {expected:,} — "
        f"a difference of {actual - expected:+,}. Something in the width, "
        f"depth, kernel sizes, block counts or head shape does not match the "
        f"design this template claims to implement. This is the check that a "
        f"parameter count measured off the model itself CANNOT make: a "
        f"self-measured total is only self-consistency."
    )

    published = reference["published"]
    if published is not None:
        drift = abs(_reference_yolox_s_parameters(80) - published) / published
        assert drift <= _PUBLISHED_TOLERANCE, (
            f"the spec transcription for {entry_name} derives "
            f"{_reference_yolox_s_parameters(80):,} parameters at 80 classes, "
            f"{drift:.1%} from the published {published:,}. The transcription "
            f"itself has drifted — fix it against the paper before trusting "
            f"the comparison above."
        )

    assert tuple(model.backbone.out_channels) == reference["backbone_out"], (
        f"{module.__name__}: backbone emits "
        f"{tuple(model.backbone.out_channels)} channels, published design has "
        f"{reference['backbone_out']}"
    )
    assert tuple(model.neck.out_channels) == reference["neck_out"], (
        f"{module.__name__}: neck emits {tuple(model.neck.out_channels)} "
        f"channels, published design has {reference['neck_out']}"
    )
    counts = _built_csp_block_counts(model)
    assert counts == reference["csp_blocks"], (
        f"{module.__name__}: backbone CSP stages hold {counts} blocks, "
        f"published design has {reference['csp_blocks']} at this depth "
        f"multiplier"
    )


def guard_predictions_are_in_original_image_coordinates(module) -> None:
    """Eval predictions must be mapped back to each input image's own frame.

    The engine's metrics compare predictions against targets in the dataset's
    pixel space, so a detector that returns boxes in its internal resized frame
    scores near-zero mAP while training loss falls normally — invisible to
    every other check here. The fixture is deliberately **non-square and
    smaller than ``image_size``**, so the internal frame is several times
    larger and an unmapped box lands far outside the image.
    """
    import torch

    model = _build(module, 3)
    # Take every candidate: at random init the class scores sit near the 0.01
    # prior, so the shipped thresholds would return an empty list and the
    # bounds assertion below would be vacuous.
    model.score_thresh = 0.0
    model.eval()

    height, width = 128, 160
    with torch.no_grad():
        predictions = model([torch.rand(3, height, width)])

    assert isinstance(predictions, list) and len(predictions) == 1
    boxes = predictions[0]["boxes"]
    assert boxes.numel(), (
        f"{module.__name__}: no predictions at score_thresh=0, so the "
        f"coordinate check below would prove nothing"
    )
    assert bool((boxes[:, 2] >= boxes[:, 0]).all() and (boxes[:, 3] >= boxes[:, 1]).all()), (
        f"{module.__name__}: predicted boxes are not valid xyxy"
    )
    worst_x = float(boxes[:, 0::2].max())
    worst_y = float(boxes[:, 1::2].max())
    assert worst_x <= width + 1e-3 and worst_y <= height + 1e-3, (
        f"{module.__name__}: predictions reach ({worst_x:.1f}, {worst_y:.1f}) "
        f"for a {width}x{height} image — the boxes are still in the model's "
        f"internal {model.input_size}-px frame. transform.postprocess() maps "
        f"them back; without it mAP is computed against the wrong frame and "
        f"nothing else in this suite notices."
    )


#: Non-square, non-uniform feature-map shapes for the head-ordering guards.
#: The shapes matter: at ``H == W`` a row-major and a column-major flatten of
#: the same map are IDENTICAL, so a transposed reshape is undetectable — the
#: same degeneracy that let an anchor-major/location-major permutation test
#: pass against both orderings at one anchor per location. Both these heads are
#: one-prior-per-location, so the only defence is a shape the template never
#: builds: it never sees a non-square feature map, because the transform makes
#: every batch square.
_ORDERING_SHAPES = ((3, 5), (2, 3), (1, 2))


def _positional_feature(torch, channels, height, width):
    """A feature map whose channel 0 holds ``y * 100 + x`` at every location, so
    an output value identifies the cell that produced it."""
    ys = torch.arange(height, dtype=torch.float32).unsqueeze(1)
    xs = torch.arange(width, dtype=torch.float32).unsqueeze(0)
    feature = torch.zeros(1, channels, height, width)
    feature[0, 0] = ys * 100.0 + xs
    return feature


def _zero_conv(torch_nn, in_channels, out_channels):
    conv = torch_nn.Conv2d(in_channels, out_channels, 1)
    torch_nn.init.zeros_(conv.weight)
    torch_nn.init.zeros_(conv.bias)
    return conv


def _select_channel_zero(torch_nn, in_channels, out_channels):
    """A 1x1 conv that copies input channel 0 into output channel 0."""
    conv = _zero_conv(torch_nn, in_channels, out_channels)
    conv.weight.data[0, 0, 0, 0] = 1.0
    return conv


def _assert_flatten_matches_grid(name, values, cells, strides):
    """``values[n]`` must equal ``y_n * 100 + x_n`` for the grid's own ``n``."""
    expected = [float(y) * 100.0 + float(x) for x, y in cells]
    actual = [float(v) for v in values]
    assert actual == expected, (
        f"{name}: the head's flattened predictions do not line up with the "
        f"anchor table it returns alongside them. Cell (y, x) codes read back "
        f"as {actual}, expected {expected} for strides {strides}. The "
        f"predictions and the anchor coordinates are flattened in DIFFERENT "
        f"orders, so every anchor is matched against another cell's "
        f"prediction — invisible on the square feature maps the template "
        f"actually builds, where the two orders coincide."
    )


# --------------------------------------------------------------------------
# yolox_s — structure
# --------------------------------------------------------------------------


def guard_yolox_head_flatten_order_matches_the_grid(module) -> None:
    """The head's flattened output and the anchor table it returns must agree.

    Rewires level 0 so its class channel 0 is literally its input feature's
    channel 0, feeds a map coding ``y * 100 + x``, and reads the codes back in
    the order the head emitted them — then compares against the cell
    coordinates the head returned in the same call. Non-square maps, for the
    reason in ``_ORDERING_SHAPES``.
    """
    import torch
    from torch import nn

    model = _build(module, 2)
    head = model.head
    in_channels = [module_.conv.in_channels for module_ in head.stems]

    features = []
    for level, (height, width) in enumerate(_ORDERING_SHAPES):
        channels = in_channels[level]
        head.stems[level] = nn.Identity()
        head.cls_convs[level] = nn.Identity()
        head.reg_convs[level] = nn.Identity()
        head.cls_preds[level] = _select_channel_zero(nn, channels, model.num_classes)
        head.reg_preds[level] = _zero_conv(nn, channels, 4)
        head.obj_preds[level] = _zero_conv(nn, channels, 1)
        features.append(_positional_feature(torch, channels, height, width))

    with torch.no_grad():
        raw, grids = head(tuple(features))

    height, width = _ORDERING_SHAPES[0]
    count = height * width
    cells = [(float(g[0]), float(g[1])) for g in grids[:count]]
    _assert_flatten_matches_grid(
        "yolox_s", raw[0, :count, 5], cells, [float(g[2]) for g in grids[:count]]
    )


def guard_yolox_decode_is_per_image_and_aligned(module) -> None:
    """Decoding is driven directly, with scores that are actually above
    threshold, at batch size two.

    A freshly built focal-loss detector predicts ``sigmoid(-4.595) = 0.01`` on
    every class, so a forward pass returns **no detections at all** and any
    check downstream of it is vacuous — the decode has nothing to get wrong.
    That is how a real defect shipped through every guard on a sibling
    template: its post-processing iterated the wrong axis and ``zip`` truncated
    silently instead of raising. Both halves of that failure are addressed
    here: synthetic head outputs with one confident detection per image, and
    **two** images, since a per-image bug is invisible at batch one by
    construction.
    """
    import torch

    model = _build(module, 2)
    classes = model.num_classes
    cells = [(index, 0) for index in range(6)]
    grids = _grid(torch, cells, 8)

    # -10 everywhere: sigmoid(-10)^2 is ~2e-9, far below score_thresh, so only
    # the two anchors set below survive and the assertions are about them.
    raw = torch.full((2, len(cells), 5 + classes), -10.0)
    raw[:, :, :4] = 0.0  # centre offset 0, log-size 0 -> a stride-sized box
    raw[0, 1, 4] = 10.0
    raw[0, 1, 5 + 2] = 10.0
    raw[1, 4, 4] = 10.0
    raw[1, 4, 5 + 1] = 10.0

    results = model._predictions(raw, grids, [(64, 64), (64, 64)])

    assert isinstance(results, list) and len(results) == 2, (
        f"yolox_s: decoding two images returned "
        f"{len(results) if isinstance(results, list) else type(results).__name__} "
        f"result(s), expected 2. The engine's handler indexes predictions "
        f"per image; a truncating zip over the wrong axis loses images "
        f"silently and is invisible at batch one."
    )
    for index, (expected_label, expected_x) in enumerate([(2, 8.0), (1, 32.0)]):
        prediction = results[index]
        assert prediction["boxes"].numel(), (
            f"yolox_s: image {index} produced no detection although its "
            f"objectness and class logits were set to +10"
        )
        best = int(prediction["scores"].argmax())
        label = int(prediction["labels"][best])
        box = prediction["boxes"][best]
        centre_x = float((box[0] + box[2]) / 2.0)
        assert label == expected_label, (
            f"yolox_s: image {index}'s top detection is class {label}, "
            f"expected {expected_label}. The score, label and box columns are "
            f"flattened independently and have come apart."
        )
        assert abs(centre_x - expected_x) < 1.0, (
            f"yolox_s: image {index}'s top detection is centred at x="
            f"{centre_x:.1f}, expected {expected_x:.1f} — the confident anchor "
            f"was paired with another anchor's box."
        )


def guard_yolox_head_is_decoupled(module) -> None:
    """The classification and regression towers must share no parameters.

    That is what "decoupled head" means, and it is YOLOX's headline change over
    YOLOv5. A coupled head — one tower feeding both 1x1 predictors — trains
    perfectly happily and reports the same loss keys, so this is checked by
    parameter identity rather than by reading the constructor.
    """
    head = _build(module, 3).head
    cls_ids = {id(p) for p in head.cls_convs.parameters()}
    reg_ids = {id(p) for p in head.reg_convs.parameters()}

    assert cls_ids and reg_ids, (
        f"expected both conv towers to hold parameters, got "
        f"{len(cls_ids)} cls / {len(reg_ids)} reg"
    )
    shared = cls_ids & reg_ids
    assert not shared, (
        f"yolox_s: the classification and regression towers share "
        f"{len(shared)} parameter tensor(s) — the head is COUPLED, not "
        f"decoupled. It would train and log identical loss keys either way."
    )
    assert len(cls_ids) == len(reg_ids), (
        f"yolox_s: asymmetric towers ({len(cls_ids)} cls vs {len(reg_ids)} "
        f"reg parameters); YOLOX's two branches are the same shape"
    )


def guard_yolox_decode_scales_by_each_levels_stride(module) -> None:
    """A zero prediction must decode to a box the size of **that anchor's**
    stride, so the three levels cover three object scales.

    Using one stride everywhere leaves every loss finite and the train step
    green; the model just cannot represent small objects.
    """
    import torch

    grids = torch.tensor([[0.0, 0.0, 8.0], [0.0, 0.0, 32.0]])
    raw = torch.zeros(1, 2, 6)
    decoded = module._decode_boxes(raw, grids)

    widths = [float(decoded[0, 0, 2]), float(decoded[0, 1, 2])]
    assert widths == pytest.approx([8.0, 32.0]), (
        f"yolox_s: a zero prediction decoded to widths {widths} at strides "
        f"(8, 32); expected [8.0, 32.0]. The per-anchor stride column is not "
        f"reaching the decode, so every level predicts at one scale."
    )
    centres = [float(decoded[0, 0, 0]), float(decoded[0, 1, 0])]
    assert centres == pytest.approx([0.0, 0.0])


# --------------------------------------------------------------------------
# yolox_s — SimOTA fixtures
# --------------------------------------------------------------------------


def _grid(torch, cells, stride):
    return torch.tensor([[float(x), float(y), float(stride)] for x, y in cells])


def _yolox_assign(module, model, gt_boxes, gt_labels, decoded, grids):
    """Call ``assign`` with neutral (all-zero) class and objectness logits, so
    the cost matrix is decided by geometry and IoU alone."""
    import torch

    count = decoded.shape[0]
    return model.assign(
        gt_boxes,
        gt_labels,
        decoded,
        torch.zeros(count),
        torch.zeros(count, model.num_classes),
        grids,
    )


def guard_yolox_class_target_is_scaled_by_the_matched_iou(module) -> None:
    """A positive's classification target is its matched IoU, not a hard 1.

    The YOLOX twin of RTMDet's ``soft_quality_target`` guard. Both detectors
    rank detections by a single score with no centreness branch, which only
    works if the classifier was trained against localisation quality. This is
    unit-tested on the named seam because the same expression inline was
    unreachable by any guard — and a hard target trains and detects perfectly
    happily, so nothing else here would notice.
    """
    import torch

    labels = torch.tensor([2, 0], dtype=torch.int64)
    ious = torch.tensor([0.4, 0.9])
    target = module._iou_aware_class_target(labels, ious, 3, torch.float32)

    assert tuple(target.shape) == (2, 3), f"unexpected target shape {target.shape}"
    assert float(target[0, 2]) == pytest.approx(0.4), (
        f"yolox_s: a positive matched at IoU 0.4 got classification target "
        f"{float(target[0, 2]):.3f}, expected 0.4. The target is not scaled by "
        f"the matched IoU, so `cls * obj` no longer carries localisation "
        f"quality and inference has nothing else to rank by."
    )
    assert float(target[1, 0]) == pytest.approx(0.9)
    assert float(target.sum()) == pytest.approx(1.3), (
        "yolox_s: the class target has mass outside the matched classes"
    )


def guard_yolox_dynamic_k_is_dynamic(module) -> None:
    """Each ground truth gets a number of anchors derived from its own IoUs.

    The fixture is built so the two GTs land on **different** k: the first has
    four strong candidates (top-k IoUs summing past 3, so ``k = 3``), the
    second exactly one weak candidate (``k = 1``). A fixed k cannot produce two
    different counts, which is what makes this fixture non-degenerate — a
    fixture where every GT collapses to ``k = 1`` would pass against a
    hard-coded 1 and prove nothing.
    """
    import torch

    model = _build(module, 2)
    grids = _grid(torch, [(0, 0), (1, 0), (2, 0), (3, 0), (6, 6), (20, 20)], 8)
    gt_boxes = torch.tensor([[12.0, 4.0, 24.0, 8.0], [52.0, 52.0, 8.0, 8.0]])
    gt_labels = torch.tensor([0, 1], dtype=torch.int64)
    decoded = torch.tensor(
        [
            [12.0, 4.0, 22.0, 7.0],
            [12.0, 4.0, 21.0, 7.0],
            [12.0, 4.0, 23.0, 7.0],
            [12.0, 4.0, 20.0, 7.0],
            [52.0, 52.0, 8.0, 8.0],
            [200.0, 200.0, 4.0, 4.0],
        ]
    )

    _, matched_labels, _, _ = _yolox_assign(
        module, model, gt_boxes, gt_labels, decoded, grids
    )
    counts = [int((matched_labels == label).sum()) for label in (0, 1)]
    assert counts[0] > counts[1] == 1, (
        f"yolox_s: SimOTA gave the two ground truths {counts} anchors. The "
        f"first has four candidates at ~0.8 IoU (dynamic k = 3) and the second "
        f"exactly one at 1.0 (dynamic k = 1), so equal counts mean k is not "
        f"being derived from the IoUs at all."
    )


def guard_yolox_centre_region_creates_candidates(module) -> None:
    """A ground truth smaller than one cell still gets candidates.

    SimOTA's candidate set is "inside the box **or** within
    ``CENTER_RADIUS`` strides of its centre". The fixture's GT is 2 px wide, so
    **no** anchor centre falls inside it — the centre rule is the only thing
    producing a candidate, which is precisely what makes this fixture able to
    test it. Without the rule the object is unlearnable and the loss is still
    finite.
    """
    import torch

    model = _build(module, 2)
    grids = _grid(torch, [(0, 0), (1, 1), (2, 2)], 8)
    gt_boxes = torch.tensor([[10.0, 10.0, 2.0, 2.0]])
    gt_labels = torch.tensor([0], dtype=torch.int64)
    decoded = torch.tensor(
        [[10.0, 10.0, 2.0, 2.0], [12.0, 12.0, 2.0, 2.0], [20.0, 20.0, 2.0, 2.0]]
    )

    candidate_mask, inside_both = model._candidate_masks(gt_boxes, grids)
    assert not bool(inside_both.any()), (
        "fixture is degenerate: no anchor should be inside this 2-px box, so "
        "inside_both must be all-False — otherwise the centre rule is not the "
        "only thing under test"
    )
    assert int(candidate_mask.sum()) > 0, (
        "yolox_s: a sub-cell ground truth produced NO candidate anchors. The "
        "centre-radius half of the candidate rule is missing, so any object "
        "smaller than a cell can never be matched — and the loss stays finite."
    )

    fg_mask, _, _, _ = _yolox_assign(
        module, model, gt_boxes, gt_labels, decoded, grids
    )
    assert int(fg_mask.sum()) > 0, (
        "yolox_s: a sub-cell ground truth got no positive anchor"
    )


def guard_yolox_in_box_rule_admits_distant_candidates(module) -> None:
    """A large ground truth gets candidates across its whole extent, not just
    near its centre.

    The symmetric partner of ``centre_region_candidates``: SimOTA's candidate
    set is "inside the box **or** near the centre", and dropping the *box* half
    is the mirror-image bug. It hurts big objects rather than small ones — a
    200-px box at stride 8 has a 20-px centre region, so keeping only the
    centre rule discards every anchor more than 2.5 cells from the middle — and
    it is just as silent. The fixture's far anchor is deep inside the box and
    far outside the centre region, which is what makes it able to test the rule.
    """
    import torch

    model = _build(module, 2)
    grids = _grid(torch, [(12, 12), (22, 22)], 8)  # centres (100, 100), (180, 180)
    gt_boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]])  # xyxy (50,50,150,150)

    # Fixture arithmetic, deliberately NOT read back out of the code under
    # test: anchor 1's centre is (22 + 0.5) * 8 = 180, which is inside the box
    # (0..200) and 80 px from its centre (100, 100) — four times the 20-px
    # centre region at stride 8. Deriving this from `_candidate_masks` output
    # instead would make the degeneracy check itself mutable, and a mutation
    # that trips a degeneracy check reads as "caught" while proving nothing.
    centre = float((grids[1, 0] + 0.5) * grids[1, 2])
    assert centre == 180.0, f"fixture is degenerate: anchor 1 centre is {centre}"
    assert 0.0 < centre < 200.0, "fixture is degenerate: anchor 1 is not in the box"
    assert centre - 100.0 > module.CENTER_RADIUS * 8.0, (
        "fixture is degenerate: anchor 1 is inside the centre region, so the "
        "in-box rule is not the only thing admitting it"
    )

    candidate_mask, _ = model._candidate_masks(gt_boxes, grids)
    assert candidate_mask.tolist() == [True, True], (
        f"yolox_s: an anchor deep inside a large ground-truth box was not a "
        f"candidate (mask {candidate_mask.tolist()}). The inside-the-box half "
        f"of the candidate rule is missing, so a big object can only ever be "
        f"matched within 2.5 cells of its centre — silently, with a finite loss."
    )


def guard_yolox_penalises_candidates_outside_the_box(module) -> None:
    """A candidate admitted only by the centre rule must lose to a
    geometrically valid one, however much better its IoU.

    The fixture's best-IoU candidate (0.99) sits outside the GT box but inside
    its centre region, and the valid candidates are far worse (0.25). Only the
    ``SIMOTA_OUTSIDE_PENALTY`` term keeps the invalid one out; drop it and the
    assigner trains the wrong anchor, with no visible symptom.
    """
    import torch

    model = _build(module, 2)
    grids = _grid(torch, [(0, 0), (1, 1), (3, 3)], 8)
    gt_boxes = torch.tensor([[12.0, 12.0, 24.0, 24.0]])
    gt_labels = torch.tensor([0], dtype=torch.int64)
    decoded = torch.tensor(
        [[12.0, 12.0, 12.0, 12.0], [12.0, 12.0, 10.0, 10.0], [12.0, 12.0, 23.9, 23.9]]
    )

    _, inside_both = model._candidate_masks(gt_boxes, grids)
    assert inside_both[0].tolist() == [True, True, False], (
        "fixture is degenerate: anchor 2 must be a candidate that is NOT "
        "inside the box, or the penalty is not under test"
    )

    fg_mask, _, _, _ = _yolox_assign(
        module, model, gt_boxes, gt_labels, decoded, grids
    )
    assert not bool(fg_mask[2]), (
        "yolox_s: the best-IoU candidate was selected even though its anchor "
        "centre is outside the ground-truth box. The inside-box-AND-centre "
        "penalty is not being applied, so SimOTA ignores its own geometric "
        "constraint — and every loss stays finite."
    )
    assert bool(fg_mask[0]), (
        "yolox_s: no geometrically valid candidate was selected either"
    )


def guard_yolox_prefers_the_better_localised_anchor(module) -> None:
    """Between two geometrically identical candidates, the better box wins.

    Both anchors sit at the same cell and stride, so every geometric term in
    the cost is identical and only ``-log(IoU)`` separates them. That isolates
    the IoU term's **sign**: flip it and SimOTA systematically trains the worst
    box it can find, with a finite loss and a falling loss curve.
    """
    import torch

    model = _build(module, 2)
    grids = _grid(torch, [(6, 6), (6, 6)], 8)
    gt_boxes = torch.tensor([[52.0, 52.0, 40.0, 40.0]])
    gt_labels = torch.tensor([0], dtype=torch.int64)
    # Same centre, same stride; IoU 0.25 versus ~0.90.
    decoded = torch.tensor([[52.0, 52.0, 20.0, 20.0], [52.0, 52.0, 38.0, 38.0]])

    fg_mask, _, _, _ = _yolox_assign(
        module, model, gt_boxes, gt_labels, decoded, grids
    )
    assert fg_mask.tolist() == [False, True], (
        f"yolox_s: given two candidates at the SAME cell and stride, one with "
        f"IoU 0.25 and one with 0.90, the assigner chose {fg_mask.tolist()} — "
        f"expected [False, True]. The IoU term in the cost is not preferring "
        f"the better-localised anchor."
    )


def guard_yolox_centre_radius_scales_with_stride(module) -> None:
    """The centre region is measured in strides, so it is wider in pixels on
    the coarse levels.

    Both anchors in the fixture sit at exactly the same pixel centre (96, 96)
    and the ground truth is 30 px away in each axis. At stride 8 the radius is
    20 px — too small; at stride 32 it is 80 px — comfortable. A constant
    radius makes the two levels indistinguishable, which is how per-level
    structure is lost while training stays green.
    """
    import torch

    model = _build(module, 2)
    grids = torch.tensor([[11.5, 11.5, 8.0], [2.5, 2.5, 32.0]])
    gt_boxes = torch.tensor([[126.0, 126.0, 2.0, 2.0]])

    candidate_mask, _ = model._candidate_masks(gt_boxes, grids)
    assert candidate_mask.tolist() == [False, True], (
        f"yolox_s: two anchors at the SAME pixel centre but strides (8, 32) "
        f"gave candidate mask {candidate_mask.tolist()}, expected "
        f"[False, True]. The centre radius is not scaled by the anchor's "
        f"stride, so the head's three levels compete for the same objects."
    )


def guard_yolox_breaks_ties_between_ground_truths(module) -> None:
    """An anchor claimed by two ground truths goes to exactly one of them.

    Observable through ``matched_ious``: it is built as
    ``(matching * ious).sum(0)``, so a doubly-claimed anchor reports the **sum**
    of two IoUs and can exceed 1.0 — which then feeds the classification
    target as a soft label above 1, and BCE against a target above 1 goes
    negative. The fixture points both GTs at the same single best anchor.
    """
    import torch

    model = _build(module, 2)
    grids = _grid(torch, [(0, 0), (1, 1)], 8)
    gt_boxes = torch.tensor([[8.0, 8.0, 10.0, 10.0], [9.0, 9.0, 14.0, 14.0]])
    gt_labels = torch.tensor([0, 1], dtype=torch.int64)
    decoded = torch.tensor([[8.5, 8.5, 10.0, 10.0], [400.0, 400.0, 4.0, 4.0]])

    fg_mask, matched_labels, _, matched_ious = _yolox_assign(
        module, model, gt_boxes, gt_labels, decoded, grids
    )
    assert int(fg_mask.sum()) == len(matched_labels) == 1, (
        f"fixture is degenerate: expected both ground truths to want the same "
        f"single anchor, got {int(fg_mask.sum())} positives"
    )
    worst = float(matched_ious.max())
    assert worst <= 1.0 + 1e-4, (
        f"yolox_s: an anchor reports a matched IoU of {worst:.3f}. An IoU "
        f"cannot exceed 1, so this anchor is assigned to BOTH ground truths "
        f"and its soft classification target is their sum — the multi-GT "
        f"tie-break is missing."
    )


# --------------------------------------------------------------------------
# rtmdet_s — structure
# --------------------------------------------------------------------------


def _rtmdet_tower_tensor(model, group: str, level: int, index: int, part: str):
    return getattr(getattr(model.head, group)[level][index], part).weight


def guard_rtmdet_head_shares_convs_and_separates_bns(module) -> None:
    """The head's conv weights are shared across levels; its BatchNorms are not.

    That is RTMDet's "SepBN" head and its defining structural feature — one
    conv tower's worth of weights serving three levels, each level keeping its
    own activation statistics. Both halves are checked, because both failure
    directions train perfectly happily: un-sharing the convs triples the head's
    parameters and changes the model, and sharing the BNs forces three very
    different activation distributions through one set of running statistics.

    Checked by storage identity (``data_ptr``), not by reading the constructor.
    """
    model = _build(module, 3)
    levels = len(model.head.strides)
    assert levels >= 3, f"expected at least three head levels, got {levels}"

    for group in ("cls_convs", "reg_convs"):
        for index in range(len(getattr(model.head, group)[0])):
            base_conv = _rtmdet_tower_tensor(model, group, 0, index, "conv")
            base_bn = _rtmdet_tower_tensor(model, group, 0, index, "norm")
            for level in range(1, levels):
                conv = _rtmdet_tower_tensor(model, group, level, index, "conv")
                bn = _rtmdet_tower_tensor(model, group, level, index, "norm")
                assert conv.data_ptr() == base_conv.data_ptr(), (
                    f"rtmdet_s: {group}[{level}][{index}].conv is a SEPARATE "
                    f"tensor from level 0's. The head is not weight-shared "
                    f"across levels, which is RTMDet's defining head design; "
                    f"an un-shared head trains fine and is a different model."
                )
                assert bn.data_ptr() != base_bn.data_ptr(), (
                    f"rtmdet_s: {group}[{level}][{index}].norm SHARES storage "
                    f"with level 0's. The norm layers must be per-level — "
                    f"that is the 'SepBN' in the head's published name, and "
                    f"sharing them forces three levels' statistics into one. "
                    f"(They are GroupNorm rather than BatchNorm here: BN's "
                    f"running stats are averaged buffers. The per-level "
                    f"SEPARATION is the design; the norm TYPE is not.)"
                )

    # named_parameters() de-duplicates by object identity, so the shared tower
    # is what the optimizer and the averaging service actually see: one copy.
    tower_names = [
        name
        for name, _ in model.head.named_parameters()
        if re.match(r"(cls|reg)_convs\.\d+\.\d+\.conv\.weight$", name)
    ]
    expected = 2 * len(model.head.cls_convs[0])
    assert len(tower_names) == expected, (
        f"rtmdet_s: named_parameters() lists {len(tower_names)} tower conv "
        f"weights ({tower_names}); a shared head has exactly {expected} — one "
        f"per stacked conv per branch, regardless of level count"
    )

    cls_ids = {id(p) for p in model.head.cls_convs.parameters()}
    reg_ids = {id(p) for p in model.head.reg_convs.parameters()}
    assert cls_ids and reg_ids and not (cls_ids & reg_ids), (
        "rtmdet_s: the classification and regression towers share parameters; "
        "RTMDet keeps them separate even though each is shared across levels"
    )


def guard_rtmdet_head_flatten_order_matches_the_priors(module) -> None:
    """The head's flattened logits and the prior table it returns must agree.

    Same construction and the same non-square shapes as the YOLOX guard, and
    the same reason: RTMDet is one prior per location, so a transposed reshape
    is invisible on the square feature maps the template actually builds. The
    prior table is in **pixels**, so the cell coordinates are recovered by
    dividing by the level's stride.
    """
    import torch
    from torch import nn

    model = _build(module, 2)
    head = model.head
    in_channels = [tower[0].conv.in_channels for tower in head.cls_convs]

    features = []
    for level, (height, width) in enumerate(_ORDERING_SHAPES):
        channels = in_channels[level]
        head.cls_convs[level] = nn.ModuleList([nn.Identity()])
        head.reg_convs[level] = nn.ModuleList([nn.Identity()])
        head.rtm_cls[level] = _select_channel_zero(nn, channels, model.num_classes)
        head.rtm_reg[level] = _zero_conv(nn, channels, 4)
        features.append(_positional_feature(torch, channels, height, width))

    with torch.no_grad():
        cls_logits, _, priors = head(tuple(features))

    height, width = _ORDERING_SHAPES[0]
    count = height * width
    stride = float(priors[0, 2])
    cells = [(float(p[0]) / stride, float(p[1]) / stride) for p in priors[:count]]
    _assert_flatten_matches_grid(
        "rtmdet_s", cls_logits[0, :count, 0], cells, [stride] * count
    )


def guard_rtmdet_decode_is_per_image_and_aligned(module) -> None:
    """Decoding is driven directly, above threshold, at batch size two.

    See the YOLOX twin for why a forward pass cannot substitute: at
    initialisation every class scores ~0.01, so the decode is handed nothing to
    get wrong and every check downstream of it is vacuous. This guard is
    duplicated rather than shared with the YOLOX one on purpose — the two
    ``_predictions`` implementations are themselves duplicates (templates
    cannot import siblings), and a copied helper that keeps only the original's
    test keeps none of its protection.
    """
    import torch

    model = _build(module, 2)
    classes = model.num_classes

    # -10 everywhere: sigmoid(-10) is 4.5e-5, below score_thresh, so only the
    # two logits set below survive the filter.
    cls_logits = torch.full((2, 6, classes), -10.0)
    cls_logits[0, 1, 2] = 10.0
    cls_logits[1, 4, 1] = 10.0
    boxes = torch.stack(
        [
            torch.tensor(
                [[float(i) * 10.0, 0.0, float(i) * 10.0 + 8.0, 8.0] for i in range(6)]
            )
        ]
        * 2
    )

    results = model._predictions(cls_logits, boxes, [(64, 64), (64, 64)])

    assert isinstance(results, list) and len(results) == 2, (
        f"rtmdet_s: decoding two images returned "
        f"{len(results) if isinstance(results, list) else type(results).__name__} "
        f"result(s), expected 2. A truncating zip over the wrong axis loses "
        f"images silently and is invisible at batch one."
    )
    for index, (expected_label, expected_x) in enumerate([(2, 14.0), (1, 44.0)]):
        prediction = results[index]
        assert prediction["boxes"].numel(), (
            f"rtmdet_s: image {index} produced no detection although its class "
            f"logit was set to +10"
        )
        best = int(prediction["scores"].argmax())
        label = int(prediction["labels"][best])
        box = prediction["boxes"][best]
        centre_x = float((box[0] + box[2]) / 2.0)
        assert label == expected_label, (
            f"rtmdet_s: image {index}'s top detection is class {label}, "
            f"expected {expected_label}. The score, label and box columns are "
            f"flattened independently and have come apart."
        )
        assert abs(centre_x - expected_x) < 1.0, (
            f"rtmdet_s: image {index}'s top detection is centred at x="
            f"{centre_x:.1f}, expected {expected_x:.1f} — the confident prior "
            f"was paired with another prior's box."
        )


def guard_rtmdet_quality_focal_loss_targets_the_matched_iou(module) -> None:
    """A positive's classification target is its **matched IoU**, not 1.0.

    That soft target is the "quality" in quality focal loss and is what lets
    RTMDet rank boxes by localisation accuracy with no objectness or centreness
    branch. Checked numerically: with a matched IoU of 0.5, the loss over a
    sweep of logits must be minimised where ``sigmoid(logit) == 0.5``, i.e. at
    0. Against a hard 1.0 target the minimum runs off to the top of the sweep.
    """
    import torch

    quality = torch.tensor([0.5])
    labels = torch.tensor([0], dtype=torch.int64)
    sweep = torch.linspace(-6.0, 6.0, 25)
    values = [
        float(
            module._quality_focal_loss(
                torch.tensor([[value, -6.0, -6.0]]), labels, quality
            )
        )
        for value in sweep
    ]
    best = float(sweep[values.index(min(values))])
    assert abs(best) < 1.0, (
        f"rtmdet_s: with a matched IoU of 0.5 the quality focal loss is "
        f"minimised at logit {best:.2f}, i.e. at a predicted score of "
        f"{torch.sigmoid(torch.tensor(best)):.2f}. A soft target regresses the "
        f"IoU, so the minimum belongs at logit 0 (score 0.5); a minimum out at "
        f"the end of the sweep means the target is a hard 1.0 and the head has "
        f"no way to express localisation quality."
    )


# --------------------------------------------------------------------------
# rtmdet_s — soft-label assigner fixtures
# --------------------------------------------------------------------------


def _rtmdet_assign(model, gt_boxes, gt_labels, boxes, priors):
    """Call ``assign`` with neutral (all-zero) class logits, so the cost matrix
    is decided by geometry and IoU alone."""
    import torch

    return model.assign(
        gt_boxes,
        gt_labels,
        boxes,
        torch.zeros(boxes.shape[0], model.num_classes),
        priors,
    )


def guard_rtmdet_dynamic_k_is_dynamic(module) -> None:
    """Each ground truth gets a number of priors derived from its own IoUs.

    Same non-degeneracy requirement as the SimOTA fixture: the first GT has
    four candidates near 0.85 IoU (``k = 3``) and the second exactly one
    (``k = 1``), so equal counts can only mean a fixed k.
    """
    import torch

    model = _build(module, 2)
    priors = torch.tensor(
        [
            [10.0, 10.0, 8.0],
            [12.0, 10.0, 8.0],
            [10.0, 12.0, 8.0],
            [12.0, 12.0, 8.0],
            [60.0, 60.0, 8.0],
            [400.0, 400.0, 8.0],
        ]
    )
    gt_boxes = torch.tensor([[0.0, 0.0, 30.0, 30.0], [55.0, 55.0, 70.0, 70.0]])
    gt_labels = torch.tensor([0, 1], dtype=torch.int64)
    boxes = torch.tensor(
        [
            [0.0, 0.0, 28.0, 28.0],
            [0.5, 0.0, 28.0, 28.5],
            [0.0, 0.5, 28.5, 28.0],
            [0.5, 0.5, 28.5, 28.5],
            [55.0, 55.0, 66.0, 66.0],
            [900.0, 900.0, 910.0, 910.0],
        ]
    )

    positive_mask, matched_gt, _ = _rtmdet_assign(
        model, gt_boxes, gt_labels, boxes, priors
    )
    counts = [int((matched_gt == index).sum()) for index in (0, 1)]
    assert counts[0] > counts[1] == 1, (
        f"rtmdet_s: the soft-label assigner gave the two ground truths "
        f"{counts} priors. The first has four candidates near 0.85 IoU "
        f"(dynamic k = 3) and the second exactly one, so equal counts mean k "
        f"is not derived from the IoUs."
    )
    assert int(positive_mask.sum()) == sum(counts)


def guard_rtmdet_soft_centre_prior_beats_a_better_iou(module) -> None:
    """A prior far from a ground truth's centre loses to a near one, even with
    a worse IoU.

    The soft centre prior is ``10 ** (distance / stride - 3)``, so it grows by
    a decade per stride of distance and dominates every other term past a few
    strides. The fixture's far prior has the better IoU (0.99 vs 0.90) and wins
    outright without the prior.
    """
    import torch

    model = _build(module, 2)
    priors = torch.tensor([[48.0, 48.0, 8.0], [10.0, 10.0, 8.0]])
    gt_boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    gt_labels = torch.tensor([0], dtype=torch.int64)
    # near prior: IoU 0.90; far prior: IoU 0.99 — deliberately the better box.
    boxes = torch.tensor([[0.0, 0.0, 90.0, 100.0], [0.0, 0.0, 99.0, 100.0]])

    positive_mask, _, _ = _rtmdet_assign(
        model, gt_boxes, gt_labels, boxes, priors
    )
    assert positive_mask.tolist() == [True, False], (
        f"rtmdet_s: got positives {positive_mask.tolist()}, expected "
        f"[True, False]. The prior 38 px further from the ground-truth centre "
        f"was chosen because its IoU is higher — the soft centre prior is not "
        f"in the cost matrix, so assignment ignores locality entirely."
    )


def guard_rtmdet_prefers_the_better_localised_prior(module) -> None:
    """Between two geometrically identical priors, the better box wins.

    Both priors sit at the same point and stride, so the soft centre prior is
    identical and only ``-log(IoU)`` separates them — which isolates that
    term's **sign**. Flip it and the assigner systematically trains the worst
    box available, while the loss falls and every structural guard stays green.
    Duplicated from the YOLOX twin on purpose: the two assigners are separate
    implementations of the same idea, and a copy that keeps only the original's
    test keeps none of its protection.
    """
    import torch

    model = _build(module, 2)
    priors = torch.tensor([[50.0, 50.0, 8.0], [50.0, 50.0, 8.0]])
    gt_boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    gt_labels = torch.tensor([0], dtype=torch.int64)
    # Same point, same stride; IoU 0.50 versus 0.95.
    boxes = torch.tensor([[0.0, 0.0, 50.0, 100.0], [0.0, 0.0, 95.0, 100.0]])

    positive_mask, _, _ = _rtmdet_assign(
        model, gt_boxes, gt_labels, boxes, priors
    )
    assert positive_mask.tolist() == [False, True], (
        f"rtmdet_s: given two priors at the SAME point and stride, one with "
        f"IoU 0.50 and one with 0.95, the assigner chose "
        f"{positive_mask.tolist()} — expected [False, True]. The IoU term in "
        f"the cost is not preferring the better-localised prior."
    )


def guard_rtmdet_centre_prior_scales_with_stride(module) -> None:
    """The centre prior divides the pixel distance by the prior's **own**
    stride, so a coarse level tolerates a distance a fine level does not.

    Both priors in the fixture sit at the same pixel position, 40 px from the
    ground-truth centre, and the fine-level one is given the *better* IoU. Only
    the per-prior stride division makes the coarse one win; a constant stride
    makes the two levels behave identically.
    """
    import torch

    model = _build(module, 2)
    priors = torch.tensor([[60.0, 60.0, 8.0], [60.0, 60.0, 32.0]])
    gt_boxes = torch.tensor([[0.0, 0.0, 200.0, 200.0]])
    gt_labels = torch.tensor([0], dtype=torch.int64)
    boxes = torch.tensor([[0.0, 0.0, 190.0, 200.0], [0.0, 0.0, 180.0, 200.0]])

    positive_mask, _, _ = _rtmdet_assign(
        model, gt_boxes, gt_labels, boxes, priors
    )
    assert positive_mask.tolist() == [False, True], (
        f"rtmdet_s: got positives {positive_mask.tolist()}, expected "
        f"[False, True]. Two priors at the SAME pixel position but strides "
        f"(8, 32) were ranked by IoU alone, so the centre prior is not "
        f"dividing by each prior's stride and the head's levels are no longer "
        f"scale-specialised."
    )


def guard_rtmdet_positives_lie_inside_their_matched_box(module) -> None:
    """Only a prior whose point falls inside a ground-truth box may match it.

    The fixture's outside prior has a far better IoU (0.99 vs 0.5) and a small
    centre distance, so without the in-box filter it wins — and the head then
    learns to fire at a point outside the object.
    """
    import torch

    model = _build(module, 2)
    priors = torch.tensor([[8.0, 8.0, 8.0], [24.0, 8.0, 8.0]])
    gt_boxes = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
    gt_labels = torch.tensor([0], dtype=torch.int64)
    boxes = torch.tensor([[0.0, 0.0, 10.0, 20.0], [0.0, 0.0, 19.8, 20.0]])

    positive_mask, matched_gt, _ = _rtmdet_assign(
        model, gt_boxes, gt_labels, boxes, priors
    )
    assert int(positive_mask.sum()) > 0, "rtmdet_s: nothing was assigned at all"
    points = priors[positive_mask][:, :2]
    matched = gt_boxes[matched_gt]
    inside = bool(
        (points[:, 0] > matched[:, 0]).all()
        and (points[:, 1] > matched[:, 1]).all()
        and (points[:, 0] < matched[:, 2]).all()
        and (points[:, 1] < matched[:, 3]).all()
    )
    assert inside, (
        f"rtmdet_s: a positive prior at {points.tolist()} is outside its "
        f"matched box {matched.tolist()}. The in-ground-truth candidate filter "
        f"is gone, so the head is trained to fire at points that are not on "
        f"the object — with a perfectly finite loss."
    )


def guard_rtmdet_breaks_ties_between_ground_truths(module) -> None:
    """A prior claimed by two ground truths goes to exactly one of them.

    Observable through the matched IoU, which is built as
    ``(matching * ious).sum(1)`` and so reports the sum of both IoUs for a
    doubly-claimed prior — a "quality" above 1, which the focal loss then uses
    as a BCE target.
    """
    import torch

    model = _build(module, 2)
    priors = torch.tensor([[8.0, 8.0, 8.0], [400.0, 400.0, 8.0]])
    gt_boxes = torch.tensor([[0.0, 0.0, 20.0, 20.0], [0.0, 0.0, 22.0, 22.0]])
    gt_labels = torch.tensor([0, 1], dtype=torch.int64)
    boxes = torch.tensor([[0.0, 0.0, 20.0, 20.0], [900.0, 900.0, 910.0, 910.0]])

    positive_mask, matched_gt, matched_ious = _rtmdet_assign(
        model, gt_boxes, gt_labels, boxes, priors
    )
    assert int(positive_mask.sum()) == 1, (
        f"fixture is degenerate: expected both ground truths to want the same "
        f"single prior, got {int(positive_mask.sum())} positives"
    )
    worst = float(matched_ious.max())
    assert worst <= 1.0 + 1e-4, (
        f"rtmdet_s: a prior reports a matched IoU of {worst:.3f}. An IoU "
        f"cannot exceed 1, so this prior is assigned to BOTH ground truths and "
        f"its soft classification target is their sum — the multi-ground-truth "
        f"tie-break is missing."
    )
    assert len(matched_gt) == 1


# --------------------------------------------------------------------------
# the guard tables, and the mutations that prove each can go red
# --------------------------------------------------------------------------

YOLOX_GUARDS = {
    "decoupled_head": guard_yolox_head_is_decoupled,
    "decode_per_level_stride": guard_yolox_decode_scales_by_each_levels_stride,
    "head_flatten_order": guard_yolox_head_flatten_order_matches_the_grid,
    "decode_per_image": guard_yolox_decode_is_per_image_and_aligned,
    "dynamic_k": guard_yolox_dynamic_k_is_dynamic,
    "centre_region_candidates": guard_yolox_centre_region_creates_candidates,
    "in_box_candidates": guard_yolox_in_box_rule_admits_distant_candidates,
    "csp_bottleneck_width": guard_yolox_csp_bottleneck_runs_at_full_branch_width,
    "prefers_better_iou": guard_yolox_prefers_the_better_localised_anchor,
    "iou_scaled_class_target": guard_yolox_class_target_is_scaled_by_the_matched_iou,
    "outside_box_penalty": guard_yolox_penalises_candidates_outside_the_box,
    "centre_radius_stride": guard_yolox_centre_radius_scales_with_stride,
    "tie_break": guard_yolox_breaks_ties_between_ground_truths,
    "positives_reach_box_branch": guard_positives_reach_the_box_regression_branch,
    "original_coordinates": guard_predictions_are_in_original_image_coordinates,
    "declared_size_measured": guard_declared_image_size_is_the_measured_edge,
    "module_tree_size": guard_module_tree_size_is_pinned,
    "published_architecture": guard_matches_the_published_architecture,
    "no_network": guard_constructs_with_no_network,
    "overfits_one_object": guard_overfits_a_single_object,
}

RTMDET_GUARDS = {
    "shared_conv_separate_bn": guard_rtmdet_head_shares_convs_and_separates_bns,
    "soft_quality_target": guard_rtmdet_quality_focal_loss_targets_the_matched_iou,
    "csp_bottleneck_width": guard_rtmdet_csp_block_runs_at_full_branch_width,
    "head_flatten_order": guard_rtmdet_head_flatten_order_matches_the_priors,
    "decode_per_image": guard_rtmdet_decode_is_per_image_and_aligned,
    "dynamic_k": guard_rtmdet_dynamic_k_is_dynamic,
    "soft_centre_prior": guard_rtmdet_soft_centre_prior_beats_a_better_iou,
    "centre_prior_stride": guard_rtmdet_centre_prior_scales_with_stride,
    "prefers_better_iou": guard_rtmdet_prefers_the_better_localised_prior,
    "inside_box_filter": guard_rtmdet_positives_lie_inside_their_matched_box,
    "tie_break": guard_rtmdet_breaks_ties_between_ground_truths,
    "positives_reach_box_branch": guard_positives_reach_the_box_regression_branch,
    "original_coordinates": guard_predictions_are_in_original_image_coordinates,
    "declared_size_measured": guard_declared_image_size_is_the_measured_edge,
    "module_tree_size": guard_module_tree_size_is_pinned,
    "published_architecture": guard_matches_the_published_architecture,
    "no_network": guard_constructs_with_no_network,
    "overfits_one_object": guard_overfits_a_single_object,
}

#: ``(name, template, anchor, replacement, guard)``. The anchor must be unique
#: in the file — ``_mutate`` refuses otherwise, so a drifted anchor is a red
#: rather than a mutation that patches nothing.
MUTATIONS = [
    # -- yolox_s ----------------------------------------------------------
    (
        "yolox/coupled_head",
        YOLOX_PATH,
        """            self.reg_convs.append(
                nn.Sequential(
                    ConvBNAct(hidden, hidden, 3, stride=1),
                    ConvBNAct(hidden, hidden, 3, stride=1),
                )
            )""",
        "            self.reg_convs.append(self.cls_convs[-1])",
        ("yolox", "decoupled_head"),
    ),
    (
        "yolox/single_stride_decode",
        YOLOX_PATH,
        "cell_x, cell_y, stride = grids[:, 0], grids[:, 1], grids[:, 2]\n"
        "    centre_x = (raw[..., 0] + cell_x) * stride",
        "cell_x, cell_y, stride = grids[:, 0], grids[:, 1], "
        "torch.full_like(grids[:, 2], 8.0)\n"
        "    centre_x = (raw[..., 0] + cell_x) * stride",
        ("yolox", "decode_per_level_stride"),
    ),
    (
        "yolox/transposed_head_flatten",
        YOLOX_PATH,
        "output = output.permute(0, 2, 3, 1).reshape(batch, height * width, channels)",
        "output = output.permute(0, 3, 2, 1).reshape(batch, height * width, channels)",
        ("yolox", "head_flatten_order"),
    ),
    (
        "yolox/decode_truncates_the_batch",
        YOLOX_PATH,
        "for boxes, class_scores, (height, width) in zip(decoded, scores, image_sizes):",
        "for boxes, class_scores, (height, width) in zip(decoded[:1], scores[:1], image_sizes):",
        ("yolox", "decode_per_image"),
    ),
    (
        "yolox/decode_misaligns_boxes",
        YOLOX_PATH,
        "            candidate_boxes = boxes[box_index]",
        "            candidate_boxes = boxes[: box_index.shape[0]]",
        ("yolox", "decode_per_image"),
    ),
    (
        "yolox/fixed_k",
        YOLOX_PATH,
        "dynamic_ks = topk_ious.sum(dim=1).int().clamp(min=1)",
        "dynamic_ks = torch.ones_like(topk_ious.sum(dim=1)).int()",
        ("yolox", "dynamic_k"),
    ),
    (
        "yolox/no_centre_rule",
        YOLOX_PATH,
        "candidate_mask = (inside_box | inside_centre).any(dim=0)",
        "candidate_mask = inside_box.any(dim=0)",
        ("yolox", "centre_region_candidates"),
    ),
    (
        "yolox/no_in_box_rule",
        YOLOX_PATH,
        "candidate_mask = (inside_box | inside_centre).any(dim=0)\n"
        "        inside_both = (",
        "candidate_mask = inside_centre.any(dim=0)\n"
        "        inside_both = (",
        ("yolox", "in_box_candidates"),
    ),
    (
        "yolox/fetches_at_construction",
        YOLOX_PATH,
        "        self.backbone = CSPDarknet()",
        '        __import__("socket").getaddrinfo("download.pytorch.org", 443)\n'
        "        self.backbone = CSPDarknet()",
        ("yolox", "no_network"),
    ),
    (
        "yolox/no_outside_penalty",
        YOLOX_PATH,
        "+ SIMOTA_OUTSIDE_PENALTY * (~inside_both).to(cls_cost.dtype)",
        "+ 0.0 * (~inside_both).to(cls_cost.dtype)",
        ("yolox", "outside_box_penalty"),
    ),
    (
        "yolox/constant_centre_radius",
        YOLOX_PATH,
        "radius = CENTER_RADIUS * stride.unsqueeze(0)",
        "radius = CENTER_RADIUS * torch.full_like(stride.unsqueeze(0), 8.0)",
        ("yolox", "centre_radius_stride"),
    ),
    (
        "yolox/no_tie_break",
        YOLOX_PATH,
        """        claimed_by = matching.sum(dim=0)
        contested = claimed_by > 1
        if bool(contested.any()):
            cheapest = torch.argmin(cost[:, contested], dim=0)
            matching[:, contested] = 0
            matching[cheapest, contested] = 1""",
        "        pass  # tie-break removed",
        ("yolox", "tie_break"),
    ),
    (
        "yolox/assign_nothing",
        YOLOX_PATH,
        "        candidate_mask, inside_both = self._candidate_masks(gt_boxes, grids)",
        "        candidate_mask, inside_both = self._candidate_masks(gt_boxes, grids)\n"
        "        candidate_mask = torch.zeros_like(candidate_mask)",
        ("yolox", "positives_reach_box_branch"),
    ),
    (
        "yolox/iou_cost_sign_flipped",
        YOLOX_PATH,
        "iou_cost = -torch.log(ious + _EPS)",
        "iou_cost = torch.log(ious + _EPS)",
        ("yolox", "prefers_better_iou"),
    ),
    (
        "yolox/hard_class_target",
        YOLOX_PATH,
        "    return one_hot * ious.to(dtype).unsqueeze(-1)",
        "    return one_hot",
        ("yolox", "iou_scaled_class_target"),
    ),
    (
        "yolox/objectness_target_zeroed",
        YOLOX_PATH,
        "            obj_targets.append(fg_mask.to(decoded.dtype))",
        "            obj_targets.append(\n"
        "                torch.zeros_like(fg_mask, dtype=decoded.dtype)\n"
        "            )",
        ("yolox", "overfits_one_object"),
    ),
    (
        "yolox/csp_bottleneck_squeezed",
        YOLOX_PATH,
        "            *[Bottleneck(hidden, hidden, 1.0, shortcut) for _ in range(n)]",
        "            *[Bottleneck(hidden, hidden, 0.5, shortcut) for _ in range(n)]",
        ("yolox", "csp_bottleneck_width"),
    ),
    (
        "yolox/csp_bottleneck_squeezed_vs_reference",
        YOLOX_PATH,
        "            *[Bottleneck(hidden, hidden, 1.0, shortcut) for _ in range(n)]",
        "            *[Bottleneck(hidden, hidden, 0.5, shortcut) for _ in range(n)]",
        ("yolox", "published_architecture"),
    ),
    (
        "yolox/depth_multiplier_moved",
        YOLOX_PATH,
        "DEPTH_MULT = 0.33",
        "DEPTH_MULT = 1.00",
        ("yolox", "published_architecture"),
    ),
    (
        "yolox/stateful_norm_returned",
        YOLOX_PATH,
        "self.norm = nn.GroupNorm(_norm_groups(out_ch), out_ch, eps=1e-3)",
        "self.norm = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.03)",
        # The mutation for `module_tree_size` now that the pin carries buffers
        # and tensor counts rather than parameters. It is the RIGHT mutation for
        # that pin: BatchNorm's running_mean/running_var/num_batches_tracked are
        # exactly the buffers the pin exists to hold at zero, because the
        # averaging service ships and averages every buffer each federated
        # round. GroupNorm has none, so a stateful norm coming back moves the
        # pinned reading and nothing else does.
        #
        # Observed red: buffers 23,178 vs pinned 0 (tensors 462 vs 240).
        # Parameters are UNCHANGED by this swap -- GroupNorm and BatchNorm both
        # carry weight+bias -- which is why `published_architecture` stays green
        # under it and this pin is the only thing that can see it.
        ("yolox", "module_tree_size"),
    ),
    (
        "yolox/width_multiplier_moved",
        YOLOX_PATH,
        "WIDTH_MULT = 0.50",
        "WIDTH_MULT = 0.75",
        # Re-pointed from `module_tree_size` to `published_architecture` when
        # the norm became GroupNorm. The pinned block no longer carries the
        # parameter total -- parameters are asserted against the PUBLISHED
        # spec instead, because a self-measured count cannot tell a wrong
        # architecture from a right one. A moved width multiplier is exactly a
        # wrong architecture, so the reference guard is where it belongs; the
        # pinned block would only have caught it by coincidence of arithmetic.
        ("yolox", "published_architecture"),
    ),
    (
        "yolox/transform_runs_at_800",
        YOLOX_PATH,
        "            min_size=self.input_size,\n"
        "            max_size=self.input_size,",
        "            min_size=800,\n            max_size=1333,",
        ("yolox", "declared_size_measured"),
    ),
    (
        "yolox/no_postprocess",
        YOLOX_PATH,
        """        detections = self._predictions(raw, grids, image_list.image_sizes)
        return self.transform.postprocess(
            detections, image_list.image_sizes, original_image_sizes
        )""",
        "        return self._predictions(raw, grids, image_list.image_sizes)",
        ("yolox", "original_coordinates"),
    ),
    # -- rtmdet_s ---------------------------------------------------------
    (
        "rtmdet/unshared_convs",
        RTMDET_PATH,
        "        self._share_convs()",
        "        pass  # _share_convs() removed",
        ("rtmdet", "shared_conv_separate_bn"),
    ),
    (
        "rtmdet/shared_bns",
        RTMDET_PATH,
        """                self.cls_convs[level][index].conv = self.cls_convs[0][index].conv
                self.reg_convs[level][index].conv = self.reg_convs[0][index].conv""",
        """                self.cls_convs[level][index].conv = self.cls_convs[0][index].conv
                self.reg_convs[level][index].conv = self.reg_convs[0][index].conv
                self.cls_convs[level][index].norm = self.cls_convs[0][index].norm
                self.reg_convs[level][index].norm = self.reg_convs[0][index].norm""",
        ("rtmdet", "shared_conv_separate_bn"),
    ),
    (
        "rtmdet/transposed_head_flatten",
        RTMDET_PATH,
        "                cls_score.permute(0, 2, 3, 1).reshape(batch, height * width, num_classes)",
        "                cls_score.permute(0, 3, 2, 1).reshape(batch, height * width, num_classes)",
        ("rtmdet", "head_flatten_order"),
    ),
    (
        "rtmdet/decode_truncates_the_batch",
        RTMDET_PATH,
        """        for image_boxes, class_scores, (height, width) in zip(
            boxes, scores, image_sizes
        ):""",
        """        for image_boxes, class_scores, (height, width) in zip(
            boxes[:1], scores[:1], image_sizes
        ):""",
        ("rtmdet", "decode_per_image"),
    ),
    (
        "rtmdet/decode_misaligns_boxes",
        RTMDET_PATH,
        "            candidate_boxes = image_boxes[box_index]",
        "            candidate_boxes = image_boxes[: box_index.shape[0]]",
        ("rtmdet", "decode_per_image"),
    ),
    (
        "rtmdet/hard_quality_target",
        RTMDET_PATH,
        "        target = quality[positive]",
        "        target = torch.ones_like(quality[positive])",
        ("rtmdet", "soft_quality_target"),
    ),
    (
        "rtmdet/fixed_k",
        RTMDET_PATH,
        "dynamic_ks = topk_ious.sum(dim=0).int().clamp(min=1)",
        "dynamic_ks = torch.ones_like(topk_ious.sum(dim=0)).int()",
        ("rtmdet", "dynamic_k"),
    ),
    (
        "rtmdet/no_centre_prior",
        RTMDET_PATH,
        "        cost = cls_cost + iou_cost + soft_centre_prior",
        "        cost = cls_cost + iou_cost",
        ("rtmdet", "soft_centre_prior"),
    ),
    (
        "rtmdet/constant_stride_prior",
        RTMDET_PATH,
        "            / candidate_priors[:, 2:3]",
        "            / torch.full_like(candidate_priors[:, 2:3], 8.0)",
        ("rtmdet", "centre_prior_stride"),
    ),
    (
        "rtmdet/no_inside_box_filter",
        RTMDET_PATH,
        "        candidate_mask = inside_gt.any(dim=1)",
        "        candidate_mask = torch.ones_like(inside_gt.any(dim=1))",
        ("rtmdet", "inside_box_filter"),
    ),
    (
        "rtmdet/no_tie_break",
        RTMDET_PATH,
        """        contested = matching.sum(dim=1) > 1
        if bool(contested.any()):
            cheapest = torch.argmin(cost[contested], dim=1)
            matching[contested] = 0
            matching[contested.nonzero(as_tuple=True)[0], cheapest] = 1""",
        "        pass  # tie-break removed",
        ("rtmdet", "tie_break"),
    ),
    (
        "rtmdet/assign_nothing",
        RTMDET_PATH,
        "        candidate_mask = inside_gt.any(dim=1)",
        "        candidate_mask = torch.zeros_like(inside_gt.any(dim=1))",
        ("rtmdet", "positives_reach_box_branch"),
    ),
    (
        "rtmdet/iou_cost_sign_flipped",
        RTMDET_PATH,
        "iou_cost = -torch.log(ious + _EPS) * ASSIGNER_IOU_WEIGHT",
        "iou_cost = torch.log(ious + _EPS) * ASSIGNER_IOU_WEIGHT",
        ("rtmdet", "prefers_better_iou"),
    ),
    (
        "rtmdet/quality_target_zeroed",
        RTMDET_PATH,
        "                    quality[positive_mask] = matched_ious",
        "                    quality[positive_mask] = torch.zeros_like(matched_ious)",
        ("rtmdet", "overfits_one_object"),
    ),
    (
        "rtmdet/fetches_at_construction",
        RTMDET_PATH,
        "        self.backbone = CSPNeXt()",
        '        __import__("socket").getaddrinfo("download.pytorch.org", 443)\n'
        "        self.backbone = CSPNeXt()",
        ("rtmdet", "no_network"),
    ),
    (
        "rtmdet/csp_block_squeezed",
        RTMDET_PATH,
        "            *[CSPNeXtBlock(mid, mid, add_identity) for _ in range(n)]",
        "            *[CSPNeXtBlock(mid, max(2, mid // 2), add_identity) for _ in range(n)]",
        ("rtmdet", "csp_bottleneck_width"),
    ),
    (
        "rtmdet/csp_block_squeezed_vs_reference",
        RTMDET_PATH,
        "            *[CSPNeXtBlock(mid, mid, add_identity) for _ in range(n)]",
        "            *[CSPNeXtBlock(mid, max(2, mid // 2), add_identity) for _ in range(n)]",
        ("rtmdet", "published_architecture"),
    ),
    (
        "rtmdet/deepen_factor_moved",
        RTMDET_PATH,
        "DEEPEN_FACTOR = 0.33",
        "DEEPEN_FACTOR = 1.00",
        ("rtmdet", "published_architecture"),
    ),
    (
        "rtmdet/stateful_norm_returned",
        RTMDET_PATH,
        "self.norm = nn.GroupNorm(_norm_groups(out_ch), out_ch, eps=1e-3)",
        "self.norm = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.03)",
        # The mutation for `module_tree_size` now that the pin carries buffers
        # and tensor counts rather than parameters. It is the RIGHT mutation for
        # that pin: BatchNorm's running_mean/running_var/num_batches_tracked are
        # exactly the buffers the pin exists to hold at zero, because the
        # averaging service ships and averages every buffer each federated
        # round. GroupNorm has none, so a stateful norm coming back moves the
        # pinned reading and nothing else does.
        #
        # Observed red: buffers 23,178 vs pinned 0 (tensors 462 vs 240).
        # Parameters are UNCHANGED by this swap -- GroupNorm and BatchNorm both
        # carry weight+bias -- which is why `published_architecture` stays green
        # under it and this pin is the only thing that can see it.
        ("rtmdet", "module_tree_size"),
    ),
    (
        "rtmdet/width_multiplier_moved",
        RTMDET_PATH,
        "WIDEN_FACTOR = 0.50",
        "WIDEN_FACTOR = 0.75",
        # See the yolox note: parameters are asserted against the published
        # spec, not against a self-measured pin, so a moved width multiplier
        # is the reference guard's to catch.
        ("rtmdet", "published_architecture"),
    ),
    (
        "rtmdet/transform_runs_at_800",
        RTMDET_PATH,
        "            min_size=self.input_size,\n"
        "            max_size=self.input_size,",
        "            min_size=800,\n            max_size=1333,",
        ("rtmdet", "declared_size_measured"),
    ),
    (
        "rtmdet/no_postprocess",
        RTMDET_PATH,
        """        detections = self._predictions(cls_logits, boxes, image_list.image_sizes)
        return self.transform.postprocess(
            detections, image_list.image_sizes, original_image_sizes
        )""",
        "        return self._predictions(cls_logits, boxes, image_list.image_sizes)",
        ("rtmdet", "original_coordinates"),
    ),
]

_GUARD_TABLES = {"yolox": YOLOX_GUARDS, "rtmdet": RTMDET_GUARDS}
_TEMPLATE_PATHS = {"yolox": YOLOX_PATH, "rtmdet": RTMDET_PATH}


def test_both_templates_exist() -> None:
    """Guard the guard: every table below is keyed on these two files."""
    for path in (YOLOX_PATH, RTMDET_PATH):
        assert path.is_file(), f"{path} is missing — this whole file is dead"


@pytest.mark.parametrize(
    "model_key,guard_name",
    [(key, name) for key, table in _GUARD_TABLES.items() for name in table],
)
def test_guard_passes_on_the_shipped_template(model_key: str, guard_name: str) -> None:
    """The per-model positive control: every guard holds on the real file."""
    module = _load(_TEMPLATE_PATHS[model_key])
    _GUARD_TABLES[model_key][guard_name](module)


@pytest.mark.parametrize(
    "name,path,anchor,replacement,target",
    MUTATIONS,
    ids=[entry[0] for entry in MUTATIONS],
)
def test_mutation_is_caught_by_its_guard(
    name: str, path: pathlib.Path, anchor: str, replacement: str, target
) -> None:
    """Point the guard at a template edited to break exactly what it checks.

    A guard that cannot be made to fail proves nothing about the code it
    covers, and for a hand-written detector "it returned a loss dict" is our
    own code answering its own question. Keeping the mutation in the suite is
    what stops a guard rotting into a tautology as the template changes.
    """
    model_key, guard_name = target
    mutated = _mutate(path, anchor, replacement)
    guard = _GUARD_TABLES[model_key][guard_name]

    with pytest.raises(AssertionError) as excinfo:
        guard(mutated)

    message = str(excinfo.value)
    assert message.strip(), f"{name}: the guard failed with an empty message"
    # The failure must be the guard's own, not a fixture-degeneracy assertion
    # (those exist in several guards and would make a mutation look caught).
    assert "fixture is degenerate" not in message, (
        f"{name}: the mutation tripped the fixture-degeneracy assertion rather "
        f"than the guard itself — the fixture no longer exercises the rule "
        f"under test:\n{message}"
    )


@pytest.mark.parametrize(
    "model_key,guard_name",
    [(key, name) for key, table in _GUARD_TABLES.items() for name in table],
)
def test_every_guard_has_a_mutation(model_key: str, guard_name: str) -> None:
    """No guard may be un-proven.

    Without this, adding a guard and forgetting its mutation leaves an
    assertion nobody has ever seen fail — which is how a wrong assigner ships
    green.
    """
    covered = {target for *_, target in MUTATIONS}
    assert (model_key, guard_name) in covered, (
        f"{model_key}/{guard_name} has no entry in MUTATIONS. Every guard here "
        f"must be shown able to go red; add the textual edit that breaks it."
    )


def test_mutations_still_train(caplog) -> None:
    """The point of the whole file, stated as a test: the mutations do NOT
    break training.

    Each assigner mutation below leaves ``model(images, targets)`` returning a
    finite loss dict, so ``test_od_torchvision_family_train_step.py`` stays
    green against every one of them. That is the reason the guards above exist,
    and asserting it here stops someone concluding the train-step test already
    covers this.
    """
    import torch

    silent = [
        entry for entry in MUTATIONS
        if entry[0]
        in {
            "yolox/fixed_k",
            "yolox/no_centre_rule",
            "yolox/no_outside_penalty",
            "yolox/assign_nothing",
            "rtmdet/fixed_k",
            "rtmdet/no_centre_prior",
            "rtmdet/no_inside_box_filter",
            "rtmdet/assign_nothing",
        }
    ]
    assert len(silent) == 8, f"expected 8 silent mutations, selected {len(silent)}"

    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0], [70.0, 70.0, 110.0, 120.0]]),
            "labels": torch.tensor([0, 1], dtype=torch.int64),
        },
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        },
    ]
    images = [torch.rand(3, 128, 160), torch.rand(3, 144, 128)]

    for name, path, anchor, replacement, _ in silent:
        module = _mutate(path, anchor, replacement)
        model = _build(module, 3)
        model.train()
        losses = model(images, targets)
        assert isinstance(losses, dict) and losses, f"{name}: no loss dict"
        for key, value in losses.items():
            assert torch.isfinite(value).all(), f"{name}: loss {key} is {value!r}"
