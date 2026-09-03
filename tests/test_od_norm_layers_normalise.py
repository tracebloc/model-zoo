"""Every OD template's normalisation layers must actually normalise a
from-scratch build (backend#3093).

The defect this pins
--------------------
``FrozenBatchNorm2d`` at construction holds ``weight=1``, ``bias=0``,
``running_mean=0``, ``running_var=1``. It therefore computes::

    (x - 0) / sqrt(1 + eps) * 1 + 0   ==   x

i.e. the identity, up to ``eps``. Those four buffers only mean anything once a
pretrained checkpoint loads real statistics into them; that is what the layer
is *for*. On a ``weights=None`` backbone there is nothing to freeze and the
layer degenerates to a no-op.

Every template in this repo builds ``weights=None`` (the hub is a closed door,
RFC-0003 D6), and no OD seed is staged yet — backend#3055 is blocked on the
store decision in backend#2659. So on the platform today, twelve shipped OD
templates train with **no backbone normalisation at all**. Not weak
normalisation: none. Measured downstream, activations reach sigma ~= 24 at the
ROI head against ~= 3 with a live BatchNorm, and the loss stays finite and
decreasing throughout — the recurring shape of defects in this area.

Note what CLAUDE.md's federated-averaging section says, because it is how this
got here: "Either freeze BN layers ... or replace with GroupNorm / LayerNorm".
Freezing is the right answer to the averaging problem (running statistics
average badly across non-IID clients) and the wrong answer from scratch, and
the two halves of that sentence are not equivalent. That file is corrected in
the same commit as this one.

What this file does NOT do
--------------------------
It does not choose a replacement norm. That is a decision, not a sweep:
GroupNorm is correct in both regimes and adds no averaged buffers, but it
changes the state_dict key set of twelve templates that exist specifically to
reproduce the torchvision checkpoint architecture for ``strict=True`` seed
loading. Recorded on backend#3093; the twelve are pinned here as known-bad in
the meantime, so a *new* template cannot acquire the defect and the twelve stop
being folklore.

Why the obvious guards do not work
----------------------------------
**Not a distributional assertion.** Anything asserting a *statistic* of a
randomly-initialised model's activations cannot separate these cases reliably —
that is trap 30, DISTRIBUTIONAL-ASSERTION-ON-RANDOM-INIT, and it has already
produced one useless guard in this epic.

**Not an identity check either**, which is the obvious structural version and
is quietly tolerance-bound. ``torch.allclose(x, layer(x))`` is ``True`` for
``FrozenBatchNorm2d`` at ``eps=0`` and at the ``1e-5`` default (measured max
deviation 3.5e-05) but **False at eps=1e-3** — where the layer is still a
no-op, merely scaled by 0.9995. A guard phrased as "is it the identity" passes
such a layer.

The property asserted instead: **a normaliser's response to a constant input
does not depend on that constant.** Feed the same layer two different constant
tensors and compare the two outputs::

    leak = max|f(b) - f(a)| / |b - a|

A layer that removes the input's location and scale gives the same output for
both, so ``leak == 0``; a layer that passes its input through gives
``leak == 1``. This is exact, needs no seed, no training run and no reference
activation statistics, and it holds for every normaliser shape — mean-and-
variance (BatchNorm/GroupNorm/LayerNorm/InstanceNorm), variance-only
(RMSNorm-style), and any of them with a learnable affine on top, since a
constant input collapses to ``bias`` either way.

Measured leaks, torch 2.11 / torchvision 0.26, ``train()`` mode, freshly
constructed:

===================================  ==========
layer                                ``leak``
===================================  ==========
``FrozenBatchNorm2d(eps=0)``         1.0
``FrozenBatchNorm2d(eps=1e-5)``      0.99999
``FrozenBatchNorm2d(eps=1e-3)``      0.9995
``BatchNorm2d`` (any eps)            0.0
``GroupNorm`` (4 .. 2048 channels)   3.1e-05
``LayerNorm`` / ``LayerNorm2d``      0.0
``InstanceNorm2d(affine=True)``      0.0
===================================  ==========

``_MAX_LEAK = 0.01`` therefore sits about two and a half orders of magnitude
above the largest true-normaliser leak and two orders below the smallest
non-normaliser one. Those extremes are pinned in
``test_the_probe_discriminates_between_the_norm_kinds`` rather than left in
this comment, so the margin is checked and not merely claimed.

The two fixture choices that decide it, and the value that kills each
---------------------------------------------------------------------
1. **The mode must be ``train()``.** A freshly constructed ``BatchNorm2d`` has
   ``running_mean=0`` / ``running_var=1`` too, so in ``eval()`` it passes its
   input straight through and leaks 1.0 — an eval-mode probe reddens
   ``faster_rcnn_resnet_v2``, ``retinanet_v2``, ``ssdlite_mobilenet`` and the
   yolo trunks, none of which are broken: they normalise on the batch while
   training, which is the regime this defect breaks. So training is the mode
   probed.
2. **The two probe constants must differ.** With ``a == b`` the denominator is
   zero and every layer trivially returns the same output for both, so *every*
   template would look like a normaliser and the rule could never fire. That is
   the degeneracy, it is asserted against directly, and
   ``test_identical_probe_constants_would_disable_the_rule`` demonstrates it.
   (A *single* constant of 0.0 is the equivalent trap in the identity-check
   formulation, where GroupNorm and a live BatchNorm both return 0.0 unchanged
   and look like identities. That formulation is not used here.)

Scope: the whole object-detection roster, both families. The yolo family builds
its own BatchNorm trunk and passes, so there is no reason to narrow this to
``torchvision_detection`` and inherit a family-selection hole.
"""

import gc

import pytest
from _od import OD_ROOT, ROOT, build_template, od_templates, template_key

#: The two constants fed through every norm layer. Deliberately far apart and
#: far from the zero-mean / unit-variance a normaliser produces.
#:
#: ⚠️ They must DIFFER — that is the whole fixture. See the module docstring
#: and ``test_identical_probe_constants_would_disable_the_rule``.
_PROBE_CONSTANTS = (7.0, 13.0)

#: How much of a constant input shift a layer may pass through and still count
#: as normalising. Not a tuned number: the measured table in the module
#: docstring has a clear gap between 3.1e-05 (worst true normaliser) and 0.9995
#: (best non-normaliser), and this sits roughly in its middle on a log scale.
_MAX_LEAK = 0.01

#: Templates whose norm layers are known NOT to normalise a from-scratch build,
#: tracked as backend#3093. Value is ``(norm class name, number of such
#: modules)`` so a PARTIAL fix — some layers swapped, some not — is as loud as
#: no fix.
#:
#: ⚠️ Asserted in BOTH directions: a listed template that has been fixed fails
#: too, with an instruction to delete its row. A list that quietly tolerates a
#: fixed entry decays into folklore nobody can audit.
#:
#: Do not add a row to silence a new template. A new non-normalising norm is a
#: bug in that template — the whole point of this file is that twelve templates
#: share one wrong default, and a twelfth would be the same defect, not a new
#: one.
NON_NORMALISING = {
    "atss_resnet": ("FrozenBatchNorm2d", 53),
    "cascade_rcnn": ("FrozenBatchNorm2d", 53),
    "centernet_resnet": ("FrozenBatchNorm2d", 53),
    "faster_rcnn_mobilenet": ("FrozenBatchNorm2d", 46),
    "faster_rcnn_mobilenet_320": ("FrozenBatchNorm2d", 46),
    "faster_rcnn_resnet": ("FrozenBatchNorm2d", 53),
    "fcos": ("FrozenBatchNorm2d", 53),
    "gfl_resnet": ("FrozenBatchNorm2d", 53),
    "retinanet": ("FrozenBatchNorm2d", 53),
    # The twelfth. `sparse_rcnn` (model-zoo#246) merged to develop between
    # this guard's CI run and its own merge, so neither PR's checks ever saw
    # the other: green on both branches, red on develop. The base-stale merge
    # race, not a new defect.
    #
    # It is listed rather than fixed, and the ratchet's own message says a new
    # identity norm is "that template's bug", so that needs an argument.
    # `sparse_rcnn` is not a template acquiring the defect AFTER this guard
    # landed; it is a twelfth instance of the same pre-existing one, authored
    # against the same `resnet50(weights=None,
    # norm_layer=misc_nn_ops.FrozenBatchNorm2d)` line as the other eleven and
    # merged in the same window. Fixing it means taking the norm decision on
    # backend#3093 — deferred precisely because GroupNorm forfeits COCO
    # seeding for this whole family (model-zoo#233 / backend#3055) — and doing
    # that for one template while eleven siblings wait would split the roster
    # for no benefit. Its head towers already use GroupNorm; it is the ResNet
    # trunk that is the no-op, exactly as for the other eleven.
    "sparse_rcnn": ("FrozenBatchNorm2d", 53),
    "tood_resnet": ("FrozenBatchNorm2d", 53),
    "vfnet_resnet": ("FrozenBatchNorm2d", 53),
}

#: The list is a RATCHET pinned by EQUALITY, not by an upper bound — the same
#: shape (and the same reasoning) as ``MAX_KNOWN_MISMATCHES`` in
#: ``test_od_declared_resolution.py``, where ``<=`` was found insufficient: it
#: blocks growth above the high-water mark but not RE-GROWTH after a fix. Fix
#: one template, delete its row, and a ``<=`` cap leaves a free slot a later
#: commit can refill with a brand-new non-normalising norm and stay green.
MAX_NON_NORMALISING = 12

#: Templates that carry NO norm module at all, with the reason. Not an
#: exemption from the rule above — a *different* question, recorded so a zero
#: finding cannot be mistaken for a pass.
#:
#: ⚠️ Asserted in both directions: a listed template that acquires a norm
#: module fails (delete its row, the rule then applies), and an unlisted
#: template with no norm modules fails rather than passing vacuously.
NORM_FREE_BY_DESIGN = {
    "ssd_vgg16": (
        "torchvision's SSDFeatureExtractorVGG follows the paper: plain VGG16 "
        "with no BatchNorm anywhere, and an L2 rescale on conv4_3 held as a "
        "bare `scale_weight` parameter rather than an nn norm module. There is "
        "no norm layer here for this rule to be about. Whether a from-scratch, "
        "norm-FREE VGG trunk is trainable at all is a real question and a "
        "separate one from backend#3093; this file does not settle it."
    ),
}


#: The roster, the ``framework`` reader and the import-and-construct all come
#: from ``tests/_od.py`` rather than being re-typed here. They were
#: byte-identical in three OD test files and had already drifted in two ways
#: (the build-module-name prefix and whether ``output_classes`` was consulted)
#: — see model-zoo#251. The per-file "second independent reader" argument still
#: holds where it was made: it is about comparing ``framework`` against
#: ``model_type`` WITHIN one file, which this file does not do.
OD_TEMPLATES = od_templates()
_stem = template_key


def _build(path):
    """Construct a template at its own declared ``output_classes``."""
    return build_template(path, prefix="norm_probe")


def _norm_types(torch):
    """Every module class that claims to normalise.

    ``_NormBase`` covers BatchNorm and InstanceNorm in one, so a template
    reaching for a variant this file has never seen is still probed instead of
    being silently skipped.
    """
    from torchvision.ops import misc as misc_nn_ops

    nn = torch.nn
    return (
        nn.modules.batchnorm._NormBase,
        nn.GroupNorm,
        nn.LayerNorm,
        misc_nn_ops.FrozenBatchNorm2d,
    )


def _candidate_shapes(torch, module):
    """Input shapes to try for one norm module, most specific first.

    Two shapes are needed and the order matters. ``nn.LayerNorm`` with
    ``normalized_shape=(C,)`` wants ``(..., C)``, while torchvision's
    ``LayerNorm2d`` subclasses it and permutes, so it wants ``(N, C, H, W)``
    and *raises* on the 2-D form. Trying the 2-D form first therefore routes
    each to the shape it actually accepts.

    Returning candidates rather than one guess is what lets ``_leaky_norms``
    insist every module was really probed: an exception used to mean "skipped,
    and counted as fine", which is how six ``LayerNorm2d`` modules went
    unverified in the first pass of this work.
    """
    nn = torch.nn
    if isinstance(module, nn.LayerNorm):
        shape = tuple(module.normalized_shape)
        candidates = [(2, *shape)]
        if len(shape) == 1:
            candidates.append((2, shape[0], 4, 4))
        return candidates
    if isinstance(module, nn.GroupNorm):
        return [(2, module.num_channels, 4, 4)]
    weight = getattr(module, "weight", None)
    running_mean = getattr(module, "running_mean", None)
    channels = None
    if weight is not None:
        channels = weight.numel()
    elif running_mean is not None:
        channels = running_mean.numel()
    elif getattr(module, "num_features", None) is not None:
        # A ``_NormBase`` with neither affine parameters nor tracked stats —
        # ``nn.InstanceNorm2d(C)`` at its defaults is exactly that — still
        # carries ``num_features``. Without this branch such a module yields
        # no candidate shape and is then hard-failed as unprobeable, which is
        # the opposite of what this file wants for a valid normaliser.
        # Observed before the fallback was added (model-zoo#251 review):
        #   AssertionError: norm module InstanceNorm2d could not be probed
        #   with any candidate input shape ... Attempts: []
        channels = int(module.num_features)
    if channels is None:
        return []
    return [(2, channels, 4, 4)]


def _leak(torch, module, shapes=None) -> float:
    """How much of a constant input shift ``module`` passes through.

    ``0`` for a layer that normalises the constant away, ``1`` for one that
    hands its input back. Raises if the module cannot be probed at all — an
    unprobeable norm is an open question, not a pass, and that raise is the
    single place this file refuses a module (``_leaky_norms`` used to
    pre-check the same condition, deriving the shapes a second time for every
    one of the several hundred modules in a detector).

    ``shapes`` lets a caller that already derived the candidates hand them
    over; ``None`` derives them here, which is what the hand-built control
    layers want.
    """
    low, high = _PROBE_CONSTANTS
    errors = []
    if shapes is None:
        shapes = _candidate_shapes(torch, module)
    for shape in shapes:
        try:
            with torch.no_grad():
                out_low = module(torch.full(shape, low))
                out_high = module(torch.full(shape, high))
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            errors.append(f"{shape}: {type(exc).__name__}: {exc}")
            continue
        return float((out_high - out_low).abs().max().item() / abs(high - low))
    raise AssertionError(
        f"norm module {type(module).__name__} could not be probed with any "
        f"candidate input shape, so this file cannot say whether it "
        f"normalises. Teach _candidate_shapes about it — do NOT let it fall "
        f"through as if it had passed. Attempts: {errors}"
    )


def _leaky_norms(torch, model) -> tuple[list[tuple[str, str, float]], int]:
    """``(non-normalising modules, number of modules probed)``.

    ``train()`` mode: see the module docstring — a fresh ``BatchNorm2d`` leaks
    1.0 in ``eval()`` and 0.0 while training, and training is the regime this
    defect breaks.
    """
    # Loop-invariant: one tuple for the whole model, not one per module. A
    # from-scratch ResNet-50 detector has several hundred modules and this
    # runs across the whole roster.
    norm_types = _norm_types(torch)
    model.train()
    leaky: list[tuple[str, str, float]] = []
    probed = 0
    for name, module in model.named_modules():
        if not isinstance(module, norm_types):
            continue
        leak = _leak(torch, module, _candidate_shapes(torch, module))
        probed += 1
        if leak > _MAX_LEAK:
            leaky.append((name, type(module).__name__, leak))
    return leaky, probed


def test_the_roster_was_found():
    """Guard the guard: this file is driven by a directory scan, and an empty
    scan would make it pass by checking nothing."""
    assert OD_TEMPLATES, (
        f"no file under {OD_ROOT} declares `framework` — the scan lost the "
        f"tree, and every assertion in this file would pass on an empty roster"
    )
    stems = {_stem(p) for p in OD_TEMPLATES}
    assert len(stems) == len(OD_TEMPLATES), (
        f"two templates share a row key: {sorted(stems)} against "
        f"{len(OD_TEMPLATES)} files — NON_NORMALISING rows would collide"
    )
    missing = sorted(set(NON_NORMALISING) - stems)
    assert not missing, (
        f"NON_NORMALISING names templates the scan did not find: {missing} — "
        f"if they were deleted, delete their rows (and lower "
        f"MAX_NON_NORMALISING) in the same commit"
    )
    unknown = sorted(set(NORM_FREE_BY_DESIGN) - stems)
    assert not unknown, (
        f"NORM_FREE_BY_DESIGN names templates the scan did not find: {unknown}"
    )
    overlap = sorted(set(NORM_FREE_BY_DESIGN) & set(NON_NORMALISING))
    assert not overlap, (
        f"{overlap} are listed as both norm-free and non-normalising; a "
        f"template cannot be both, and the two lists would disagree silently"
    )


@pytest.mark.parametrize("path", OD_TEMPLATES, ids=lambda p: str(p.relative_to(ROOT)))
def test_norm_layers_normalise_a_from_scratch_build(path):
    """Both directions in one place.

    Unlisted template: every norm module must normalise.
    Listed template: the finding must be EXACTLY what was recorded — same norm
    class, same count. A fix (or a partial fix) fails here with an instruction
    to delete the row, which is the same both-directions discipline as
    ``test_od_declared_resolution.py``. It is asserted per template rather than
    also restated in a second loop on purpose: the restatement would rebuild
    twelve ResNet-50 detectors for an assertion already made here, and this
    suite runs on a 16 GB machine.
    """
    torch = pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")

    stem = _stem(path)
    _, model = _build(path)
    try:
        leaky, probed = _leaky_norms(torch, model)
    finally:
        del model
        gc.collect()

    reason = NORM_FREE_BY_DESIGN.get(stem)
    if reason is not None:
        assert probed == 0, (
            f"{stem} is listed in NORM_FREE_BY_DESIGN but now carries {probed} "
            f"norm module(s). Delete its row — the rule applies to it now. The "
            f"recorded reason was: {reason}"
        )
        return

    assert probed, (
        f"{stem} has no norm module at all, so this file checked nothing on "
        f"it. That is either a real finding — a from-scratch trunk with no "
        f"normalisation anywhere — or an architecture that genuinely has none, "
        f"in which case add a row to NORM_FREE_BY_DESIGN saying which. A "
        f"vacuous pass is not an answer."
    )

    known = NON_NORMALISING.get(stem)
    if known is None:
        assert not leaky, (
            f"{stem}: {len(leaky)} of {probed} norm module(s) do not normalise "
            f"a from-scratch build — shift the input by a constant and the "
            f"output shifts with it (leak > {_MAX_LEAK}), so the layer is a "
            f"no-op and this model trains with no normalisation there at all. "
            f"First few (name, class, leak): {leaky[:3]}.\n"
            f"This is the FrozenBatchNorm2d defect in backend#3093: its "
            f"weight=1 / bias=0 / running_mean=0 / running_var=1 buffers are "
            f"meaningful only once a pretrained checkpoint loads real "
            f"statistics into them, and no OD seed is staged (backend#3055). "
            f"Use a norm that is correct from scratch too — GroupNorm "
            f"normalises per sample with no running statistics, so it needs no "
            f"checkpoint and adds no buffers for the averaging service to "
            f"ship. Do not add a row here to silence this."
        )
        return

    found_classes = sorted({cls for _, cls, _ in leaky})
    assert (found_classes, len(leaky)) == ([known[0]], known[1]), (
        f"{stem} is recorded in NON_NORMALISING as {known[1]} x {known[0]}, "
        f"but this build has {len(leaky)} non-normalising module(s) of "
        f"class(es) {found_classes or ['none']} out of {probed} probed.\n"
        f"  - FIXED (none left)? Delete its row from NON_NORMALISING, lower "
        f"MAX_NON_NORMALISING to {MAX_NON_NORMALISING - 1}, and update the "
        f"equality pin in test_the_non_normalising_list_only_ever_shrinks — "
        f"all in this commit. The guard then holds it correct forever.\n"
        f"  - PARTIALLY fixed? Some layers were swapped and some were not; the "
        f"rest still train unnormalised. Finish the template.\n"
        f"  - Changed some other way? backend#3093 needs updating before this "
        f"row does."
    )


def test_the_non_normalising_list_only_ever_shrinks():
    """The ratchet.

    Without it, the cheapest way to green a newly-broken template is to add its
    name to ``NON_NORMALISING`` — the exact failure the list exists to prevent,
    performed on the list itself.

    Legal edits are: delete a row, lower ``MAX_NON_NORMALISING``, and update
    the pin below, all in one commit. Adding a row fails here, so a twelfth
    non-normalising template has to be argued for rather than absorbed.
    """
    assert len(NON_NORMALISING) == MAX_NON_NORMALISING, (
        f"NON_NORMALISING holds {len(NON_NORMALISING)} entries "
        f"({sorted(NON_NORMALISING)}) against a pinned {MAX_NON_NORMALISING}.\n"
        f"  - GREW? A norm that does not normalise in a new template is that "
        f"template's bug. Pick one that works from scratch instead of listing "
        f"it.\n"
        f"  - SHRANK? Good: backend#3093 fixed one. Lower MAX_NON_NORMALISING "
        f"to {len(NON_NORMALISING)} and update the equality pin below in this "
        f"same commit.\n"
        f"Asserted by EQUALITY, not `<=`: an upper bound would let a fix free "
        f"a slot a later commit could quietly refill."
    )
    assert MAX_NON_NORMALISING == 12, (
        f"MAX_NON_NORMALISING is {MAX_NON_NORMALISING}, not the 12 recorded. "
        f"Raising it defeats the ratchet, and it has been raised exactly once "
        f"— 11 to 12 for `sparse_rcnn`, with the argument on its row. "
        f"Lowering it is correct once backend#3093 fixes a template, and "
        f"belongs in the same commit that lowers it."
    )


def test_the_probe_discriminates_between_the_norm_kinds():
    """Control: the probe must answer differently for the norm kinds on the
    roster, and the measured margin either side of ``_MAX_LEAK`` must hold.

    This is the assertion that stops the whole file from being vacuous. If
    ``_leak`` regressed to "everything normalises", every unlisted template
    above would still pass and only the twelve recorded rows would notice; if
    it regressed the other way, the failure would read like a template bug.
    Both directions are pinned here on hand-built layers, with no template
    involved.

    ``FrozenBatchNorm2d`` is checked at three ``eps`` values on purpose.
    ``eps=1e-3`` is the case a plain ``torch.allclose(x, layer(x))`` identity
    check gets WRONG — the layer is still a no-op but scaled by 0.9995, so
    allclose says "not an identity" while this probe still says "does not
    normalise".
    """
    torch = pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")
    from torchvision.models.convnext import LayerNorm2d
    from torchvision.ops import misc as misc_nn_ops

    nn = torch.nn

    # --- must be flagged: leaks the input straight through -----------------
    for eps in (0.0, 1e-5, 1e-3):
        frozen = misc_nn_ops.FrozenBatchNorm2d(64, eps=eps)
        frozen.train()
        leak = _leak(torch, frozen)
        assert leak > _MAX_LEAK, (
            f"a freshly constructed FrozenBatchNorm2d(eps={eps}) passes its "
            f"input through (weight=1, bias=0, running_mean=0, running_var=1) "
            f"and the probe must say so; measured leak {leak:.3g} against a "
            f"threshold of {_MAX_LEAK}. This is the defect the file exists "
            f"for — if the probe cannot see it, nothing here means anything."
        )
        assert leak > 0.99, (
            f"FrozenBatchNorm2d(eps={eps}) leaked {leak:.6g}, not the ~1.0 "
            f"recorded in the module docstring's table. The margin above "
            f"_MAX_LEAK={_MAX_LEAK} is what makes the threshold safe; if it "
            f"has collapsed, re-measure the table before touching this."
        )

    # --- must NOT be flagged: normalises the constant away -----------------
    clean = {
        # nn.InstanceNorm2d at ITS DEFAULTS: affine=False, so no `weight`,
        # and track_running_stats=False, so no `running_mean` either. A
        # perfectly valid normaliser that exposes its channel count only as
        # `num_features` — the latent gap review on model-zoo#251.
        "InstanceNorm2d(defaults)": nn.InstanceNorm2d(64),
        "BatchNorm2d": nn.BatchNorm2d(64),
        "BatchNorm2d(eps=1e-3)": nn.BatchNorm2d(64, eps=1e-3),
        "GroupNorm(32,64)": nn.GroupNorm(32, 64),
        "GroupNorm(32,2048)": nn.GroupNorm(32, 2048),
        "LayerNorm(768)": nn.LayerNorm(768),
        "LayerNorm2d(96)": LayerNorm2d(96),
        "InstanceNorm2d": nn.InstanceNorm2d(64, affine=True),
    }
    for label, layer in clean.items():
        layer.train()
        leak = _leak(torch, layer)
        assert leak <= _MAX_LEAK, (
            f"{label} normalises per batch or per sample and must NOT be "
            f"flagged; measured leak {leak:.3g} against a threshold of "
            f"{_MAX_LEAK}. If BatchNorm2d is flagged, three templates that are "
            f"fine (faster_rcnn_resnet_v2, retinanet_v2, ssdlite_mobilenet) go "
            f"red for no reason — note that the same layer leaks 1.0 in "
            f"eval() mode, which is why the probe runs in train()."
        )
        assert leak < 1e-3, (
            f"{label} leaked {leak:.3g}, far above the 3.1e-05 worst case "
            f"recorded in the module docstring's table. _MAX_LEAK's lower "
            f"margin has collapsed; re-measure the table rather than raising "
            f"the threshold."
        )

    # A fresh BatchNorm2d in eval() is exactly the false positive the train()
    # choice avoids. Pinned so the mode cannot be changed by accident.
    eval_bn = nn.BatchNorm2d(64)
    eval_bn.eval()
    assert _leak(torch, eval_bn) > _MAX_LEAK, (
        "a freshly constructed BatchNorm2d in eval() mode passes its input "
        "through (running_mean=0, running_var=1). If it does not, the reason "
        "_leaky_norms calls model.train() no longer holds and the module "
        "docstring needs rewriting."
    )


def test_identical_probe_constants_would_disable_the_rule():
    """Name the fixture value that would make this rule unable to fire, and
    prove it has not been chosen.

    With ``a == b`` the probe compares a layer's output against itself: every
    layer on the roster, normalising or not, returns the same thing for the
    same input, so the leak is 0 everywhere, all twenty-four templates look
    clean, and the twelve recorded rows become indistinguishable from the
    eleven genuinely clean ones (plus the one that is norm-free by design). Demonstrated on the worst case — the layer this file
    exists to catch — rather than asserted in a comment.
    """
    torch = pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")
    from torchvision.ops import misc as misc_nn_ops

    low, high = _PROBE_CONSTANTS
    assert low != high, (
        f"_PROBE_CONSTANTS holds two equal values ({low}, {high}), which makes "
        f"every layer's leak 0 and this whole file unable to fire. See below."
    )

    frozen = misc_nn_ops.FrozenBatchNorm2d(64)
    frozen.train()
    with torch.no_grad():
        same = torch.full((2, 64, 4, 4), low)
        degenerate = float(
            (frozen(same) - frozen(same.clone())).abs().max().item()
        )
    assert degenerate == 0.0, (
        "probing FrozenBatchNorm2d with one constant twice yields a leak of "
        "zero — i.e. 'normalises' — which is the degeneracy _PROBE_CONSTANTS "
        "avoids by holding two different values. If this no longer holds, the "
        "docstring's reasoning about the fixture needs rewriting, not this "
        "assertion."
    )
    assert _leak(torch, frozen) > _MAX_LEAK, (
        "the same layer, probed with the two DIFFERENT constants, must be "
        "flagged — that contrast is the fixture doing its job"
    )
