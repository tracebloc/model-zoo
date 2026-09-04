"""Every OD template's declared ``image_size`` must be the resolution it runs at
(backend#3058, filed from backend#2982).

The defect this pins
--------------------
``image_size`` is not decoration: the SDK hands it to the edge to size the
dataset. A torchvision detector then applies its own
``GeneralizedRCNNTransform``, which **upscales anything below ``min_size``
straight back to ``min_size``**. So a template declaring 448 while its transform
runs at 800 produces this pipeline:

    dataset delivers 448x448  ->  transform upscales to 800  ->  model trains

The model trains on 448-resolution content stretched to 800. It pays the resize
twice and throws away the detail it would have had if the dataset had delivered
800 natively. Nothing errors, nothing warns, and the only symptom is accuracy
that is worse than the architecture should give — which is indistinguishable
from "detection is hard".

Three shipped templates HAD it (``faster_rcnn_resnet``, ``fcos``,
``retinanet``, all 448 against a transform at 800). **All three are fixed** —
backend#3058, model-zoo#265 — and ``KNOWN_MISMATCHES`` is empty with its ratchet
at zero. This file landed as the guard first and is now also the record of the
fix: it stops a *new* template acquiring the defect, and there is no longer any
template it excuses.

The before/after mAP measurement backend#3058 originally wanted was NOT gated
on: it needs seeded weights from the blocked backend#2659, both arms score ~0
from random init, and an upscale cannot recover detail a downscale discarded —
so it would have established magnitude, not direction. It rides on
backend#3048's sweep once seeds exist.

Asserted in both directions
---------------------------
``KNOWN_MISMATCHES`` is not a skip list. A template in it that has been *fixed*
fails too, with an instruction to delete its row. That is the RFC's
``EXPECTED_RED`` discipline: a list that quietly tolerates a fixed entry decays
into a list nobody trusts, and the next person cannot tell which rows are real.

The declared-vs-built comparison is TAUTOLOGICAL for most templates
-------------------------------------------------------------------
That is the hole this file was rewritten to close (recorded on backend#3058),
and it is worth stating plainly because the check reads as correct.

Comparing a template's declared ``image_size`` against the resolution its own
built model runs at only bites a template that hardcodes a resolution
*different* from its declaration. Any template that wires its own
``image_size`` into its own transform — the normal, correct pattern — satisfies
it **by construction**, because both sides of the comparison come from the same
literal. The check compares a value with itself.

The evidence is concrete: ``efficientdet_d0`` sat on disk declaring
``image_size = 448`` under 20 green tests, including this one, while actually
running at 512. Two of three new templates carried a wrong declaration past it.
Neither was caught here; both were caught by their author pinning against a
published literal per template instead. So the check's real coverage was
exactly the three legacy files already in ``KNOWN_MISMATCHES``, and every future
template inherited the hole. Trap 31, SELF-CONSISTENT-NUMBER: a check whose
expectation is derived from the code under test can only confirm internal
consistency.

The three numbers, and where each comes from
--------------------------------------------
The fix is a third number, from a source the template cannot influence:

1. ``declared``    — the template's ``image_size`` literal (what the SDK hands
   the edge to size the dataset).
2. ``built``       — what the constructed model's own transform resizes to.
3. ``published``   — ``PUBLISHED_RESOLUTION`` below: the resolution the
   *architecture* is specified at. **Independent of the template by
   construction** — nothing in ``model_zoo/`` feeds it.

All three must agree, and **two comparisons are enough to say so**:
``declared == built`` (the original check) and ``declared == published`` (the
new one) together imply ``built == published``. A third test asserting that
pairing directly was written and then removed — it added no coverage and
rebuilt all twenty torchvision detectors, the heavy swin/convnext ones
included, to re-derive an implication (review on model-zoo#252). Stated here
so it is not helpfully re-added: if you find yourself wanting it, check
whether the two existing comparisons already give it to you.

A wrong declaration can therefore no longer hide behind a matching transform
(``efficientdet_d0``'s hole), and a wrong transform can no longer hide behind a
declaration edited to match it — the mirror image, which nothing checked at
all: ``declared == built`` still passes, but ``declared == published`` now
fails.

The obvious objection to a hand-written spec table is that it can be edited to
match a bad template — moving the tautology rather than removing it. So 20 of
its 24 rows are **re-derived at test time** rather than trusted: 9 by
constructing the torchvision builder the template mirrors and reading its
transform, 8 by reading ``min_size``'s default off the torchvision detector
class the template instantiates without overriding, 3 from the vendored engine
family contract. Editing one of those rows to match a bad template turns this
file red. The remaining 3 — ``cascade_rcnn``, ``centernet_resnet``,
``efficientdet_d0`` and ``sparse_rcnn``, all of which build their own transform
from their own ``image_size`` — are hand-written literals with citations, and
that set is pinned by equality so a new template cannot join it by omission.

Adding a row cannot silence anything either: ``PUBLISHED_RESOLUTION`` requires
a row for **every** template with no default and no fallback, and the ratchet on
``KNOWN_MISMATCHES`` still refuses new exemptions.

``test_the_declared_vs_built_comparison_is_self_consistent_by_construction``
demonstrates the tautology on a stub rather than asserting it in this
docstring, so the reasoning above is checked and cannot rot.

Scope: the whole OD roster, both families. The declared-vs-built comparison is
still ``torchvision_detection``-only — the ``yolo`` family has no ``transform``
to read — but the declared-vs-published comparison needs no model at all, so
there is no reason for the yolo templates to go unchecked. Their anchor is the
engine's family contract rather than a paper; see ``PUBLISHED_RESOLUTION``.
"""

import gc
import importlib.util
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parent.parent
OD_ROOT = ROOT / "model_zoo" / "object_detection"
CONTRACTS = pathlib.Path(__file__).parent / "contracts" / "tracebloc_engine"

FAMILY = "torchvision_detection"

#: Templates whose declared ``image_size`` is known NOT to match the resolution
#: they run at, tracked as backend#3058. Value is the declared/effective pair, so
#: a partial change is as loud as no change.
#:
#: **EMPTY as of model-zoo#265**, which fixed the last three
#: (``faster_rcnn_resnet``, ``fcos``, ``retinanet`` — all 448 against a transform
#: at 800). This dict now exists only so that ADDING a row is still the thing the
#: ratchet refuses. Same shape as ``NON_NORMALISING`` in
#: ``test_od_norm_layers_normalise.py`` after model-zoo#262 emptied it.
#:
#: ⚠️ Asserted in BOTH directions — fixing a listed template fails this file
#: until its row is deleted. Do not add a row to silence a new template; a new
#: mismatch is a bug in that template, not a new known issue.
KNOWN_MISMATCHES: dict[str, tuple[int, int]] = {}

#: The list is a RATCHET: it may only ever shrink, and its length is pinned by
#: EQUALITY rather than by an upper bound.
#:
#: Added after a fail-ability sweep, then tightened after review. Four scenarios
#: were checked first: a new template acquiring the defect, a listed one being
#: fixed but left on the list, a listed one changing to a different wrong value
#: (all caught), and adding a row for a newly-broken template (SURVIVED). An
#: exception list that can grow is not a guard, it is a habit, so the count was
#: pinned.
#:
#: ⚠️ Pinning it with `<=` was NOT sufficient, and the hole is worth naming
#: because the weaker version reads as correct. `<=` blocks growth above the
#: high-water mark but not RE-GROWTH after a fix: fix a template, delete its row,
#: and the length drops to 2 while the cap stays 3 — both assertions still pass,
#: and a later commit can drop a brand-new mismatch into the freed slot and stay
#: green. That is exactly the evasion the ratchet exists to stop. The sweep
#: missed it because it tested ADDING a row above the cap and never
#: DELETE-then-RE-ADD.
#:
#: With equality, deleting a row forces this number down in the same commit, and
#: the `MAX_KNOWN_MISMATCHES == <n>` pin below keeps that a conscious,
#: reviewable edit rather than a silent one. The legal edit is therefore
#: *enforced* rather than merely documented.
MAX_KNOWN_MISMATCHES = 0

#: How each row of ``PUBLISHED_RESOLUTION`` is anchored — i.e. what would have
#: to change for the row to legitimately change. Three of the four kinds are
#: **re-derived at test time** from something outside ``model_zoo/``, so
#: editing such a row to match a bad template turns this file red instead of
#: silencing it. That is the difference between a spec table and a wish list.
#:
#: ``BUILDER``       — the template mirrors a torchvision builder. The
#:                     expectation is read out of ``torchvision.models.detection``
#:                     by CONSTRUCTING that builder and reading its transform.
#: ``CLASS_DEFAULT`` — the template instantiates a torchvision detector class
#:                     without passing ``min_size``, so the resolution IS
#:                     torchvision's documented reference default. Read out of
#:                     the class signature.
#: ``ENGINE``        — the platform's contract for the family overrides the
#:                     architecture's published resolution. Only ``yolo``: the
#:                     engine's ``YoloHandler`` takes a grid tensor at a fixed
#:                     input size, so a yolo template is bound by the handler,
#:                     not by its paper. Read out of the vendored engine
#:                     contract.
#: ``MODERN_YOLO``   — the template is one of the hand-written modern YOLOs,
#:                     every one of which is published and evaluated at the
#:                     same resolution. The row carries no number of its own:
#:                     it is checked against ``MODERN_YOLO_RESOLUTION``, one
#:                     cited family fact shared by all of them, the way the
#:                     ``ENGINE`` rows are checked against the engine contract.
#:                     Added by model-zoo#258 because ``UNVERIFIABLE_LITERALS``
#:                     had reached the 8 its own message named as the limit and
#:                     asked the next author for a derivation source for this
#:                     family instead of a ninth row.
#: ``LITERAL``       — the architecture has no reference implementation in our
#:                     stack AND shares its authority with nothing else, so the
#:                     row is a hand-written number with a citation. **These are
#:                     the only rows that could be edited to match a bad
#:                     template**, which is why the set of them is pinned by
#:                     equality below rather than open-ended.
BUILDER = "torchvision-builder"
CLASS_DEFAULT = "torchvision-class-default"
ENGINE = "engine-contract"
MODERN_YOLO = "modern-yolo-family"
LITERAL = "published-literal"

#: The resolution every modern-YOLO scale is published, trained and evaluated
#: at, and the citation for it. ONE fact, anchoring FOUR rows.
#:
#: This is the "derivation source for the hand-written family" the
#: ``UNVERIFIABLE_LITERALS`` pin asked the next author for (model-zoo#237) —
#: delivered rather than deferred with a ninth row. It works the same way the
#: ``ENGINE`` kind does: one authority, many rows, so an individual row can no
#: longer be quietly edited to match a bad template. It must equal this
#: constant, and changing this constant moves all four rows at once and shows up
#: as such in a diff.
#:
#: ⚠️ WHAT THIS IS NOT. It is not a re-derivation from our own stack the way
#: BUILDER and CLASS_DEFAULT are — nothing here constructs an upstream object
#: and reads its transform, because these templates have no upstream object in
#: our dependency set to construct. The honest claim is narrower: four
#: unreviewable literals have been collapsed into ONE reviewable literal plus a
#: family membership test. That is a real reduction in the soft spot, and it is
#: deliberately not described as more than that.
MODERN_YOLO_RESOLUTION = 640
MODERN_YOLO_CITATION = (
    "Every modern-YOLO scale is published, trained and evaluated at 640x640 on "
    "MS COCO: YOLOX (Ge et al. 2021) sec. 3 test size 640; Ultralytics YOLOv8 "
    "(2023) default imgsz=640; YOLOv9 (Wang et al. 2024) sec. 4.1; YOLOv10 "
    "(Wang et al. 2024, NeurIPS, arXiv:2405.14458) results table Test Size 640; "
    "Ultralytics YOLO11 (2024) cfg/models/11/yolo11.yaml, whose per-scale "
    "summary comment carries the parameter count and the GFLOPs on ONE line, "
    "and whose s row's 21.7 GFLOPs is reproduced at 640 and at no other edge "
    "(13.9 at 512, 31.3 at 768, measured with ultralytics 8.3.0) — so the "
    "9,458,752-parameter figure on that line is quoted at 640; "
    "Ultralytics YOLO12 (2025) cfg/models/12/yolo12.yaml the same way, and its "
    "s row's 21.7 GFLOPs is likewise reproduced at 640 and at no other "
    "32-divisible edge (19.6 at 608, 23.9 at 672, measured with ultralytics "
    "8.3.78), with YOLOv12 (Tian, Ye & Doermann, arXiv:2502.12524) evaluated at "
    "640 on MS COCO; "
    "cfg/default.yaml sets imgsz: 640. "
    "Each template's own published parameter figure is quoted at that scale."
)

#: The resolution each architecture is SPECIFIED at, with its anchor and the
#: detail needed to re-derive or review it. This is the number that makes this
#: file non-tautological: it comes from outside ``model_zoo/`` entirely.
#:
#: ⚠️ EVERY OD template needs a row — asserted in both directions, with no
#: default and no ``.get`` fallback. That is deliberate: a lookup that falls
#: back to "no opinion" is how a new template inherits the hole this table
#: exists to close.
PUBLISHED_RESOLUTION: dict[str, tuple[int, str, str]] = {
    # --- mirrors a torchvision builder: re-derived by building it ----------
    "faster_rcnn_resnet": (800, BUILDER, "fasterrcnn_resnet50_fpn"),
    "faster_rcnn_resnet_v2": (800, BUILDER, "fasterrcnn_resnet50_fpn_v2"),
    "faster_rcnn_mobilenet": (800, BUILDER, "fasterrcnn_mobilenet_v3_large_fpn"),
    "faster_rcnn_mobilenet_320": (320, BUILDER, "fasterrcnn_mobilenet_v3_large_320_fpn"),
    "retinanet": (800, BUILDER, "retinanet_resnet50_fpn"),
    "retinanet_v2": (800, BUILDER, "retinanet_resnet50_fpn_v2"),
    "fcos": (800, BUILDER, "fcos_resnet50_fpn"),
    "ssd_vgg16": (300, BUILDER, "ssd300_vgg16"),
    "ssdlite_mobilenet": (320, BUILDER, "ssdlite320_mobilenet_v3_large"),
    # --- instantiates a torchvision detector class and inherits its
    # --- reference default: re-derived from the class signature -----------
    "faster_rcnn_convnext_small": (800, CLASS_DEFAULT, "FasterRCNN"),
    "faster_rcnn_swin_t": (800, CLASS_DEFAULT, "FasterRCNN"),
    "fcos_convnext_small": (800, CLASS_DEFAULT, "FCOS"),
    "fcos_swin_t": (800, CLASS_DEFAULT, "FCOS"),
    "atss_resnet": (800, CLASS_DEFAULT, "RetinaNet"),
    "gfl_resnet": (800, CLASS_DEFAULT, "RetinaNet"),
    "tood_resnet": (800, CLASS_DEFAULT, "RetinaNet"),
    "vfnet_resnet": (800, CLASS_DEFAULT, "RetinaNet"),
    # --- yolo family: anchored to the ENGINE, not to the papers. YOLOv5-s and
    # --- YOLOv8 are published at 640, but the engine's YoloHandler takes a
    # --- fixed-size grid tensor and these templates resize their output to a
    # --- 7x7 grid to match it, so the handler wins. That the v5/v8 templates
    # --- are forced into a v1-shaped grid at all is a separate defect from
    # --- this one; recorded on backend#3058, not fixed here.
    "yolo_v1": (448, ENGINE, "YOLOv1 (Redmon et al. 2016) is 448x448, and the engine's yolo contract agrees"),
    "yolo_v5": (448, ENGINE, "engine yolo family contract: fixed 448 input, 7x7 grid"),
    "yolo_v8": (448, ENGINE, "engine yolo family contract: fixed 448 input, 7x7 grid"),
    # --- no reference implementation in our stack: hand-written literal.
    # --- All three build their OWN GeneralizedRCNNTransform from their own
    # --- image_size, which is exactly the self-consistent pattern that made
    # --- the old check tautological — so these are the rows where the
    # --- citation is doing the work, and the ones a reviewer should check.
    "cascade_rcnn": (800, LITERAL, "Cascade R-CNN (Cai & Vasconcelos 2018) trains at 800/1333, the R50-FPN reference recipe"),
    "sparse_rcnn": (800, LITERAL, "Sparse R-CNN (Sun et al. 2021) sec. 4.1 trains at 800/1333, the R50-FPN reference recipe"),
    "centernet_resnet": (512, LITERAL, "CenterNet (Zhou et al. 2019) sec. 4: 512x512 input"),
    "efficientdet_d0": (512, LITERAL, "EfficientDet (Tan et al. 2020) table 1: D0 input is 512x512"),
    # NOT the same template as `yolo_v8` above, and the two must not be
    # collapsed. `yolo_v8` is forced into the engine's v1-shaped 7x7 grid
    # contract at 448; `yolov8_s` (model-zoo#253) is the genuine multi-scale
    # YOLOv8-S, anchor-free with a DFL box branch, and Ultralytics specifies it
    # at 640 -- which is also the `imgsz` default its published 11,166,560
    # parameter summary is measured at. A row that read 448 here would be
    # describing the wrong architecture.
    # the genuine multi-scale YOLOv8-S (model-zoo#253), anchor-free with a DFL box branch — NOT `yolo_v8`, which the engine's contract fixes at 448 in a v1-shaped 7x7 grid
    "yolov8_s": (MODERN_YOLO_RESOLUTION, MODERN_YOLO, "the genuine multi-scale YOLOv8-S (model-zoo#253), anchor-free with a DFL box branch — NOT `yolo_v8`, which the engine's contract fixes at 448 in a v1-shaped 7x7 grid. Ultralytics YOLOv8 (2023) default imgsz=640; the yolov8s.yaml scale its published 11,166,560-parameter summary is quoted at"),
    # Same distinction as `yolov8_s` above: this is the GELAN YOLOv9-S
    # (model-zoo#255), not a member of the engine's fixed-448 yolo contract.
    # YOLOv9 (Wang et al. 2024) sec. 4.1 trains and evaluates on MS COCO at
    # 640x640, and 640 is also the scale `yolov9s.yaml`'s own 7318368-parameter
    # header is quoted at -- the anchor #255's count is verified against.
    # the GELAN YOLOv9-S (model-zoo#255)
    "yolov9_s": (MODERN_YOLO_RESOLUTION, MODERN_YOLO, "the GELAN YOLOv9-S (model-zoo#255). YOLOv9 (Wang et al. 2024) sec. 4.1 trains and evaluates at 640x640 on MS COCO; the yolov9s.yaml scale its published 7,318,368-parameter header is quoted at"),
    # Same distinction as `yolov8_s`/`yolov9_s` above and worth repeating,
    # because `yolox` reads like a yolo-family name and is not one: this is the
    # multi-scale anchor-free YOLOX-S (model-zoo#237), which declares
    # `torchvision_detection`, NOT a member of the engine's fixed-448 7x7-grid
    # `yolo` contract. A row of 448 here would pin the wrong architecture.
    # the multi-scale anchor-free YOLOX-S (model-zoo#237); the name reads like the legacy yolo family and is not in it
    "yolox_s": (MODERN_YOLO_RESOLUTION, MODERN_YOLO, "the multi-scale anchor-free YOLOX-S (model-zoo#237); the name reads like the legacy yolo family and is not in it. YOLOX (Ge et al. 2021) sec. 3 and the official Megvii README standard-models table: YOLOX-s is size 640, the scale its published 9.0M-parameter row is quoted at"),
    # the NMS-free dual-assignment YOLOv10-S (model-zoo#258). ⚠️ Its published
    # parameter figure has THREE variants and this row's citation is quoted at
    # the dual-head one; do not "correct" it to the README's 7.2M, which is the
    # fused one2one-only deployed graph. The resolution is 640 in every variant.
    "yolov10_s": (MODERN_YOLO_RESOLUTION, MODERN_YOLO, "the NMS-free dual-assignment YOLOv10-S (model-zoo#258). YOLOv10 (Wang et al. 2024, NeurIPS, arXiv:2405.14458) results table gives Test Size 640 for every scale; the scale its published 8,128,272-parameter dual-head summary is quoted at"),
    # the C3k2/C2PSA YOLO11-S (model-zoo#263). ⚠️ THE STEM HAS NO `v`: upstream
    # dropped it at this generation (`yolo11.yaml`, `yolo11s.pt`), so `yolov11_s`
    # is a name that has never existed and searching for it finds nothing.
    # Unlike `yolov10_s` above there is only ONE published figure worth quoting:
    # unfused 9,458,752 against fused 9,443,760, a 0.16% gap the docs table's
    # one-decimal "9.4M" cannot distinguish either way.
    "yolo11_s": (MODERN_YOLO_RESOLUTION, MODERN_YOLO, "the C3k2/C2PSA YOLO11-S (model-zoo#263), NMS-based unlike its yolov10_s neighbour. Ultralytics YOLO11 (2024) cfg/models/11/yolo11.yaml quotes '9458752 parameters, 9458736 gradients, 21.7 GFLOPs' on one summary line per scale, and 21.7 GFLOPs is reproduced only at 640 — the resolution is recovered from the same line as the parameter count rather than asserted alongside it"),
    # the A2C2f/Area-Attention YOLOv12-S (model-zoo#266). ⚠️ THE STEM KEEPS THE
    # `v` AND THAT IS THE OPPOSITE CALL TO `yolo11_s` ABOVE: this generation has
    # TWO upstreams that disagree about the name. The authors' paper and repo are
    # `yolov12` (sunsmarterjie/yolov12, yolov12s.pt); Ultralytics integrated it
    # as `yolo12`. So both spellings are real here, where `yolov11` never was.
    # ⚠️ AND THERE ARE THREE PUBLISHED PARAMETER FIGURES. This row's citation
    # quotes the ULTRALYTICS one, which is what the template is built against
    # and the only one that lives in an installable package. The authors' v1.0
    # tag says 9,285,632 for the same topology (its `Conv` signature reads the
    # padding argument as `bias`, giving every Area-Attention positional
    # encoding a bias: +1,536 here); their current `main` is YOLOv12-turbo at
    # 9,127,424 / 19.7 GFLOPs and is a DIFFERENT architecture — grouped
    # downsample convs at yaml layers 1 and 3. Do not reconcile this row
    # against either.
    "yolov12_s": (MODERN_YOLO_RESOLUTION, MODERN_YOLO, "the attention-centric YOLOv12-S (model-zoo#266) — R-ELAN backbone with Area Attention, no SPPF and no C2PSA, NMS-based. Ultralytics YOLO12 (2025) cfg/models/12/yolo12.yaml quotes '9,284,096 parameters, 9,284,080 gradients, 21.7 GFLOPs' on one summary line per scale, and 21.7 GFLOPs is reproduced at 640 and at no other 32-divisible edge (19.6 at 608, 23.9 at 672, 13.9 at 512, 31.2 at 768, measured with ultralytics 8.3.78) — so the resolution is recovered from the same line that carries the parameter count. Tian, Ye & Doermann, arXiv:2502.12524; cfg/default.yaml sets imgsz: 640 independently"),
    # RTMDet is NOT a YOLO — CSPNeXt backbone, mmdetection lineage — so it does
    # not join the family anchor above even though its resolution matches.
    # Membership is the thing being asserted, not the number.
    "rtmdet_s": (640, LITERAL, "RTMDet (Lyu et al. 2022); mmdetection configs/rtmdet/README.md COCO table gives the RTMDet-s row as input size 640, 8.89M parameters, 44.6 box AP"),
}

#: The rows whose expectation is a hand-written literal rather than re-derived.
#: Pinned by EQUALITY, and small on purpose: 5 of 25. A new template landing
#: here has to be added deliberately, with a citation and a reviewer looking at
#: it — not by omission from the derivation maps, which is how a "spec" table
#: quietly degrades back into whatever the templates already said.
#:
#: The pin is doing its job rather than being maintenance: `sparse_rcnn`
#: (model-zoo#246) and `yolov8_s` (model-zoo#253) both arrived on `develop`
#: while this PR was open, and each forced a deliberate cited edit here instead
#: of being absorbed. That is the whole design — but it also means this set
#: grows once per hand-written template, so if it ever passes ~8 the honest
#: response is a derivation source for the hand-written family, not a longer
#: list.
UNVERIFIABLE_LITERALS = frozenset(
    {
        "cascade_rcnn",
        "centernet_resnet",
        "efficientdet_d0",
        "sparse_rcnn",
        "rtmdet_s",
    }
)

#: The rows anchored to the modern-YOLO family fact. Pinned by EQUALITY like the
#: literal set, so a template cannot join the family — and inherit its
#: resolution without its own review — merely by being named after a YOLO.
MODERN_YOLO_ROWS = frozenset(
    {
        "yolo11_s",
        "yolov8_s",
        "yolov9_s",
        "yolov10_s",
        "yolov12_s",
        "yolox_s",
    }
)


def _schema_path() -> pathlib.Path:
    paths = sorted(
        CONTRACTS.glob("object_detection_families.v*.json"),
        key=lambda p: int(re.search(r"\.v(\d+)\.json$", p.name).group(1)),
    )
    assert paths, f"no vendored OD families schema under {CONTRACTS}"
    return paths[-1]


def _family_values() -> frozenset[str]:
    """The family name plus every alias.

    Resolved by family, never by the literal ``"torchvision_detection"``:
    ``faster_rcnn_resnet.py`` declares the legacy ``rcnn`` alias, and it is one
    of the three templates this file exists to pin — so a literal-keyed scan
    would silently drop the most important row.
    """
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    entries = [f for f in schema["families"] if f["family"] == FAMILY]
    assert entries, f"{_schema_path().name}: no {FAMILY!r} family entry"
    return frozenset(v.strip().lower() for v in {FAMILY, *entries[0].get("aliases", [])})


FAMILY_VALUES = _family_values()


def _read_model_type(path: pathlib.Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(r'^\s*model_type\s*=\s*["\'](\w*)["\']', text, re.MULTILINE)
    return match.group(1) if match else None


def _other_family_values() -> frozenset[str]:
    """Accepted values routing to a family that is NOT this one.

    The schema says object detection has exactly two families, so this plus
    ``FAMILY_VALUES`` partitions the vocabulary — which is what lets the guard
    below assert COVERAGE without a floor to recompute (backend#2982).
    """
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    values: set[str] = set()
    for entry in schema["families"]:
        if entry["family"] == FAMILY:
            continue
        values |= {entry["family"], *entry.get("aliases", [])}
    return frozenset(v.strip().lower() for v in values)


OTHER_FAMILY_VALUES = _other_family_values()


def _declares_framework(path: pathlib.Path) -> str | None:
    """The module-level ``framework``, or ``None`` for a support module.

    A SECOND, INDEPENDENT regex from ``_read_model_type``: the partition below
    compares the two readers' verdicts, and one reader answering for both would
    make that comparison vacuous the moment it broke.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(r'^\s*framework\s*=\s*["\'](\w*)["\']', text, re.MULTILINE)
    return match.group(1) if match else None


def _od_templates() -> list[pathlib.Path]:
    return [p for p in sorted(OD_ROOT.rglob("*.py")) if _declares_framework(p)]


FAMILY_TEMPLATES = [
    p for p in sorted(OD_ROOT.rglob("*.py"))
    if (_read_model_type(p) or "").strip().lower() in FAMILY_VALUES
]

OD_TEMPLATES = _od_templates()


def _stem(path: pathlib.Path) -> str:
    """Row key for a template.

    The yolo templates are all named ``model.py`` under a versioned directory,
    so a bare ``path.stem`` would collide three ways and the published-resolution
    table would silently hold one row for three architectures.
    """
    return path.parent.name if path.stem == "model" else path.stem


def _read_declared_image_size(path: pathlib.Path) -> int | None:
    """The module-level ``image_size`` literal, read STATICALLY.

    Read without importing on purpose: the declared-vs-published comparison
    needs no model, so it must not need torch either. That is what lets it
    cover the yolo family — and it means the comparison cannot be affected by
    anything the template does at construction time.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    # Anchored at COLUMN ZERO, not ``^\s*``. ``\s*`` consumes leading
    # indentation, so an indented ``image_size = N`` — a class attribute, a
    # local in a builder — would match as if it were the module-level
    # declaration, and the reader's own test claims it does not. No current
    # template trips it, so this was latent; the guarantee is now the one
    # stated (review on model-zoo#252). All 23 OD templates declare
    # ``image_size`` at column zero, so nothing on the roster moves.
    match = re.search(r"^image_size\s*=\s*(\d+)", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def _build(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(
        re.sub(r"\W", "_", f"resolution_{path.stem}"), path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = getattr(module, "main_class", None) or getattr(module, "main_method", None)
    assert entry, f"{path}: neither main_class nor main_method is defined"
    return module, getattr(module, entry)(3)


def _effective_resolution(model):
    """What the model's own transform actually resizes to.

    Two shapes: the SSDs are genuinely fixed-size (``fixed_size=(300, 300)``),
    while the R-CNN/RetinaNet/FCOS family uses ``min_size``/``max_size`` and
    upscales below the minimum.
    """
    transform = getattr(model, "transform", None)
    if transform is None:
        return None
    fixed = getattr(transform, "fixed_size", None)
    if fixed:
        return int(fixed[0])
    min_size = transform.min_size
    if isinstance(min_size, (list, tuple)):
        return int(min_size[0])
    return int(min_size)


def test_family_templates_were_found():
    """Guard the guard: an empty scan would make this file pass by checking
    nothing, and it is driven by a file scan plus a schema lookup."""
    assert "rcnn" in FAMILY_VALUES, (
        f"{_schema_path().name}: {FAMILY!r} lost its legacy 'rcnn' alias — "
        f"faster_rcnn_resnet declares it, so this scan just stopped covering "
        f"the only template on the legacy alias"
    )
    # This was `len(FAMILY_TEMPLATES) >= 4` — a floor every roster PR is
    # invited to raise, i.e. a shared literal with the same serialisation cost
    # the census literal had (backend#2982). It is a PARTITION now: the schema
    # publishes exactly two OD families, so every OD template belongs to this
    # one or to `yolo`, and adding a template moves both sides at once.
    templates = _od_templates()
    assert templates, (
        f"no file under {OD_ROOT} declares `framework` — the scan lost the tree, "
        f"and this file would pass by checking nothing"
    )
    other = {
        p
        for p in templates
        if (_read_model_type(p) or "").strip().lower() in OTHER_FAMILY_VALUES
    }
    assert other, (
        f"no OD template routes to a family other than {FAMILY!r}; the yolo "
        f"roster is part of the tree, so this means model_type reading broke"
    )
    uncovered = sorted(
        str(p.relative_to(ROOT)) for p in set(templates) - other - set(FAMILY_TEMPLATES)
    )
    assert not uncovered, (
        f"OD template(s) in neither the {FAMILY!r} roster this file checks nor "
        f"any other family — they declare a model_type outside the schema's "
        f"vocabulary, or none at all, so no resolution check reaches them: "
        f"{uncovered}"
    )
    unexpected = sorted(
        str(p.relative_to(ROOT)) for p in set(FAMILY_TEMPLATES) - set(templates)
    )
    assert not unexpected, (
        f"the {FAMILY!r} roster contains files that do not declare `framework` — "
        f"a support module cannot be a template: {unexpected}"
    )
    missing = set(KNOWN_MISMATCHES) - {p.stem for p in FAMILY_TEMPLATES}
    assert not missing, (
        f"KNOWN_MISMATCHES names templates the scan did not find: {sorted(missing)} "
        f"— if they were deleted, delete their rows too"
    )


@pytest.mark.parametrize(
    "path", FAMILY_TEMPLATES, ids=lambda p: str(p.relative_to(ROOT))
)
def test_declared_image_size_is_the_resolution_the_model_runs_at(path):
    torch = pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")
    del torch

    module, model = _build(path)
    declared = int(module.image_size)
    effective = _effective_resolution(model)
    assert effective is not None, (
        f"{path.name}: no .transform to read an effective resolution from — if "
        f"this template resizes some other way, this guard needs to learn about it"
    )

    known = KNOWN_MISMATCHES.get(path.stem)
    if known is None:
        assert declared == effective, (
            f"{path.name} declares image_size = {declared} but its transform "
            f"runs at {effective}. The SDK hands image_size to the edge to size "
            f"the dataset, and the transform then rescales to {effective} — so "
            f"the model trains on {declared}-resolution content resized to "
            f"{effective}, paying the resize twice and losing detail. Declare "
            f"{effective}, or set the transform's min_size/fixed_size to "
            f"{declared} if that is really the intent. See backend#3058."
        )
        return

    # A known mismatch: assert it is STILL exactly what was recorded.
    assert (declared, effective) == known, (
        f"{path.name} is listed in KNOWN_MISMATCHES as declared={known[0]} / "
        f"effective={known[1]}, but is now declared={declared} / "
        f"effective={effective}.\n"
        f"  - If this was FIXED (declared == effective), delete its row from "
        f"KNOWN_MISMATCHES; the guard then holds it correct forever.\n"
        f"  - If it changed some other way, backend#3058 needs updating before "
        f"this row does."
    )


def test_known_mismatches_are_all_still_mismatched():
    """The other direction, stated once rather than per template.

    A row that has quietly become correct makes the whole list untrustworthy —
    the next reader cannot tell which entries are real defects and which are
    stale. This fails loudly with an instruction to prune.
    """
    pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")

    stale = []
    for stem, (recorded_declared, recorded_effective) in sorted(KNOWN_MISMATCHES.items()):
        matches = [p for p in FAMILY_TEMPLATES if p.stem == stem]
        assert matches, f"KNOWN_MISMATCHES row {stem!r} matches no template"
        module, model = _build(matches[0])
        declared = int(module.image_size)
        effective = _effective_resolution(model)
        if declared == effective:
            stale.append(f"{stem} (now consistently {declared})")
        else:
            assert (declared, effective) == (recorded_declared, recorded_effective), (
                f"{stem}: recorded {recorded_declared}/{recorded_effective}, "
                f"found {declared}/{effective}"
            )

    assert not stale, (
        f"these KNOWN_MISMATCHES entries are no longer mismatched: "
        f"{', '.join(stale)}. Delete their rows — the guard will then hold them "
        f"correct, which is the point. Leaving a fixed row in place makes the "
        f"list decay into folklore. See backend#3058."
    )


def test_the_known_mismatch_list_only_ever_shrinks():
    """The ratchet.

    Without this, the cheapest way to make a new declared/effective mismatch go
    green is to add its name to ``KNOWN_MISMATCHES`` — which is exactly the
    failure the list exists to prevent, performed on the list itself. Verified
    by mutation: adding a row for a newly-broken template silenced every other
    assertion in this file.

    Asserted by EQUALITY rather than ``<=``. The weaker form reads as correct
    and is not: it blocks growth above the high-water mark but not *re-growth
    after a fix*. Fix a template, delete its row, and the length drops below the
    cap — both a ``<=`` bound and the ``MAX == n`` pin still pass, leaving a free
    slot a later commit can refill with a brand-new mismatch and stay green.

    **The ratchet is now AT ITS FLOOR** — empty, pinned at 0 (model-zoo#265).
    There is no legal edit to these two values left: the only edit the original
    design permitted was *downward*, and down is where they are. A new
    declared/effective mismatch is a bug in the template that introduced it, not
    a row to add here.

    Historically the legal edit was: delete a row, lower
    ``MAX_KNOWN_MISMATCHES``, and update the pin below — all in one commit. That
    is recorded because it is what got us to zero, not because it is still
    available.
    """
    assert len(KNOWN_MISMATCHES) == MAX_KNOWN_MISMATCHES, (
        f"KNOWN_MISMATCHES holds {len(KNOWN_MISMATCHES)} entries "
        f"({sorted(KNOWN_MISMATCHES)}) against a pinned {MAX_KNOWN_MISMATCHES}.\n"
        f"  - GREW? A declared/effective mismatch in a NEW template is a bug in "
        f"that template — fix its image_size instead of listing it.\n"
        f"  - SHRANK? No longer possible: the ratchet is at its floor (0) as "
        f"of model-zoo#265, so a shrink means a row was added and removed, not "
        f"that a template was fixed.\n"
        f"This is asserted by EQUALITY, not `<=`, on purpose: an upper bound "
        f"would let a fix free a slot that a later commit could quietly refill "
        f"with a brand-new mismatch."
    )
    assert MAX_KNOWN_MISMATCHES == 0, (
        f"MAX_KNOWN_MISMATCHES is {MAX_KNOWN_MISMATCHES}, not the 0 this ratchet "
        f"reached when backend#3058 fixed the last three templates "
        f"(faster_rcnn_resnet, fcos, retinanet). It may only ever go DOWN, and "
        f"it is already at the floor: there is no legal edit to this number "
        f"left. A new declared/effective mismatch is a bug in the template that "
        f"introduced it, not a row to add here."
    )



def test_every_od_template_has_a_published_resolution_row():
    """The table must be COMPLETE, in both directions.

    Completeness is the whole anti-tautology mechanism. A lookup that falls
    back to "no opinion" for an unlisted template is exactly how
    ``efficientdet_d0`` went 20 tests green while declaring the wrong number —
    so there is no default, no ``.get``, and a template with no row fails here
    with what to do about it.
    """
    assert OD_TEMPLATES, (
        f"no file under {OD_ROOT} declares `framework` — the scan lost the "
        f"tree, and this table would be trivially complete"
    )
    stems = [_stem(p) for p in OD_TEMPLATES]
    assert len(set(stems)) == len(stems), (
        f"two OD templates share a row key ({sorted(stems)}) — "
        f"PUBLISHED_RESOLUTION rows would collide and one architecture would "
        f"be checked against another's specification"
    )

    missing = sorted(set(stems) - set(PUBLISHED_RESOLUTION))
    assert not missing, (
        f"OD template(s) with no PUBLISHED_RESOLUTION row: {missing}.\n"
        f"Add one: the resolution the ARCHITECTURE is specified at, per its "
        f"paper or reference recipe, with the citation — not the number the "
        f"template happens to declare, and not the number its transform "
        f"happens to run at. Deriving the expectation from the template is "
        f"the tautology this table exists to remove (backend#3058)."
    )
    extra = sorted(set(PUBLISHED_RESOLUTION) - set(stems))
    assert not extra, (
        f"PUBLISHED_RESOLUTION names templates the scan did not find: {extra} "
        f"— if they were deleted, delete their rows too"
    )

    # Anchor-KIND validity is deliberately not re-asserted here:
    # ``test_the_anchor_kinds_partition_the_roster`` owns that partition over
    # this same dict, and two tests failing on one edit tells the reader
    # nothing the first one did not (review on model-zoo#252). This test owns
    # COMPLETENESS.
    for stem, (value, anchor, citation) in sorted(PUBLISHED_RESOLUTION.items()):
        assert isinstance(value, int) and value > 0, f"{stem}: bad resolution {value!r}"
        assert citation.strip(), (
            f"{stem}: empty citation. A row without one is an unreviewable "
            f"literal, which is the thing this table replaces."
        )


@pytest.mark.parametrize(
    "path", OD_TEMPLATES, ids=lambda p: str(p.relative_to(ROOT))
)
def test_declared_image_size_matches_the_published_specification(path):
    """The independent check: ``declared`` against ``published``.

    Needs no model and no torch — which is why it reaches the yolo family that
    the declared-vs-built comparison cannot see, and why nothing the template
    does at construction time can influence its expectation.

    ``KNOWN_MISMATCHES`` is read here as well as by the declared-vs-built
    comparison — one exemption list, one ratchet, both comparisons. It is now
    **empty** (model-zoo#265), so this test exempts nothing: every template's
    declared value is checked against its published specification.
    """
    stem = _stem(path)
    declared = _read_declared_image_size(path)
    assert declared is not None, (
        f"{path.name}: no module-level `image_size` literal found. The SDK "
        f"hands this to the edge to size the dataset; a template without one "
        f"cannot be checked and probably cannot be trained."
    )
    row = PUBLISHED_RESOLUTION.get(stem)
    assert row is not None, (
        f"{stem} has no PUBLISHED_RESOLUTION row, so there is nothing to "
        f"check its declaration against. See "
        f"test_every_od_template_has_a_published_resolution_row for what to "
        f"add and why — indexing the dict directly here raised a bare "
        f"KeyError for every parametrized case and buried that message "
        f"(review on model-zoo#252)."
    )
    published, anchor, citation = row

    known = KNOWN_MISMATCHES.get(stem)
    if known is not None:
        recorded_declared, recorded_effective = known
        assert declared == recorded_declared, (
            f"{stem} is recorded in KNOWN_MISMATCHES as declaring "
            f"{recorded_declared}, but declares {declared}.\n"
            f"  - Changed to the published {published}? Then it is FIXED: "
            f"delete its row, lower MAX_KNOWN_MISMATCHES, update the equality "
            f"pin — all in this commit.\n"
            f"  - Changed to something else? backend#3058 needs updating "
            f"before this row does."
        )
        assert published == recorded_effective, (
            f"{stem}: KNOWN_MISMATCHES records its real resolution as "
            f"{recorded_effective} while PUBLISHED_RESOLUTION says the "
            f"architecture is specified at {published} ({citation}). The two "
            f"anchors disagree, so one of them is wrong — resolve that before "
            f"trusting either."
        )
        return

    assert declared == published, (
        f"{stem} declares image_size = {declared}, but {anchor} says this "
        f"architecture runs at {published} ({citation}).\n"
        f"`image_size` is what the SDK hands the edge to size the dataset, so "
        f"a declaration below the architecture's specified resolution makes "
        f"the edge deliver less detail than the model is built to consume, and "
        f"one above it makes the edge ship pixels the model immediately "
        f"discards. Either way the resize is paid twice.\n"
        f"⚠️ Do NOT 'fix' this by editing the PUBLISHED_RESOLUTION row to "
        f"match the template. That number is anchored outside this repo on "
        f"purpose — matching it to the code under test is precisely the "
        f"tautology backend#3058 was reopened about."
    )


@pytest.mark.parametrize(
    "stem",
    sorted(s for s, (_, a, _) in PUBLISHED_RESOLUTION.items() if a == BUILDER),
)
def test_builder_anchored_rows_are_re_derived_from_torchvision(stem):
    """Build the torchvision builder this template mirrors and read its
    resolution back, rather than trusting the row.

    This is what makes a BUILDER row unfalsifiable by editing: to green a
    template that declares the wrong number, you would have to change
    torchvision.
    """
    torch = pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")
    from torchvision.models import detection as tv_detection

    recorded, _, builder_name = PUBLISHED_RESOLUTION[stem]
    builder = getattr(tv_detection, builder_name, None)
    assert builder is not None, (
        f"{stem}: torchvision.models.detection has no {builder_name!r} — the "
        f"builder was renamed or removed, so this row has lost its anchor. "
        f"Find the new name, or move the row to LITERAL with a citation and "
        f"add it to UNVERIFIABLE_LITERALS."
    )
    model = builder(weights=None, weights_backbone=None)
    try:
        derived = _effective_resolution(model)
    finally:
        del model
        gc.collect()
    del torch

    assert derived is not None, f"{builder_name}: built model exposes no transform"
    assert derived == recorded, (
        f"{stem}: PUBLISHED_RESOLUTION records {recorded}, but "
        f"torchvision's {builder_name} actually runs at {derived}.\n"
        f"The row is wrong, or torchvision changed its reference default. "
        f"Fix the row to {derived} — and then check the template, because its "
        f"declared image_size was being compared against the wrong number."
    )


def test_class_default_anchored_rows_are_re_derived_from_torchvision():
    """Read ``min_size``'s default out of the torchvision detector class each
    of these templates instantiates without overriding it.

    Same purpose as the builder check, by signature rather than construction —
    these templates pass a custom backbone to a stock detector class, so there
    is no builder to call, but the reference default is still torchvision's and
    still machine-readable.
    """
    pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    pytest.importorskip("torchvision", reason="torchvision not installed in this CI job")
    import inspect

    from torchvision.models.detection.faster_rcnn import FasterRCNN
    from torchvision.models.detection.fcos import FCOS
    from torchvision.models.detection.retinanet import RetinaNet

    classes = {"FasterRCNN": FasterRCNN, "FCOS": FCOS, "RetinaNet": RetinaNet}
    rows = {
        stem: (value, detail)
        for stem, (value, anchor, detail) in PUBLISHED_RESOLUTION.items()
        if anchor == CLASS_DEFAULT
    }
    assert rows, (
        "no CLASS_DEFAULT rows left, so this check verifies nothing — delete "
        "it along with the last row if the templates all moved anchors"
    )

    for stem, (recorded, class_name) in sorted(rows.items()):
        cls = classes.get(class_name)
        assert cls is not None, (
            f"{stem}: this file does not know the torchvision class "
            f"{class_name!r}; import it above rather than dropping the row"
        )
        params = inspect.signature(cls).parameters
        assert "min_size" in params, (
            f"{stem}: {class_name} no longer takes min_size, so the row has "
            f"lost its anchor and the template's inherited resolution needs "
            f"re-establishing from scratch"
        )
        derived = params["min_size"].default
        assert isinstance(derived, int), (
            f"{stem}: {class_name}.min_size default is {derived!r}, not an int "
            f"— it is no longer a fixed reference resolution"
        )
        assert derived == recorded, (
            f"{stem}: PUBLISHED_RESOLUTION records {recorded}, but "
            f"{class_name}'s min_size default — which is what this template "
            f"inherits, since it passes none — is {derived}. Fix the row, then "
            f"check the template."
        )


def test_the_anchor_kinds_partition_the_roster():
    """Every row is anchored, and the un-re-derivable ones are pinned.

    The set of LITERAL rows is the file's soft spot: those are the only
    expectations that could be edited to match a bad template. Pinning the set
    by equality means a new template cannot join it silently — which is the
    failure mode that would let this table decay back into the tautology it
    replaced.
    """
    kinds = {stem: anchor for stem, (_, anchor, _) in PUBLISHED_RESOLUTION.items()}
    unknown = {
        s: k
        for s, k in kinds.items()
        if k not in (BUILDER, CLASS_DEFAULT, ENGINE, MODERN_YOLO, LITERAL)
    }
    assert not unknown, f"rows with an unrecognised anchor kind: {unknown}"

    literals = {s for s, k in kinds.items() if k == LITERAL}
    assert literals == UNVERIFIABLE_LITERALS, (
        f"the LITERAL rows are {sorted(literals)}, but UNVERIFIABLE_LITERALS "
        f"pins {sorted(UNVERIFIABLE_LITERALS)}.\n"
        f"  - GREW? A new template whose expectation cannot be re-derived is a "
        f"reviewable decision, not a default. Check first whether it mirrors a "
        f"torchvision builder or inherits a class default — most do — and only "
        f"then add it here WITH a citation.\n"
        f"  - SHRANK? Good: a row became re-derivable. Update this set in the "
        f"same commit."
    )
    assert len(UNVERIFIABLE_LITERALS) == 5, (
        f"UNVERIFIABLE_LITERALS holds {len(UNVERIFIABLE_LITERALS)} rows, not "
        f"the 5 recorded. Growing it weakens every claim in this file's "
        f"docstring about the table being independent, so each addition is a "
        f"deliberate edit here with a citation — which is what the pin is for.\n"
        f"\n"
        f"History: 3 -> 4 `sparse_rcnn` (model-zoo#246), 4 -> 5 `yolov8_s` "
        f"(model-zoo#253), 5 -> 6 `yolov9_s` (model-zoo#255), 6 -> 8 "
        f"`yolox_s` and `rtmdet_s` together (model-zoo#237), then 8 -> 5 when "
        f"`yolov10_s` (model-zoo#258) arrived.\n"
        f"\n"
        f"⚠️ THAT LAST STEP WENT DOWN, AND ON PURPOSE. At 8 this message said "
        f"the next addition should come with a derivation source for the "
        f"hand-written family rather than a ninth row, and that raising the "
        f"number again without one was 'the decay this pin exists to make "
        f"visible'. So #256 did not add a ninth row: the four genuine "
        f"modern-YOLO templates (`yolov8_s`, `yolov9_s`, `yolov10_s`, "
        f"`yolox_s`) moved onto the MODERN_YOLO anchor, which checks them "
        f"against ONE cited family fact — MODERN_YOLO_RESOLUTION — the way the "
        f"ENGINE kind checks the legacy yolo rows against the engine's "
        f"contract. Four unreviewable literals became one reviewable literal "
        f"plus a membership test.\n"
        f"\n"
        f"What is left here is the genuinely one-off five: `cascade_rcnn`, "
        f"`sparse_rcnn`, `centernet_resnet`, `efficientdet_d0` and "
        f"`rtmdet_s`. None is a YOLO, none shares an authority with another, "
        f"and none has a torchvision builder or class default to re-derive "
        f"against — so for these the citation really is the only thing "
        f"standing between the row and the tautology this file removes.\n"
        f"\n"
        f"NOTE for whoever hits this next: a new modern YOLO (YOLO11, YOLOv12) "
        f"belongs in MODERN_YOLO_ROWS, not here, and costs this set nothing. A "
        f"new NON-YOLO hand-written detector does land here — and if that "
        f"pushes this back toward 8, the answer is the same as it was: find "
        f"the shared authority, not a longer list."
    )
    for stem in UNVERIFIABLE_LITERALS:
        citation = PUBLISHED_RESOLUTION[stem][2]
        assert len(citation.split()) >= 4, (
            f"{stem}: citation {citation!r} is too thin to review. A LITERAL "
            f"row's only defence is that a human can check it against the "
            f"source; name the paper and where in it."
        )


def test_the_modern_yolo_rows_are_anchored_to_one_family_fact():
    """The MODERN_YOLO rows must all equal ``MODERN_YOLO_RESOLUTION``.

    This is the derivation source the ``UNVERIFIABLE_LITERALS`` pin asked for
    (model-zoo#237), and it is what stops the hand-written modern-YOLO family
    contributing one unreviewable literal per template forever. It works like
    the ``ENGINE`` kind: a single authority with a single citation, and every
    row checked against it rather than against itself.

    Both directions are pinned:

    * every MODERN_YOLO row equals the family constant, so an individual row
      cannot be edited to match a template that resizes to something else —
      only the shared constant can move, and that moves all four visibly;
    * membership is pinned by EQUALITY, so a template cannot inherit the
      family's resolution without its own review just by being named after a
      YOLO. That matters in the other direction too: ``rtmdet_s`` is 640 and is
      deliberately NOT in the family, because CSPNeXt/mmdetection is a separate
      lineage and its agreement with 640 is a coincidence of the era rather
      than the same published fact.
    """
    rows = {stem for stem, (_, anchor, _) in PUBLISHED_RESOLUTION.items()
            if anchor == MODERN_YOLO}
    assert rows == MODERN_YOLO_ROWS, (
        f"the MODERN_YOLO rows are {sorted(rows)}, but MODERN_YOLO_ROWS pins "
        f"{sorted(MODERN_YOLO_ROWS)}.\n"
        f"  - GREW? A template joining this family inherits a resolution "
        f"without its own citation being reviewed. That is fine for a genuine "
        f"modern YOLO and wrong for anything else — say which it is here.\n"
        f"  - SHRANK? A template left the family; it needs its own anchor, "
        f"probably LITERAL with a citation, in this same commit."
    )
    assert rows, "the MODERN_YOLO kind has no rows, so this guard is vacuous"

    for stem in sorted(rows):
        value = PUBLISHED_RESOLUTION[stem][0]
        assert value == MODERN_YOLO_RESOLUTION, (
            f"{stem} is anchored to the modern-YOLO family but declares "
            f"{value}, not the family's {MODERN_YOLO_RESOLUTION}. A row in "
            f"this family does not get its own number — that is the entire "
            f"point of the family anchor. If this template genuinely runs at "
            f"another resolution it is not a member: give it a LITERAL row "
            f"with its own citation and add it to UNVERIFIABLE_LITERALS."
        )

    # The per-row notes are free text, and the LITERAL thinness check does not
    # reach this kind — so a row could join the family carrying no provenance
    # at all while the shared citation did all the work. Each row keeps the
    # paper/section reference it had before it was folded into the family.
    for stem in sorted(rows):
        note = PUBLISHED_RESOLUTION[stem][2]
        assert len(note.split()) >= 8, (
            f"{stem}: its MODERN_YOLO note {note!r} is too thin to review. "
            f"Being in the family fixes the NUMBER; the note still has to say "
            f"which architecture this is and where its 640 is published, "
            f"because that is what a reviewer checks."
        )

    assert len(MODERN_YOLO_CITATION.split()) >= 20, (
        f"MODERN_YOLO_CITATION is {len(MODERN_YOLO_CITATION.split())} words. It "
        f"is the ONLY thing defending four rows at once, so it has to name each "
        f"architecture and where the 640 comes from — a thinner citation here "
        f"is worse than four separate thin ones, because it is trusted four "
        f"times."
    )
    # Every family member must be NAMED in the shared citation. Without this
    # the citation could describe three architectures while anchoring four, and
    # the fourth would be riding on a fact nobody wrote down for it.
    lowered = MODERN_YOLO_CITATION.lower()
    unnamed = sorted(
        stem for stem in rows
        if stem.split("_")[0].replace("v", "v") not in lowered
    )
    assert not unnamed, (
        f"MODERN_YOLO_CITATION does not name {unnamed}, so those rows are "
        f"anchored to a fact that does not mention them. Extend the citation "
        f"in the same commit that adds the row."
    )


def _engine_fixed_input_sizes(notes: str) -> set[int]:
    """The fixed input size(s) the engine's family notes actually declare.

    Matched against the FIELD — the contract's phrasing is "fixed 448 input" —
    not against any 3-4 digit token in the prose. ``\\b(\\d{3,4})\\b`` accepted
    an incidental number (an issue id like 3058, a year, or the papers' own
    640), so a wrong ENGINE row could pass merely because 448 appeared
    somewhere else in the sentence (review on model-zoo#252).

    Extracted as a function purely so
    ``test_the_engine_anchor_reads_the_field_not_the_prose`` can point it at
    adversarial notes. Inline, the loose pattern and this one agree on today's
    contract text, so loosening it back was a mutation NOTHING could see —
    which is the only reason this is not still a one-liner.
    """
    return {int(n) for n in re.findall(r"fixed\s+(\d{3,4})\s+input", notes)}


def test_the_engine_anchor_reads_the_field_not_the_prose():
    """Control: the anchor must not be satisfiable by stray digits.

    The two patterns agree on the contract as it reads today, so this pins the
    difference on prose that separates them — otherwise the tightening is
    unenforced and reverts silently.
    """
    contract_today = (
        "Grid-tensor output + external loss, fixed 448 input. The family name "
        "equals the declared value, so it needs no alias."
    )
    assert _engine_fixed_input_sizes(contract_today) == {448}

    # The same declaration, with incidental numbers a future editor might add.
    chatty = (
        "Grid-tensor output + external loss, fixed 448 input. See backend#3058; "
        "note YOLOv5-s and YOLOv8 are published at 640 upstream, 2026 revision."
    )
    assert _engine_fixed_input_sizes(chatty) == {448}, (
        "the anchor picked up a number that is not the declared fixed input. "
        "With `\\b(\\d{3,4})\\b` this returns {448, 3058, 640}, so an ENGINE row "
        "of 640 — the papers' resolution, which is exactly the wrong answer "
        "for this family — would pass. Anchor to the field."
    )

    reworded = "Grid-tensor output, a fixed input of 448."
    assert _engine_fixed_input_sizes(reworded) == set(), (
        "the extractor claimed to find a size in prose that does not use the "
        "phrasing it matches; it must come back empty so the caller's 'no "
        "longer names a fixed <N> input' assertion fires and someone updates "
        "the pattern deliberately"
    )


def test_the_engine_anchored_rows_match_the_vendored_contract():
    """The yolo rows are anchored to a file the ENGINE owns, not to a literal.

    ``PUBLISHED_RESOLUTION``'s yolo rows override the architectures' published
    640 because the engine's ``YoloHandler`` takes a fixed-size grid tensor.
    That is a claim about the engine, so it is checked against the vendored
    contract: if the engine's fixed input size changes, this fails here instead
    of leaving three rows quietly wrong.
    """
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    yolo = [f for f in schema["families"] if f["family"] == "yolo"]
    assert yolo, f"{_schema_path().name}: no 'yolo' family entry to anchor against"
    notes = yolo[0].get("notes", "")
    sizes = _engine_fixed_input_sizes(notes)
    assert sizes, (
        f"{_schema_path().name}: the yolo family's notes no longer name a "
        f"'fixed <N> input' ({notes!r}), so the ENGINE-anchored rows in "
        f"PUBLISHED_RESOLUTION have nothing to be anchored to. Either the "
        f"contract rephrased it — match the new phrasing, or better, read it "
        f"from a machine-readable field if one now exists — or the family's "
        f"input is no longer fixed, in which case those rows need a different "
        f"anchor entirely. Do NOT widen this back to bare digits: that made "
        f"the anchor satisfiable by any number in the prose."
    )

    engine_rows = {
        stem: value
        for stem, (value, anchor, _) in PUBLISHED_RESOLUTION.items()
        if anchor == ENGINE
    }
    assert engine_rows, (
        "no ENGINE-anchored rows left, so this check verifies nothing — if the "
        "yolo templates moved to a paper anchor, delete this test with them"
    )
    disagree = {s: v for s, v in engine_rows.items() if v not in sizes}
    assert not disagree, (
        f"ENGINE-anchored rows {disagree} disagree with the fixed input size "
        f"the vendored engine contract names ({sorted(sizes)}): "
        f"{notes!r}. The engine is the authority for this family; update the "
        f"rows, and check whether the templates themselves still match."
    )


def test_the_declared_vs_built_comparison_is_self_consistent_by_construction():
    """Demonstrate the tautology this file was rewritten to close.

    A template that wires its own ``image_size`` into its own transform — the
    normal, correct pattern — satisfies the declared-vs-built comparison at ANY
    value, including a wrong one. Shown on a stub with no template and no torch
    involved, so the module docstring's reasoning is checked rather than
    asserted.

    ``efficientdet_d0`` is the real instance: it declared 448 while running at
    512, sat under 20 green tests including that comparison, and was caught
    only when its author pinned it against the published 512 by hand.
    """
    import types

    wrong = 448  # what efficientdet_d0 declared
    right = PUBLISHED_RESOLUTION["efficientdet_d0"][0]
    assert right != wrong, (
        "this demonstration needs a published value that differs from the "
        "declaration it is contrasted with; efficientdet_d0's row now reads "
        f"{right}, so re-pick the example rather than deleting the test"
    )

    # A template that consumes its own declaration: one literal, two places.
    # This is what every correctly-written template looks like.
    self_consistent = types.SimpleNamespace(
        transform=types.SimpleNamespace(
            min_size=wrong, max_size=wrong * 2, fixed_size=None
        )
    )
    built = _effective_resolution(self_consistent)
    assert built == wrong, (
        f"_effective_resolution read {built} back from a transform built at "
        f"{wrong}; the demonstration below depends on it reading the stub's "
        f"own value"
    )
    # declared == built, so the declared-vs-built comparison PASSES ...
    # ... on a declaration that is simply wrong for this architecture.
    assert built != right, (
        f"declared-vs-built passes a template declaring {wrong} while the "
        f"architecture is specified at {right}. That gap is exactly what the "
        f"declared-vs-published comparison exists to see. If the two numbers "
        f"ever coincide, this demonstration stops demonstrating anything and "
        f"needs a different example rather than deleting."
    )


def test_the_two_readers_are_independent_and_discriminate(tmp_path):
    """Guard the guard above: the partition compares two readers' verdicts, and
    if both collapsed to "always None" the roster and the expected roster would
    go empty together and it would pass on nothing (backend#2982)."""
    support = tmp_path / "loss.py"
    support.write_text("import torch\n\n\ndef loss(a, b):\n    return a - b\n", "utf-8")
    assert _declares_framework(support) is None
    assert _read_model_type(support) is None

    template = tmp_path / "model.py"
    template.write_text(
        'framework = "pytorch"\nmodel_type = "torchvision_detection"\n'
        "image_size = 640\n",
        "utf-8",
    )
    assert _declares_framework(template) == "pytorch"
    assert _read_model_type(template) == "torchvision_detection"

    # The third reader, added with the published-resolution table. If it
    # silently returned None the declared-vs-published comparison would fail
    # loudly rather than pass — but a reader that answered the WRONG number
    # would not, so both directions are pinned.
    assert _read_declared_image_size(template) == 640
    assert _read_declared_image_size(support) is None

    # Both halves of the "module level" claim, each enforced. The indented
    # half was previously only CLAIMED: the reader used `^\s*image_size`, and
    # `\s*` happily consumes indentation, so an indented assignment ahead of
    # the real one would have won. Latent — no template trips it — but the
    # docstring promised something the regex did not deliver (review on
    # model-zoo#252).
    commented = tmp_path / "commented.py"
    commented.write_text("# image_size = 111\nimage_size = 512\n", "utf-8")
    assert _read_declared_image_size(commented) == 512, (
        "the reader must not pick up a commented-out assignment ahead of the "
        "real module-level one"
    )

    indented = tmp_path / "indented.py"
    indented.write_text(
        "class _Cfg:\n    image_size = 111\n\n\nimage_size = 512\n", "utf-8"
    )
    assert _read_declared_image_size(indented) == 512, (
        "the reader picked up an INDENTED image_size ahead of the "
        "module-level one — a class attribute or a builder local is not the "
        "value the SDK hands the edge, and treating it as such would compare "
        "the published resolution against an unrelated number"
    )

    only_indented = tmp_path / "only_indented.py"
    only_indented.write_text("class _Cfg:\n    image_size = 111\n", "utf-8")
    assert _read_declared_image_size(only_indented) is None, (
        "a file whose ONLY image_size is indented declares none at module "
        "level; returning 111 here would invent a declaration"
    )
