#!/usr/bin/env python3
"""Acceptance sweep: every ``torchvision_detection`` OD template trains a cycle
and infers a well-formed payload, with a PER-TEMPLATE report (backend#3048).

    python tools/od_acceptance_sweep.py --steps 8 --experiments 2 \
        --out sweep.json --markdown sweep.md

WHAT THIS IS, AND WHAT IT IS NOT (backend#3048's own framing)
-------------------------------------------------------------
`e2e-test-agent`'s `harness/od/probes.py` (backend#2975) is a STRUCTURAL probe
set that runs on every PR: count units, registry resolution, family resolution,
sidecar pairing, over a two-image synthetic fixture. It trains zero models. This
is the other artifact -- an ACCEPTANCE SWEEP that runs when epic backend#1169
claims to be finished. #3048 is explicit that the two must not be merged:
folding them makes the fast one slow and the thorough one skippable.

Which is also why this is a TOOL and not a pytest module. model-zoo CI's torch
job is `timeout-minutes: 30` on a 2-core `ubuntu-latest`, and a multi-experiment
cycle sweep over the roster is measured in hours (see COST below). What lives in
CI is `tests/test_od_acceptance_sweep.py` -- every assertion in this file, each
one fired at a payload that must break it. The sweep itself is run on demand.

LOCAL-ONLY, BY DECISION, AND THE LIVE LEG IS A SEPARATE PR
-----------------------------------------------------------
This needs no cluster, no backend and no credentials, which is what makes it
runnable today: the e2e harness's `secrets.local.json` does not exist on any
machine we have, and `harness/od/scenario.py` has no `--live` path at all (by
its own design). The live half -- ingest, an experiment per template on a dev
edge, the REST assertions -- is `harness/od_sweep/` in `e2e-test-agent`, and it
waits on a credential. Nothing here pretends to be that.

THE TWO VERDICTS ARE NOT ONE VERDICT
------------------------------------
#3048's checklist asks for "finite loss and non-zero mAP". The second half is
NOT ACHIEVABLE TODAY and this file does not pretend otherwise. backend#3055
(host COCO seeds) is blocked on backend#2659 (a store location and a credential),
so every template here builds from RANDOM INITIALISATION -- including the three
that declare a seed, because a declared seed that is not hosted is not a seed.
backend#3055 states the fallback: #3048 must say, per template, whether it ran
seeded or from scratch. So the report carries:

  mechanical  -- did it train a cycle and return a well-formed payload?
  quality     -- is the detection quality measurable, and what did it measure?
  cause       -- WHY quality is pending, derived per run (see below)

A template that trains and infers cleanly from scratch is a REAL PASS on the
mechanical criterion and a PENDING on the quality one. Reporting one green
verdict by quietly lowering the bar is the shape of every false-green this epic
has produced -- most memorably an audit that scored `mask_rcnn` "uploadable"
when it could not be uploaded at all, because it measured dispatch reachability
and stopped there.

THE CAUSE COLUMN IS DERIVED PER RUN, NEVER BAKED IN
---------------------------------------------------
Measured on develop at 2026-09-04: 11 of 25 templates return ZERO boxes on both
fixture images from random init, and the other 14 return score-threshold noise
(100/200/300 boxes). For those 11, "low mAP" is the wrong description -- there
is no payload to score. That distinction is a separate column.

It is NOT a constant in this file. backend#3093 is fixing the from-scratch
normalisation (`norm_layer=FrozenBatchNorm2d` is a bit-exact identity on a
`weights=None` build, so those backbones train with no normalisation at all),
and when it lands some of the 11 are expected to start emitting boxes. A
hardcoded list would then report last week's reality. The cause is computed from
the payload this run produced, the same way the seeding column is computed by
calling `check_dump_coverage.survey()` rather than transcribing its verdict.

HYPERPARAMETERS COME FROM THE ENGINE, NOT FROM THIS FILE
--------------------------------------------------------
The engine builds the optimizer as, verbatim
(`core/frameworks/pytorch_adapter.py::_construct_optimizer`)::

    return optimizer_class(model.parameters(), lr=learning_rate)

`lr` ONLY -- no momentum -- and `process_learning_rate` defaults to `0.001`.
This matters concretely: a first draft of this sweep used
`SGD(lr=1e-3, momentum=0.9)` and `centernet_resnet` went NaN in 12 steps. Under
the engine's actual construction it does not. A harness that picks its own
optimizer measures the harness. `--engine <checkout>` re-derives these from the
engine source and FAILS on disagreement, rather than trusting the constants
below (the `harness/od/scenario.py` reachable-checkout pattern).

RESOLUTION IS RECORDED, NOT ASSUMED (backend#3058)
--------------------------------------------------
`faster_rcnn_resnet`, `fcos` and `retinanet` declare `image_size = 448` and
override nothing, so they fall through to `GeneralizedRCNNTransform`'s default
`min_size=800`/`max_size=1333` -- verified on develop: no `min_size`, `max_size`
or `GeneralizedRCNNTransform` in any of the three. backend#3058 corrects the
declaration to 800, after which a native-aspect source lands near 800x1067
instead of an upscaled square 800x800, about a third more pixels.

So a cost figure is meaningless without the resolution it was measured at, and
`observed_input_shape` records the POST-TRANSFORM batched tensor per template
rather than the declared size. That is the number the network actually saw.
Worth knowing before reading any cost table: the transform normalises the input
away, so feeding it a small image does not make a cheap measurement -- 128x160
upscales to a batched 928x1024, which is LARGER than what 448x448 produces.

COST, measured on develop (Apple silicon, CPU, torch 2.12.1, 2-image batch)
---------------------------------------------------------------------------
Per template, steady-state train step: median 2.2s, min 0.37s (`yolov8_s`),
max 44.4s (`faster_rcnn_convnext_small`). One step + one eval across all 25 is
108s. A 20-step cycle x3 experiments is ~141 minutes, of which 86 belong to the
two convnext templates alone. `--skip-slow` drops that pair; it is opt-in and
named in the report's `skipped` list, because #3048 forbids a silent cap.

WHY LOSS *DECREASE* IS RECORDED AND NOT GATED ON
------------------------------------------------
It is the obvious second half of "finite loss", and it would be a false
assurance. `tests/test_od_norm_layers_normalise.py` establishes that twelve
shipped OD templates train with NO backbone normalisation at all
(`FrozenBatchNorm2d` on a `weights=None` build is a bit-exact identity), and
records that for those twelve "the loss stays finite and decreasing throughout
-- the recurring shape of defects in this area". A gate on decreasing loss would
therefore be green on all twelve while they are broken, and would read as
evidence that they are not. `loss_first`/`loss_last`/`loss_decreased` are
columns; the gate is finiteness, gradient reachability and payload shape.

For the same reason nothing here asserts a STATISTIC of a randomly-initialised
model's activations -- that is the norm file's trap 30,
DISTRIBUTIONAL-ASSERTION-ON-RANDOM-INIT, and it is that file's job, not this
one's.

NOT COVERED HERE: THE THREE `yolo` TEMPLATES
--------------------------------------------
`yolo_v1/`, `yolo_v5/` and `yolo_v8/` route to the `yolo` family, which speaks a
different contract: `model(x)` returns a raw grid tensor `(B, S1, S2, C)` and the
loss is an external `Custom_loss(preds, targets)` over grid-ENCODED targets. The
encode/decode lives in the engine's `YoloHandler`
(`core/utils/object_detection_utils.py:991`) and model-zoo vendors only the
engine's *schema*, not the engine. Covering them from here would mean
reimplementing that encoder, i.e. asserting against a second copy of the
contract instead of the contract. They are 3 of the roster's 28 templates and
they are named in the report's `uncovered` block so the number is never read as
25-of-25.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Sibling tool module, same directory -- the `check_engine_pin_drift.py` idiom,
# so this runs as `python tools/od_acceptance_sweep.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
# And `tests/_od.py`, which is the SSoT for "which OD templates exist" and "how
# to construct one". Reaching into `tests/` from `tools/` is the precedent
# `tools/seed_index.py` set for exactly this reason -- it reads
# `tests/test_model_contract.py` rather than "add a THIRD copy" of the CI RAM
# skip list. `_od.py`'s own docstring says the point of it landing is that "the
# count of copies stops growing", and it had already caught a copy that
# silently narrowed a roster scan (model-zoo#251). A fourth copy here is the
# defect that module exists to stop.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from _od import (  # noqa: E402  -- after the sys.path insert
    OD_ROOT,
    build_template,
    od_templates,
    template_key,
)
from check_dump_coverage import (  # noqa: E402
    EXPECTS_SEED,
    NO_SEED,
    survey as dump_survey,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_DIR = REPO_ROOT / "tests" / "contracts" / "tracebloc_engine"

#: The family this sweep exercises. `yolo` is a different contract entirely --
#: see the module docstring's "NOT COVERED HERE".
FAMILY = "torchvision_detection"

# ---------------------------------------------------------------------------
# Engine-derived hyperparameters. Constants here, re-derived by `--engine`.
# ---------------------------------------------------------------------------

#: `core/frameworks/pytorch_adapter.py::_construct_optimizer`:
#:     return optimizer_class(model.parameters(), lr=learning_rate)
#: The absence of momentum is the load-bearing part -- see the module docstring.
ENGINE_OPTIMIZER_KWARGS: Dict[str, Any] = {}

#: `core/frameworks/pytorch_adapter.py::process_learning_rate`, the `"custom"`
#: branch's `initial_learning_rate = 0.001  # Default initial learning rate`.
ENGINE_DEFAULT_LR = 0.001

#: `optimizers = {"sgd": torch.optim.SGD, ...}` -- the platform's default.
ENGINE_OPTIMIZER_NAME = "sgd"

#: Where the two facts above live, for `--engine` to check and for the report to
#: cite. Relative to a `tracebloc-engine` checkout.
ENGINE_ADAPTER_PATH = "core/frameworks/pytorch_adapter.py"

#: Measured cost outliers (module docstring's COST). `--skip-slow` only.
#: An explicit set rather than a measured threshold: deciding "slow" by timing
#: the run would make the skip list depend on the machine, and a skip list that
#: differs between two runs of the same sweep is not reportable.
SLOW_TEMPLATES = frozenset({"faster_rcnn_convnext_small", "fcos_convnext_small"})

#: Quality-pending causes, derived per run by `_quality_cause`.
CAUSE_EMPTY_PAYLOAD = "empty-payload"
CAUSE_RANDOM_SCORES = "random-init-scores"
CAUSE_MEASURABLE = "measurable"


# ---------------------------------------------------------------------------
# Roster derivation -- the same partition `tests/test_od_torchvision_family_
# train_step.py` establishes, and pinned equal to it by
# `tests/test_od_acceptance_sweep.py::test_the_sweep_roster_is_the_contract_
# tests_roster`. Two readers rather than one import because a tool must run
# standalone; the test is what stops them drifting.
# ---------------------------------------------------------------------------


def schema_path() -> Path:
    """Newest vendored ``object_detection_families.v<N>.json``.

    Globbed and numerically sorted, not pinned by name: the schema is version-
    bumped in place when the engine's vocabulary narrows (v1 -> v2 dropped
    ``hf_transformer`` with backend#2973), and pinning a filename breaks on the
    rename instead of adopting it. Numeric key so v10 does not sort before v2.
    """
    paths = sorted(
        CONTRACTS_DIR.glob("object_detection_families.v*.json"),
        key=lambda p: int(re.search(r"\.v(\d+)\.json$", p.name).group(1)),
    )
    if not paths:
        raise SystemExit(f"no vendored OD families schema under {CONTRACTS_DIR}")
    return paths[-1]


def family_values() -> frozenset:
    """Declaration values routing to ``FAMILY`` -- the name plus every alias.

    THE ALIAS IS NOT OPTIONAL. `rcnn` is a live legacy alias and
    `faster_rcnn_resnet.py` declares it; selecting on the literal string
    `"torchvision_detection"` silently drops that template from the sweep.
    Normalized the way the engine's `resolve_family` does.
    """
    schema = json.loads(schema_path().read_text(encoding="utf-8"))
    entries = [f for f in schema["families"] if f["family"] == FAMILY]
    if not entries:
        raise SystemExit(f"{schema_path().name}: no {FAMILY!r} family entry")
    values = {FAMILY, *entries[0].get("aliases", [])}
    return frozenset(v.strip().lower() for v in values)


def read_model_type(path: Path) -> Optional[str]:
    """The module-level ``model_type``, read statically -- no import.

    This one reader stays local rather than moving into `tests/_od.py`, and that
    is the same call `_od.py` itself documents: it holds the helpers deciding
    WHICH templates exist and HOW to build one, while `_read_model_type` stays
    per-file precisely because two files compare two readers' verdicts to prove
    a roster partition is not vacuous. Here the second reader is the
    `framework` regex inside `_od.od_templates`, and the partition it forms with
    this one is asserted in `tests/test_od_acceptance_sweep.py`
    (`test_uncovered_templates_are_reported_and_are_the_yolo_family`).

    Read statically because file SELECTION must not cost a model build: several
    of these templates allocate hundreds of MB on import.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(r'^\s*model_type\s*=\s*["\']([^"\']*)["\']', text, re.MULTILINE)
    return match.group(1) if match else None


def _read_int_decl(path: Path, key: str) -> Optional[int]:
    """A module-level integer declaration (``image_size``, ``output_classes``)."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(rf"^\s*{key}\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def family_templates(values: Optional[frozenset] = None) -> List[Path]:
    """The templates this sweep covers."""
    accepted = family_values() if values is None else values
    return [
        p
        for p in od_templates()
        if (read_model_type(p) or "").strip().lower() in accepted
    ]


def uncovered_templates(values: Optional[frozenset] = None) -> List[Path]:
    """Templates outside ``FAMILY`` -- reported, never silently dropped."""
    accepted = family_values() if values is None else values
    return [
        p
        for p in od_templates()
        if (read_model_type(p) or "").strip().lower() not in accepted
    ]


# ---------------------------------------------------------------------------
# Seeding column -- computed by calling the survey, not transcribed from it.
# ---------------------------------------------------------------------------


def seeding_index() -> Dict[str, str]:
    """``{stem: seeded-or-scratch phrase}`` for object_detection.

    Calls `check_dump_coverage.survey()` rather than restating its verdict:
    a hand-written seeded/scratch column is a second source of truth that goes
    stale the first time a dump is staged.

    EVERY TEMPLATE IS "scratch" TODAY, INCLUDING `EXPECTS_SEED` ONES. This tool
    builds `MyModel()` and loads no weights, and it could not load them anyway:
    backend#2659 has not hosted the dumps. `EXPECTS_SEED` therefore means "a
    seed is declared and waiting", which is a different report row from
    "random-initialised by design" and is kept distinct for that reason.
    """
    index: Dict[str, str] = {}
    for key, record in dump_survey(REPO_ROOT).items():
        category, _, stem = key.partition("/")
        if category != "object_detection":
            continue
        status = record.get("status")
        if status == EXPECTS_SEED:
            index[stem] = "scratch (seed declared, not hosted -- backend#2659)"
        elif status == NO_SEED:
            index[stem] = "scratch (random-init by design)"
        else:
            index[stem] = f"scratch ({status})"
    return index


# ---------------------------------------------------------------------------
# Model construction and the platform's optimizer
# ---------------------------------------------------------------------------


def make_optimizer(torch, model, lr: float = ENGINE_DEFAULT_LR):
    """The engine's optimizer, built the engine's way.

    `optimizer_class(model.parameters(), lr=learning_rate)` and nothing else --
    `ENGINE_OPTIMIZER_KWARGS` is empty and is passed through explicitly so the
    emptiness is visible at the call site rather than implied by its absence.
    """
    return torch.optim.SGD(model.parameters(), lr=lr, **ENGINE_OPTIMIZER_KWARGS)


def verify_engine_hyperparameters(engine_root: Path) -> List[str]:
    """Re-derive the optimizer contract from an engine checkout.

    Returns a list of DISAGREEMENTS. The `harness/od/scenario.py` pattern: the
    constants in this file are a recording, and a recording that no longer
    matches its source is a failure, not a fallback. Read as text rather than
    imported -- the engine is not a dependency of this repo and importing it
    would pull its whole runtime in to check two literals.
    """
    adapter = engine_root / ENGINE_ADAPTER_PATH
    if not adapter.is_file():
        return [f"{adapter} does not exist -- is {engine_root} a tracebloc-engine checkout?"]
    text = adapter.read_text(encoding="utf-8")
    problems: List[str] = []

    if not re.search(
        r"return\s+optimizer_class\(\s*model\.parameters\(\)\s*,\s*lr\s*=\s*learning_rate\s*\)",
        text,
    ):
        problems.append(
            f"{ENGINE_ADAPTER_PATH}: `_construct_optimizer`'s CPU return no longer reads "
            f"`optimizer_class(model.parameters(), lr=learning_rate)` -- if it grew a "
            f"momentum/weight-decay kwarg, ENGINE_OPTIMIZER_KWARGS must grow it too or "
            f"this sweep is measuring an optimizer the platform does not use"
        )
    lr_match = re.search(r"initial_learning_rate\s*=\s*([0-9.eE+-]+)", text)
    if not lr_match:
        problems.append(
            f"{ENGINE_ADAPTER_PATH}: no `initial_learning_rate = <float>` found; "
            f"ENGINE_DEFAULT_LR={ENGINE_DEFAULT_LR} can no longer be confirmed"
        )
    elif float(lr_match.group(1)) != ENGINE_DEFAULT_LR:
        problems.append(
            f"{ENGINE_ADAPTER_PATH}: default lr is {lr_match.group(1)}, this sweep uses "
            f"{ENGINE_DEFAULT_LR}"
        )
    if f'"{ENGINE_OPTIMIZER_NAME}": torch.optim.SGD' not in text:
        problems.append(
            f"{ENGINE_ADAPTER_PATH}: the optimizer table no longer maps "
            f"{ENGINE_OPTIMIZER_NAME!r} to torch.optim.SGD"
        )
    return problems


# ---------------------------------------------------------------------------
# The fixture batch
# ---------------------------------------------------------------------------


def make_batch(torch, image_size: int, num_classes: int):
    """Two images at the template's DECLARED resolution, and their targets.

    Images at `image_size` because that is what the platform delivers and this
    is an acceptance sweep, not a unit test choosing a cheap input. (It is also
    not the expensive choice: `GeneralizedRCNNTransform` rescales to
    `min_size=800` regardless, so a small input is upscaled rather than saved --
    see the module docstring.)

    TARGETS CARRY ONLY `boxes` AND `labels`. That is the intersection of the two
    producers -- the engine's `image_detection_dataset_pytorch` emits
    `{boxes, labels, area, iscrowd}` and the SDK's dummy RCNN dataset emits
    `{boxes, labels}` -- so a template that trains against these trains against
    either, and one needing more than that is broken by definition. This is the
    contract `mask_rcnn.py` failed (backend#2988).

    LABELS SPAN THE FULL MODEL-SPACE RANGE `[1, num_classes]`. The engine's
    `TorchvisionDetectionHandler` shifts a dataset-space label `[0, C-1]` up past
    torchvision's background index before the model sees it (backend#3062), so a
    template in this family is handed `[1, C]` and must allocate `C + 1` head
    channels. Spanning only `[1, C-1]` would let a template that allocated
    `num_classes` channels pass here and raise on the last class at train time.

    THE SECOND IMAGE HAS ZERO OBJECTS -- the engine's explicit target for an
    unannotated image, shapes and dtypes matched to that dataset. #3048 requires
    that an image with nothing in it neither crashes nor is silently dropped.
    """
    images = [
        torch.rand(3, image_size, image_size),
        torch.rand(3, image_size, image_size),
    ]
    span = min(image_size - 1, 120)
    targets = [
        {
            "boxes": torch.tensor(
                [[5.0, 5.0, span * 0.5, span * 0.5], [span * 0.5, span * 0.5, span, span]]
            ),
            "labels": torch.tensor([1, num_classes], dtype=torch.int64),
        },
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        },
    ]
    return images, targets


def observed_input_shape(model, images, targets) -> Optional[List[int]]:
    """The POST-TRANSFORM batched tensor shape, or ``None`` if no transform.

    The resolution the network actually saw, which is not the declared
    `image_size` for any template using `GeneralizedRCNNTransform`'s defaults --
    the backend#3058 point. Recorded per template so a cost figure can never be
    read at the wrong resolution.

    Instruments `model.transform.forward` for one call and restores it, rather
    than reimplementing the resize: a reimplementation is a second copy of the
    thing being measured.
    """
    transform = getattr(model, "transform", None)
    if transform is None or not hasattr(transform, "forward"):
        return None
    captured: Dict[str, Any] = {}
    original = transform.forward

    def _capture(imgs, tgts=None):
        result = original(imgs, tgts)
        try:
            captured["shape"] = list(result[0].tensors.shape)
        except (AttributeError, IndexError, TypeError):
            pass
        return result

    transform.forward = _capture
    try:
        model(images, targets)
    finally:
        transform.forward = original
    return captured.get("shape")


# ---------------------------------------------------------------------------
# The assertions. Each returns FINDINGS -- a list of strings, empty when the
# property holds -- so every one of them is callable from the guard tests with
# a payload built to break it. A check that has never fired is not evidence.
# ---------------------------------------------------------------------------


def train_step_findings(torch, losses: Any, step: int) -> List[str]:
    """One train step's loss dict against the handler's contract.

    The handler calls `sum(losses.values())`, so: a non-empty dict of finite
    SCALAR tensors. Each clause is separate because "the loss was bad" and "the
    loss was not a loss" are different failures with different causes.
    """
    findings: List[str] = []
    if not isinstance(losses, dict):
        return [f"step {step}: train mode returned {type(losses).__name__}, not a dict"]
    if not losses:
        return [f"step {step}: train mode returned an EMPTY loss dict"]
    for key, value in sorted(losses.items()):
        if not torch.is_tensor(value):
            findings.append(f"step {step}: loss {key!r} is {type(value).__name__}, not a tensor")
            continue
        if value.ndim != 0:
            findings.append(
                f"step {step}: loss {key!r} has shape {tuple(value.shape)}, not a scalar"
            )
            continue
        if not bool(torch.isfinite(value).all()):
            findings.append(f"step {step}: loss {key!r} is {value.item()}, not finite")
    return findings


def payload_findings(torch, preds: Any, n_images: int) -> List[str]:
    """The eval payload against #3048's inference checklist.

    Boxes in pixel xyxy; `scores`/`labels` ALIGNED with `boxes`; and an image the
    detector finds nothing on present as a zero-row entry rather than dropped.

    ON THE LAST POINT, precisely: this asserts the payload has one entry PER
    IMAGE. An empty entry is a PASS -- a detector finding nothing is a legitimate
    result and #3048 asks only that it "neither crashes nor silently drops the
    record". A check that required a non-empty payload would fail 11 of 25
    templates for being untrained, which is a quality question and belongs in
    the other column.
    """
    findings: List[str] = []
    if not isinstance(preds, list):
        return [f"eval returned {type(preds).__name__}, not a list"]
    if len(preds) != n_images:
        findings.append(
            f"eval returned {len(preds)} predictions for {n_images} images -- "
            f"{n_images - len(preds)} record(s) DROPPED"
        )
    for i, pred in enumerate(preds):
        if not isinstance(pred, dict):
            findings.append(f"prediction {i} is {type(pred).__name__}, not a dict")
            continue
        missing = {"boxes", "scores", "labels"} - set(pred)
        if missing:
            findings.append(f"prediction {i} is missing {sorted(missing)}")
            continue
        boxes, scores, labels = pred["boxes"], pred["scores"], pred["labels"]
        if not torch.is_tensor(boxes) or boxes.ndim != 2 or boxes.shape[-1] != 4:
            shape = tuple(boxes.shape) if torch.is_tensor(boxes) else type(boxes).__name__
            findings.append(f"prediction {i} boxes are {shape}, expected (N, 4)")
            continue
        n = boxes.shape[0]
        for name, tensor in (("scores", scores), ("labels", labels)):
            if not torch.is_tensor(tensor):
                findings.append(
                    f"prediction {i} {name} is {type(tensor).__name__}, not a tensor"
                )
            elif tensor.shape[0] != n:
                findings.append(
                    f"prediction {i} has {n} boxes but {tensor.shape[0]} {name} -- "
                    f"{name} are not aligned with boxes"
                )
        if n and not bool(
            (boxes[:, 2] >= boxes[:, 0]).all() and (boxes[:, 3] >= boxes[:, 1]).all()
        ):
            findings.append(
                f"prediction {i} boxes are not xyxy (x2<x1 or y2<y1); the metrics read "
                f"them as xyxy pixels"
            )
    return findings


def gradient_findings(names_without_grad: Sequence[str]) -> List[str]:
    """Parameters not REACHABLE FROM THE LOSS through the autograd graph.

    The prose is deliberately that and not "parameters that are trained". What
    the check tests is `p.grad is not None` after a backward, which means exactly
    "autograd reached it" -- a parameter whose gradient is present but ZERO
    satisfies this, and 9 of 25 templates have such parameters at baseline
    (unassigned FPN levels and per-level scales on a 2-image batch). The
    non-zero form is therefore not shippable as a gate, and `n_zero_grad` is
    reported as DATA instead.

    Why this check earns its place: a module left CONSTRUCTED but dropped out of
    `forward()` is invisible to parameter count, to `state_dict` keys, to tensor
    shapes and to the loss keys -- every property the existing contract test
    checks. It is not invisible to the gradient graph. Verified by mutating
    `yolox_s.py`'s `Conv.forward` from `self.act(self.norm(self.conv(x)))` to
    `self.act(self.conv(x))`: the existing train-step test passed, and this
    check went from 0 to 148 of 240 parameters unreachable.
    """
    if not names_without_grad:
        return []
    shown = list(names_without_grad)[:5]
    return [
        f"{len(names_without_grad)} trainable parameter(s) are not reachable from the "
        f"loss through the autograd graph (a module constructed but dropped from "
        f"forward() looks exactly like this): {shown}"
        + ("..." if len(names_without_grad) > 5 else "")
    ]


def measure_gradient_reachability(torch, model, optimizer, images, targets):
    """One FRESH backward, then which trainable parameters have no gradient.

    THE FRESH BACKWARD IS THE WHOLE POINT AND IT IS EASY TO GET WRONG. Gradients
    ACCUMULATE, so `p.grad is not None` measured after a multi-step cycle is
    satisfied by a parameter that received a gradient only on step 1 and has been
    unreachable ever since -- the check would silently pass on exactly the defect
    it exists to catch. `zero_grad(set_to_none=True)` clears them to None (plain
    `zero_grad()` would leave zero TENSORS, which are `not None` and would make
    this check vacuous), and only then is one backward run and read.
    """
    optimizer.zero_grad(set_to_none=True)
    model.train()
    losses = model(images, targets)
    total = sum(losses.values())
    total.backward()

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_grad = [n for n, p in trainable if p.grad is None]
    zero_grad = [
        n for n, p in trainable if p.grad is not None and not bool(p.grad.abs().sum())
    ]
    return len(trainable), no_grad, zero_grad


def _quality_cause(n_preds: Sequence[int], seeded: bool) -> str:
    """Why detection quality is or is not measurable, from THIS run's payload.

    Derived, never a constant -- backend#3093's normalisation fix is expected to
    move which templates emit boxes at all, and a baked list would then report a
    reality that has changed. See the module docstring.
    """
    if seeded:
        return CAUSE_MEASURABLE
    return CAUSE_EMPTY_PAYLOAD if sum(n_preds) == 0 else CAUSE_RANDOM_SCORES


# ---------------------------------------------------------------------------
# One experiment, one template
# ---------------------------------------------------------------------------


def run_experiment(
    torch, path: Path, *, steps: int, num_classes: int, seed: int, image_size: int
) -> Dict[str, Any]:
    """Train ``steps`` steps and infer once. Never raises -- returns the failure.

    #3048 asks for "the failure with its cause for anything red", so an
    exception is DATA to be reported per template, not a reason to abandon the
    sweep 6 templates in.
    """
    record: Dict[str, Any] = {"seed": seed, "steps_requested": steps}
    started = time.perf_counter()
    try:
        torch.manual_seed(seed)
        _, model = build_template(path, num_classes, prefix="od_sweep")
        record["params_m"] = round(
            sum(p.numel() for p in model.parameters()) / 1e6, 1
        )
        images, targets = make_batch(torch, image_size, num_classes)
        record["observed_input_shape"] = observed_input_shape(model, images, targets)

        optimizer = make_optimizer(torch, model)
        model.train()
        findings: List[str] = []
        history: List[float] = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            step_findings = train_step_findings(torch, losses, step)
            findings.extend(step_findings)
            if step == 0:
                record["loss_keys"] = sorted(losses) if isinstance(losses, dict) else []
            if step_findings:
                break
            total = sum(losses.values())
            history.append(float(total.detach()))
            total.backward()
            optimizer.step()

        record["steps_run"] = len(history)
        if history:
            record["loss_first"] = round(history[0], 4)
            record["loss_last"] = round(history[-1], 4)
            record["loss_min"] = round(min(history), 4)
            record["loss_decreased"] = history[-1] < history[0]
            record["loss_ratio"] = (
                round(history[-1] / history[0], 4) if history[0] else None
            )

        n_trainable, no_grad, zero_grad = measure_gradient_reachability(
            torch, model, optimizer, images, targets
        )
        record["n_trainable"] = n_trainable
        record["n_no_grad"] = len(no_grad)
        record["n_zero_grad"] = len(zero_grad)
        findings.extend(gradient_findings(no_grad))

        model.eval()
        with torch.no_grad():
            preds = model(images)
        findings.extend(payload_findings(torch, preds, len(images)))
        record["n_preds"] = [
            int(p["boxes"].shape[0])
            for p in preds
            if isinstance(p, dict) and torch.is_tensor(p.get("boxes"))
        ]
        record["findings"] = findings
        record["mechanical"] = not findings
    except Exception as error:  # noqa: BLE001 -- the failure IS the deliverable
        record["findings"] = [f"{type(error).__name__}: {error}"]
        record["mechanical"] = False
    record["wall_s"] = round(time.perf_counter() - started, 2)
    return record


def sweep(
    *,
    steps: int,
    experiments: int,
    num_classes: int,
    only: Optional[Sequence[str]] = None,
    skip_slow: bool = False,
    engine_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the sweep and return the report structure."""
    try:
        import torch
    except ImportError as error:  # pragma: no cover -- environment, not logic
        raise SystemExit(f"this sweep needs torch: {error}") from error

    accepted = family_values()
    covered = family_templates(accepted)
    if not covered:
        raise SystemExit(
            f"the {FAMILY!r} roster is empty -- the scan lost the tree, and a sweep "
            f"over nothing would report success having checked nothing"
        )

    engine_problems = (
        verify_engine_hyperparameters(engine_root) if engine_root else []
    )
    if engine_problems:
        raise SystemExit(
            "engine hyperparameter contract disagrees with this sweep's constants:\n  "
            + "\n  ".join(engine_problems)
        )

    seeds = seeding_index()
    selected, skipped = [], []
    for path in covered:
        key = template_key(path)
        if only and key not in only:
            continue
        if skip_slow and key in SLOW_TEMPLATES:
            skipped.append(
                {"template": key, "reason": "--skip-slow (measured cost outlier)"}
            )
            continue
        selected.append(path)

    rows: List[Dict[str, Any]] = []
    for path in selected:
        image_size = _read_int_decl(path, "image_size")
        row: Dict[str, Any] = {
            "template": template_key(path),
            "path": str(path.relative_to(REPO_ROOT)),
            "model_type": (read_model_type(path) or "").strip().lower(),
            "declared_image_size": image_size,
            "seeding": seeds.get(path.stem, "unknown (not in the dump survey)"),  # survey keys on stem
            "seeded": False,  # backend#2659: no OD seed is hosted, so never True
        }
        if image_size is None:
            row["experiments"] = []
            row["mechanical"] = False
            row["findings"] = ["no module-level `image_size` declaration to size the batch"]
        else:
            runs = [
                run_experiment(
                    torch,
                    path,
                    steps=steps,
                    num_classes=num_classes,
                    seed=seed,
                    image_size=image_size,
                )
                for seed in range(experiments)
            ]
            row["experiments"] = runs
            verdicts = {bool(r["mechanical"]) for r in runs}
            row["mechanical"] = all(r["mechanical"] for r in runs)
            row["flaky"] = len(verdicts) > 1
            row["findings"] = sorted({f for r in runs for f in r.get("findings", [])})
            row["cycles_run"] = sum(r.get("steps_run", 0) for r in runs)
            row["wall_s"] = round(sum(r.get("wall_s", 0.0) for r in runs), 2)
            first = runs[0]
            row["observed_input_shape"] = first.get("observed_input_shape")
            row["n_preds"] = first.get("n_preds", [])
            row["n_zero_grad"] = first.get("n_zero_grad")
            row["params_m"] = first.get("params_m")
            row["loss_first"] = first.get("loss_first")
            row["loss_last"] = first.get("loss_last")
            row["loss_decreased"] = first.get("loss_decreased")
        row["quality"] = "pending"
        row["quality_cause"] = _quality_cause(row.get("n_preds") or [], row["seeded"])
        rows.append(row)

    return {
        "ticket": "tracebloc/backend#3048",
        "scope": "local-only; the live dev leg is e2e-test-agent harness/od_sweep/",
        "family": FAMILY,
        "family_values": sorted(accepted),
        "schema": schema_path().name,
        "steps": steps,
        "experiments": experiments,
        "num_classes": num_classes,
        "optimizer": {
            "name": ENGINE_OPTIMIZER_NAME,
            "lr": ENGINE_DEFAULT_LR,
            "extra_kwargs": ENGINE_OPTIMIZER_KWARGS,
            "derived_from": f"tracebloc-engine {ENGINE_ADAPTER_PATH}",
            "verified_against_checkout": str(engine_root) if engine_root else None,
        },
        "roster": {
            "od_templates_total": len(od_templates()),
            "covered_by_this_sweep": len(covered),
            "run": len(rows),
            "skipped": skipped,
            "uncovered": [
                {
                    "template": str(p.relative_to(OD_ROOT)),
                    "model_type": (read_model_type(p) or "").strip().lower(),
                    "reason": (
                        "the `yolo` family speaks a different contract (raw grid tensor "
                        "+ external Custom_loss over grid-encoded targets); the "
                        "encode/decode lives in the engine's YoloHandler and model-zoo "
                        "vendors only the schema"
                    ),
                }
                for p in uncovered_templates(accepted)
            ],
        },
        "templates": rows,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_HEADERS = (
    "template",
    "seeded / scratch",
    "mechanical",
    "quality",
    "cause",
    "cycles",
    "loss first -> last",
    "grad-reachable",
    "payload",
    "resolution",
    "failure + cause",
)


def markdown(report: Dict[str, Any]) -> str:
    """The per-template table. #3048: "All models pass" with no table is the
    shape of every false-green this epic has produced."""
    out: List[str] = []
    roster = report["roster"]
    out.append(f"# OD acceptance sweep -- {report['ticket']}")
    out.append("")
    out.append(f"- scope: **{report['scope']}**")
    out.append(
        f"- roster: **{roster['run']} run** of {roster['covered_by_this_sweep']} in family "
        f"`{report['family']}`, out of {roster['od_templates_total']} OD templates total"
    )
    out.append(
        f"- optimizer: `{report['optimizer']['name']}` lr={report['optimizer']['lr']} "
        f"extra_kwargs={report['optimizer']['extra_kwargs'] or '{}'} "
        f"(derived from {report['optimizer']['derived_from']})"
    )
    out.append(
        f"- cycle: {report['steps']} steps x {report['experiments']} experiments per template"
    )
    out.append("")
    passed = sum(1 for r in report["templates"] if r["mechanical"])
    out.append(
        f"**mechanical: {passed}/{len(report['templates'])} pass. "
        f"quality: 0/{len(report['templates'])} measurable** -- no OD seed is hosted "
        f"(backend#2659 blocks backend#3055), so every row below ran from random "
        f"initialisation and quality is pending with a stated cause."
    )
    out.append("")
    out.append("| " + " | ".join(_HEADERS) + " |")
    out.append("|" + "|".join("---" for _ in _HEADERS) + "|")
    for row in report["templates"]:
        loss = (
            f"{row.get('loss_first')} -> {row.get('loss_last')}"
            if row.get("loss_first") is not None
            else "-"
        )
        grad = (
            "yes" if row.get("n_zero_grad") is None else f"yes ({row['n_zero_grad']} zero-grad)"
        )
        if not row["mechanical"]:
            grad = "see failure"
        shape = row.get("observed_input_shape")
        resolution = (
            f"{'x'.join(str(d) for d in shape[-2:])} (declared {row.get('declared_image_size')})"
            if shape
            else f"declared {row.get('declared_image_size')}, no transform"
        )
        cells = (
            f"`{row['template']}`",
            row["seeding"],
            ("PASS" if row["mechanical"] else "**FAIL**")
            + (" (FLAKY)" if row.get("flaky") else ""),
            row["quality"],
            row["quality_cause"],
            str(row.get("cycles_run", 0)),
            loss,
            grad,
            str(row.get("n_preds", [])),
            resolution,
            "; ".join(row.get("findings", [])) or "-",
        )
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    if roster["skipped"]:
        out.append("## Skipped (named, not silently capped)")
        out.append("")
        for entry in roster["skipped"]:
            out.append(f"- `{entry['template']}` -- {entry['reason']}")
        out.append("")
    if roster["uncovered"]:
        out.append("## Not covered by this sweep")
        out.append("")
        for entry in roster["uncovered"]:
            out.append(
                f"- `{entry['template']}` (`{entry['model_type']}`) -- {entry['reason']}"
            )
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=8, help="train steps per cycle")
    parser.add_argument(
        "--experiments",
        type=int,
        default=2,
        help="cycles per template -- #3048 asks for more than one lucky run",
    )
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument(
        "--only", action="append", help="template stem; repeatable"
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help=f"skip the measured cost outliers ({', '.join(sorted(SLOW_TEMPLATES))}); "
        f"they are named in the report's skipped list",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        help="tracebloc-engine checkout; re-derives the optimizer contract and FAILS "
        "on disagreement with this file's constants",
    )
    parser.add_argument("--out", type=Path, help="write the report as JSON")
    parser.add_argument("--markdown", type=Path, help="write the per-template table")
    args = parser.parse_args(argv)

    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.experiments < 1:
        parser.error("--experiments must be at least 1")

    report = sweep(
        steps=args.steps,
        experiments=args.experiments,
        num_classes=args.num_classes,
        only=args.only,
        skip_slow=args.skip_slow,
        engine_root=args.engine,
    )
    if args.out:
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    table = markdown(report)
    if args.markdown:
        args.markdown.write_text(table + "\n", encoding="utf-8")
    print(table)

    failed = [r["template"] for r in report["templates"] if not r["mechanical"]]
    if failed:
        print(f"\nmechanical FAILURES: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
