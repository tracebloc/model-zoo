#!/usr/bin/env python3
"""Hold the zoo's templates to the memory of the smallest client, at their defaults.

A ready-made model must train as published on any client tracebloc admits a run
on. The tracebloc SDK estimates a run's peak memory before it starts
(``tracebloc.training.memory_estimate``) and warns when that estimate does not
fit the client's per-run memory with margin. This tool asks the SDK that same
question for each template, at the template's own defaults, against
``BUDGET_BYTES``: a template the SDK would warn about is a finding.

A template no batch fits without LoRA is listed in ``NOT_FITTING`` with what
the SDK's sizing does with it on that budget, and the check asks the SDK that
too: a listed template the SDK sizes otherwise, or fits without LoRA, is a
finding.

The estimate and its margin are the SDK's, never restated here: only the
budget and the templates' defaults are this repo's. The SDK is imported when
the check runs, so reading a template's defaults needs nothing but the
standard library.

    python3 tools/template_memory.py          # every template, one line each
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
ZOO = "model_zoo"
INDEX = "model_zoo/index.v1.json"

#: The smallest per-run memory a tracebloc client admits a training run at: its
#: smallest training size, 4 GiB. A template's defaults must fit it.
BUDGET_BYTES = 4 * 1024**3

#: The oldest SDK whose estimate and sizing this check reads.
MIN_SDK = (3, 0, 0)

#: What the SDK's sizing does with a template on ``BUDGET_BYTES``: fits it
#: without LoRA, turns LoRA on to fit it, or refuses it.
FITS = "fits"
LORA = "lora"
REFUSED = "refused"

#: PyTorch templates no batch fits without LoRA, by what the SDK's sizing does
#: with them on the budget. Their defaults are for clients with more memory: a
#: ``REFUSED`` one trains only on such a client.
NOT_FITTING = {
    "causal_language_modeling/pytorch/nanogpt_style_scratch": LORA,
    "image_classification/pytorch/aimv2": REFUSED,
    "masked_language_modeling/pytorch/bert_large_mlm": REFUSED,
    "object_detection/pytorch/sparse_rcnn": REFUSED,
    "object_detection/pytorch/yolo_v1/model": REFUSED,
    "semantic_segmentation/pytorch/dpt": REFUSED,
    "sentence_pair_classification/pytorch/bert_base_uncased": LORA,
    "text_classification/pytorch/bert_base_uncased": LORA,
    "text_classification/pytorch/bert_base_uncased_scratch": LORA,
    "text_classification/pytorch/electra": LORA,
    "text_classification/pytorch/gte_modernbert": REFUSED,
    "text_classification/pytorch/modernbert": REFUSED,
    "text_classification/pytorch/qwen2_5_0_5b": REFUSED,
    "time_series_forecasting/pytorch/timesfm": REFUSED,
    "token_classification/pytorch/deberta_v3_token": LORA,
    "token_classification/pytorch/modernbert_token": LORA,
    "token_classification/pytorch/xlm_roberta_token": LORA,
}


@dataclass(frozen=True)
class Defaults:
    """What a template declares the estimate is taken at."""

    template: str
    category: str
    batch_size: int
    input_size: Optional[int]


#: ``(defaults, path) -> (needed bytes, the SDK's warning or None)``.
Estimator = Callable[[Defaults, Path], "tuple[int, Optional[str]]"]
#: ``(defaults, path) -> (FITS, LORA or REFUSED; the batch the SDK sizes to)``.
Sizer = Callable[[Defaults, Path], "tuple[str, int]"]


def _pytorch_ids(root: Path) -> "set[str]":
    rows = json.loads((root / INDEX).read_text(encoding="utf-8"))["templates"]
    return {row["id"] for row in rows if row["framework"] == "pytorch"}


def checked(root: Path) -> "tuple[list[str], list[str]]":
    """``(templates, findings)``: the index's PyTorch templates but not
    ``NOT_FITTING``, and a finding per ``NOT_FITTING`` entry the index does not
    list as a PyTorch template."""
    ids = _pytorch_ids(root)
    findings = [
        f"{name}: listed in NOT_FITTING but not a checked template"
        for name in sorted(NOT_FITTING.keys() - ids)
    ]
    return sorted(ids - NOT_FITTING.keys()), findings


def _constants(path: Path) -> dict:
    """The template's literal module-level constants, read without importing it."""
    found = {}
    for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    continue
    return found


def read_defaults(root: Path, template: str) -> Defaults:
    """The defaults ``template`` declares. Raises ValueError saying what is wrong."""
    path = root / ZOO / f"{template}.py"
    if not path.is_file():
        raise ValueError(f"no template at {path.relative_to(root)}")
    consts = _constants(path)
    category = consts.get("category")
    if category != template.split("/", 1)[0]:
        raise ValueError(f"declares category {category!r}")
    batch = consts.get("batch_size")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError(f"batch_size {batch!r} is not a positive int")
    # A text template's input size is its sequence length, an image template's
    # its image size; which one the estimate reads for a category is the SDK's.
    size = consts.get("sequence_length", consts.get("image_size"))
    if isinstance(size, (list, tuple)) and size:
        size = size[0]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        size = None
    return Defaults(template, category, batch, size)


def _sdk_version(text: str) -> tuple:
    parts = []
    for piece in text.split(".")[:3]:
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _sdk():
    """The SDK's estimate and sizing, and a reader of a template's model and
    optimizer as ``start()`` takes them. Raises RuntimeError when the installed
    SDK is older than ``MIN_SDK``."""
    from importlib.metadata import version
    import importlib.util

    installed = version("tracebloc")
    if _sdk_version(installed) < MIN_SDK:
        raise RuntimeError(
            f"tracebloc {installed} is installed; this check reads the estimate of "
            f"{'.'.join(map(str, MIN_SDK))} or later"
        )
    from tracebloc.training import memory_estimate, memory_sizing
    from tracebloc.training.plan import (
        _DEFAULT_OPTIMIZER,
        _TEXT_CATEGORIES,
        _TEXT_DEFAULT_OPTIMIZER,
    )

    def read(defaults: Defaults, path: Path):
        spec = importlib.util.spec_from_file_location("zoo_template", path)
        module = importlib.util.module_from_spec(spec)
        # A folder template imports its siblings (``loss.py``) by bare name.
        sys.path.insert(0, str(path.parent))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(path.parent))
        if hasattr(module, "MyModel"):
            model = module.MyModel()
        else:
            model = getattr(module, module.main_class)()
        shape = memory_estimate.describe_model(model)
        if shape is None:
            raise ValueError("the SDK cannot read the model")
        text = defaults.category in _TEXT_CATEGORIES
        if text and defaults.input_size is None:
            # The SDK would assume a length rather than read the template's.
            raise ValueError("declares no sequence_length")
        return shape, _TEXT_DEFAULT_OPTIMIZER if text else _DEFAULT_OPTIMIZER

    return memory_estimate, memory_sizing, read


def sdk_estimator() -> Estimator:
    """The SDK's own estimate, as ``start()`` takes it. Raises RuntimeError when
    the installed SDK is older than ``MIN_SDK``."""
    memory_estimate, _, read = _sdk()

    def estimate(defaults: Defaults, path: Path):
        shape, optimizer = read(defaults, path)
        peak = memory_estimate.estimate_peak_memory(
            shape,
            category=defaults.category,
            batch_size=defaults.batch_size,
            input_size=defaults.input_size,
            optimizer=optimizer,
        )
        return (
            memory_estimate.needed_bytes(peak),
            memory_estimate.memory_fit_warning(peak, BUDGET_BYTES),
        )

    return estimate


def sdk_sizer() -> Sizer:
    """The SDK's own sizing of a template's defaults on ``BUDGET_BYTES``, as
    linking it sizes them. Raises RuntimeError when the installed SDK is older
    than ``MIN_SDK``."""
    _, memory_sizing, read = _sdk()
    verdicts = {memory_sizing.LORA_ON: LORA, memory_sizing.REFUSED: REFUSED}

    def size(defaults: Defaults, path: Path):
        shape, optimizer = read(defaults, path)
        sizing = memory_sizing.size_batch(
            shape,
            category=defaults.category,
            batch_size=defaults.batch_size,
            max_batch_size=None,
            data_cap=None,
            input_size=defaults.input_size,
            optimizer=optimizer,
            lora=False,
            lora_r=None,
            q_lora=False,
            batch_explicit=False,
            lora_explicit=False,
            limit_bytes=BUDGET_BYTES,
            where="memory",
        )
        return verdicts.get(sizing.outcome, FITS), sizing.batch_size

    return size


def check_listed(root: Path, sizer: Sizer) -> "tuple[list[str], list[str]]":
    """``(report lines, findings)`` for ``NOT_FITTING``: one line per entry the
    index lists, and a finding per entry the SDK sizes otherwise than listed.
    Entries the index does not list are :func:`checked`'s findings."""
    lines, findings = [], []
    ids = _pytorch_ids(root)
    for template, listed in sorted(NOT_FITTING.items()):
        if template not in ids:
            continue
        try:
            defaults = read_defaults(root, template)
            verdict, batch = sizer(defaults, root / ZOO / f"{template}.py")
        except Exception as error:  # noqa: BLE001 -- every failure is a finding by name
            findings.append(f"{template}: {error}")
            continue
        lines.append(f"{verdict}  {template}: batch {batch} (listed)")
        if verdict == FITS:
            findings.append(
                f"{template}: listed in NOT_FITTING, but the SDK fits it at batch "
                f"{batch} without LoRA; lower its defaults and take it off the list"
            )
        elif verdict != listed:
            findings.append(
                f"{template}: listed in NOT_FITTING as {listed}, but the SDK's "
                f"sizing says {verdict}"
            )
    return lines, findings


def check(
    root: Path, estimator: Estimator, templates: Optional[Sequence[str]] = None
) -> "tuple[list[str], list[str]]":
    """``(report lines, findings)``: one line per template, and one finding per
    template that is missing, unreadable, or warned about at its defaults.
    ``templates`` defaults to :func:`checked`'s, with its findings. An empty
    template list is a finding: a check of nothing passes nothing."""
    lines, findings = [], []
    if templates is None:
        templates, findings = checked(root)
    if not templates:
        findings.append("no template is checked")
    for template in templates:
        try:
            defaults = read_defaults(root, template)
            needed, warning = estimator(defaults, root / ZOO / f"{template}.py")
        except Exception as error:  # noqa: BLE001 -- every failure is a finding by name
            findings.append(f"{template}: {error}")
            continue
        figure = f"{needed / 1024**3:.2f} GiB at batch {defaults.batch_size}, size {defaults.input_size}"
        lines.append(f"{'OVER' if warning else 'fits'}  {template}: {figure}")
        if warning:
            findings.append(
                f"{template}: ~{figure}, over the {BUDGET_BYTES / 1024**3:.0f} GiB "
                f"budget with the SDK's margin; lower its defaults"
            )
    return lines, findings


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    lines, findings = check(ROOT, sdk_estimator())
    listed_lines, listed_findings = check_listed(ROOT, sdk_sizer())
    lines += listed_lines
    findings += listed_findings
    for line in lines:
        print(line)
    for finding in findings:
        print(f"FINDING {finding}", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
