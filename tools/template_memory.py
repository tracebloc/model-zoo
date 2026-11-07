#!/usr/bin/env python3
"""Hold the zoo's templates to the memory of the smallest client, at their defaults.

A ready-made model must train as published on any client tracebloc admits a run
on. The tracebloc SDK estimates a run's peak memory before it starts
(``tracebloc.training.memory_estimate``) and warns when that estimate does not
fit the client's per-run memory with margin. This tool asks the SDK that same
question for each template, at the template's own defaults, against
``BUDGET_BYTES``: a template the SDK would warn about is a finding.

The estimate and its margin are the SDK's, never restated here: only the
budget and the templates' defaults are this repo's. The SDK is imported when
the check runs, so reading a template's defaults needs nothing but the
standard library.

    python3 tools/template_memory.py          # every checked template, one line each
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
ZOO = "model_zoo"

#: The smallest per-run memory a tracebloc client admits a training run at: its
#: smallest training size, 4 GiB. A template's defaults must fit it.
BUDGET_BYTES = 4 * 1024**3

#: The oldest SDK whose estimate this check reads.
MIN_SDK = (1, 2, 88)

#: The templates held to the budget, as ``<category>/<framework>/<name>``.
CHECKED = (
    "multi_label_text_classification/pytorch/esm2_35m",
    "text_regression/pytorch/esm2_35m",
)


@dataclass(frozen=True)
class Defaults:
    """What a template declares the estimate is taken at."""

    template: str
    category: str
    batch_size: int
    input_size: Optional[int]


#: ``(defaults, path) -> (needed bytes, the SDK's warning or None)``.
Estimator = Callable[[Defaults, Path], "tuple[int, Optional[str]]"]


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


def sdk_estimator() -> Estimator:
    """The SDK's own estimate, as ``start()`` takes it. Raises RuntimeError when
    the installed SDK is older than ``MIN_SDK``."""
    from importlib.metadata import version
    import importlib.util

    installed = version("tracebloc")
    if _sdk_version(installed) < MIN_SDK:
        raise RuntimeError(
            f"tracebloc {installed} is installed; this check reads the estimate of "
            f"{'.'.join(map(str, MIN_SDK))} or later"
        )
    from tracebloc.training import memory_estimate
    from tracebloc.training.plan import (
        _DEFAULT_OPTIMIZER,
        _TEXT_CATEGORIES,
        _TEXT_DEFAULT_OPTIMIZER,
    )

    def estimate(defaults: Defaults, path: Path):
        spec = importlib.util.spec_from_file_location("zoo_template", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
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
        peak = memory_estimate.estimate_peak_memory(
            shape,
            category=defaults.category,
            batch_size=defaults.batch_size,
            input_size=defaults.input_size,
            optimizer=_TEXT_DEFAULT_OPTIMIZER if text else _DEFAULT_OPTIMIZER,
        )
        return (
            memory_estimate.needed_bytes(peak),
            memory_estimate.memory_fit_warning(peak, BUDGET_BYTES),
        )

    return estimate


def check(
    root: Path, estimator: Estimator, templates: Sequence[str] = CHECKED
) -> "tuple[list[str], list[str]]":
    """``(report lines, findings)``: one line per template, and one finding per
    template that is missing, unreadable, or warned about at its defaults. An
    empty template list is a finding: a check of nothing passes nothing."""
    lines, findings = [], []
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
    for line in lines:
        print(line)
    for finding in findings:
        print(f"FINDING {finding}", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
