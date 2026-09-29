"""A folder template's loss must accept its own model's output.

A folder template ships ``model.py`` and ``loss.py`` side by side. The engine
and the SDK's ``configure_loss`` build both with no arguments, ``MyModel()`` and
``Custom_loss()``, so nothing ties the loss's class count to the model's
``output_classes``. A loss that defaults to 10 classes against
``output_classes = 3`` passed every other check and failed only on a user's
first training step: ``yolo_v5`` raised "Expected 20 channels but got 13", and
``yolo_v8``'s factory emitted 20 channels against a 3-class target.

So this builds every folder template the way the engine does, with no
arguments, and runs one forward pass plus the loss and its backward pass. The
folders are found on disk (every ``loss.py`` under ``model_zoo/``), not listed
here, so a new folder template is covered the day it lands.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import pytest

torch = pytest.importorskip("torch")

ROOT = pathlib.Path(__file__).resolve().parents[1]
FOLDERS = sorted(p.parent for p in (ROOT / "model_zoo").rglob("loss.py"))


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_folder_templates_are_found():
    assert FOLDERS, "no folder template with a loss.py under model_zoo/"


@pytest.mark.parametrize("folder", FOLDERS, ids=lambda p: str(p.relative_to(ROOT / "model_zoo")))
def test_bare_loss_accepts_bare_model_output(folder):
    model_mod = _load(folder / "model.py", f"_folder_model_{folder.name}")
    loss_mod = _load(folder / "loss.py", f"_folder_loss_{folder.name}")
    model = getattr(model_mod, model_mod.main_class)()
    loss_fn = loss_mod.Custom_loss()

    torch.manual_seed(0)
    size = model_mod.image_size
    out = model(torch.randn(1, 3, size, size))

    channels = model_mod.output_classes + 5 * loss_fn.B
    assert loss_fn.C == model_mod.output_classes, (
        f"Custom_loss() counts {loss_fn.C} classes, model.py declares "
        f"output_classes = {model_mod.output_classes}"
    )
    assert out.shape[-1] == channels, (
        f"{model_mod.main_class}() emits {out.shape[-1]} channels per cell, "
        f"output_classes + 5 * B = {channels}"
    )

    loss = loss_fn(out, torch.zeros(*out.shape[:-1], channels))
    assert math.isfinite(float(loss))
    loss.backward()
