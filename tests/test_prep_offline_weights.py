"""Mutation tests for tools/prep_offline_weights.py's offline verification.

The tool's whole value is that a template which still downloads at build
time makes it go RED. A verifier that stays green for such a template is
vacuous — so these tests exercise the ``--verify-ship`` subprocess exactly
as the tool invokes it (same environment builder), against throwaway
templates written to a temp dir:

* a genuinely offline template must pass (positive control — proves a red
  mutant is red for the right reason, not because the harness is broken);
* a template that fetches a checkpoint over the network must fail;
* a template whose parameters don't match the dump must fail (the strict
  load is the platform's seed-load contract).

The fetching template lives only in tmp_path, never under model_zoo/.
"""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

ROOT = pathlib.Path(__file__).parent.parent
TOOL = ROOT / "tools" / "prep_offline_weights.py"

OFFLINE_TEMPLATE = """\
import torch.nn as nn

framework = "pytorch"
main_class = "MyModel"
category = "tabular_classification"


def MyModel():
    return nn.Linear(4, 2)
"""

# A template that still pulls a checkpoint over the network at build time —
# through torch.hub, which consults none of the HF offline variables.
FETCHING_TEMPLATE = """\
import torch
import torch.nn as nn

framework = "pytorch"
main_class = "MyModel"
category = "image_classification"


def MyModel():
    torch.hub.load_state_dict_from_url(
        "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    )
    return nn.Linear(4, 2)
"""


def _tool_module():
    spec = importlib.util.spec_from_file_location("prep_offline_weights", TOOL)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_verify_ship(ship: pathlib.Path, state: pathlib.Path, tmp_path: pathlib.Path):
    """Invoke --verify-ship the way the tool's main mode does: fresh
    subprocess, offline env set before interpreter start, empty caches."""
    cache = tmp_path / "empty-cache"
    cache.mkdir(exist_ok=True)
    return subprocess.run(
        [sys.executable, str(TOOL), "--verify-ship", str(ship), "--state", str(state)],
        env=_tool_module()._offline_env(str(cache)),
        capture_output=True,
        text=True,
        check=False,  # each test asserts the returncode it expects
    )


def test_offline_template_passes(tmp_path):
    ship = tmp_path / "offline_template.py"
    ship.write_text(OFFLINE_TEMPLATE)
    state = tmp_path / "state.pt"
    torch.save(torch.nn.Linear(4, 2).state_dict(), state)

    result = _run_verify_ship(ship, state, tmp_path)

    assert result.returncode == 0, result.stderr
    assert "strict load" in result.stdout


def test_network_fetching_template_goes_red(tmp_path):
    """Mutation test: a build-time download must fail the verification."""
    ship = tmp_path / "fetching_template.py"
    ship.write_text(FETCHING_TEMPLATE)
    state = tmp_path / "state.pt"
    torch.save(torch.nn.Linear(4, 2).state_dict(), state)

    result = _run_verify_ship(ship, state, tmp_path)

    assert result.returncode != 0, (
        "verification stayed green for a template that downloads at build time:\n"
        + result.stdout
    )
    assert "strict load" not in result.stdout
    assert "network access blocked" in result.stderr


def test_mismatched_state_dict_goes_red(tmp_path):
    """The strict load is the seed-load contract: wrong keys/shapes = red."""
    ship = tmp_path / "offline_template.py"
    ship.write_text(OFFLINE_TEMPLATE)
    state = tmp_path / "state.pt"
    torch.save(torch.nn.Linear(8, 3).state_dict(), state)

    result = _run_verify_ship(ship, state, tmp_path)

    assert result.returncode != 0
    assert "strict load" not in result.stdout
