"""Tests for tools/verify_dumps_against_engine_pin.py — the CI dump gate.

The gate's whole value is that a dump which will NOT strict-load into its
shipped template under the engine's pin makes it go RED, and that a manifest
whose ``built_with`` disagrees with the installed engine pin also goes red. A
gate that stays green for those is vacuous, so these tests exercise every
verdict against throwaway synthetic templates + dumps written to a temp dir
(no transformers/timm needed — a plain nn.Module reproduces the key-layout
contract that matters).
"""

import importlib.util
import json
import pathlib

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

ROOT = pathlib.Path(__file__).parent.parent
TOOL = ROOT / "tools" / "verify_dumps_against_engine_pin.py"

TEMPLATE = """\
from torch import nn

framework = "pytorch"
main_class = "MyModel"


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
"""

BROKEN_TEMPLATE = """\
framework = "pytorch"
main_class = "MyModel"


class MyModel:
    def __init__(self):
        raise RuntimeError("cannot build under this engine pin")
"""


class _Ref(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)


def _tool():
    spec = importlib.util.spec_from_file_location("verify_dumps_against_engine_pin", TOOL)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_dumps(root: pathlib.Path):
    dumps = root / "dist"
    dumps.mkdir()
    (root / "good.py").write_text(TEMPLATE)
    (root / "broken.py").write_text(BROKEN_TEMPLATE)
    torch.save(_Ref().state_dict(), dumps / "good_weights.pkl")
    bad = _Ref().state_dict()
    del bad["fc.bias"]
    torch.save(bad, dumps / "mismatch_weights.pkl")
    return dumps


def test_selftest_entrypoint_passes():
    """The built-in --selftest asserts every verdict end to end."""
    assert _tool()._selftest() == 0


def _manifest(mod, dumps_entries, **built_with):
    return json.dumps({"schema": 2, "built_with": built_with, "dumps": dumps_entries})


def test_categorises_and_fails_closed_on_bad_dumps(tmp_path):
    mod = _tool()
    dumps = _write_dumps(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        _manifest(
            mod,
            [
                {"name": "good", "template": "good.py", "weights": "good_weights.pkl"},
                {"name": "mismatch", "template": "good.py", "weights": "mismatch_weights.pkl"},
                {"name": "broken", "template": "broken.py", "weights": "good_weights.pkl"},
                {"name": "gone", "template": "good.py", "weights": "absent_weights.pkl"},
            ],
            torch=mod._installed_version("torch"),
        )
    )
    rc = mod.run_sweep(manifest, dumps, tmp_path, tmp_path / "report.json", False, True)
    assert rc == 1

    cats = {r["name"]: r["category"] for r in json.loads((tmp_path / "report.json").read_text())["results"]}
    assert cats == {
        "good": mod.OK,
        "mismatch": mod.KEY_MISMATCH,
        "broken": mod.BUILD_FAIL,
        "gone": mod.MISSING,
    }


def test_provenance_drift_is_red(tmp_path):
    """A built_with that disagrees with the installed engine pin fails closed —
    this is the 'engine transformers bump' alarm."""
    mod = _tool()
    dumps = _write_dumps(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        _manifest(
            mod,
            [{"name": "good", "template": "good.py", "weights": "good_weights.pkl"}],
            transformers="9.9.9",
        )
    )
    rc = mod.run_sweep(manifest, dumps, tmp_path, tmp_path / "report.json", False, True)
    assert rc == 1


def test_all_ok_is_green(tmp_path):
    mod = _tool()
    dumps = _write_dumps(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        _manifest(
            mod,
            [{"name": "good", "template": "good.py", "weights": "good_weights.pkl"}],
            torch=mod._installed_version("torch"),
        )
    )
    rc = mod.run_sweep(manifest, dumps, tmp_path, tmp_path / "report.json", False, True)
    assert rc == 0


def test_absent_manifest_armed_green_but_red_when_required(tmp_path):
    mod = _tool()
    dumps = _write_dumps(tmp_path)
    missing = tmp_path / "nope.json"
    assert mod.run_sweep(missing, dumps, tmp_path, tmp_path / "r.json", False, True) == 0
    assert mod.run_sweep(missing, dumps, tmp_path, tmp_path / "r2.json", True, True) == 2
