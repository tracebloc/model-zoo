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


def _installed_built_with(mod):
    """The provenance block that matches whatever is installed in THIS env, so a
    genuinely-clean manifest is expressible regardless of which pins the test
    interpreter happens to carry. A key installed here but omitted would (rightly)
    read as drift."""
    return {
        k: mod._installed_version(k)
        for k in mod._PROVENANCE_KEYS
        if mod._installed_version(k) is not None
    }


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
            **_installed_built_with(mod),
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


def test_partial_built_with_is_red(tmp_path):
    """A NON-empty built_with block that declares torch correctly but OMITS a pin
    the engine actually installs must fail closed — this is the finding's exact
    "partial block, for example only torch" hole. (An entirely-absent block is a
    separate, already-covered failure.)"""
    mod = _tool()
    installed = _installed_built_with(mod)
    omitted = [k for k in mod._PROVENANCE_KEYS if k != "torch" and k in installed]
    if not omitted:
        pytest.skip("this interpreter installs only torch; omission drift not reproducible")
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
    assert rc == 1
    problems = json.loads((tmp_path / "report.json").read_text())["provenance_problems"]
    assert any(
        k in p and "absent from the manifest" in p for p in problems for k in omitted
    )


def test_too_large_template_is_skipped_not_built(tmp_path):
    """A dump whose template is too large to construct in CI RAM is reported
    SKIPPED_RAM: the build is not attempted (so it can't OOM the sweep and take
    every other dump down with it) and it does not redden the gate."""
    mod = _tool()
    dumps = _write_dumps(tmp_path)
    huge_template = "model_zoo/" + next(iter(mod._TOO_LARGE_FOR_CI_RAM))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        _manifest(
            mod,
            [
                # template file intentionally NOT created: the RAM skip must fire
                # before any build/import is attempted.
                {"name": "huge", "template": huge_template, "weights": "good_weights.pkl"},
                {"name": "good", "template": "good.py", "weights": "good_weights.pkl"},
            ],
            **_installed_built_with(mod),
        )
    )
    rc = mod.run_sweep(manifest, dumps, tmp_path, tmp_path / "report.json", False, True)
    assert rc == 0
    cats = {
        r["name"]: r["category"]
        for r in json.loads((tmp_path / "report.json").read_text())["results"]
    }
    assert cats["huge"] == mod.SKIPPED_RAM
    assert cats["good"] == mod.OK


def test_too_large_entries_exist():
    """Every _TOO_LARGE_FOR_CI_RAM entry must name a real template under
    model_zoo/, or it silently skips nothing. Keeps the set in lockstep with the
    tree (mirrors test_model_contract.py's own existence guard)."""
    mod = _tool()
    model_root = ROOT / "model_zoo"
    for entry in mod._TOO_LARGE_FOR_CI_RAM:
        assert (model_root / entry).is_file(), (
            f"_TOO_LARGE_FOR_CI_RAM entry {entry!r} does not exist under model_zoo/"
        )
