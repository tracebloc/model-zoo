"""Tests for tools/verify_dumps_against_engine_pin.py — the CI dump gate.

The gate's whole value is that a dump which will NOT strict-load into its
shipped template under the engine's pin makes it go RED, and that a manifest
whose ``built_with`` disagrees with the installed engine pin also goes red. A
gate that stays green for those is vacuous, so these tests exercise every
verdict against throwaway synthetic templates + dumps written to a temp dir
(no transformers/timm needed — a plain nn.Module reproduces the key-layout
contract that matters).
"""

import hashlib
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


CATEGORY = "image_classification"


def _template_dir(root: pathlib.Path, category: str = CATEGORY) -> pathlib.Path:
    d = root / "model_zoo" / category / "pytorch"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_dumps(root: pathlib.Path):
    """A throwaway zoo + flat dist/, laid out as `fetch-all` lays it out.

    Each entry NAME is its dump-directory name, so it resolves (via
    seed_index) to `model_zoo/<category>/pytorch/<name>.py` — the canonical
    manifest carries no template path, the name IS the mapping."""
    dumps = root / "dist"
    dumps.mkdir()
    tdir = _template_dir(root)
    for stem in ("good", "mismatch", "gone"):
        (tdir / f"{stem}.py").write_text(TEMPLATE)
    (tdir / "broken.py").write_text(BROKEN_TEMPLATE)
    torch.save(_Ref().state_dict(), dumps / "good_weights.pkl")
    torch.save(_Ref().state_dict(), dumps / "broken_weights.pkl")
    bad = _Ref().state_dict()
    del bad["fc.bias"]
    torch.save(bad, dumps / "mismatch_weights.pkl")
    return dumps


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(dumps: pathlib.Path, name: str, **extra) -> dict:
    """One canonical `entries` record for `<name>_weights.pkl` in dist/."""
    f = dumps / f"{name}_weights.pkl"
    sha = _sha(f) if f.exists() else "0" * 64
    return {"file": f.name, "sha256": sha, "size_bytes": 1, **extra}


def test_selftest_entrypoint_passes():
    """The built-in --selftest asserts every verdict end to end."""
    assert _tool()._selftest() == 0


def _manifest(mod, entries, **built_with):
    """The canonical schema-2 shape: `entries`, keyed by dump-directory name."""
    return json.dumps(
        {"schema": 2, "prefix": "zoo-weights", "built_with": built_with, "entries": entries}
    )


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
            {name: _entry(dumps, name) for name in ("good", "mismatch", "broken", "gone")},
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
            {"good": _entry(dumps, "good")},
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
            {"good": _entry(dumps, "good")},
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
            {"good": _entry(dumps, "good")},
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
    huge = pathlib.PurePosixPath(next(iter(mod._TOO_LARGE_FOR_CI_RAM)))
    # The template must EXIST now — an entry resolves to its template by name
    # before anything else — but it must never be BUILT: importing it raises.
    huge_dir = _template_dir(tmp_path, huge.parts[0])
    (huge_dir / huge.name).write_text("raise RuntimeError('the RAM skip did not fire')\n")
    torch.save(_Ref().state_dict(), dumps / f"{huge.stem}_weights.pkl")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        _manifest(
            mod,
            {huge.stem: _entry(dumps, huge.stem), "good": _entry(dumps, "good")},
            **_installed_built_with(mod),
        )
    )
    rc = mod.run_sweep(manifest, dumps, tmp_path, tmp_path / "report.json", False, True)
    assert rc == 0
    cats = {
        r["name"]: r["category"]
        for r in json.loads((tmp_path / "report.json").read_text())["results"]
    }
    assert cats[huge.stem] == mod.SKIPPED_RAM
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


def test_too_large_set_matches_contract_suite():
    """The verifier's _TOO_LARGE_FOR_CI_RAM must stay byte-equal to the contract
    suite's set. Two hand-maintained copies would otherwise drift, and a newly
    skipped large template would still be BUILT here and could OOM the sweep."""
    mod = _tool()
    spec = importlib.util.spec_from_file_location(
        "_contract_for_skipset", ROOT / "tests" / "test_model_contract.py"
    )
    assert spec and spec.loader
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    assert mod._TOO_LARGE_FOR_CI_RAM == contract._TOO_LARGE_FOR_CI_RAM


def test_present_manifest_with_empty_dumps_is_red(tmp_path):
    """A PRESENT manifest that declares no dumps must fail closed (exit 2): it
    protects nothing while looking green. Only an ABSENT manifest is the intended
    armed-green no-op. (Name kept; the empty collection is now the canonical
    `entries` dict.)"""
    mod = _tool()
    dumps = _write_dumps(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": 2, "built_with": {}, "entries": {}}))
    rc = mod.run_sweep(manifest, dumps, tmp_path, tmp_path / "report.json", False, True)
    assert rc == 2


# --------------------------------------------------------------------------
# The canonical manifest's per-entry keys: status, category, built_with
# --------------------------------------------------------------------------
# Each of these is a key backend's manifest carries today (or is adding), and
# each one is a way for a sweep to be wrong while looking right: a retired seed
# counted as a failure (the gate is then switched off), an all-retired manifest
# read as green, a shared stem resolved to the wrong category's head, an
# entry's own provenance ignored in favour of the shared block.


def _run(mod, tmp_path, dumps, entries, *, built_with=None, name="manifest.json"):
    manifest = tmp_path / name
    manifest.write_text(
        _manifest(mod, entries, **(built_with if built_with is not None else _installed_built_with(mod)))
    )
    report = tmp_path / f"{name}.report.json"
    rc = mod.run_sweep(manifest, dumps, tmp_path, report, False, True)
    return rc, json.loads(report.read_text()) if report.exists() else None


def _cats(report):
    return {r["name"]: r["category"] for r in report["results"]}


def test_retired_entries_are_neither_verified_nor_failures(tmp_path, capsys):
    """A retired entry has no template and no bytes in dist/ (it is not
    fetched) — both would be failures for a live entry. Retired, they are
    neither, and they are NAMED on a NOT GATED line."""
    mod = _tool()
    dumps = _write_dumps(tmp_path)
    rc, report = _run(
        mod,
        tmp_path,
        dumps,
        {
            "good": _entry(dumps, "good"),
            "detr": _entry(dumps, "detr", status="retired"),
        },
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert _cats(report) == {"good": mod.OK, "detr": mod.RETIRED_ENTRY}
    assert "NOT GATED: 1 retired entr(ies)" in out and "detr" in out, out


def test_a_manifest_of_only_retired_entries_is_not_a_green_sweep(tmp_path, capsys):
    mod = _tool()
    dumps = _write_dumps(tmp_path)
    rc, _ = _run(mod, tmp_path, dumps, {"detr": _entry(dumps, "detr", status="retired")})
    err = capsys.readouterr().err
    assert rc == 2, err
    assert "no LIVE entries" in err and "detr" in err, err


def test_an_undefined_status_is_red_by_name(tmp_path, capsys):
    mod = _tool()
    dumps = _write_dumps(tmp_path)
    rc, report = _run(
        mod,
        tmp_path,
        dumps,
        {"good": _entry(dumps, "good"), "mismatch": _entry(dumps, "mismatch", status="retried")},
    )
    assert rc == 1
    assert _cats(report)["mismatch"] == mod.BAD_STATUS


class _Wide(nn.Module):
    """A second head shape for the same stem in another category."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 5)


WIDE_TEMPLATE = TEMPLATE.replace("nn.Linear(4, 3)", "nn.Linear(4, 5)")


def _shared_stem_zoo(tmp_path):
    """`twin` ships in two categories with different heads — the
    bert_base_uncased shape — and the dump is the WIDE one's."""
    dumps = tmp_path / "dist"
    dumps.mkdir()
    (_template_dir(tmp_path, "image_classification") / "twin.py").write_text(TEMPLATE)
    (_template_dir(tmp_path, "text_classification") / "twin.py").write_text(WIDE_TEMPLATE)
    torch.save(_Wide().state_dict(), dumps / "twin_weights.pkl")
    return dumps


def test_a_recorded_category_selects_between_templates_sharing_a_stem(tmp_path):
    mod = _tool()
    dumps = _shared_stem_zoo(tmp_path)
    rc, report = _run(
        mod, tmp_path, dumps, {"twin": _entry(dumps, "twin", category="text_classification")}
    )
    assert rc == 0, report
    (result,) = report["results"]
    assert result["category"] == mod.OK
    assert result["template"] == "model_zoo/text_classification/pytorch/twin.py"


def test_a_shared_stem_with_no_recorded_category_is_refused_not_picked(tmp_path):
    """Picking the first category alphabetically is backend's old
    find_template bug: the dump then loads against the wrong head."""
    mod = _tool()
    dumps = _shared_stem_zoo(tmp_path)
    rc, report = _run(mod, tmp_path, dumps, {"twin": _entry(dumps, "twin")})
    assert rc == 1
    assert _cats(report) == {"twin": mod.NO_TEMPLATE}


def test_a_recorded_category_the_name_contradicts_is_refused(tmp_path):
    """`sentence_pair_` files a dump under sentence_pair_classification; an
    entry recording another category is two claims disagreeing."""
    mod = _tool()
    dumps = tmp_path / "dist"
    dumps.mkdir()
    (_template_dir(tmp_path, "sentence_pair_classification") / "twin.py").write_text(TEMPLATE)
    (_template_dir(tmp_path, "text_classification") / "twin.py").write_text(TEMPLATE)
    torch.save(_Ref().state_dict(), dumps / "sentence_pair_twin_weights.pkl")
    rc, report = _run(
        mod,
        tmp_path,
        dumps,
        {"sentence_pair_twin": _entry(dumps, "sentence_pair_twin", category="text_classification")},
    )
    assert rc == 1
    (result,) = report["results"]
    assert result["category"] == mod.NO_TEMPLATE
    assert "disagree" in result["detail"], result


def test_an_entrys_own_built_with_replaces_the_shared_block(tmp_path):
    """backend check_provenance.resolve_built_with: own block REPLACES the
    shared one. So a drifted shared block does not redden an entry that
    describes itself correctly — and a correct shared block does not hide an
    entry whose own block drifts."""
    mod = _tool()
    dumps = _write_dumps(tmp_path)
    clean = _installed_built_with(mod)
    drifted = dict(clean, transformers="9.9.9")

    rc, report = _run(
        mod, tmp_path, dumps, {"good": _entry(dumps, "good", built_with=clean)}, built_with=drifted
    )
    assert rc == 0, report["provenance_problems"]

    rc, report = _run(
        mod,
        tmp_path,
        dumps,
        {"good": _entry(dumps, "good", built_with=drifted)},
        built_with=clean,
        name="m2.json",
    )
    assert rc == 1
    assert any(p.startswith("good (own built_with) transformers") for p in report["provenance_problems"])


def test_an_own_built_with_is_never_topped_up_from_the_shared_block(tmp_path):
    """REPLACE, not merge: an own block that omits a package the engine pin
    installs is a partial block, and the shared block — true of OTHER dumps —
    must not fill the gap."""
    mod = _tool()
    clean = _installed_built_with(mod)
    omittable = [k for k in clean if k != "torch"]
    if not omittable:
        pytest.skip("this interpreter installs only torch; a partial own block is not expressible")
    dumps = _write_dumps(tmp_path)
    partial = {k: v for k, v in clean.items() if k != omittable[0]}
    rc, report = _run(
        mod, tmp_path, dumps, {"good": _entry(dumps, "good", built_with=partial)}, built_with=clean
    )
    assert rc == 1
    assert any(
        p.startswith(f"good (own built_with) {omittable[0]}") and "absent" in p
        for p in report["provenance_problems"]
    ), report["provenance_problems"]


def test_a_present_but_empty_own_built_with_is_refused(tmp_path, capsys):
    mod = _tool()
    dumps = _write_dumps(tmp_path)
    rc, _ = _run(mod, tmp_path, dumps, {"good": _entry(dumps, "good", built_with={})})
    assert rc == 2
    assert "present but empty" in capsys.readouterr().err


def test_retired_entries_are_left_out_of_provenance(tmp_path):
    """A retired dump's built_with is still TRUE about who built those bytes,
    and it is expected to drift from today's pin — that is often why it was
    retired. It must not redden the gate it is exempt from."""
    mod = _tool()
    dumps = _write_dumps(tmp_path)
    rc, report = _run(
        mod,
        tmp_path,
        dumps,
        {
            "good": _entry(dumps, "good"),
            "old": _entry(dumps, "old", status="retired", built_with={"transformers": "0.0.1"}),
        },
    )
    assert rc == 0, report["provenance_problems"]


def test_the_retired_dumps_list_shape_is_refused_by_name(tmp_path, capsys):
    """Nothing ever wrote the `dumps` list. Refused as a schema, not read as a
    stub — the advice for the two differs."""
    mod = _tool()
    dumps = _write_dumps(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 2,
                "built_with": _installed_built_with(mod),
                "dumps": [{"name": "good", "template": "x.py", "weights": "good_weights.pkl"}],
            }
        )
    )
    rc = mod.run_sweep(manifest, dumps, tmp_path, tmp_path / "r.json", False, True)
    err = capsys.readouterr().err
    assert rc == 2
    assert "'dumps' list" in err and "nothing ever wrote it" in err, err
    assert "stub" not in err, err


def test_a_file_that_is_not_a_bare_name_is_malformed(tmp_path):
    mod = _tool()
    dumps = _write_dumps(tmp_path)
    rc, report = _run(
        mod, tmp_path, dumps, {"good": dict(_entry(dumps, "good"), file="../good_weights.pkl")}
    )
    assert rc == 1
    assert _cats(report) == {"good": mod.MALFORMED}


def test_the_cli_sweeps_a_canonical_manifest(tmp_path):
    """Through the real entry point, as the workflow calls it: defaults for
    --manifest/--dumps-dir resolve from the file's own location, so the tool
    is copied (with its two sibling modules) into a checkout-shaped dir."""
    import shutil
    import subprocess
    import sys

    mod = _tool()
    (tmp_path / "tools").mkdir()
    for name in ("verify_dumps_against_engine_pin.py", "check_dump_coverage.py", "seed_index.py"):
        shutil.copy2(ROOT / "tools" / name, tmp_path / "tools" / name)
    dumps = _write_dumps(tmp_path)
    (tmp_path / "manifest.json").write_text(
        _manifest(
            mod,
            {"good": _entry(dumps, "good"), "detr": _entry(dumps, "detr", status="retired")},
            **_installed_built_with(mod),
        )
    )
    proc = subprocess.run(
        [sys.executable, "tools/verify_dumps_against_engine_pin.py", "--require-manifest"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "All 1 live dump(s) verify" in proc.stdout, proc.stdout
    assert "NOT GATED: 1 retired" in proc.stdout, proc.stdout
