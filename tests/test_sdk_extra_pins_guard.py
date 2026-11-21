# mutates: tools/check_sdk_extra_pins.py
"""The SDK-extra pin guard reads the real mirror and refuses a pin outside the SDK's range.

The SDK's wheel is not downloaded here: each test builds a wheel whose
``[pytorch]`` extra is derived from the REAL ``.github/requirements/pytorch.txt``
(a caret range around each pin), so the guard is exercised on the repo's own
mirror. The rows break one thing at a time and assert the exact finding.
"""

from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = "tools/check_sdk_extra_pins.py"
MIRROR = ROOT / ".github" / "requirements" / "pytorch.txt"
SDK = ROOT / ".github" / "requirements" / "sdk.txt"
CI = ROOT / ".github" / "workflows" / "ci.yml"
_PINNED = ("torch", "torchvision", "transformers", "timm", "peft")


def _mirror_versions() -> dict[str, str]:
    out = {}
    for ln in MIRROR.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9._-]+)==([^\s#]+)", ln.strip())
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def _sdk_version() -> str:
    m = re.search(r"^tracebloc\[pytorch\]==(\S+)", SDK.read_text(), re.MULTILINE)
    assert m, "sdk.txt no longer pins tracebloc[pytorch]"
    return m.group(1)


def _wheel(tmp: Path, ranges: dict[str, str], version: str | None = None, extra="pytorch") -> Path:
    lines = ["Metadata-Version: 2.1", "Name: tracebloc", f"Version: {version or _sdk_version()}"]
    lines += [f'Requires-Dist: {p}{r}; extra == "{extra}"' for p, r in ranges.items()]
    path = tmp / "tracebloc-x-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("tracebloc-x.dist-info/METADATA", "\n".join(lines) + "\n")
    return path


def _agreeing_ranges() -> dict[str, str]:
    v = _mirror_versions()
    return {p: f">={v[p]}" for p in _PINNED}


def _run(wheel: Path, *extra: str):
    return subprocess.run(
        [sys.executable, "-I", str(ROOT / TOOL), "--wheel", str(wheel), *extra],
        capture_output=True, text=True, cwd=ROOT,
    )


def test_the_real_mirror_is_inside_a_range_built_from_it(tmp_path):
    r = _run(_wheel(tmp_path, _agreeing_ranges()))
    assert r.returncode == 0, r.stderr


def test_ci_runs_the_guard_before_the_sdk_install():
    text = CI.read_text()
    job = text[text.index("  sdk-upload:"):]
    assert "tools/check_sdk_extra_pins.py" in job
    assert job.index("pip download") < job.index("tools/check_sdk_extra_pins.py")
    assert job.index("tools/check_sdk_extra_pins.py") < job.index("pip install -r .github/requirements/test.txt")


def test_row_a_narrowed_torch_cap_below_the_mirror_is_refused(tmp_path):
    v = _mirror_versions()["torch"]
    ranges = _agreeing_ranges()
    assert _run(_wheel(tmp_path, ranges)).returncode == 0
    ranges["torch"] = f"<{v},>=1.0"  # the mutation: the cap now excludes the pin
    r = _run(_wheel(tmp_path, ranges))
    assert r.returncode == 1, r.stderr
    assert f"torch=={v} is outside the SDK's range" in r.stderr


@pytest.mark.parametrize(
    ("pin", "spec", "holds"),
    [
        ("2.0.0", "2.0.0", True),  # the pin is the range's own lower bound
        ("2.0.5", "2.0.0", True),
        ("2.1", "2.0", True),
        ("2.1.0", "2.0.0", False),  # the mutation: another minor
        ("3.0.0", "2.0", False),  # the mutation: another major
        ("1.9.0", "2.0.0", False),
    ],
)
def test_row_a_compatible_release_range_is_judged_on_the_zero_stripped_pin(pin, spec, holds):
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import check_sdk_extra_pins as tool
    finally:
        sys.path.remove(str(ROOT / "tools"))
    assert tool._holds("~=", tool._ver(pin), spec) is holds


@pytest.mark.parametrize("pkg", _PINNED)
def test_row_every_pinned_package_is_held_not_only_torch(tmp_path, pkg):
    v = _mirror_versions()[pkg]
    ranges = _agreeing_ranges()
    ranges[pkg] = f">{v}"
    r = _run(_wheel(tmp_path, ranges))
    assert r.returncode == 1, r.stderr
    assert f"{pkg}=={v} is outside" in r.stderr


def test_row_a_wheel_at_another_version_than_sdk_txt_cannot_be_told(tmp_path):
    r = _run(_wheel(tmp_path, _agreeing_ranges(), version="0.0.1"))
    assert r.returncode == 2
    assert "CANNOT TELL" in r.stderr


def test_row_a_wheel_without_a_pytorch_extra_cannot_be_told(tmp_path):
    r = _run(_wheel(tmp_path, _agreeing_ranges(), extra="sklearn"))
    assert r.returncode == 2
    assert "extra 'pytorch'" in r.stderr


def test_row_an_unreadable_specifier_cannot_be_told(tmp_path):
    ranges = _agreeing_ranges()
    ranges["torch"] = "===2.15.0"
    r = _run(_wheel(tmp_path, ranges))
    assert r.returncode == 2
    assert "cannot read specifier" in r.stderr


def test_row_no_package_in_common_cannot_be_told(tmp_path):
    r = _run(_wheel(tmp_path, {"requests": ">=2"}))
    assert r.returncode == 2
    assert "no package is named by both" in r.stderr
