"""Every checked template trains, at its defaults, within the smallest client's memory,
and every listed one is sized by the SDK as ``NOT_FITTING`` says.

Runs ``tools/template_memory.py`` with the tracebloc SDK's own estimate, so it
needs the SDK: like ``test_sdk_upload_from_clone.py`` it runs in the
``sdk-upload`` CI job (``MODEL_ZOO_SDK_UPLOAD=1``) and skips elsewhere. There a
missing or too-old SDK is a failure, never a skip. Run it locally with::

    pip install -r .github/requirements/sdk.txt
    MODEL_ZOO_SDK_UPLOAD=1 pytest tests/test_template_memory.py
"""

import importlib.util
import os
import pathlib
import sys

import pytest

if os.environ.get("MODEL_ZOO_SDK_UPLOAD") != "1":
    pytest.skip(
        "runs in the sdk-upload CI job (MODEL_ZOO_SDK_UPLOAD=1, SDK from "
        ".github/requirements/sdk.txt)",
        allow_module_level=True,
    )

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = "tools/template_memory.py"


def _load():
    spec = importlib.util.spec_from_file_location("template_memory", ROOT / TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load()


def test_every_checked_template_fits_at_its_defaults():
    templates, _ = tool.checked(ROOT)
    lines, findings = tool.check(ROOT, tool.sdk_estimator())
    assert not findings, "\n".join(findings)
    assert len(lines) == len(templates) > 0


def test_every_listed_template_is_sized_as_listed():
    lines, findings = tool.check_listed(ROOT, tool.sdk_sizer())
    assert not findings, "\n".join(findings)
    assert len(lines) == len(tool.NOT_FITTING) > 0
