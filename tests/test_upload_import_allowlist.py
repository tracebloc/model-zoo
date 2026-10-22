"""Every template imports only what the backend's upload scan admits.

WHY THIS TEST EXISTS
--------------------
The backend runs Bandit over every uploaded model file, together with its own
plugin (TBT001), and refuses the upload on ANY finding. TBT001's first rule is
an import allowlist. A file whose imports reach a top-level name outside
``ALLOWED_PYTHON_PACKAGE_IMPORTS`` is refused. That includes the standard
library: ``from typing import List`` or ``import copy`` is enough. Nothing here
checked this, so ``cascade_rcnn.py`` / ``sparse_rcnn.py`` (``typing``) and
``yolov10_s.py`` (``copy``) shipped and could not be uploaded (model-zoo#142).

The allowlist is vendored in
``tests/contracts/tracebloc_backend/upload_import_allowlist.v1.json`` (refresh
recipe in that directory's README). This test reads it and walks every
template's AST the way Bandit builds its per-file import set: every ``import``
and ``from ... import`` statement, at ANY depth. An import inside a function
body counts the same as one at the top of the file.

Two narrower checks cover the non-import Bandit findings a template has
tripped: B610 (Django ``QuerySet.extra``) matches any call whose name is
``extra``, so a loop variable called ``extra`` refused ``efficientdet_d0.py``;
B110 matches an ``except`` whose whole body is ``pass``, which kept the
stability-selection templates out.

Stdlib only, no network, no Bandit install: runs in every CI framework job.
"""
from __future__ import annotations

import ast
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ZOO = ROOT / "model_zoo"
CONTRACT = (
    ROOT / "tests" / "contracts" / "tracebloc_backend" / "upload_import_allowlist.v1.json"
)

#: Templates KNOWN to be refused today, each with the issue that owns the
#: decision. This list may only shrink. A listed template that has become clean
#: fails the test below, so its row cannot outlive the fix. Do not add a row to
#: make a new template pass: an import the gate refuses is that template's bug.
KNOWN_REFUSED: dict[str, set[str]] = {}

#: Templates PARKED because the gate refuses them: moved out of ``model_zoo/`` to
#: the same path under ``parked/``, so no walker, and no row of ``index.v1.json``,
#: offers them (``parked/README.md`` says what brings each back). Each row records
#: what the gate refuses, by ``_gate_refusals``. A parked template the gate would
#: admit fails below: un-park it.
PARKED_ROOT = ROOT / "parked"
PARKED = {
    "tabular_classification/sklearn/random_forest_stability.py": {"B110"},
}


def _allowed() -> frozenset[str]:
    data = json.loads(CONTRACT.read_text())
    return frozenset(data["allowed_top_level_packages"])


def _templates() -> list[pathlib.Path]:
    return sorted(ZOO.rglob("*.py"))


def _refused_imports(source: str, allowed: frozenset[str]) -> set[str]:
    """Top-level names *source* imports that the gate would refuse."""
    refused = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # A relative import resolves against a sibling file, and a template
            # is uploaded alone: there is no sibling, so the gate refuses it.
            if node.level:
                refused.add("." * node.level + (node.module or ""))
                continue
            names = [node.module or ""]
        else:
            continue
        for name in names:
            top = name.split(".", 1)[0]
            if top not in allowed:
                refused.add(top)
    return refused


def _try_except_pass_lines(source: str) -> list[int]:
    """Lines of the handlers Bandit's B110 flags under the default config the scan
    runs with: a bare ``except:`` or ``except Exception:`` whose whole body is ``pass``."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ExceptHandler) or len(node.body) != 1:
            continue
        broad = node.type is None or getattr(node.type, "id", None) == "Exception"
        if broad and isinstance(node.body[0], ast.Pass):
            lines.append(node.lineno)
    return lines


def _gate_refusals(source: str, allowed: frozenset[str]) -> set[str]:
    """What the gate refuses in *source*: imports by top-level name, B110 by its id."""
    refused = _refused_imports(source, allowed)
    if _try_except_pass_lines(source):
        refused.add("B110")
    return refused


def test_the_contract_and_the_roster_are_not_empty() -> None:
    """A control: an empty allowlist or an empty glob would pass everything below vacuously."""
    assert {"torch", "sklearn"} <= _allowed()
    assert len(_templates()) > 100


def test_the_walk_sees_what_the_gate_refuses() -> None:
    """A control that the walk catches each import shape that refused a real template."""
    allowed = _allowed()
    probes = {
        "from typing import List\n": {"typing"},
        "import copy\n": {"copy"},
        "def f():\n    import os.path\n": {"os"},
        "from __future__ import annotations\n": {"__future__"},
        "from . import sibling\n": {"."},
        "import torch.nn\nfrom collections import OrderedDict\n": set(),
    }
    for source, expected in probes.items():
        assert _refused_imports(source, allowed) == expected, source


@pytest.mark.parametrize(
    "path", _templates(), ids=lambda p: str(p.relative_to(ZOO))
)
def test_template_imports_only_allowlisted_packages(path: pathlib.Path) -> None:
    rel = str(path.relative_to(ZOO))
    refused = _refused_imports(path.read_text(), _allowed())
    if rel in KNOWN_REFUSED:
        assert refused == KNOWN_REFUSED[rel], (
            f"{rel} is listed in KNOWN_REFUSED with {sorted(KNOWN_REFUSED[rel])} but "
            f"now refuses {sorted(refused)}. If it is clean, delete its row; the "
            "list only shrinks."
        )
        return
    assert not refused, (
        f"{rel} imports {sorted(refused)}, which the backend's upload scan refuses "
        "(TBT001 allowlist, tests/contracts/tracebloc_backend/"
        "upload_import_allowlist.v1.json). Standard-library modules are refused "
        "too: use built-in generics instead of `typing`, and build a second "
        "module instead of `copy.deepcopy`."
    )


@pytest.mark.parametrize(
    "path", _templates(), ids=lambda p: str(p.relative_to(ZOO))
)
def test_template_makes_no_call_named_extra(path: pathlib.Path) -> None:
    """Bandit B610 flags any call named ``extra``, and one finding refuses the upload."""
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == "extra":
                hits.append(node.lineno)
    assert not hits, (
        f"{path.relative_to(ZOO)} calls something named `extra` at line(s) {hits}; "
        "Bandit's B610 flags it and the backend refuses the upload. Rename it."
    )


def test_the_b110_walk_sees_what_bandit_flags() -> None:
    """A control on each handler shape, flagged or not, under Bandit's default config."""
    probes = {
        "try:\n    f()\nexcept:\n    pass\n": [3],
        "try:\n    f()\nexcept Exception:\n    pass\n": [3],
        "try:\n    f()\nexcept ValueError:\n    pass\n": [],
        "try:\n    f()\nexcept Exception:\n    g()\n": [],
        "try:\n    f()\nexcept Exception:\n    pass\n    g()\n": [],
    }
    for source, expected in probes.items():
        assert _try_except_pass_lines(source) == expected, source


@pytest.mark.parametrize(
    "path", _templates(), ids=lambda p: str(p.relative_to(ZOO))
)
def test_template_has_no_try_except_pass(path: pathlib.Path) -> None:
    """Bandit B110 flags ``except: pass``, and one finding refuses the upload."""
    hits = _try_except_pass_lines(path.read_text())
    assert not hits, (
        f"{path.relative_to(ZOO)} swallows an exception with `pass` at line(s) {hits}; "
        "Bandit's B110 flags it and the backend refuses the upload. Handle the case "
        "the exception stands for, or let it raise."
    )


def _index_ids() -> set[str]:
    return {row["id"] for row in json.loads((ZOO / "index.v1.json").read_text())["templates"]}


def test_the_parked_roster_is_exactly_the_parked_directory() -> None:
    """Every row names a file under ``parked/``, and every file there has a row.

    Derived both ways, so a stale row and an unlisted parked file are each red.
    """
    on_disk = {p.relative_to(PARKED_ROOT).as_posix() for p in PARKED_ROOT.rglob("*.py")}
    assert on_disk == set(PARKED)


@pytest.mark.parametrize("rel", sorted(PARKED))
def test_a_parked_template_is_not_offered(rel: str) -> None:
    """Not under ``model_zoo/``, not in the index, not seen by the template walk."""
    assert not (ZOO / rel).exists(), f"{rel} is parked but also under model_zoo/"
    assert rel.removesuffix(".py") not in _index_ids(), f"{rel} is parked but indexed"
    assert ZOO / rel not in _templates()


@pytest.mark.parametrize("rel", sorted(PARKED))
def test_a_parked_template_is_still_refused(rel: str) -> None:
    """The reason it was parked still holds; once the gate admits it, un-park it."""
    refused = _gate_refusals((PARKED_ROOT / rel).read_text(), _allowed())
    assert refused == PARKED[rel], (
        f"parked/{rel} is parked for {sorted(PARKED[rel])} but the gate now refuses "
        f"{sorted(refused)}. If that is nothing, un-park it (parked/README.md)."
    )
