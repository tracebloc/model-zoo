"""The zoo's category directories, held to the ingestor's published category enum.

WHY THIS FILE EXISTS
--------------------
The platform publishes one list of task categories and several repos keep a
copy. Nothing compared this repo's copy -- the directories under ``model_zoo/``
-- to anything. It matched the published list by luck, not by a check. The
logic lives in ``tools/ingest_category_contract.py``; its docstring has the
full argument. This file is the gate.

THREE ASSERTIONS, KEPT APART so a red run names which one broke:

* ``test_every_published_category_has_a_zoo_directory`` -- a category the
  producer accepts with no directory here: shipped with no models.
* ``test_every_zoo_directory_is_a_published_category`` -- a directory here the
  producer does not accept: a zombie. A single ``==`` would say only "these
  differ"; the fixes are different, so the findings are too.
* ``test_each_model_declares_the_category_of_its_directory`` -- a module whose
  ``category`` disagrees with the directory it sits in.

Those three read the VENDORED enum and need no network. A check against a
vendored copy agrees with itself the day the copy goes stale, so the fourth --
``test_the_vendored_enum_equals_the_producers_develop`` -- holds the copy to
the producer. It is marked ``ingest_producer``, deselected by default
(``conftest.py``), and selected explicitly by ci.yml's ``category-contract``
job, which hands it a checkout of the producer. With no checkout it FAILS: zero
comparisons is not agreement.

The rest drive the helpers with synthetic inputs, and assert on WHICH failure
fired -- a check that reddens for a different reason than the one under test is
indistinguishable from a proof.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import urllib.error

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_TOOL = ROOT / "tools" / "ingest_category_contract.py"


def _load():
    spec = importlib.util.spec_from_file_location("ingest_category_contract", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load()

#: Env var naming a checkout of the producer at its ``develop``. Set only by the
#: CI job that selects the marked test.
PRODUCER_CHECKOUT_ENV = "INGEST_PRODUCER_CHECKOUT"

DIRECTORIES = gate.zoo_directories()
PUBLISHED = gate.load_vendored()
DECLARED = gate.declared_categories()


# ── fail closed: the populations are not empty ─────────────────────────────


def test_the_populations_are_not_empty():
    """An empty listing makes every assertion below vacuously true, and that is
    indistinguishable in a log from agreement."""
    assert DIRECTORIES, f"no category directories under {gate.MODEL_ROOT}"
    assert PUBLISHED, f"no categories in {gate.VENDORED}"
    assert DECLARED, f"no model modules under {gate.MODEL_ROOT}"


def test_every_category_directory_holds_a_model():
    """A directory is only a category if something in it declares one. Without
    this, an empty directory satisfies the tree == enum check on its own."""
    populated = {directory for directory, _ in DECLARED.values()}
    empty = sorted(DIRECTORIES - populated)
    assert not empty, f"category directories with no model module in them: {empty}"


# ── the tree against the published enum, both directions ───────────────────


def test_the_vendored_enum_lists_no_category_twice():
    """Set equality hides a duplicate: {a, b} == {a, b} whether or not `a` was
    listed twice."""
    assert not gate.duplicates(PUBLISHED), f"{gate.VENDORED.name} repeats {gate.duplicates(PUBLISHED)}"


def test_every_published_category_has_a_zoo_directory():
    missing = sorted(set(PUBLISHED) - DIRECTORIES)
    assert not missing, (
        f"the published category enum accepts {missing}, and model_zoo/ has no directory for "
        f"{'it' if len(missing) == 1 else 'them'}: a category shipped with no models. Add "
        f"model_zoo/<category>/ with at least one template, or -- if the category is being "
        f"retired -- retire it in the producer first and refresh {gate.VENDORED.name}."
    )


def test_every_zoo_directory_is_a_published_category():
    zombies = sorted(DIRECTORIES - set(PUBLISHED))
    assert not zombies, (
        f"model_zoo/ has {zombies}, which the published category enum does not accept: a "
        f"zombie directory. A dataset of that category cannot be ingested, so nothing can "
        f"train these templates. Remove the directory, or add the category in the producer "
        f"first and refresh {gate.VENDORED.name}."
    )


# ── each module's declared category, asserted INTO the directory set ───────


@pytest.mark.parametrize("rel", sorted(DECLARED), ids=str)
def test_each_model_declares_the_category_of_its_directory(rel):
    directory, category = DECLARED[rel]
    assert category == directory, (
        f"model_zoo/{rel} declares category = {category!r} but sits in model_zoo/{directory}/. "
        f"The directory is the category; the declaration must match it."
    )


# ── the vendored copy against the producer ─────────────────────────────────


@pytest.mark.ingest_producer
def test_the_vendored_enum_equals_the_producers_develop():
    """The layer that stops a stale vendored copy from passing.

    Reads a checkout of the producer, not the network: ci.yml mints the read
    token and checks the producer out in a job that runs none of this repo's
    code, and hands only the schema file across."""
    root = os.environ.get(PRODUCER_CHECKOUT_ENV)
    if not root:
        pytest.fail(
            f"could not look: {PRODUCER_CHECKOUT_ENV} is not set, so there is no producer "
            f"checkout to compare {gate.VENDORED.name} against. This test is selected only "
            f"where the producer is supposed to be available; its absence there is a failure, "
            f"not a skip."
        )
    rel, body = gate.select_from_tree(pathlib.Path(root))
    upstream = gate.categories_from(body, rel)
    findings = gate.compare(upstream, PUBLISHED, f"the producer ({gate.PRODUCER_REF})", gate.VENDORED.name)
    assert not findings, (
        "the vendored category enum is STALE:\n  "
        + "\n  ".join(findings)
        + "\nRefresh it with `python3 tools/ingest_category_contract.py --write`, then reconcile "
        "model_zoo/ against the other tests in this file."
    )


# ── the helpers, driven with synthetic input ───────────────────────────────


def _tree(tmp_path, spec):
    """``spec`` is ``{relative path: source}``."""
    for rel, src in spec.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    return tmp_path


def test_directories_are_derived_and_ignore_caches_and_files(tmp_path):
    root = _tree(
        tmp_path,
        {
            "image_classification/m.py": "",
            "__pycache__/x.pyc": "",
            ".hidden/x": "",
            "TOKENIZERS.md": "",
        },
    )
    assert gate.zoo_directories(root) == {"image_classification"}


def test_support_files_are_not_models_and_computed_categories_are_findings(tmp_path):
    root = _tree(
        tmp_path,
        {
            "a/good.py": 'framework = "pytorch"\ncategory = "a"\n',
            "a/sub/loss.py": "import torch\n",
            "a/annotated.py": 'framework: str = "pytorch"\ncategory: str = "a"\n',
            "a/nocat.py": 'framework = "pytorch"\n',
            "a/computed.py": 'framework = "pytorch"\ncategory = "a" + ""\n',
            "a/nested.py": 'framework = "pytorch"\ndef f():\n    category = "a"\n',
        },
    )
    got = gate.declared_categories(root)
    assert "a/sub/loss.py" not in got, "a module declaring neither field is a support file"
    assert got["a/good.py"] == ("a", "a")
    assert got["a/annotated.py"] == ("a", "a")
    assert got["a/nocat.py"] == ("a", None), "a model with no category must surface as None"
    assert got["a/computed.py"][1].startswith("<not a string literal")
    assert got["a/nested.py"] == ("a", None), "only MODULE-level assignments count"


def _schema(categories) -> bytes:
    return json.dumps({"properties": {"category": {"enum": categories}}}).encode()


@pytest.mark.parametrize(
    "document, fragment",
    [
        (b"{}", "has no `properties`"),
        (b'{"properties": {}}', "has no `properties/category`"),
        (b'{"properties": {"category": {}}}', "has no `properties/category/enum`"),
        (b"[]", "has no `properties`"),
        (_schema([]), "EMPTY or non-list"),
        (_schema({}), "EMPTY or non-list"),
        (_schema(None), "EMPTY or non-list"),
        (_schema(["a", 3]), "not non-empty strings"),
        (_schema(["a", ""]), "not non-empty strings"),
        (b"not json", "not valid JSON"),
    ],
)
def test_a_document_that_is_not_the_contract_is_a_refusal(document, fragment):
    """``bool({})`` is False: a malformed payload is exactly the shape that takes
    a permissive branch and reports agreement it never established."""
    with pytest.raises(gate.Refusal) as exc:
        gate.categories_from(document, "u")
    assert fragment in str(exc.value)


def test_compare_names_each_direction_and_duplicates_separately():
    assert gate.compare(["a", "b"], ["b", "a"], "P", "Z") == [], "set equality, not order"
    assert gate.compare(["a", "b", "c"], ["a", "b"], "P", "Z") == ["only P has: ['c']"]
    assert gate.compare(["a"], ["a", "z"], "P", "Z") == ["only Z has: ['z']"]
    assert gate.compare(["a", "a"], ["a"], "P", "Z") == ["P lists ['a'] more than once"]


# ── dual-path selection: from a checkout ───────────────────────────────────

NEW, OLD = gate.CANDIDATE_PATHS


def test_the_new_path_is_tried_first():
    assert NEW.startswith("tracebloc_ingestor/contracts/"), gate.CANDIDATE_PATHS
    assert OLD == "tracebloc_ingestor/schema/ingest.v1.json"


def test_tree_selection_prefers_the_newest_present_candidate(tmp_path):
    _tree(tmp_path, {NEW: "new", OLD: "old"})
    assert gate.select_from_tree(tmp_path) == (NEW, b"new")


def test_tree_selection_falls_back_when_the_newest_is_absent(tmp_path):
    _tree(tmp_path, {OLD: "old"})
    assert gate.select_from_tree(tmp_path) == (OLD, b"old")


def test_tree_selection_refuses_naming_every_path_when_all_are_absent(tmp_path):
    with pytest.raises(gate.Refusal) as exc:
        gate.select_from_tree(tmp_path)
    for rel in gate.CANDIDATE_PATHS:
        assert rel in str(exc.value)


def test_tree_selection_is_fatal_from_a_present_candidate_that_is_not_a_file(tmp_path):
    (tmp_path / NEW).mkdir(parents=True)
    _tree(tmp_path, {OLD: "old"})
    with pytest.raises(gate.Refusal) as exc:
        gate.select_from_tree(tmp_path)
    assert NEW in str(exc.value) and "not a file" in str(exc.value), "must NOT fall through to the old path"


def test_tree_selection_refuses_a_missing_checkout(tmp_path):
    with pytest.raises(gate.Refusal, match="nothing was fetched"):
        gate.select_from_tree(tmp_path / "absent")


# ── dual-path selection: over HTTP (the refresh) ───────────────────────────


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _opener(answers):
    """``answers`` maps a candidate path to bytes (200) or an int (HTTP error)."""
    seen = []

    def opener(req, timeout):
        rel = next(p for p in gate.CANDIDATE_PATHS if f"/contents/{p}?" in req.full_url)
        seen.append(rel)
        answer = answers[rel]
        if isinstance(answer, int):
            raise urllib.error.HTTPError(req.full_url, answer, "x", {}, None)
        if isinstance(answer, Exception):
            raise answer
        return _Resp(answer)

    opener.seen = seen
    return opener


def test_http_a_404_or_410_selects_the_next_candidate():
    for status in (404, 410):
        op = _opener({NEW: status, OLD: b"old"})
        assert gate.fetch_http("o", "develop", None, op) == (OLD, b"old")
        assert op.seen == [NEW, OLD]


def test_http_prefers_the_newest_and_stops_there():
    op = _opener({NEW: b"new", OLD: b"old"})
    assert gate.fetch_http("o", "develop", None, op) == (NEW, b"new")
    assert op.seen == [NEW]


@pytest.mark.parametrize("status", [401, 403, 500, 502])
def test_http_any_other_status_is_fatal_naming_the_candidate(status):
    op = _opener({NEW: status, OLD: b"old"})
    with pytest.raises(gate.Refusal) as exc:
        gate.fetch_http("o", "develop", None, op)
    assert NEW in str(exc.value) and str(status) in str(exc.value)
    assert op.seen == [NEW], "could-not-look must never fall through to the next candidate"


def test_http_a_transport_failure_is_fatal_naming_the_candidate():
    op = _opener({NEW: urllib.error.URLError("dns"), OLD: b"old"})
    with pytest.raises(gate.Refusal) as exc:
        gate.fetch_http("o", "develop", None, op)
    assert NEW in str(exc.value)


def test_http_all_absent_is_a_refusal_naming_every_path():
    op = _opener({NEW: 404, OLD: 404})
    with pytest.raises(gate.Refusal) as exc:
        gate.fetch_http("o", "develop", None, op)
    for rel in gate.CANDIDATE_PATHS:
        assert rel in str(exc.value)


# ── the marker is deselected by default and selectable ─────────────────────


def _run_pytest(where, *args):
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *args],
        cwd=where,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return proc.returncode, proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""


def test_the_producer_test_is_deselected_unless_selected(tmp_path):
    """The marked test must not run in the ordinary ``pytest tests/`` -- no job
    but one has a producer checkout -- and must run, and FAIL without one, when
    selected. Driven through the real hook file in a throwaway session, not
    restated."""
    (tmp_path / "conftest.py").write_text((ROOT / "tests" / "ingest_producer_marker.py").read_text())
    (tmp_path / "test_probe.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.ingest_producer\n"
        "def test_live():\n"
        "    pytest.fail('could not look')\n\n"
        "def test_offline():\n"
        "    pass\n"
    )
    rc, summary = _run_pytest(tmp_path)
    assert rc == 0 and "1 passed" in summary and "1 deselected" in summary, summary
    rc, summary = _run_pytest(tmp_path, "-m", "ingest_producer")
    assert rc == 1 and "1 failed" in summary and "1 deselected" in summary, summary
    rc, summary = _run_pytest(tmp_path, "-m", "not ingest_producer")
    assert rc == 0 and "1 passed" in summary, summary


# ── the CI wiring: read back out of ci.yml, not restated ───────────────────

CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text()


def _job(name):
    """The text of one top-level job in ci.yml. No PyYAML: this suite runs in
    envs that do not install it."""
    lines = CI.splitlines()
    start = lines.index(f"  {name}:")
    body = []
    for line in lines[start + 1 :]:
        if line.startswith("  ") and not line.startswith("   ") and line.strip() and not line.lstrip().startswith("#"):
            break
        body.append(line)
    return "\n".join(body)


def _block(job_text, key):
    """The lines of a ``key: |`` literal block inside a job."""
    lines = job_text.splitlines()
    at = next(i for i, line in enumerate(lines) if line.strip() == f"{key}: |")
    indent = None
    out = []
    for line in lines[at + 1 :]:
        stripped = line.strip()
        this = len(line) - len(line.lstrip())
        if indent is None:
            indent = this
        if not stripped or this < indent:
            break
        out.append(stripped)
    return out


def test_the_fetch_checks_out_every_candidate_path_in_order():
    """The workflow's sparse checkout is the only list of candidates besides
    ``CANDIDATE_PATHS``. A path added to one and not the other is a candidate
    the gate says it tries and does not fetch."""
    got = _block(_job("fetch-ingest-schema"), "sparse-checkout")
    assert got == ["/" + p for p in gate.CANDIDATE_PATHS], got


def test_the_fetch_uploads_every_candidate_path_and_a_root_anchor():
    got = _block(_job("fetch-ingest-schema"), "path")
    assert got == ["_producer/PRODUCER_COMMIT"] + [f"_producer/{p}" for p in gate.CANDIDATE_PATHS], got


def test_the_gate_job_runs_when_the_fetch_fails_and_selects_the_marker():
    """A job skipped because a job it needs failed reports SKIPPED, which reads
    as passing -- the fail-open this job exists to avoid."""
    job = _job("category-contract")
    assert "needs: fetch-ingest-schema" in job
    assert "    if: ${{ !cancelled() }}" in job, "the job must run when the fetch failed"
    assert "pytest tests/test_zoo_category_contract.py -m ingest_producer" in job
    assert f"{PRODUCER_CHECKOUT_ENV}: _producer" in job


# ── control ────────────────────────────────────────────────────────────────


def test_the_instrument_can_disagree():
    """Every tree assertion above runs against a pairing that agrees today. If
    ``compare`` returned [] unconditionally, the producer test would still pass.
    This is the pair that cannot both hold -- synthetic, so it names the
    instrument and nothing else when it fails."""
    assert gate.compare(["a", "b"], ["b", "a"], "P", "Z") == []
    assert gate.compare(["a", "b"], ["a"], "P", "Z") != []
