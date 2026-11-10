"""Every vendored contract copy is registered, pinned, and held to its pin.

Offline, in ci.yml's required ``category-contract``: a copy names its producer,
the commit and source blobs it was derived from, and the digest of its contract
fields as ``tools/contract_copies.py --write`` wrote them. The comparison with
the producer's moving branches is ``.github/workflows/contract-drift.yml``'s,
required nowhere; the tests below hold that workflow to the registry, so a
source the registry adds cannot go unread there.
"""

from __future__ import annotations

import re
import subprocess

import contract_copies_fixtures as fx
import pytest
import test_vendored_contract_carries_no_internal_refs as refs
import workflow_text as wf

ROOT = fx.ROOT
tool = fx.load_tool()
CONTRACTS = ROOT / "tests" / "contracts"
DRIFT = ROOT / ".github" / "workflows" / "contract-drift.yml"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _listed() -> set[str]:
    return {
        p.relative_to(CONTRACTS).as_posix()
        for p in CONTRACTS.rglob("*")
        if p.is_file() and p.name != "README.md"
    }


# ── the committed copies ───────────────────────────────────────────────────


def test_every_copy_under_tests_contracts_is_registered():
    listed = _listed()
    assert listed, "no copies under tests/contracts -- nothing was checked"
    registered = {c.file for c in tool.COPIES} | set(tool.HELD_ELSEWHERE)
    assert listed - registered == set(), "a copy no tool holds"
    assert registered - listed == set(), "a registered copy that is not there"


def test_a_copy_held_elsewhere_names_a_tool_that_exists():
    for file, holder in tool.HELD_ELSEWHERE.items():
        assert (ROOT / holder).is_file(), f"{file} is held by {holder}, which does not exist"


@pytest.mark.parametrize("copy", tool.COPIES, ids=lambda c: c.file)
def test_every_copy_is_held_to_its_pin(copy):
    doc = tool.load_doc(CONTRACTS / copy.file)
    assert tool.pin_findings(doc, copy) == []


def test_an_absent_copy_is_a_refusal(tmp_path):
    with pytest.raises(tool.Refusal):
        tool.load_doc(tmp_path / "absent.json")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(tool.Refusal):
        tool.load_doc(tmp_path / "bad.json")


def test_a_copy_without_its_contract_field_is_a_refusal():
    with pytest.raises(tool.Refusal):
        tool.pin_findings({"generated_from": {}}, tool.COPIES[0])


def test_blob_sha_is_gits(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"some bytes\n")
    git = subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True, check=True)
    assert tool.blob_sha(path.read_bytes()) == git.stdout.strip()


# ── the derivations ────────────────────────────────────────────────────────


def test_each_derivation_reads_the_producers_shape():
    src = {r: {p: t.encode() for p, t in f.items()} for r, f in fx.sources().items()}
    by = {c.file: c.derive({p: src[c.repo][p] for p in c.paths}) for c in tool.COPIES}
    assert by["tracebloc_backend/model_type_choices.v1.json"] == {
        "choices": [
            {"constant": "MODEL_TYPE_ALPHA", "value": "alpha", "label": "Alpha Model"},
            {"constant": "MODEL_TYPE_BETA", "value": "beta", "label": "Beta Model"},
        ],
        "accepted_model_type_values": ["alpha", "beta"],
    }
    assert by["tracebloc_backend/upload_import_allowlist.v1.json"] == {"allowed_top_level_packages": ["math", "torch"]}
    assert by["tracebloc_engine/object_detection_families.v2.json"] == {
        "version": 2,
        "families": [{"family": "alpha", "aliases": ["a"], "training_supported": True, "handler": "h.alpha"}],
        "accepted_model_type_values": ["alpha"],
        "not_accepted": {"examples": ["gamma"]},
    }
    assert by["tracebloc_sdk/sklearn_model_types.v1.json"] == {
        "accepted_by_family": {"classifier": ["linear", "naive", "tree"], "regression": ["linear", "svm"]}
    }


@pytest.mark.parametrize("copy", tool.COPIES, ids=lambda c: c.file)
@pytest.mark.parametrize("body", [b"", b"x = 1\n", b"{}", b"\xff"], ids=["empty", "no-symbol", "empty-json", "binary"])
def test_a_source_that_is_not_the_producers_is_a_refusal(copy, body):
    with pytest.raises(tool.Refusal):
        copy.derive({p: body for p in copy.paths})


def test_a_duplicated_value_is_a_refusal():
    dup = 'ALLOWED_PYTHON_PACKAGE_IMPORTS = ("torch", "torch")\n'
    with pytest.raises(tool.Refusal):
        tool.derive_upload_imports({"x.py": dup.encode()})


@pytest.mark.parametrize("literal", ['"torch"', '{"torch": 1}', "1"])
def test_an_allowlist_that_is_not_a_sequence_is_a_refusal(literal):
    with pytest.raises(tool.Refusal):
        tool.derive_upload_imports({"x.py": f"ALLOWED_PYTHON_PACKAGE_IMPORTS = {literal}\n".encode()})


def test_a_model_type_compared_either_way_round_is_derived():
    src = {
        "skl_tabular_base.py": b"""\
class B:
    _ensemble_prefixes = {}

    def average_estimators(self, models, model_type, ensemble_cls):
        if "linear" == model_type or self.model_type in ("ridge",):
            return 1
""",
    }
    assert tool.derive_sklearn_types(src)["accepted_by_family"] == {
        "classifier": ["linear", "ridge"],
        "regression": ["linear", "ridge"],
    }


def test_the_scrub_is_the_public_copy_tests_pattern():
    assert tool.ISSUE_REF.pattern == refs.ISSUE_REF.pattern


def test_a_partial_tree_still_reads_the_rest_with_a_token(monkeypatch, tmp_path):
    trees = fx.producer_tree(tmp_path / "p")
    seen = []
    monkeypatch.setattr(tool, "_token", lambda: "t0ken")
    monkeypatch.setattr(tool, "resolve", lambda repo, ref, token: seen.append(token) or (_ for _ in ()).throw(tool.Refusal("stop")))
    assert tool.run(["--tree", f"backend={trees['backend']}"]) == 2
    assert seen == ["t0ken"]


# ── the comparison, end to end, on a producer checkout ─────────────────────


def test_copies_derived_from_a_tree_agree_with_it(tmp_path):
    trees = fx.producer_tree(tmp_path / "p")
    copies = fx.rendered_copies(tool, trees, tmp_path / "c")
    assert tool.run(["--copies", str(copies), *fx.tree_args(trees)]) == 0


def test_a_producer_change_is_drift_naming_the_copy_and_field_only(tmp_path, capsys):
    trees = fx.producer_tree(tmp_path / "p")
    copies = fx.rendered_copies(tool, trees, tmp_path / "c")
    imports = trees["backend"] / "bandit_tracebloc_model_validation/bandit_plugins/import_names.py"
    imports.write_text('ALLOWED_PYTHON_PACKAGE_IMPORTS = ("torch", "math", "zz_unpublished")\n')
    assert tool.run(["--copies", str(copies), *fx.tree_args(trees)]) == 1
    err = capsys.readouterr().err
    assert "DRIFT: tracebloc_backend/upload_import_allowlist.v1.json: allowed_top_level_packages" in err
    assert "zz_unpublished" not in err, "a producer value reached the log"


def test_an_absent_source_is_a_refusal_not_a_verdict(tmp_path):
    trees = fx.producer_tree(tmp_path / "p")
    copies = fx.rendered_copies(tool, trees, tmp_path / "c")
    (trees["tracebloc-engine"] / "core/schema/object_detection_families.v2.json").unlink()
    assert tool.run(["--copies", str(copies), *fx.tree_args(trees)]) == 2


def test_write_scrubs_a_tracker_reference_from_an_upstream_note(tmp_path):
    trees = fx.producer_tree(tmp_path / "p")
    copies = fx.rendered_copies(tool, trees, tmp_path / "c")
    text = (copies / "tracebloc_engine/object_detection_families.v2.json").read_text()
    assert "#12" not in text and tool.PLACEHOLDER in text


@pytest.mark.parametrize("argv", [["--write"], ["--write", "--ref", "v1.2.3"], ["--write", "--ref", "develop", "--tree", "backend=/x"]])
def test_write_takes_a_producer_branch_over_the_network_only(argv):
    assert tool.run(argv) == 2


# ── where the checks run ───────────────────────────────────────────────────


def test_the_required_check_runs_this_file_whatever_the_step_before_it_did():
    job = wf.job_block(CI.read_text(encoding="utf-8"), "category-contract")
    step = next((s for s in wf.steps(job) if "pytest tests/test_contract_copies_held.py" in s), None)
    assert step is not None, "category-contract does not run tests/test_contract_copies_held.py"
    assert "if: ${{ !cancelled() }}" in step


def _drift() -> str:
    return wf.strip_comments(DRIFT.read_text(encoding="utf-8"))


def test_the_drift_run_pairs_each_branch_with_the_producers_same_branch():
    text = _drift()
    assert re.search(r"branch: \[(.*)\]", text).group(1).split(", ") == list(tool.BRANCHES)
    assert 'git archive "origin/$BRANCH" tests/contracts' in text
    assert re.search(r"(?m)^  (pull_request|pull_request_target|push):", text) is None


@pytest.mark.parametrize("repo", sorted({c.repo for c in tool.COPIES}))
def test_the_drift_run_checks_out_every_source_of_every_copy(repo):
    steps = [s for s in wf.steps(wf.job_block(_drift(), "drift")) if f"/{repo}\n" in s and "repository:" in s]
    assert len(steps) == 1, f"no single checkout of {repo}"
    step = steps[0]
    assert "ref: ${{ matrix.branch }}" in step
    assert f"path: _producer/{repo}" in step
    assert "persist-credentials: false" in step
    for copy in (c for c in tool.COPIES if c.repo == repo):
        for p in copy.paths:
            assert f"            /{p}\n" in step, f"{copy.file}'s source {p} is not checked out"
    assert f"--tree {repo}=_producer/{repo}" in _drift()
    token = next(s for s in wf.steps(wf.job_block(_drift(), "drift")) if "create-github-app-token" in s)
    assert f"            {repo}\n" in token, f"the token cannot read {repo}"


def test_the_drift_log_never_asks_for_producer_values():
    assert "--show" not in _drift()
    assert wf.publishers(wf.job_block(_drift(), "drift")) == []
