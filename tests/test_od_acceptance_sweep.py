"""Guard the guards in ``tools/od_acceptance_sweep.py`` (backend#3048).

The sweep itself is NOT a test — model-zoo CI's torch job is
``timeout-minutes: 30`` on a 2-core runner and the sweep is measured in hours
(one 20-step cycle x3 experiments over the roster is ~141 minutes, 86 of which
belong to the two convnext templates). So the sweep runs on demand and THIS
file is what runs in CI: every assertion the sweep makes, fired at an input
built to break it.

WHY THAT IS THE WHOLE POINT. #3048 exists because this epic keeps producing
false greens — most memorably an audit that scored ``mask_rcnn`` "uploadable"
when it could not be uploaded at all. A sweep that reports 25 greens is worth
nothing unless its checks can be shown going red, and shown on every run rather
than once when they were written. That is the convention
``harness/od/scenario.py`` established with its pair-checks and the one
``test_check_dump_coverage.py`` uses here: a probe that decayed into a constant
fails the next time anyone runs CI, not the next time someone re-reads it.

Every check below therefore comes with BOTH directions — an input it must
refuse and an input it must accept. A one-sided test passes just as happily
against ``return []``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).parent.parent
TOOLS = ROOT / "tools"

sys.path.insert(0, str(TOOLS))

sweep_mod = importlib.import_module("od_acceptance_sweep")


def _contract_test_module():
    """``tests/test_od_torchvision_family_train_step.py``, imported by path.

    Imported rather than duplicated because the next test asserts the sweep's
    roster EQUALS that file's roster. The two readers are independent
    implementations on purpose (a tool has to run standalone), and this is what
    stops them drifting into two different answers about what the roster is.
    """
    path = ROOT / "tests" / "test_od_torchvision_family_train_step.py"
    spec = importlib.util.spec_from_file_location("od_contract_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------


def test_the_sweep_roster_is_the_contract_tests_roster() -> None:
    """The sweep covers exactly the family the single-step contract test covers.

    If these two ever disagree, one of them is sweeping a template the other
    thinks is out of family — and the disagreement would be invisible, because
    each file's own scan would still look self-consistent.
    """
    contract = _contract_test_module()
    assert sweep_mod.FAMILY == contract.FAMILY
    assert sweep_mod.family_values() == contract.FAMILY_VALUES
    mine = {p.resolve() for p in sweep_mod.family_templates()}
    theirs = {p.resolve() for p in contract.FAMILY_TEMPLATES}
    assert mine == theirs, (
        f"sweep-only: {sorted(p.name for p in mine - theirs)}; "
        f"contract-test-only: {sorted(p.name for p in theirs - mine)}"
    )


def test_the_roster_is_not_empty_and_the_alias_is_in_play() -> None:
    """Guard the guard: an empty scan makes the sweep report success having
    checked nothing — the silent-green shape of backend#1859.

    The alias clause is not decoration. ``rcnn`` is a live legacy value and
    ``faster_rcnn_resnet.py`` declares it, so selecting on the literal string
    ``"torchvision_detection"`` drops that template silently.
    """
    assert "rcnn" in sweep_mod.family_values()
    assert sweep_mod.family_templates()
    assert sweep_mod.od_templates()


def test_uncovered_templates_are_reported_and_are_the_yolo_family() -> None:
    """The 3/28 this sweep cannot cover must be enumerated, never dropped.

    #3048 forbids a silent cap, so "25 of 25 pass" has to be impossible to
    write: the uncovered set is derived from the same partition as the covered
    set, so it cannot be empty while the yolo templates exist.
    """
    covered = {p.resolve() for p in sweep_mod.family_templates()}
    uncovered = {p.resolve() for p in sweep_mod.uncovered_templates()}
    everything = {p.resolve() for p in sweep_mod.od_templates()}
    assert covered | uncovered == everything
    assert not covered & uncovered
    assert uncovered, "no OD template routes outside the family — model_type reading broke"
    for path in uncovered:
        assert sweep_mod.read_model_type(path).strip().lower() == "yolo"


# ---------------------------------------------------------------------------
# The loss-dict assertion, both directions
# ---------------------------------------------------------------------------


def test_train_step_findings_accepts_a_valid_loss_dict() -> None:
    torch = pytest.importorskip("torch")
    good = {"classification": torch.tensor(1.5), "bbox_regression": torch.tensor(0.25)}
    assert sweep_mod.train_step_findings(torch, good, 0) == []


@pytest.mark.parametrize(
    "payload, expected_substring",
    [
        pytest.param([], "not a dict", id="not-a-dict"),
        pytest.param({}, "EMPTY loss dict", id="empty-dict"),
        pytest.param({"cls": 1.5}, "not a tensor", id="python-float"),
        pytest.param(None, "not a dict", id="none"),
    ],
)
def test_train_step_findings_refuses_a_malformed_loss_dict(payload, expected_substring) -> None:
    torch = pytest.importorskip("torch")
    findings = sweep_mod.train_step_findings(torch, payload, 3)
    assert findings, f"{payload!r} was accepted"
    assert any(expected_substring in f for f in findings), findings


def test_train_step_findings_refuses_a_non_scalar_loss() -> None:
    """The handler calls ``sum(losses.values())``, so a vector loss is not a
    loss — it sums to a tensor that ``.backward()`` refuses."""
    torch = pytest.importorskip("torch")
    findings = sweep_mod.train_step_findings(
        torch, {"cls": torch.tensor([1.0, 2.0])}, 0
    )
    assert any("not a scalar" in f for f in findings), findings


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_train_step_findings_refuses_a_non_finite_loss(bad) -> None:
    """The half of #3048's train criterion that is achievable today.

    All three, not just NaN: an overflowing loss diverges to +/-Inf before it
    becomes NaN, and a check written only against NaN misses the step where it
    actually went wrong.
    """
    torch = pytest.importorskip("torch")
    findings = sweep_mod.train_step_findings(
        torch, {"cls": torch.tensor(float(bad))}, 7
    )
    assert any("not finite" in f for f in findings), findings
    assert any("step 7" in f for f in findings), "the failing step is not named"


# ---------------------------------------------------------------------------
# The inference-payload assertion, both directions
# ---------------------------------------------------------------------------


def _payload(torch, n_boxes: int) -> dict:
    """A well-formed prediction with ``n_boxes`` boxes, derived not transcribed:
    ``scores`` and ``labels`` are sized FROM ``boxes`` so a valid fixture cannot
    accidentally be a misaligned one."""
    boxes = torch.tensor([[1.0, 2.0, 30.0, 40.0]]).repeat(n_boxes, 1)
    return {
        "boxes": boxes,
        "scores": torch.rand(boxes.shape[0]),
        "labels": torch.ones(boxes.shape[0], dtype=torch.int64),
    }


def test_payload_findings_accepts_a_well_formed_payload() -> None:
    torch = pytest.importorskip("torch")
    preds = [_payload(torch, 3), _payload(torch, 1)]
    assert sweep_mod.payload_findings(torch, preds, 2) == []


def test_an_image_the_detector_finds_nothing_on_is_a_PASS() -> None:
    """#3048's exact wording is that an empty detection must "neither crash nor
    silently drop the record" — it does not require a detection.

    This is the positive control that keeps the sweep honest in the direction it
    would be tempting to cheat. 11 of 25 templates return zero boxes on both
    fixture images from random init (measured on develop, 2026-09-04); a check
    that required a non-empty payload would fail all 11 for being untrained,
    which is a QUALITY question and belongs in the other column. Without this
    test, someone tightening the payload check "to make it stronger" would
    silently convert 11 quality-pendings into mechanical failures.
    """
    torch = pytest.importorskip("torch")
    empty_both = [_payload(torch, 0), _payload(torch, 0)]
    assert sweep_mod.payload_findings(torch, empty_both, 2) == []


def test_payload_findings_refuses_a_dropped_record() -> None:
    """The other half of #3048's empty-image clause, and the one an eval loop
    that filters empties would trip: one entry PER IMAGE, always."""
    torch = pytest.importorskip("torch")
    findings = sweep_mod.payload_findings(torch, [_payload(torch, 2)], 2)
    assert any("DROPPED" in f for f in findings), findings


def test_payload_findings_refuses_a_non_list() -> None:
    torch = pytest.importorskip("torch")
    findings = sweep_mod.payload_findings(torch, _payload(torch, 1), 1)
    assert any("not a list" in f for f in findings), findings


@pytest.mark.parametrize("key", ["boxes", "scores", "labels"])
def test_payload_findings_refuses_a_missing_key(key) -> None:
    torch = pytest.importorskip("torch")
    pred = _payload(torch, 2)
    del pred[key]
    findings = sweep_mod.payload_findings(torch, [pred], 1)
    assert any(key in f and "missing" in f for f in findings), findings


@pytest.mark.parametrize("key", ["scores", "labels"])
def test_payload_findings_refuses_scores_or_labels_misaligned_with_boxes(key) -> None:
    """#3048 asks for ``scores``/``labels`` ALIGNED. A payload whose scores are
    one short is not merely malformed — the metrics zip them against boxes, so
    every box after the gap is scored with the wrong confidence."""
    torch = pytest.importorskip("torch")
    pred = _payload(torch, 4)
    pred[key] = pred[key][:3]
    findings = sweep_mod.payload_findings(torch, [pred], 1)
    assert any("not aligned" in f for f in findings), findings


def test_payload_findings_refuses_boxes_of_the_wrong_shape() -> None:
    torch = pytest.importorskip("torch")
    pred = _payload(torch, 2)
    pred["boxes"] = torch.rand(2, 5)
    findings = sweep_mod.payload_findings(torch, [pred], 1)
    assert any("expected (N, 4)" in f for f in findings), findings


def test_payload_findings_refuses_boxes_that_are_not_xyxy() -> None:
    """Pixel xyxy is #3048's wording and the metrics' assumption. A cxcywh or
    xywh payload passes every shape check and scores near zero for a reason
    nobody would find — so the ORDERING is asserted, not just the width."""
    torch = pytest.importorskip("torch")
    pred = _payload(torch, 1)
    pred["boxes"] = torch.tensor([[30.0, 40.0, 1.0, 2.0]])  # x2 < x1, y2 < y1
    findings = sweep_mod.payload_findings(torch, [pred], 1)
    assert any("not xyxy" in f for f in findings), findings


# ---------------------------------------------------------------------------
# The gradient-reachability assertion — the sweep's functional check
# ---------------------------------------------------------------------------


def _stub_detector(torch, *, bypass_norm: bool):
    """A minimal duck-typed ``torchvision_detection`` model.

    ``bypass_norm=True`` builds the defect this check exists for: a submodule
    left CONSTRUCTED but dropped out of ``forward()``. That mutation is
    invisible to parameter count, ``state_dict`` keys, tensor shapes and loss
    keys — every property the single-step contract test asserts. Verified
    against a real template as well as this stub: mutating ``yolox_s.py``'s
    ``Conv.forward`` from ``self.act(self.norm(self.conv(x)))`` to
    ``self.act(self.conv(x))`` left ``test_family_template_trains_and_evals
    [yolox_s.py]`` passing while 148 of 240 parameters went unreachable.

    A stub rather than that template here because CI should not build a detector
    to assert a property of autograd.
    """
    nn = torch.nn

    class Stub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.norm = nn.GroupNorm(2, 4)
            self.head = nn.Conv2d(4, 1, 1)
            self._bypass = bypass_norm

        def forward(self, images, targets=None):
            batch = torch.stack(list(images))
            feature = self.conv(batch)
            if not self._bypass:
                feature = self.norm(feature)
            logits = self.head(feature)
            if self.training:
                return {"classification": logits.abs().mean()}
            return [
                {
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }
                for _ in images
            ]

    return Stub()


def _stub_inputs(torch):
    images = [torch.rand(3, 8, 8), torch.rand(3, 8, 8)]
    targets = [
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}
        for _ in images
    ]
    return images, targets


def test_the_bypassed_module_mutation_is_invisible_to_the_older_checks() -> None:
    """Establish that the gradient check is not redundant.

    Parameter count, ``state_dict`` keys and the loss keys are identical across
    the mutation, so the properties the existing contract test asserts cannot
    tell these two models apart. If that ever stops being true this test fails
    and the gradient check may be removable — which is the honest way to find
    out, rather than keeping a guard nobody can justify.
    """
    torch = pytest.importorskip("torch")
    intact = _stub_detector(torch, bypass_norm=False)
    broken = _stub_detector(torch, bypass_norm=True)
    images, targets = _stub_inputs(torch)

    assert sum(p.numel() for p in intact.parameters()) == sum(
        p.numel() for p in broken.parameters()
    )
    assert set(intact.state_dict()) == set(broken.state_dict())
    intact.train(), broken.train()
    assert set(intact(images, targets)) == set(broken(images, targets))


def test_gradient_reachability_finds_a_constructed_but_bypassed_module() -> None:
    torch = pytest.importorskip("torch")
    model = _stub_detector(torch, bypass_norm=True)
    optimizer = sweep_mod.make_optimizer(torch, model)
    images, targets = _stub_inputs(torch)

    n_trainable, no_grad, _ = sweep_mod.measure_gradient_reachability(
        torch, model, optimizer, images, targets
    )
    assert n_trainable
    assert sorted(no_grad) == ["norm.bias", "norm.weight"], no_grad
    assert sweep_mod.gradient_findings(no_grad), "the finding was not raised"


def test_gradient_reachability_accepts_an_intact_model() -> None:
    """The other direction. Without this, the check could report every model
    broken and the test above would still pass."""
    torch = pytest.importorskip("torch")
    model = _stub_detector(torch, bypass_norm=False)
    optimizer = sweep_mod.make_optimizer(torch, model)
    images, targets = _stub_inputs(torch)

    _, no_grad, _ = sweep_mod.measure_gradient_reachability(
        torch, model, optimizer, images, targets
    )
    assert no_grad == []
    assert sweep_mod.gradient_findings(no_grad) == []


def test_gradient_reachability_is_not_fooled_by_accumulated_gradients() -> None:
    """The trap that makes or breaks this check, asserted directly.

    Gradients ACCUMULATE. Measured naively after a multi-step cycle,
    ``p.grad is not None`` is satisfied by a parameter that received a gradient
    on step 1 and has been unreachable ever since — so the check would pass on
    precisely the defect it exists to catch, and it would pass on every template.

    Here the module is bypassed only AFTER a real backward has populated
    ``norm``'s gradients. A ``measure_gradient_reachability`` that did not clear
    to None first would report zero unreachable parameters.
    """
    torch = pytest.importorskip("torch")
    model = _stub_detector(torch, bypass_norm=False)
    optimizer = sweep_mod.make_optimizer(torch, model)
    images, targets = _stub_inputs(torch)

    model.train()
    model(images, targets)["classification"].backward()
    assert model.norm.weight.grad is not None, "precondition: grads are populated"

    model._bypass = True
    _, no_grad, _ = sweep_mod.measure_gradient_reachability(
        torch, model, optimizer, images, targets
    )
    assert sorted(no_grad) == ["norm.bias", "norm.weight"], (
        f"stale gradients masked the unreachable parameters: {no_grad} — "
        f"zero_grad(set_to_none=True) is what makes this check non-vacuous"
    )


def test_zero_grad_without_set_to_none_would_make_the_check_vacuous() -> None:
    """Why the ``set_to_none=True`` is spelled out rather than left default.

    Plain ``zero_grad()`` leaves zero TENSORS behind, and a zero tensor is
    ``not None``. This pins the library behaviour the sweep's choice depends on,
    so a torch release that changed the default cannot quietly neuter the check.
    """
    torch = pytest.importorskip("torch")
    model = _stub_detector(torch, bypass_norm=False)
    optimizer = sweep_mod.make_optimizer(torch, model)
    images, targets = _stub_inputs(torch)

    model.train()
    model(images, targets)["classification"].backward()

    optimizer.zero_grad(set_to_none=False)
    assert model.norm.weight.grad is not None, (
        "zero_grad(set_to_none=False) no longer leaves a zero tensor; the sweep's "
        "reliance on set_to_none=True can be revisited"
    )
    optimizer.zero_grad(set_to_none=True)
    assert model.norm.weight.grad is None


def test_zero_grad_parameters_are_data_and_not_a_failure() -> None:
    """``n_zero_grad`` is reported, never gated on.

    Measured baseline on develop: 9 of 25 templates have trainable parameters
    whose gradient is present but exactly zero — unassigned FPN levels and
    per-level scales on a two-image batch. Gating on non-zero gradients would
    be red on 9 templates for a legitimate reason, so the prose says
    "reachable from the loss through the autograd graph", which is what
    ``grad is not None`` actually tests, and nothing stronger.
    """
    torch = pytest.importorskip("torch")

    class ZeroGrad(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = torch.nn.Parameter(torch.ones(2))
            # Reachable from the loss, but multiplied by zero — grad is a zero
            # tensor, not None.
            self.scaled = torch.nn.Parameter(torch.ones(2))

        def forward(self, images, targets=None):
            return {"loss": (self.used + self.scaled * 0.0).sum()}

    model = ZeroGrad()
    optimizer = sweep_mod.make_optimizer(torch, model)
    _, no_grad, zero_grad = sweep_mod.measure_gradient_reachability(
        torch, model, optimizer, [], None
    )
    assert no_grad == [], "a zero gradient is not an absent gradient"
    assert zero_grad == ["scaled"]
    assert sweep_mod.gradient_findings(no_grad) == [], (
        "a zero-gradient parameter was reported as a mechanical failure"
    )


# ---------------------------------------------------------------------------
# The optimizer contract
# ---------------------------------------------------------------------------


def test_the_sweep_builds_the_engine_s_optimizer_and_no_other() -> None:
    """No momentum, and lr from the engine's default.

    This is the check that stops the sweep measuring itself. A first draft used
    ``momentum=0.9`` and ``centernet_resnet`` went NaN in 12 steps; under the
    engine's actual construction it does not. The failure that mattered was in
    the harness, and it was reported as a template defect until this was pinned.
    """
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    optimizer = sweep_mod.make_optimizer(torch, model)

    assert isinstance(optimizer, torch.optim.SGD)
    assert sweep_mod.ENGINE_OPTIMIZER_KWARGS == {}, (
        "the engine passes lr and nothing else; adding a kwarg here changes what "
        "every reported number means"
    )
    group = optimizer.param_groups[0]
    assert group["lr"] == sweep_mod.ENGINE_DEFAULT_LR == 0.001
    assert group["momentum"] == 0
    assert group["weight_decay"] == 0


def test_the_engine_hyperparameter_check_fires_on_a_drifted_adapter(tmp_path) -> None:
    """The ``--engine`` guard, shown going red — and shown going green.

    The faithful fixture is BUILT FROM the sweep's own declared constants rather
    than hand-typed, so this cannot pass by transcribing a literal that has
    since moved.
    """
    adapter = tmp_path / sweep_mod.ENGINE_ADAPTER_PATH
    adapter.parent.mkdir(parents=True)
    faithful = (
        f'        optimizers = {{"{sweep_mod.ENGINE_OPTIMIZER_NAME}": torch.optim.SGD}}\n'
        f"        initial_learning_rate = {sweep_mod.ENGINE_DEFAULT_LR}\n"
        "        return optimizer_class(model.parameters(), lr=learning_rate)\n"
    )
    adapter.write_text(faithful, encoding="utf-8")
    assert sweep_mod.verify_engine_hyperparameters(tmp_path) == []

    adapter.write_text(
        faithful.replace(
            "lr=learning_rate)", "lr=learning_rate, momentum=0.9)"
        ),
        encoding="utf-8",
    )
    problems = sweep_mod.verify_engine_hyperparameters(tmp_path)
    assert any("momentum" in p for p in problems), problems

    adapter.write_text(faithful.replace("= 0.001", "= 0.05"), encoding="utf-8")
    assert any("default lr" in p for p in problems + sweep_mod.verify_engine_hyperparameters(tmp_path))

    assert sweep_mod.verify_engine_hyperparameters(tmp_path / "nope"), (
        "a missing adapter must be a problem, not a silent pass"
    )


# ---------------------------------------------------------------------------
# The report's columns
# ---------------------------------------------------------------------------


def test_the_quality_cause_is_derived_from_the_payload_not_a_template_list() -> None:
    """The cause column must track reality, not 2026-09-04's snapshot.

    backend#3093 is fixing from-scratch normalisation
    (``norm_layer=FrozenBatchNorm2d`` is a bit-exact identity on a
    ``weights=None`` build), and when it lands some of the 11 zero-box templates
    are expected to start emitting boxes. A hardcoded list would then report the
    wrong cause with total confidence. So the same inputs must produce different
    causes purely from the payload.
    """
    assert sweep_mod._quality_cause([0, 0], seeded=False) == sweep_mod.CAUSE_EMPTY_PAYLOAD
    assert sweep_mod._quality_cause([3, 0], seeded=False) == sweep_mod.CAUSE_RANDOM_SCORES
    assert sweep_mod._quality_cause([], seeded=False) == sweep_mod.CAUSE_EMPTY_PAYLOAD
    assert sweep_mod._quality_cause([0, 0], seeded=True) == sweep_mod.CAUSE_MEASURABLE
    assert (
        sweep_mod.CAUSE_EMPTY_PAYLOAD
        != sweep_mod.CAUSE_RANDOM_SCORES
        != sweep_mod.CAUSE_MEASURABLE
    )


def test_the_seeding_column_covers_the_roster_and_reports_nothing_seeded() -> None:
    """Derived by calling ``check_dump_coverage.survey()``, not transcribed.

    And the #2659 fact stated as a property: no OD seed is hosted, the sweep
    loads no weights, so every row is "scratch". A template that declares a
    seed is a DIFFERENT row from one that is random-init by design — #3055 asks
    #3048 to say which, so the two phrasings must stay distinguishable.
    """
    index = sweep_mod.seeding_index()
    stems = {p.stem for p in sweep_mod.family_templates()}
    missing = stems - set(index)
    assert not missing, f"templates absent from the dump survey: {sorted(missing)}"
    assert all(v.startswith("scratch") for v in index.values()), index
    declared = {k for k, v in index.items() if "seed declared" in v}
    by_design = {k for k, v in index.items() if "by design" in v}
    assert declared and by_design, (
        f"the two seeding states collapsed into one phrasing: {sorted(set(index.values()))}"
    )
    assert not declared & by_design


def test_the_markdown_report_is_per_template_and_names_what_ran() -> None:
    """#3048: "All models pass" with no table is the shape of every false-green
    this epic has produced. So the renderer is asserted to emit a ROW PER
    TEMPLATE and to carry the uncovered block."""
    report = {
        "ticket": "tracebloc/backend#3048",
        "scope": "local-only",
        "family": "torchvision_detection",
        "steps": 8,
        "experiments": 2,
        "optimizer": {
            "name": "sgd",
            "lr": 0.001,
            "extra_kwargs": {},
            "derived_from": "engine",
        },
        "roster": {
            "od_templates_total": 28,
            "covered_by_this_sweep": 25,
            "run": 2,
            "skipped": [{"template": "slow_one", "reason": "--skip-slow"}],
            "uncovered": [
                {"template": "yolo_v1/model.py", "model_type": "yolo", "reason": "other contract"}
            ],
        },
        "templates": [
            {
                "template": "green_one",
                "seeding": "scratch (random-init by design)",
                "status": "PASS",
                "quality": "pending",
                "quality_cause": "empty-payload",
                "cycles_run": 16,
                "loss_first": 2.0,
                "loss_last": 1.0,
                "n_zero_grad": 3,
                "n_preds": [0, 0],
                "observed_input_shape": [2, 3, 800, 800],
                "declared_image_size": 448,
                "findings": [],
                "divergence": [],
            },
            {
                "template": "red_one",
                "seeding": "scratch (seed declared, not hosted -- backend#2659)",
                "status": "FAIL",
                "quality": "pending",
                "quality_cause": "random-init-scores",
                "cycles_run": 0,
                "n_preds": [4],
                "declared_image_size": 640,
                "findings": ["step 0: loss 'cls' is nan, not finite"],
                "divergence": [],
            },
            {
                "template": "divergent_one",
                "seeding": "scratch (random-init by design)",
                "status": "DIVERGENT",
                "quality": "pending",
                "quality_cause": "empty-payload",
                "cycles_run": 6,
                "loss_first": 101.0,
                "loss_last": 1.8e17,
                "n_zero_grad": 5,
                "n_preds": [0, 0],
                "declared_image_size": 300,
                "findings": [],
                "divergence": ["loss DIVERGED: 101.0 -> 1.8e+17, 1.78e+15x its first step"],
            },
        ],
    }
    table = sweep_mod.markdown(report)

    assert "`green_one`" in table and "`red_one`" in table
    assert "**FAIL**" in table, "a red template is not marked as one"
    assert "**DIVERGENT**" in table, "a diverging template is not marked as one"
    assert "DIVERGED" in table, "the divergence note is not carried into the table"
    assert "not finite" in table, "#3048 requires the failure WITH its cause"
    assert "yolo_v1/model.py" in table, "the uncovered templates are not named"
    assert "slow_one" in table, "a skipped template is not named"
    assert "empty-payload" in table and "random-init-scores" in table
    assert "800x800 (declared 448)" in table, (
        "the resolution the network actually saw is not recorded — a cost or metric "
        "figure at an unstated resolution is unreadable (backend#3058)"
    )
    # THREE COUNTS, not a ratio: a "N/M pass" line is exactly what would let the
    # top line read "2/3 mechanical" while one of them is diverging.
    assert "1 pass, 1 divergent, 1 fail (of 3)" in table, table.splitlines()[8]
    # The report must never claim quality it does not have.
    assert "quality: 0/3 measurable" in table
    assert "Only the 1 `PASS` rows meet" in table, (
        "the report does not say that DIVERGENT fails the exit criterion"
    )


def test_the_slow_skip_list_is_opt_in_and_named() -> None:
    """A cost cap that is not in the report is a silent cap, which #3048
    forbids. ``SLOW_TEMPLATES`` must be real templates, so the list cannot rot
    into a set of names that skip nothing."""
    keys = {sweep_mod.template_key(p) for p in sweep_mod.family_templates()}
    assert sweep_mod.SLOW_TEMPLATES
    unknown = sweep_mod.SLOW_TEMPLATES - keys
    assert not unknown, f"--skip-slow names templates that do not exist: {sorted(unknown)}"


# ---------------------------------------------------------------------------
# The divergence state (backend#3048's third mechanical status)
# ---------------------------------------------------------------------------


def test_a_loss_that_grows_101x_is_divergent_and_99x_is_not() -> None:
    """The decision boundary, from both sides, on CONSTRUCTED inputs.

    Deliberately not a roster row that happens to sit near the line: a template
    drifting past a threshold would silently turn this into a test of that
    template rather than of the threshold.
    """
    assert sweep_mod.divergence_findings(1.0, 101.0, steps_run=3)
    assert sweep_mod.divergence_findings(1.0, 99.0, steps_run=3) == []

    assert sweep_mod.classify_status([], sweep_mod.divergence_findings(1.0, 101.0, 3)) == (
        sweep_mod.STATUS_DIVERGENT
    )
    assert sweep_mod.classify_status([], sweep_mod.divergence_findings(1.0, 99.0, 3)) == (
        sweep_mod.STATUS_PASS
    )


def test_the_divergence_boundary_is_strict() -> None:
    """``> DIVERGENCE_FACTOR``, exactly as the prose says.

    Pinned because "more than 100x" and ">= 100x" differ by one row and the
    difference is invisible unless someone asserts it.
    """
    factor = sweep_mod.DIVERGENCE_FACTOR
    assert sweep_mod.divergence_findings(1.0, factor, steps_run=3) == []
    assert sweep_mod.divergence_findings(1.0, factor * 1.000001, steps_run=3)


def test_the_real_ssd_vgg16_numbers_are_divergent_and_the_real_passers_are_not() -> None:
    """The observed values this state exists for, and its nearest true negative.

    Measured on develop 2026-09-04 under the engine's own optimizer at declared
    resolution. ``vfnet_resnet`` at 1.13x is the widest genuine RISE among
    passing templates, so this also records that nothing legitimate sits within
    two orders of magnitude of the line.
    """
    assert sweep_mod.divergence_findings(101.6198, 1.8000589330631885e17, steps_run=3)
    assert sweep_mod.divergence_findings(2.3879, 2.6922, steps_run=3) == []
    assert sweep_mod.divergence_findings(825.6479, 4.794597911645026e29, steps_run=3)


def test_fail_takes_precedence_over_divergent() -> None:
    """A NaN loss AND a grown loss is a FAIL, never softened to DIVERGENT."""
    findings = ["step 2: loss 'heatmap' is nan, not finite"]
    divergence = sweep_mod.divergence_findings(1.0, 1e30, steps_run=3)
    assert divergence
    assert sweep_mod.classify_status(findings, divergence) == sweep_mod.STATUS_FAIL
    assert sweep_mod.classify_status(findings, []) == sweep_mod.STATUS_FAIL


def test_a_one_step_run_is_not_a_clean_pass() -> None:
    """``--steps 1`` makes the comparison vacuous, so it must not read as passed.

    first and last are the same measurement, so a naive ``last > 100 * first``
    reports every one-step run as non-divergent — a green on a check that never
    ran, which is the false-green shape this whole file exists to prevent.
    """
    for steps in (0, 1):
        notes = sweep_mod.divergence_findings(5.0, 5.0, steps_run=steps)
        assert notes, f"steps_run={steps} was treated as a real measurement"
        assert "UNCHECKED" in notes[0]
        assert sweep_mod.classify_status([], notes) == sweep_mod.STATUS_DIVERGENT
    # Two steps is enough to compare, and a flat loss is not divergent.
    assert sweep_mod.divergence_findings(5.0, 5.0, steps_run=2) == []


def test_a_non_positive_first_step_loss_is_undecidable_not_passed() -> None:
    """The defensive branch, fired rather than left unexercised.

    ``100 * first`` is zero or negative for a non-positive baseline, so the test
    would fire on everything or nothing. These losses are sums of non-negative
    terms, so it is unreachable on today's roster — which is exactly why it needs
    a test rather than a comment.
    """
    for first in (0.0, -3.5):
        notes = sweep_mod.divergence_findings(first, 10.0, steps_run=3)
        assert notes, f"first={first} silently passed"
        assert "UNDECIDABLE" in notes[0]
        assert sweep_mod.classify_status([], notes) == sweep_mod.STATUS_DIVERGENT


def test_a_missing_loss_history_is_not_a_pass() -> None:
    notes = sweep_mod.divergence_findings(None, None, steps_run=3)
    assert notes and "UNCHECKED" in notes[0]
    assert sweep_mod.classify_status([], notes) == sweep_mod.STATUS_DIVERGENT


def test_experiments_aggregate_worst_of_not_first_or_majority() -> None:
    """The behaviour, not just the ordering constant.

    This test previously asserted only that ``_STATUS_SEVERITY`` is ordered, and
    a mutation replacing the whole aggregation with ``runs[0]["status"]`` left it
    GREEN — a guard whose docstring claimed worst-of while the check tested a
    dict. Asserting the function is what makes the claim true.

    Every case here has the bad run in a position that first-of and majority
    would both miss.
    """
    agg = sweep_mod.aggregate_status
    PASS, DIV, FAIL = sweep_mod.STATUS_PASS, sweep_mod.STATUS_DIVERGENT, sweep_mod.STATUS_FAIL

    assert agg([PASS, PASS]) == PASS
    # first-of would say PASS; majority-of-3 would say PASS.
    assert agg([PASS, DIV]) == DIV
    assert agg([PASS, PASS, DIV]) == DIV
    assert agg([PASS, PASS, FAIL]) == FAIL
    # FAIL outranks DIVERGENT wherever it sits.
    assert agg([DIV, FAIL]) == FAIL
    assert agg([FAIL, DIV]) == FAIL
    assert agg([DIV, DIV]) == DIV

    sev = sweep_mod._STATUS_SEVERITY
    assert sev[FAIL] > sev[DIV] > sev[PASS]


def test_divergent_exits_non_zero_just_like_fail() -> None:
    """A sweep containing a diverging template is not a clean run.

    Asserted on the exit STATUS, not the table: CI, a Makefile target and a
    shell pipeline all read the former. A three-state gate whose middle state
    exits 0 would report the row honestly and still let every automated caller
    treat the sweep as green.
    """
    def report(*statuses):
        return {"templates": [{"status": st} for st in statuses]}

    assert sweep_mod.exit_code(report("PASS", "PASS")) == 0
    assert sweep_mod.exit_code(report("PASS", "DIVERGENT")) == 1
    assert sweep_mod.exit_code(report("PASS", "FAIL")) == 1
    assert sweep_mod.exit_code(report("DIVERGENT")) == 1
    assert sweep_mod.exit_code(report()) == 0


def test_sweep_aggregates_with_aggregate_status() -> None:
    """`sweep` must actually CALL the worst-of helper.

    Extracting `aggregate_status` made the rule testable, but it also made it
    possible for `sweep` to stop using it and for every test above to stay green.
    Read statically rather than by running a sweep, which needs torch and hours:
    the point is only that the call site exists and that nothing has gone back to
    indexing a run directly.
    """
    source = (ROOT / "tools" / "od_acceptance_sweep.py").read_text(encoding="utf-8")
    assert 'row["status"] = aggregate_status(' in source, (
        "sweep() no longer aggregates experiment statuses through aggregate_status; "
        "worst-of is not being applied even though its unit test still passes"
    )
    assert 'row["status"] = runs[0]' not in source
