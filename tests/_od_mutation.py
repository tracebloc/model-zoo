"""The mutation-sweep harness shared by the YOLO template suites.

``test_yolov8_s.py``, ``test_yolov9_s.py``, ``test_yolov10_s.py``,
``test_yolo11_s.py`` and ``test_yolov12_s.py`` each prove a hand-written
detector with the same machinery: a ``GUARDS`` table, a ``MUTATIONS`` table of
``(name, anchor, replacement, target_guard)`` rows, a loader that executes the
template from a real file, and the same handful of sweep tests. It lives here
once so the five suites cannot drift apart.

What stays in each file
-----------------------
The guards, the two tables, the silent-mutation list and every test that is
about that one template. Each file binds the shared tests to its own tables by
assignment::

    test_no_mutation_baseline = harness.no_mutation_baseline(GUARDS, _load)

pytest names a test after the module attribute it is bound to, so every node
ID (``tests/test_yolov8_s.py::test_mutation_is_caught_by_its_guard[...]``) is
the one the file had before; only the code behind it moved. The file's
``pytestmark`` still applies, because marks follow the module the test is
collected from.

Not a general test-utils module: it holds the mutation sweep and nothing else.
``test_od_hand_written_detectors.py`` runs a different sweep (one guard table
over many templates) and keeps its own loader.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import tempfile

import pytest

# --------------------------------------------------------------------------
# loading, mutating
# --------------------------------------------------------------------------


def exec_source(source: str, stem: str):
    """Import ``source`` as a fresh module.

    Written to a real file rather than ``exec``-ed into a dict so tracebacks
    point somewhere, and so the module behaves exactly as the uploaded template
    does — a template is a file, and the SDK loads it as one.
    """
    with tempfile.TemporaryDirectory(prefix="tb-mutate-") as raw:
        target = pathlib.Path(raw) / f"{stem}.py"
        target.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(stem, target)
        assert spec and spec.loader, f"cannot build a spec for {target}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Kept so a guard that needs a SECOND, independently-configured copy of
        # the module under test can re-exec the same source (``reload``).
        # Reading the file again would silently hand the guard the PRISTINE
        # template while the mutation sweep believed it was testing a mutated
        # one — a guard pointed at unmutated code reports "passed", which is
        # indistinguishable from a survivor and reads as the opposite of what
        # it means.
        module.__tb_source__ = source
        return module


def loaders(template: pathlib.Path):
    """Return ``(load, mutate)`` for one template file.

    ``load()`` executes the shipped template. ``mutate(anchor, replacement)``
    executes it with ``anchor`` replaced by ``replacement``.

    THE ANCHOR MUST APPEAR EXACTLY ONCE, and that assertion is the whole point.
    A patch whose anchor has drifted silently changes nothing, the guard is then
    pointed at a pristine template, and the mutation reports "passed" — which is
    indistinguishable from a genuine survivor and reads as the opposite of what
    it means. So a drifted anchor fails here instead.
    """

    def load():
        return exec_source(template.read_text(encoding="utf-8"), template.stem)

    def mutate(anchor: str, replacement: str):
        source = template.read_text(encoding="utf-8")
        occurrences = source.count(anchor)
        assert occurrences == 1, (
            f"{template.name}: mutation anchor occurs {occurrences} times, "
            f"expected 1. The template moved under the mutation; re-anchor it "
            f"rather than loosening this check.\nanchor:\n{anchor}"
        )
        mutated = source.replace(anchor, replacement)
        assert mutated != source, "the replacement is identical to the anchor"
        return exec_source(mutated, f"mutated_{template.stem}")

    return load, mutate


def reload(module):
    """A fresh copy of whatever source ``module`` was built from.

    Used by the guards that rebuild at another published scale: they mutate
    module-level tables, so they need their own copy — and it must be the same
    source, mutated or not.
    """
    source = getattr(module, "__tb_source__", None)
    assert source, (
        f"{module.__name__}: no recorded source to re-exec. A guard rebuilding "
        f"at another scale must not fall back to reading the template file, or "
        f"it would test pristine code under a mutation."
    )
    return exec_source(source, f"reloaded_{module.__name__}")


def build(module, num_classes: int, input_size=None):
    """Construct the template's declared entry point."""
    entry_name = getattr(module, "main_class", None) or getattr(
        module, "main_method", None
    )
    assert entry_name, f"{module.__name__}: no main_class / main_method"
    entry = getattr(module, entry_name)
    if input_size is None:
        return entry(num_classes)
    return entry(num_classes, input_size)


# --------------------------------------------------------------------------
# the sweep tests, bound to one file's tables
# --------------------------------------------------------------------------


def _named(name: str):
    """Give a factory-made test the name it is bound to, so ``repr`` and
    introspection show it. A traceback still names the frame ``test`` in this
    file; the failing node ID is what says which template and row."""

    def rename(function):
        function.__name__ = function.__qualname__ = name
        return function

    return rename


def template_is_a_torchvision_detector(template: pathlib.Path):
    @_named("test_the_template_exists")
    def test() -> None:
        """Guard the guard: every table in this file is keyed on one file."""
        assert template.is_file(), f"{template} is missing — this whole file is dead"
        source = template.read_text(encoding="utf-8")
        assert re.search(
            r'^\s*model_type\s*=\s*"torchvision_detection"', source, re.MULTILINE
        ), (
            f"{template.name} must declare model_type = 'torchvision_detection'. "
            f"The legacy 'yolo' family is a fixed 7x7 grid at 448px with one "
            f"object per cell and an external loss.py, and it is frozen."
        )
        assert "yolo" != (
            re.search(r'^\s*model_type\s*=\s*"(\w*)"', source, re.MULTILINE).group(1)
        )

    return test


def no_mutation_baseline(guards: dict, load):
    @_named("test_no_mutation_baseline")
    def test() -> None:
        """The zero row of the sweep.

        A mutation sweep without a no-mutation baseline cannot distinguish "the
        guard caught the mutation" from "the guard fails on everything", and a
        ``mutate`` anchor that no longer applies would otherwise report a
        pristine template as a caught mutation. This runs the whole table
        against the unmutated file in ONE module load, so the baseline is cheap
        and always present.

        It overlaps the per-guard positive control below, which runs each
        guard on its own fresh load. What only this test does is run every
        guard against the SAME module object, in table order.
        """
        module = load()
        failures = []
        for name, guard in guards.items():
            try:
                guard(module)
            except AssertionError as error:  # pragma: no cover — reported, not hit
                failures.append(f"{name}: {error}")
        assert not failures, (
            "guard(s) fail on the UNMUTATED template, so every 'mutation caught' "
            "result below is meaningless:\n" + "\n\n".join(failures)
        )

    return test


def guard_passes_on_the_shipped_template(guards: dict, load):
    @_named("test_guard_passes_on_the_shipped_template")
    @pytest.mark.parametrize("guard_name", sorted(guards))
    def test(guard_name: str) -> None:
        """The per-guard positive control, so a failure names one guard."""
        guards[guard_name](load())

    return test


def mutation_is_caught_by_its_guard(guards: dict, mutations: list, mutate):
    @_named("test_mutation_is_caught_by_its_guard")
    @pytest.mark.parametrize(
        "name,anchor,replacement,target",
        mutations,
        ids=[entry[0] for entry in mutations],
    )
    def test(name: str, anchor: str, replacement: str, target: str) -> None:
        """Point ONE named guard at a template edited to break exactly what it
        checks.

        THE ATTRIBUTION IS THE POINT, and it is why this runs a single guard
        rather than the whole table. A sweep that runs every guard against
        every mutation proves that *something* reddened, not that the guard
        under test did — so a guard that is defined and never actually
        exercised looks covered. Here the mutation names its guard and only
        that guard runs, so a red is attributable and a green is a genuine
        survivor.

        A guard that cannot be made to fail proves nothing about the code it
        covers, and for a hand-written detector "it returned a loss dict" is
        our own code answering its own question. Keeping the mutation in the
        suite is what stops a guard rotting into a tautology as the template
        changes.
        """
        mutated = mutate(anchor, replacement)
        guard = guards[target]

        with pytest.raises(AssertionError) as excinfo:
            guard(mutated)

        message = str(excinfo.value)
        assert message.strip(), f"{name}: the guard failed with an empty message"
        # The failure must be the guard's own, not a fixture-degeneracy
        # assertion: several guards carry those, and a mutation that trips one
        # has told us the fixture stopped exercising the rule, not that the
        # rule is guarded.
        assert "fixture is degenerate" not in message and "fixture:" not in message, (
            f"{name}: the mutation tripped a fixture-degeneracy assertion rather "
            f"than the guard itself — the fixture no longer exercises the rule "
            f"under test:\n{message}"
        )

    return test


def every_guard_has_a_mutation(guards: dict, mutations: list):
    @_named("test_every_guard_has_a_mutation")
    @pytest.mark.parametrize("guard_name", sorted(guards))
    def test(guard_name: str) -> None:
        """No guard may be un-proven.

        Without this, adding a guard and forgetting its mutation leaves an
        assertion nobody has ever seen fail — which is how a wrong assigner
        ships green.
        """
        covered = {target for *_, target in mutations}
        assert guard_name in covered, (
            f"{guard_name} has no entry in MUTATIONS. Every guard here must be "
            f"shown able to go red; add the textual edit that breaks it."
        )

    return test


def every_mutation_targets_a_real_guard(guards: dict, mutations: list):
    @_named("test_every_mutation_targets_a_real_guard")
    def test() -> None:
        """The other direction: a mutation naming a renamed guard would be
        skipped silently by the parametrized test's ``KeyError``-free lookup."""
        unknown = sorted({target for *_, target in mutations} - set(guards))
        assert not unknown, f"MUTATIONS name guards that do not exist: {unknown}"

    return test


def mutation_names_are_unique(mutations: list):
    @_named("test_mutation_names_are_unique")
    def test() -> None:
        """Two mutations sharing a name make the sweep's report ambiguous — the
        pytest id collides and one of them is indistinguishable from the other
        in any table published from this suite."""
        names = [entry[0] for entry in mutations]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        assert not duplicates, f"duplicate mutation names: {duplicates}"

    return test


def silent_mutations_still_train(
    mutations: list, silent: frozenset, mutate, *, not_silent: frozenset | None = None
):
    """``not_silent``, when the file derives ``silent`` by exclusion, is checked
    for stale entries: an exclusion that matches nothing quietly shrinks the
    sweep."""

    @_named("test_silent_mutations_still_train")
    def test() -> None:
        """The point of this whole file, stated as a test: the mutations do NOT
        break training.

        Each mutation named in ``_SILENT_MUTATIONS`` leaves
        ``model(images, targets)`` returning a finite loss dict and
        ``model(images)`` returning well-formed predictions, so the family
        train-step test stays green against every one of them. That is the
        reason the guards exist, and asserting it here stops someone
        concluding the family test already covers this template.
        """
        import torch

        selected = [entry for entry in mutations if entry[0] in silent]
        assert len(selected) == len(silent), (
            f"_SILENT_MUTATIONS names "
            f"{sorted(silent - {e[0] for e in mutations})} which are not "
            f"in MUTATIONS"
        )
        if not_silent is not None:
            stale = sorted(not_silent - {entry[0] for entry in mutations})
            assert not stale, (
                f"_NOT_SILENT excludes {stale}, which are not in MUTATIONS — an "
                f"exclusion that matches nothing quietly shrinks this sweep"
            )

        targets = [
            {
                "boxes": torch.tensor(
                    [[10.0, 10.0, 60.0, 60.0], [70.0, 70.0, 110.0, 120.0]]
                ),
                "labels": torch.tensor([1, 3], dtype=torch.int64),
            },
            {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            },
        ]
        images = [torch.rand(3, 128, 160), torch.rand(3, 144, 128)]

        for name, anchor, replacement, _ in selected:
            module = mutate(anchor, replacement)
            model = build(module, 3, 128)
            model.train()
            losses = model(images, targets)
            assert isinstance(losses, dict) and losses, f"{name}: no loss dict"
            for key, value in losses.items():
                assert torch.isfinite(value).all(), f"{name}: loss {key} is {value!r}"
            model.eval()
            with torch.no_grad():
                predictions = model(images)
            assert isinstance(predictions, list) and len(predictions) == len(images), (
                f"{name}: eval returned {predictions!r}"
            )
            for prediction in predictions:
                assert {"boxes", "scores", "labels"} <= set(prediction), (
                    f"{name}: eval prediction is missing keys"
                )
            del model, module

    return test
