"""One template per non-object-detection category survives a few train steps.

``test_model_contract.py`` proves a template constructs; constructing is not
training. A template whose forward pass detaches its head, emits the wrong
shape for its category's loss, or produces a non-finite loss still constructs.
This file runs a few optimizer steps on tiny synthetic input for one
representative template per category and asserts that every step's loss is
finite, every trainable parameter receives a gradient and moves, and one
eval-mode forward pass returns the train-mode output shape, finite.

Each template's input shape and class count come from its own header
(``output_classes``, ``num_feature_points``, ``sequence_length``,
``forecast_horizon``, ``image_size``), so a header the platform rewrites is the
one under test. U-Net is the exception: it takes any multiple of 16, so it runs
at 64px rather than its declared size. Object detection has its own file,
``test_od_torchvision_family_train_step.py``.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ZOO = ROOT / "model_zoo"
STEPS = 3


def _load(path: pathlib.Path):
    name = re.sub(r"\W", "_", f"train_step_{path.relative_to(ZOO)}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(module):
    entry = getattr(module, "main_class", None) or module.main_method
    return getattr(module, entry)()


# Each batch builder takes (torch, module) and returns (inputs, target, loss_fn);
# inputs is a tuple passed positionally to the model.


def _image_classification(torch, m):
    size = m.image_size
    x = torch.rand(2, 3, size, size)
    y = torch.tensor([0, m.output_classes - 1])
    return (x,), y, torch.nn.functional.cross_entropy


def _semantic_segmentation(torch, m):
    # Any multiple of 16 is a valid U-Net input; 64 keeps the step to well under a second.
    x = torch.rand(2, 3, 64, 64)
    y = torch.randint(0, m.output_classes, (2, 64, 64))
    return (x,), y, torch.nn.functional.cross_entropy


def _text_classification(torch, m):
    n = max(4, m.output_classes)  # every class, the last included, appears in the batch
    ids = torch.randint(1, 1000, (n, m.sequence_length))
    mask = torch.ones_like(ids)
    mask[1, m.sequence_length // 2 :] = 0
    y = torch.arange(n) % m.output_classes
    return (ids, mask), y, torch.nn.functional.cross_entropy


def _tabular_classification(torch, m):
    n = max(8, m.output_classes)  # every class, the last included, appears in the batch
    x = torch.randn(n, m.num_feature_points)
    x[0, 0] = float("nan")  # the platform's tables carry missing values
    y = torch.arange(n) % m.output_classes
    return (x,), y, torch.nn.functional.cross_entropy


def _time_series_classification(torch, m):
    n = max(4, m.output_classes)  # every class, the last included, appears in the batch
    x = torch.randn(n, m.sequence_length, m.num_feature_points)
    x[1, m.sequence_length // 2 :] = 0.0  # zero post-padding, as the dataset emits it
    y = torch.arange(n) % m.output_classes
    return (x,), y, torch.nn.functional.cross_entropy


def _time_series_forecasting(torch, m):
    x = torch.randn(4, m.sequence_length, m.num_feature_points)
    y = torch.randn(4, m.forecast_horizon)
    return (x,), y, torch.nn.functional.mse_loss


#: One representative per category: the smallest template, so the whole file
#: adds seconds to the pytorch job.
TORCH_CASES = {
    "image_classification/pytorch/simple_cnn.py": _image_classification,
    "semantic_segmentation/pytorch/unet.py": _semantic_segmentation,
    "text_classification/pytorch/simple_text.py": _text_classification,
    "tabular_classification/pytorch/fcn.py": _tabular_classification,
    "time_series_classification/pytorch/gru.py": _time_series_classification,
    "time_series_forecasting/pytorch/gru.py": _time_series_forecasting,
}

SKLEARN_CASES = ["tabular_classification/sklearn/logistic_regression.py"]


def _assert_trains(torch, model, batch, name: str, optimizer_cls=None) -> None:
    """Run ``STEPS`` optimizer steps, then one eval forward pass.

    Factored out so the guard can be pointed at a model or optimizer known to
    break it, below. Every trainable parameter must receive a finite gradient
    and end the run changed: a detached or dead submodule fails here.
    """
    inputs, target, loss_fn = batch
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert params, f"{name}: no trainable parameters"
    before = {n: p.detach().clone() for n, p in params.items()}
    optimizer = (optimizer_cls or torch.optim.Adam)(params.values(), lr=1e-2)

    model.train()
    for _ in range(STEPS):
        optimizer.zero_grad()
        out = model(*inputs)
        if target.is_floating_point():
            # A regression loss broadcasts a mis-shaped output instead of raising.
            assert out.shape == target.shape, (
                f"{name}: output {tuple(out.shape)} vs target {tuple(target.shape)}"
            )
        loss = loss_fn(out, target)
        assert loss.ndim == 0, f"{name}: loss has shape {tuple(loss.shape)}"
        assert torch.isfinite(loss), f"{name}: loss is {loss.item()}"
        loss.backward()
        no_grad = sorted(n for n, p in params.items() if p.grad is None)
        assert not no_grad, f"{name}: parameters received no gradient: {no_grad[:5]}"
        bad = sorted(n for n, p in params.items() if not torch.isfinite(p.grad).all())
        assert not bad, f"{name}: non-finite gradients in {bad[:5]}"
        optimizer.step()

    unchanged = sorted(n for n, p in params.items() if torch.equal(p.detach(), before[n]))
    assert not unchanged, f"{name}: parameters did not move in {STEPS} steps: {unchanged[:5]}"

    model.eval()
    with torch.no_grad():
        pred = model(*inputs)
    assert pred.shape == out.shape, (
        f"{name}: eval output {tuple(pred.shape)} differs from train output {tuple(out.shape)}"
    )
    assert torch.isfinite(pred).all(), f"{name}: eval output is not finite"


def test_cases_name_one_template_per_category() -> None:
    """Guard the table: each entry exists, is a template, and sits in the
    category it is keyed under, so a rename cannot leave a case pointing nowhere."""
    for rel in [*TORCH_CASES, *SKLEARN_CASES]:
        path = ZOO / rel
        assert path.is_file(), f"{rel}: no such template"
        text = path.read_text(encoding="utf-8")
        declared = re.search(r'^category\s*=\s*["\'](\w+)["\']', text, re.MULTILINE)
        assert declared and declared.group(1) == rel.split("/")[0], (
            f"{rel}: declares category {declared and declared.group(1)!r}"
        )


@pytest.mark.parametrize("rel", sorted(TORCH_CASES))
def test_template_trains(rel: str) -> None:
    torch = pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    torch.manual_seed(0)
    module = _load(ZOO / rel)
    model = _build(module)
    _assert_trains(torch, model, TORCH_CASES[rel](torch, module), rel)


@pytest.mark.parametrize("rel", SKLEARN_CASES)
def test_sklearn_template_fits(rel: str) -> None:
    """A fit on a tiny, linearly separable table must learn it.

    A template that fits without learning (a constant predictor, a pipeline
    step that drops every feature) predicts one class and fails the accuracy bar.
    """
    pytest.importorskip("sklearn", reason="scikit-learn not installed in this CI job")
    selected = os.environ.get("MODEL_ZOO_FRAMEWORKS")
    if selected is not None and "sklearn" not in {s.strip() for s in selected.split(",")}:
        pytest.skip("MODEL_ZOO_FRAMEWORKS leaves sklearn templates to another job")
    import numpy as np

    module = _load(ZOO / rel)
    classes = module.output_classes
    assert module.num_feature_points > classes, f"{rel}: too few features to separate"
    rng = np.random.default_rng(0)
    y = np.arange(20 * classes) % classes
    x = rng.normal(scale=0.1, size=(len(y), module.num_feature_points))
    x[np.arange(len(y)), y] += 3.0  # class k lights up feature k
    x[0, -1] = np.nan  # the platform's tables carry missing values

    model = _build(module).fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (len(y), classes) and np.isfinite(proba).all(), rel
    accuracy = float((model.predict(x) == y).mean())
    assert accuracy >= 0.9, f"{rel}: training accuracy {accuracy:.2f} on separable data"


def test_guard_rejects_a_step_that_moves_nothing() -> None:
    """The guard, shown going red: an optimizer whose step is a no-op."""
    torch = pytest.importorskip("torch", reason="pytorch not installed in this CI job")

    class NoStep(torch.optim.SGD):
        def step(self, closure=None):
            return None

    torch.manual_seed(0)
    rel = "tabular_classification/pytorch/fcn.py"
    module = _load(ZOO / rel)
    batch = _tabular_classification(torch, module)
    with pytest.raises(AssertionError, match="did not move"):
        _assert_trains(torch, _build(module), batch, rel, optimizer_cls=NoStep)


def test_guard_rejects_a_detached_layer_and_a_non_finite_loss() -> None:
    torch = pytest.importorskip("torch", reason="pytorch not installed in this CI job")
    torch.manual_seed(0)
    nn = torch.nn

    class Detached(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Linear(4, 4)
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            return self.head(self.body(x).detach())

    batch = ((torch.randn(4, 4),), torch.tensor([0, 1, 0, 1]), nn.functional.cross_entropy)
    with pytest.raises(AssertionError, match="no gradient"):
        _assert_trains(torch, Detached(), batch, "detached")

    class Exploding(Detached):
        def forward(self, x):
            return self.head(self.body(x)) * float("nan")

    with pytest.raises(AssertionError, match="loss is nan"):
        _assert_trains(torch, Exploding(), batch, "exploding")


def test_guard_rejects_a_template_that_learns_nothing(monkeypatch) -> None:
    """The sklearn half of the guard: a constant predictor fails the accuracy bar."""
    pytest.importorskip("sklearn", reason="scikit-learn not installed in this CI job")
    from sklearn.dummy import DummyClassifier

    real_load = _load

    def constant(path):
        module = real_load(path)
        module.MyModel = lambda: DummyClassifier(strategy="most_frequent")
        module.main_method = "MyModel"
        return module

    monkeypatch.setitem(globals(), "_load", constant)
    with pytest.raises(AssertionError, match="training accuracy"):
        test_sklearn_template_fits(SKLEARN_CASES[0])
