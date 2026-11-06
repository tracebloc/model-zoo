"""``image_classification/pytorch/multichannel_cnn.py`` takes its channel count
from one header.

tracebloc fills a picked template's ``input_channels`` with the dataset's
channel count, as it fills ``output_classes``: it replaces the integer of the
one module-level, single-line ``input_channels = <int>`` statement and nothing
else. So the template must declare it exactly that way, and its first
convolution must take whatever count that line says. Every channel count a
dataset may have (1 to 64) builds, runs a forward pass, and the model fits a
small batch, at the template's own image size and at others.
"""

import ast
import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")

ROOT = pathlib.Path(__file__).parent.parent
TEMPLATE = ROOT / "model_zoo" / "image_classification" / "pytorch" / "multichannel_cnn.py"


def _header(source, name):
    """The one module-level ``name = <int>`` statement, as tracebloc reads it."""
    tree = ast.parse(source)
    found = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    assert len(found) == 1, f"{name}: {len(found)} module-level assignments"
    (node,) = found
    assert isinstance(node, ast.Assign) and len(node.targets) == 1, name
    assert node.col_offset == 0 and node.lineno == node.end_lineno, name
    assert isinstance(node.value, ast.Constant) and type(node.value.value) is int, name
    return node


def _rendered(tmp_path, **counts):
    """The template with each ``name=count`` written into its header, built."""
    lines = TEMPLATE.read_text(encoding="utf-8").split("\n")
    for name, count in counts.items():
        node = _header("\n".join(lines), name)
        lines[node.lineno - 1] = f"{name} = {count}"
    path = tmp_path / f"multichannel_cnn_{'_'.join(map(str, counts.values()))}.py"
    path.write_text("\n".join(lines), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _first_conv(model):
    return next(m for m in model.modules() if isinstance(m, torch.nn.Conv2d))


def test_the_counts_are_headers_tracebloc_can_fill():
    source = TEMPLATE.read_text(encoding="utf-8")
    for name in ("input_channels", "output_classes", "image_size", "batch_size"):
        _header(source, name)


@pytest.mark.parametrize("channels", [1, 3, 4, 13, 64])
def test_the_first_convolution_takes_the_header(tmp_path, channels):
    module = _rendered(tmp_path, input_channels=channels, output_classes=5)
    model = getattr(module, module.main_class)()
    assert _first_conv(model).in_channels == channels
    size = module.image_size
    out = model(torch.randn(2, channels, size, size))
    assert out.shape == (2, 5)


@pytest.mark.parametrize("size", [32, 97])
def test_any_image_size_runs(tmp_path, size):
    module = _rendered(tmp_path, input_channels=4)
    model = getattr(module, module.main_class)()
    assert model(torch.randn(2, 4, size, size)).shape == (2, module.output_classes)


def test_it_fits_a_small_four_channel_batch(tmp_path):
    torch.manual_seed(0)
    module = _rendered(tmp_path, input_channels=4, output_classes=2)
    model = getattr(module, module.main_class)()
    size = module.image_size
    labels = torch.arange(module.batch_size) % 2
    # The class is in the fourth channel alone, which an RGB model never reads.
    images = torch.rand(module.batch_size, 4, size, size) * 0.1
    images[:, 3] += labels.view(-1, 1, 1).float()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    first = loss_fn(model(images), labels).item()
    for _ in range(30):
        optimiser.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimiser.step()
    assert loss.item() < first / 4
