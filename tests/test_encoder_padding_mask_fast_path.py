"""Token templates built on ``nn.TransformerEncoder`` must not reach the
nested-tensor fast path when given a padding mask.

In eval with no gradient, ``nn.TransformerEncoder.forward`` checks the padding
mask with ``torch._nested_tensor_from_mask_left_aligned`` before it checks the
device, and that op has no Apple GPU (MPS) kernel: validation on a Mac GPU
raised ``NotImplementedError`` for simple_token and tiny_mlm. The fast path is
off only when ``enable_nested_tensor=False`` is passed to the constructor:
``__init__`` copies it into ``use_nested_tensor``, which is what ``forward``
reads, so setting ``encoder.enable_nested_tensor = False`` afterwards changes
nothing.

CI has no Apple GPU. The op is patched to raise MPS's own error, so a template
that still reaches it fails here on the CPU exactly as it did on the Mac.
"""

import importlib.util
import pathlib
import re
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

ROOT = pathlib.Path(__file__).parent.parent
MODEL_ROOT = ROOT / "model_zoo"

#: The two templates the Mac GPU met it with, named here independently of the
#: scan below, so a scan that finds nothing cannot pass.
MET_ON_THE_MAC = {
    "token_classification/pytorch/simple_token.py",
    "masked_language_modeling/pytorch/tiny_mlm.py",
}

#: torch's message when MPS has no kernel for the op.
MPS_ERROR = (
    "The operator 'aten::_nested_tensor_from_mask_left_aligned' is not currently "
    "implemented for the MPS device."
)


def _token_encoder_templates() -> list[pathlib.Path]:
    """Every template that builds an ``nn.TransformerEncoder`` and takes an
    ``attention_mask``: the inputs that reach the padding-mask check."""
    found = []
    for path in sorted(MODEL_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "nn.TransformerEncoder(" in text and re.search(
            r"def forward\(self, input_ids, attention_mask", text
        ):
            found.append(path)
    return found


TEMPLATES = _token_encoder_templates()


def _key(path: pathlib.Path) -> str:
    return path.relative_to(MODEL_ROOT).as_posix()


def _build(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(re.sub(r"\W", "_", path.stem), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    return getattr(module, module.main_class)()


def test_the_scan_finds_the_templates_the_mac_met():
    assert MET_ON_THE_MAC <= {_key(p) for p in TEMPLATES}


@pytest.mark.parametrize("path", TEMPLATES, ids=_key)
def test_an_eval_forward_with_padding_never_reaches_the_op(path):
    model = _build(path).eval()
    vocab = min(
        m.num_embeddings for m in model.modules() if isinstance(m, torch.nn.Embedding)
    )
    input_ids = torch.randint(1, min(vocab, 100), (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    attention_mask[:, 12:] = 0  # padded, left-aligned: what a tokenizer gives

    calls = []

    def no_mps_kernel(*args, **kwargs):
        calls.append(1)
        raise NotImplementedError(MPS_ERROR)

    with (
        patch.object(torch, "_nested_tensor_from_mask_left_aligned", no_mps_kernel),
        torch.no_grad(),
    ):
        model(input_ids=input_ids, attention_mask=attention_mask)
    assert calls == [], f"{_key(path)} reached the nested-tensor mask check"


@pytest.mark.parametrize("path", TEMPLATES, ids=_key)
def test_every_encoder_has_the_fast_path_off(path):
    encoders = [
        m for m in _build(path).modules() if isinstance(m, torch.nn.TransformerEncoder)
    ]
    assert encoders, f"{_key(path)} built no nn.TransformerEncoder"
    assert all(not e.use_nested_tensor for e in encoders), _key(path)


def test_no_template_sets_the_flag_after_construction():
    # The assignment form is a no-op (see the module docstring).
    offenders = [
        _key(p)
        for p in sorted(MODEL_ROOT.rglob("*.py"))
        if re.search(r"\.enable_nested_tensor\s*=", p.read_text(encoding="utf-8"))
    ]
    assert offenders == []
