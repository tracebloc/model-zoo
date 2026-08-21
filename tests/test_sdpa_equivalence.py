"""Forward-equivalence tests for the SDPA attention conversion (backend#2090).

Hand-rolled attention (explicit softmax(QK^T/sqrt(d))·V) in zoo templates is
routed through torch.nn.functional.scaled_dot_product_attention, which
dispatches to fused backends per hardware. These tests are the acceptance
bar for that conversion: for each converted module, run the original manual
formula and the SDPA path on the same weights and the same random input, and
assert the outputs match in fp32 at tight tolerance.
"""

import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")

ROOT = pathlib.Path(__file__).parent.parent


def _load(rel_path):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manual_alibi_attention(attn_module, x, key_padding_mask=None):
    """The pre-conversion formula, verbatim: explicit scaled QK^T, additive
    ALiBi bias, masked_fill, softmax, then the value matmul. Runs on the
    converted module's own weights so any drift is the conversion's."""
    B, S, _ = x.shape
    qkv = attn_module.qkv(x).reshape(B, S, 3, attn_module.num_heads, attn_module.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * attn_module.scale

    positions = torch.arange(S, device=x.device)
    distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
    alibi = -distance.unsqueeze(0) * attn_module.slopes.unsqueeze(-1).unsqueeze(-1)
    attn = attn + alibi.unsqueeze(0)

    if key_padding_mask is not None:
        attn = attn.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

    attn = torch.softmax(attn, dim=-1)
    # dropout is identity in eval mode, matching dropout_p=0.0 on the SDPA path

    out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
    return attn_module.out_proj(out)


@pytest.mark.parametrize("with_padding_mask", [False, True])
def test_alibi_attention_matches_manual_formula(with_padding_mask):
    mod = _load("model_zoo/masked_language_modeling/pytorch/relative_position_mlm.py")

    torch.manual_seed(0)
    attn = mod._ALiBiMultiheadAttention(hidden_size=64, num_heads=4, dropout=0.1)
    attn.eval()

    B, S = 3, 17
    x = torch.randn(B, S, 64)
    key_padding_mask = None
    if with_padding_mask:
        # ragged padding tails; no fully-padded row (all-masked rows are
        # NaN under both formulations, so they prove nothing)
        key_padding_mask = torch.zeros(B, S, dtype=torch.bool)
        key_padding_mask[0, 12:] = True
        key_padding_mask[2, 5:] = True

    with torch.no_grad():
        expected = _manual_alibi_attention(attn, x, key_padding_mask)
        actual = attn(x, key_padding_mask)

    assert expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_relative_position_mlm_full_forward_is_finite_with_padding():
    """End-to-end guard: the converted layer inside the full model still
    produces finite logits for padded batches (mask plumbing intact)."""
    mod = _load("model_zoo/masked_language_modeling/pytorch/relative_position_mlm.py")

    torch.manual_seed(0)
    model = mod.RelativePositionMLM(
        vocab_size=100, hidden_size=32, num_layers=2, num_heads=4, intermediate_size=64
    )
    model.eval()

    input_ids = torch.randint(0, 100, (2, 10))
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    attention_mask[1, 6:] = 0

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    assert logits.shape == (2, 10, 100)
    assert torch.isfinite(logits[0]).all()
    assert torch.isfinite(logits[1, :6]).all()


@pytest.mark.parametrize("half_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_padding_mask", [False, True])
def test_alibi_mask_dtype_follows_the_query(half_dtype, with_padding_mask):
    """SDPA's contract for a float mask is "the same type as query, key,
    value". The ALiBi bias is built in fp32, so on half weights it has to be
    cast before the call — otherwise the fused kernels see a mask they will
    not take (ROCm's efficient attention warns and refuses outright) and the
    path silently falls back to math. The old add-after-matmul formulation
    promoted dtypes for free; SDPA does not. Bugbot, #165.
    """
    mod = _load("model_zoo/masked_language_modeling/pytorch/relative_position_mlm.py")

    torch.manual_seed(0)
    attn = mod._ALiBiMultiheadAttention(hidden_size=64, num_heads=4, dropout=0.1)
    attn.eval().to(half_dtype)

    B, S = 2, 9
    x = torch.randn(B, S, 64, dtype=half_dtype)
    key_padding_mask = None
    if with_padding_mask:
        key_padding_mask = torch.zeros(B, S, dtype=torch.bool)
        key_padding_mask[1, 6:] = True

    seen = {}
    real_sdpa = mod.F.scaled_dot_product_attention

    def _spy(q, k, v, *args, **kwargs):
        seen["q"] = q.dtype
        seen["mask"] = kwargs["attn_mask"].dtype
        return real_sdpa(q, k, v, *args, **kwargs)

    mod.F.scaled_dot_product_attention = _spy
    try:
        with torch.no_grad():
            out = attn(x, key_padding_mask)
    finally:
        mod.F.scaled_dot_product_attention = real_sdpa

    assert seen["mask"] == seen["q"] == half_dtype
    assert out.dtype == half_dtype
    assert torch.isfinite(out).all()
