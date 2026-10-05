"""`tools/seed_contract.py audit` — the check that a seed about to be hosted
(or already hosted) carries none of its template's SEED_EXCLUDED_PREFIXES.

model-zoo#135: prod served `ssdlite_mobilenet` / `ssd_vgg16` (and every other
declaring template sampled) with the class head still in, so any
`output_classes` other than the dump's was refused on a size mismatch. Nothing
looked at the bytes being published: `verify_backbone_seeds.py` was run on the
strip output, not on what the store serves.
"""

import pathlib
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

ROOT = pathlib.Path(__file__).parent.parent
TOOL = ROOT / "tools" / "seed_contract.py"

HEADED = """\
SEED_EXCLUDED_PREFIXES = ("fc.",)
framework = "pytorch"
"""
HEADLESS = 'framework = "pytorch"\n'


class _Ref(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(4, 4)
        self.fc = nn.Linear(4, 3)


def _zoo(tmp_path):
    tdir = tmp_path / "model_zoo" / "image_classification" / "pytorch"
    tdir.mkdir(parents=True)
    (tdir / "headed.py").write_text(HEADED)
    (tdir / "headless.py").write_text(HEADLESS)
    return tmp_path


def _audit(zoo, weights):
    return subprocess.run(
        [sys.executable, str(TOOL), "--zoo", str(zoo), "audit", "--weights", str(weights)],
        capture_output=True,
        text=True,
    )


def _full():
    return _Ref().state_dict()


def _backbone():
    return {k: v for k, v in _full().items() if not k.startswith("fc.")}


def test_a_dump_that_still_carries_its_declared_head_is_red(tmp_path):
    zoo = _zoo(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    torch.save(_full(), dist / "headed_weights.pkl")
    torch.save(_full(), dist / "headless_weights.pkl")
    out = _audit(zoo, dist)
    assert out.returncode == 1, out.stdout + out.stderr
    assert "headed: 2 declared head key(s)" in out.stderr
    # A template that declares nothing has nothing to strip: never flagged.
    assert "headless:" not in out.stderr


def test_backbone_only_seeds_pass_in_the_strip_layout(tmp_path):
    """`strip` writes `<dir>/<dir>_weights.pkl`; `fetch-all` writes flat."""
    zoo = _zoo(tmp_path)
    seeds = tmp_path / "seeds"
    (seeds / "headed").mkdir(parents=True)
    torch.save(_backbone(), seeds / "headed" / "headed_weights.pkl")
    out = _audit(zoo, seeds)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "1 clean, 0 carrying" in out.stdout


def test_an_empty_directory_audits_nothing_and_is_not_a_pass(tmp_path):
    zoo = _zoo(tmp_path)
    (tmp_path / "dist").mkdir()
    out = _audit(zoo, tmp_path / "dist")
    assert out.returncode == 2
    assert "nothing audited" in out.stderr


def test_a_dump_with_no_template_is_red(tmp_path):
    zoo = _zoo(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    torch.save(_backbone(), dist / "nobody_weights.pkl")
    out = _audit(zoo, dist)
    assert out.returncode == 1
    assert "unresolved" in out.stderr


def test_every_declaring_template_names_a_head_the_audit_can_match():
    """Plant the check over the population: every real template that declares
    SEED_EXCLUDED_PREFIXES declares non-empty string prefixes, so the audit's
    startswith test means what it says for all of them (an empty or bare-string
    declaration would match everything or the wrong keys)."""
    sys.path.insert(0, str(ROOT / "tools"))
    from seed_index import build_index, read_prefixes

    declaring = 0
    for candidates in build_index(ROOT).values():
        for _, path in candidates:
            prefixes = read_prefixes(path)
            if prefixes is None:
                continue
            declaring += 1
            assert prefixes, f"{path}: empty SEED_EXCLUDED_PREFIXES"
            for p in prefixes:
                assert isinstance(p, str) and len(p) > 1 and p.endswith("."), (path, p)
    assert declaring >= 50, declaring
