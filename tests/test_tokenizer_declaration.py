"""Every NLP template names the tokenizer that travels with it, and the file it
names is one the training engine can load.

WHY A DECLARATION, AND WHY THIS TEST
------------------------------------
The HuggingFace hub is a closed door for training (see tests/conftest.py), so the
tokenizer an NLP model trains with has to be UPLOADED with the model: the SDK
sends it as the ``upload_tokenizer`` part and the engine loads that file as the
federation's single source of truth. Which file that is used to live in two
places at once -- a naming convention (``<model>_tokenizer.json`` beside the
template) and a per-README upload snippet -- and any tool copying a template
had to know both. Twelve templates in this tree follow neither: they rely on a
shared, differently named file in their directory (``tokenizer.json``,
``bert_vocab_tokenizer.json``, ...), which a convention-based resolver never
finds, so those models upload with no tokenizer and stop at training time.

So the template SAYS which file it ships, the way it already says ``framework``
and ``category``::

    tokenizer_file = "simple_text_tokenizer.json"

-- a bare filename, resolved in the template's own directory. Anything that
uploads a zoo template (the SDK, an automation, a person copying the snippet)
reads that one line instead of knowing a convention.

WHAT IS ASSERTED, and where each rule comes from
------------------------------------------------
* every template in an NLP directory declares ``tokenizer_file`` (an NLP
  directory is one that holds a tokenizer JSON -- derived from the tree, not
  from a list of task names);
* the declared value is a bare filename and the file exists beside the template;
* the file is a HuggingFace ``tokenizers`` JSON (``model.vocab`` present), the
  shape the engine's ``load_tokenizer_from_file`` builds a
  ``PreTrainedTokenizerFast`` from;
* its vocabulary carries the special-token ROLES the engine validates before
  training -- a pad token for every task, a mask token for masked language
  modeling -- accepting any of the surface forms the engine accepts (see
  ``_ENGINE_TOKEN_FORMS``);
* its largest token id fits the template's declared vocabulary size, because
  the engine refuses to add a missing special token (that would mint an id past
  the embedding table) and a vocabulary wider than the table is an IndexError
  at training time;
* every tokenizer JSON in an NLP directory is named by at least one template,
  so a file nothing ships cannot sit there looking like coverage.

All of it is stdlib: this test runs in every CI framework job, including the
ones with no torch installed.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "model_zoo"

#: The surface forms the training engine accepts for each special-token ROLE,
#: mirrored from tracebloc-engine ``core/utils/text_utils.py`` (``PAD_TOKEN_FORMS``,
#: ``MASK_TOKEN_FORMS``) at ``4ef9cb3b``. A role is satisfied by ANY one form, so
#: a sentencepiece or BPE tokenizer passes on ``<pad>`` / ``<|endoftext|>`` where
#: the WordPiece literal ``[PAD]`` is absent. Restated here because the engine is
#: not checked out in this repo's CI; the ref is the claim, refresh it when the
#: engine's tuple moves.
_ENGINE_TOKEN_FORMS = {
    "pad": ("[PAD]", "<pad>", "<|endoftext|>", "</s>"),
    "mask": ("[MASK]", "<mask>"),
}

#: Which roles each task's engine strategy requires of a contributor file:
#: every NLP strategy resolves a pad token (``resolve_pad_token_or_raise``);
#: masked language modeling additionally requires a mask token.
_REQUIRED_ROLES = {
    "masked_language_modeling": ("pad", "mask"),
}
_DEFAULT_ROLES = ("pad",)

_DECLARATION = re.compile(r'^\s*tokenizer_file\s*=\s*["\']([^"\']+)["\']\s*$', re.MULTILINE)
_VOCAB_SIZE = re.compile(r"^\s*_?(?:VOCAB_SIZE|vocab_size)\s*=\s*(\d+)\s*$", re.MULTILINE)
_FRAMEWORK = re.compile(r'^\s*framework\s*=\s*["\'](\w+)["\']', re.MULTILINE)


def _nlp_dirs() -> list[pathlib.Path]:
    """Directories that hold at least one tokenizer JSON: the NLP surface,
    derived from the tree rather than from a list of task names."""
    dirs = sorted({p.parent for p in MODEL_ROOT.rglob("*.json")
                   if p.parent != MODEL_ROOT})
    assert dirs, "no tokenizer JSON anywhere under model_zoo/ -- this suite would be vacuous"
    return dirs


def _templates(directory: pathlib.Path) -> list[pathlib.Path]:
    """Model files in one directory: `.py` files declaring a framework."""
    return sorted(p for p in directory.glob("*.py")
                  if _FRAMEWORK.search(p.read_text(encoding="utf-8")))


def _declaration(path: pathlib.Path) -> str | None:
    matches = _DECLARATION.findall(path.read_text(encoding="utf-8"))
    assert len(matches) <= 1, f"{path}: more than one tokenizer_file declaration"
    return matches[0] if matches else None


def _nlp_templates() -> list[pathlib.Path]:
    out = []
    for directory in _nlp_dirs():
        out.extend(_templates(directory))
    assert out, "no template found in any NLP directory"
    return out


def _ids(path: pathlib.Path) -> str:
    return str(path.relative_to(ROOT))


@pytest.mark.parametrize("path", _nlp_templates(), ids=_ids)
def test_every_nlp_template_declares_its_tokenizer(path: pathlib.Path) -> None:
    declared = _declaration(path)
    assert declared, (
        f"{path}: sits in a directory that ships tokenizer files but declares no "
        "`tokenizer_file`. Name the file this model trains with -- a bare filename "
        "in this directory -- the way the template names `framework`.")


@pytest.mark.parametrize("path", _nlp_templates(), ids=_ids)
def test_the_declared_file_exists_beside_the_template(path: pathlib.Path) -> None:
    declared = _declaration(path)
    if declared is None:
        pytest.skip("no declaration; the previous test reports it")
    assert "/" not in declared and "\\" not in declared and not declared.startswith("."), (
        f"{path}: `tokenizer_file` must be a bare filename in the template's own "
        f"directory, got {declared!r}")
    assert declared.endswith(".json"), f"{path}: {declared!r} is not a tokenizers JSON"
    assert (path.parent / declared).is_file(), (
        f"{path}: declares tokenizer_file={declared!r} but {path.parent / declared} "
        "does not exist -- the model would upload without a tokenizer and stop at "
        "training time")


def _tokenizer_json(path: pathlib.Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict) and "model" in document, (
        f"{path}: not a HuggingFace tokenizers file (no top-level `model`)")
    vocab = document["model"].get("vocab")
    assert vocab, f"{path}: the tokenizer model carries no vocabulary"
    return document


def _vocab_ids(document: dict) -> dict[str, int]:
    vocab = document["model"]["vocab"]
    if isinstance(vocab, dict):
        return {str(k): int(v) for k, v in vocab.items()}
    # Unigram models carry [token, score] pairs; the id is the position.
    return {str(entry[0]): i for i, entry in enumerate(vocab)}


@pytest.mark.parametrize("path", _nlp_templates(), ids=_ids)
def test_the_declared_file_carries_the_roles_the_engine_requires(path: pathlib.Path) -> None:
    declared = _declaration(path)
    if declared is None or not (path.parent / declared).is_file():
        pytest.skip("no usable declaration; earlier tests report it")
    document = _tokenizer_json(path.parent / declared)
    vocab = _vocab_ids(document)
    added = {t.get("content") for t in document.get("added_tokens", [])}
    category = path.parent.parent.name
    for role in _REQUIRED_ROLES.get(category, _DEFAULT_ROLES):
        forms = _ENGINE_TOKEN_FORMS[role]
        assert any(f in vocab or f in added for f in forms), (
            f"{path}: {declared} carries none of {forms} for the {role!r} role, and "
            "the engine refuses to add one -- a token minted at training time would "
            "sit past the embedding table")


@pytest.mark.parametrize("path", _nlp_templates(), ids=_ids)
def test_the_vocabulary_fits_the_templates_embedding_table(path: pathlib.Path) -> None:
    declared = _declaration(path)
    if declared is None or not (path.parent / declared).is_file():
        pytest.skip("no usable declaration; earlier tests report it")
    sizes = _VOCAB_SIZE.findall(path.read_text(encoding="utf-8"))
    if not sizes:
        pytest.skip("the template declares no vocab size to compare against "
                    "(a pretrained architecture carries its own)")
    declared_size = int(sizes[0])
    document = _tokenizer_json(path.parent / declared)
    ids = list(_vocab_ids(document).values())
    ids += [int(t["id"]) for t in document.get("added_tokens", []) if "id" in t]
    assert max(ids) < declared_size, (
        f"{path}: {declared} has a token id {max(ids)} but the template sizes its "
        f"embedding table at {declared_size} -- an IndexError at training time")


@pytest.mark.parametrize("directory", _nlp_dirs(), ids=_ids)
def test_every_tokenizer_file_is_named_by_a_template(directory: pathlib.Path) -> None:
    named = {d for d in (_declaration(p) for p in _templates(directory)) if d}
    present = {p.name for p in directory.glob("*.json")}
    orphans = sorted(present - named)
    assert not orphans, (
        f"{directory}: {orphans} are shipped by no template. A tokenizer file "
        "nothing names is either dead or a model's missing declaration.")


def test_the_declaration_reader_sees_what_it_should() -> None:
    """Non-vacuity for the regex: it must read a real declaration and refuse a
    commented-out or malformed one, or every test above passes over `None`."""
    assert _DECLARATION.findall('framework = "pytorch"\ntokenizer_file = "x_tokenizer.json"\n') == ["x_tokenizer.json"]
    assert _DECLARATION.findall("# tokenizer_file = \"x.json\"\n") == []
    assert _DECLARATION.findall("tokenizer_file = None\n") == []
    assert _VOCAB_SIZE.findall("_VOCAB_SIZE = 30522\nvocab_size = 5\n") == ["30522", "5"]


# ---------------------------------------------------------------------------
# The shared bert-vocab copies are verified copies, not generated files
# ---------------------------------------------------------------------------

_BERT_REFERENCE = MODEL_ROOT / "text_classification" / "pytorch" / "bert_base_uncased_tokenizer.json"


def _without_post_processor(document: dict) -> dict:
    return {k: v for k, v in document.items() if k != "post_processor"}


def _bert_vocab_copies() -> list[pathlib.Path]:
    """Every tokenizer JSON that IS the bert-base-uncased tokenizer -- identical
    to the pretrained template's own file in everything but the post-processor
    -- other than that reference file itself. Derived by comparing documents,
    not by listing names; a pretrained template's tokenizer that merely shares
    the vocabulary (its own padding/truncation or model block) is not a copy."""
    reference = _without_post_processor(_tokenizer_json(_BERT_REFERENCE))
    copies = []
    for directory in _nlp_dirs():
        for path in sorted(directory.glob("*.json")):
            if path == _BERT_REFERENCE:
                continue
            document = json.loads(path.read_text(encoding="utf-8"))
            if _without_post_processor(document) == reference:
                copies.append(path)
    assert len(copies) >= 4, (
        f"only {len(copies)} bert-vocab copy found; the assertions below would be vacuous")
    return copies


def _variant(path: pathlib.Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    return "encoder" if document.get("post_processor") else "decoder"


def test_every_copy_is_the_reference_tokenizer_in_everything_but_the_post_processor() -> None:
    """The derivation IS the claim: normalizer, pre-tokenizer, decoder, model
    (the 30522 WordPiece vocabulary), added tokens and the baked padding all
    equal bert-base-uncased's own file. Spelled out per key so a drift names
    the field."""
    reference = _tokenizer_json(_BERT_REFERENCE)
    for path in _bert_vocab_copies():
        document = json.loads(path.read_text(encoding="utf-8"))
        for key in ("normalizer", "pre_tokenizer", "decoder", "model", "added_tokens",
                    "padding", "truncation"):
            assert document.get(key) == reference.get(key), f"{path}: `{key}` differs"
        padding = document.get("padding") or {}
        assert padding.get("pad_id") == 0 and padding.get("pad_token") == "[PAD]", (
            f"{path}: padding is not baked in as [PAD] id 0")


def test_the_copies_come_in_exactly_an_encoder_and_a_decoder_variant() -> None:
    variants = {_variant(p) for p in _bert_vocab_copies()}
    assert variants == {"encoder", "decoder"}, (
        f"expected an encoder-style and a decoder-style variant, found {sorted(variants)}")
    reference_pp = _tokenizer_json(_BERT_REFERENCE).get("post_processor")
    for path in _bert_vocab_copies():
        document = json.loads(path.read_text(encoding="utf-8"))
        if _variant(path) == "encoder":
            assert document["post_processor"] == reference_pp, (
                f"{path}: an encoder-style copy with a post-processor that is not "
                "bert-base-uncased's [CLS] ... [SEP] template")
        else:
            assert document.get("post_processor") is None


def test_every_from_scratch_nlp_template_names_a_copy_or_its_own_pretrained_file() -> None:
    """A template sized to `vocab_size = 30522` names either one of the copies
    above or a pretrained tokenizer with that vocabulary -- never a file whose
    ids do not fit; the fit test above holds the numbers, this holds the set."""
    copies = set(_bert_vocab_copies()) | {_BERT_REFERENCE}
    named = 0
    for path in _nlp_templates():
        text = path.read_text(encoding="utf-8")
        sizes = _VOCAB_SIZE.findall(text)
        declared = _declaration(path)
        if not sizes or declared is None or int(sizes[0]) != 30522:
            continue
        named += 1
        target = path.parent / declared
        document = json.loads(target.read_text(encoding="utf-8"))
        assert _vocab_ids(document) == _vocab_ids(_tokenizer_json(_BERT_REFERENCE)), (
            f"{path}: sized to 30522 but names {declared}, whose vocabulary is not "
            "bert-base-uncased's")
    assert named >= 10, f"only {named} template(s) sized to 30522 -- vacuous"
