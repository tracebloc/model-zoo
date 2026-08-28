#!/usr/bin/env python3
"""Resolve a dump directory to the ONE template it belongs to.

WHY THIS IS ITS OWN MODULE. A template stem is not unique: `bert_base_uncased`
ships in BOTH `text_classification` and `sentence_pair_classification`, with a
dump apiece. Indexing by basename means one silently wins and both dumps resolve
to the same file — which `derive_seed_excluded.py` already avoided by keying on
`category/stem`, and which `seed_contract.py` and `verify_backbone_seeds.py`
then reintroduced by keying on the stem (Bugbot, model-zoo#217).

Fixing it in two places separately would leave a third copy free to drift, so the
resolution lives here once and both tools import it.

AMBIGUITY IS AN ERROR, NOT A PICK. The old code called `.setdefault()`, so a
collision resolved to whichever category sorted first and said nothing. A tool
that strips head keys from the wrong template's declaration produces a seed that
is wrong in a way nobody can see, so an unresolvable stem raises instead.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CONSTANT = "SEED_EXCLUDED_PREFIXES"

#: Dump directories are named `<stem>` except where a stem collides, in which
#: case the category is prefixed. Kept as an explicit map rather than inferred:
#: a rule like "strip up to the last underscore" would corrupt every stem that
#: merely contains one.
DUMP_DIR_CATEGORY_PREFIXES = {
    "sentence_pair_": "sentence_pair_classification",
}

#: The other half of a collision: when one of the colliding dumps is named for
#: the BARE stem, nothing in the name says which category it is.
#:
#: THIS IS A MAP AND NOT A DEFAULT ON PURPOSE. "Bare name means whichever
#: category has no prefix" would resolve today and rot the moment a third
#: `bert_base_uncased` appears -- silently, in a tool whose whole job is to pick
#: the right template. An entry here is one line, and a missing one raises with
#: the fix in the message.
DUMP_DIR_CATEGORY = {
    "bert_base_uncased": "text_classification",
}


#: `tests/test_model_contract.py` is the SOURCE OF TRUTH for which templates are
#: too big to build on a CI runner (`gemma_2`'s fp32 tensors alone are ~10.5 GB
#: on a ~16 GB runner). `verify_dumps_against_engine_pin.py` already mirrors it
#: by hand; rather than add a THIRD copy, read the real one.
_CONTRACT_TEST = "tests/test_model_contract.py"
_RAM_SKIP_CONSTANT = "_TOO_LARGE_FOR_CI_RAM"


class AmbiguousTemplate(RuntimeError):
    """A dump directory maps to more than one template, with no disambiguator."""


def ci_ram_skips(repo: Path) -> frozenset:
    """Templates that must not be BUILT in CI, as `<category>/<fw>/<name>.py`.

    Parsed out of the test module with `ast` — importing it would drag in pytest
    and execute collection to read one set literal. Returns an empty set if the
    constant cannot be found, and the caller treats that as "skip nothing":
    a build that OOMs is loud, whereas silently skipping everything would be a
    green sweep that checked nothing.
    """
    path = repo / _CONTRACT_TEST
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except OSError:
        return frozenset()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == _RAM_SKIP_CONSTANT:
                    try:
                        return frozenset(ast.literal_eval(node.value))
                    except ValueError:
                        return frozenset()
    return frozenset()


def read_prefixes(path: Path) -> Optional[Tuple[str, ...]]:
    """The template's declared head prefixes, or None if it declares none.

    Read with `ast` rather than by importing: reading a module-level tuple does
    not need a model, and every template here builds a large one on import.
    Parsing also cannot execute a template as a side effect of asking it a
    question.
    """
    for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == CONSTANT:
                    return tuple(ast.literal_eval(node.value))
    return None


def zoo_root(zoo: Path) -> Path:
    return zoo / "model_zoo" if (zoo / "model_zoo").is_dir() else zoo


def build_index(zoo: Path) -> Dict[str, List[Tuple[str, Path]]]:
    """`{stem: [(category, path), ...]}` — every category a stem appears in."""
    root = zoo_root(zoo)
    index: Dict[str, List[Tuple[str, Path]]] = {}
    for category in sorted(p.name for p in root.iterdir() if p.is_dir()):
        pytorch = root / category / "pytorch"
        if not pytorch.is_dir():
            continue
        for path in sorted(pytorch.glob("*.py")):
            index.setdefault(path.stem, []).append((category, path))
    return index


def resolve(
    index: Dict[str, List[Tuple[str, Path]]], dump_dir: str
) -> Tuple[str, Path]:
    """The `(category, path)` a dump directory belongs to.

    Raises `AmbiguousTemplate` rather than guessing — see the module docstring.
    """
    stem, category = dump_dir, DUMP_DIR_CATEGORY.get(dump_dir)
    for prefix, prefixed_category in DUMP_DIR_CATEGORY_PREFIXES.items():
        if dump_dir.startswith(prefix):
            stem, category = dump_dir[len(prefix) :], prefixed_category
            break

    candidates = index.get(stem, [])
    if not candidates:
        raise AmbiguousTemplate(f"{dump_dir}: no template named {stem}.py in the zoo")

    if category is not None:
        for found_category, path in candidates:
            if found_category == category:
                return found_category, path
        raise AmbiguousTemplate(
            f"{dump_dir}: names category {category}, but {stem}.py exists only in "
            f"{[c for c, _ in candidates]}"
        )

    if len(candidates) == 1:
        return candidates[0]

    raise AmbiguousTemplate(
        f"{dump_dir}: {stem}.py exists in {[c for c, _ in candidates]} and the dump "
        f"directory names no category. Add a DUMP_DIR_CATEGORY_PREFIXES entry, or "
        f"rename the dump directory to `<category-prefix>_{stem}`."
    )
