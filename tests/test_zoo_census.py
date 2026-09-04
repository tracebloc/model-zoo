"""Tests for ``tests/zoo_census.py`` — the derivation that replaced the census literal.

WHY THIS FILE IS THE POINT (backend#2982)
-----------------------------------------
Replacing a hand-maintained literal with a derivation has exactly one failure
mode, and it is silent: if the derivation counts the same files the tool counts,
by the same rule, then both sides go wrong together and the assertion in
``test_check_dump_coverage.py`` passes while the gate is blind. The old literal
could not fail that way — a human had to type it.

So the derivation earns its place only by DIVERGING from the tool where the tool
is wrong. Every test below is that proof: a synthetic tree the tool mis-reads,
and an assertion that ``zoo_census`` reads it differently. If someone later
rewrites ``zoo_census.census()`` as ``dict(check_dump_coverage.survey(zoo))``,
these fail — which is the guard on the guard.

The three divergences, in the order they matter:

1. ``survey()`` globs ``pytorch/*.py`` one level deep; ``zoo_census`` recurses.
2. ``classify()`` substring-matches the RAW source, so a line wrap splitting
   ``no weight file`` declassifies a template; ``zoo_census`` normalises first.
3. ``classify()`` surveys a file iff it contains ``"Offline variant"``;
   ``zoo_census`` also accepts ``SEED_EXCLUDED_PREFIXES``, which is code rather
   than prose and therefore survives a docstring edit.
"""

import importlib.util
import pathlib
import sys

import zoo_census

ROOT = pathlib.Path(__file__).parent.parent
TOOL = ROOT / "tools" / "check_dump_coverage.py"

SEED_EXPECTING = '''\
"""A model.

Offline variant: built from the inlined config. Upload the matched
``{stem}_weights.pkl`` via ``weights=True``.
"""
framework = "pytorch"
'''

NO_SEED = '''\
"""A model.

Offline variant: a scratch template random-initializes by design, so there is
no weight file: upload with ``weights=False``.
"""
framework = "pytorch"
'''

UNMIGRATED = '''\
"""A model that still fetches from the hub."""
framework = "pytorch"
'''

#: The `tft.py` shape: an unmigrated from-scratch template whose CODE contains a
#: keyword argument ending in `weights=False`. A bare substring test enrols it in
#: the census and reddens the real tree for nothing.
UNMIGRATED_WITH_NEED_WEIGHTS = '''\
"""A from-scratch model. Nothing offline about it, nothing hosted for it."""
framework = "pytorch"


def forward(self, h):
    out, _ = self.attn(h, h, h, need_weights=False)
    return out
'''


def _tool():
    spec = importlib.util.spec_from_file_location("check_dump_coverage", TOOL)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(ROOT / "tools"))
    return mod


def _zoo(tmp_path, templates):
    """``templates`` is ``{"<category>/<path under pytorch>": <source>}``.

    The key may carry directory separators, which is how the nested-template
    cases below are built.
    """
    root = tmp_path / "model_zoo"
    for key, source in templates.items():
        category, relative = key.split("/", 1)
        path = root / category / "pytorch" / f"{relative}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source.format(stem=pathlib.Path(relative).name), "utf-8")
    return tmp_path


# --- the derivation agrees with the tool where the tool is right ------------


def test_the_two_sides_agree_on_a_plain_tree(tmp_path):
    """The baseline. Divergence has to be EARNED by a defect, not constant."""
    zoo = _zoo(
        tmp_path,
        {
            "object_detection/faster_rcnn": SEED_EXPECTING,
            "object_detection/ssdlite": NO_SEED,
            "image_classification/simple_cnn": UNMIGRATED,
        },
    )
    surveyed = {k: v["status"] for k, v in _tool().survey(zoo).items()}
    assert zoo_census.census(zoo) == surveyed == {
        "object_detection/faster_rcnn": "EXPECTS_SEED",
        "object_detection/ssdlite": "NO_SEED",
    }


def test_the_derivation_discriminates_rather_than_accepting_everything(tmp_path):
    """A predicate that returns True for every file would make the census
    assertion pass by counting the whole tree. Assert that it says NO to
    something it visited."""
    zoo = _zoo(
        tmp_path,
        {
            "object_detection/faster_rcnn": SEED_EXPECTING,
            "image_classification/simple_cnn": UNMIGRATED,
        },
    )
    visited = zoo_census.template_files(zoo)
    assert len(visited) == 2
    assert set(zoo_census.census(zoo)) == {"object_detection/faster_rcnn"}


def test_the_real_zoo_census_is_not_empty():
    """An enumeration that found nothing would make the set comparison in
    test_check_dump_coverage.py vacuously true against an equally broken tool.
    The tool fail-closes on an empty survey; this is the same refusal here."""
    census = zoo_census.census(ROOT)
    assert census, "zoo_census found no migrated templates in the real zoo"
    visited = zoo_census.template_files(ROOT)
    assert 0 < len(census) < len(visited), (
        f"census {len(census)} of {len(visited)} visited files — a census equal "
        f"to the whole tree means the footprint test stopped discriminating"
    )


# --- divergence 1: the tool globs one level, this recurses -------------------


def test_a_migrated_template_in_a_subdirectory_is_seen_here_and_not_by_the_tool(tmp_path):
    """`survey()` does `(category / "pytorch").glob("*.py")` — ONE level. The
    tree already has `object_detection/pytorch/yolo_v1/model.py`, so a migrated
    template one directory down is not hypothetical, and the old literal could
    never have caught it: the count simply never moved."""
    zoo = _zoo(tmp_path, {"object_detection/centernet/model": SEED_EXPECTING})
    assert _tool().survey(zoo) == {}, "the tool's glob got deeper on its own"
    assert set(zoo_census.census(zoo)) == {"object_detection/centernet/model"}


def test_the_nested_key_is_one_the_tool_cannot_emit(tmp_path):
    """The keys have to differ, or a nested template would silently satisfy the
    set comparison by colliding with its flat namesake."""
    zoo = _zoo(
        tmp_path,
        {
            "object_detection/model": SEED_EXPECTING,
            "object_detection/centernet/model": SEED_EXPECTING,
        },
    )
    assert set(zoo_census.census(zoo)) == {
        "object_detection/model",
        "object_detection/centernet/model",
    }
    assert set(_tool().survey(zoo)) == {"object_detection/model"}


# --- divergence 2: whitespace ----------------------------------------------

WRAPPED_NO_SEED = '''\
"""A model.

Offline variant: a scratch template random-initializes by design, so there is no
weight file: upload with ``weights=False``.
"""
framework = "pytorch"
'''


def test_a_line_wrap_inside_no_weight_file_declassifies_only_the_tool(tmp_path):
    """THE TRAP, as a test. `classify()` needs BOTH `weights=False` and the
    literal `no weight file`; a wrap between "no" and "weight" breaks the second,
    so a NO_SEED template silently becomes UNCLASSIFIED — or, if it also names a
    dump, flips status. `zoo_census` normalises whitespace first, keeps the
    NO_SEED verdict, and the two sides then disagree, which is the red."""
    zoo = _zoo(tmp_path, {"object_detection/ssdlite": WRAPPED_NO_SEED})
    tool = _tool()
    surveyed = {k: v["status"] for k, v in tool.survey(zoo).items()}
    assert surveyed == {"object_detection/ssdlite": tool.UNCLASSIFIED}
    assert zoo_census.census(zoo) == {"object_detection/ssdlite": zoo_census.NO_SEED}


WRAPPED_MARKER = '''\
"""A model.

Something something offline
variant: built from the inlined config. Upload the matched
``{stem}_weights.pkl`` via ``weights=True``.
"""
framework = "pytorch"
'''


def test_a_line_wrap_inside_the_marker_itself_still_lands_in_the_census(tmp_path):
    """`"Offline variant"` split across lines removes a template from the survey
    entirely. The old literal caught that as a count drop; this catches it by
    name, because the dump the template names is a second footprint anyway."""
    zoo = _zoo(tmp_path, {"text_classification/bert": WRAPPED_MARKER})
    assert _tool().survey(zoo) == {}
    assert set(zoo_census.census(zoo)) == {"text_classification/bert"}


# --- divergence 3: footprints beyond the docstring --------------------------

SEED_CONSTANT_ONLY = '''\
"""A model. The migration paragraph that used to be here has been deleted."""
framework = "pytorch"

SEED_EXCLUDED_PREFIXES = ("classifier.",)
'''


def test_deleting_the_docstring_paragraph_does_not_hide_a_seed_declaring_template(tmp_path):
    """The direction the old literal bought and a naive derivation would lose.

    `SEED_EXCLUDED_PREFIXES` is written by `tools/seed_contract.py apply`, not by
    an author's prose, so it survives a docstring rewrite. 53 of the 61 migrated
    templates carry it; for those, un-migrating by deleting the paragraph is
    still red. See zoo_census's docstring for the eight that do not carry it.
    """
    zoo = _zoo(tmp_path, {"text_classification/bert": SEED_CONSTANT_ONLY})
    assert _tool().survey(zoo) == {}
    census = zoo_census.census(zoo)
    assert set(census) == {"text_classification/bert"}
    assert census["text_classification/bert"] == zoo_census.UNCLASSIFIED


def test_the_footprint_is_named_so_the_failure_says_what_to_fix(tmp_path):
    zoo = _zoo(tmp_path, {"text_classification/bert": SEED_CONSTANT_ONLY})
    assert zoo_census.why(zoo)["text_classification/bert"] == [
        "declares SEED_EXCLUDED_PREFIXES"
    ]


# --- the false positive that would redden the real tree ---------------------


def test_a_need_weights_kwarg_does_not_enrol_an_unmigrated_template(tmp_path):
    """`time_series_forecasting/tft.py` calls `self.attn(..., need_weights=False)`.
    A bare `weights=false` substring test matches it, which would put a
    from-scratch template in the census and fail the real tree. The lookbehind in
    `UPLOAD_KWARG` is what stops that, and this is what stops the lookbehind
    being removed as noise."""
    zoo = _zoo(
        tmp_path,
        {"time_series_forecasting/tft": UNMIGRATED_WITH_NEED_WEIGHTS},
    )
    assert zoo_census.census(zoo) == {}
    assert _tool().survey(zoo) == {}


def test_an_upload_instruction_with_spaces_around_the_equals_still_counts(tmp_path):
    """`weights = False` in prose is the same instruction. Normalisation is
    one-space, not zero-space, so the kwarg pattern has to tolerate the space
    itself rather than relying on it being gone."""
    source = '''\
"""A model.

Offline variant. It random-initialises, there is no weight file: upload with
``weights = False``.
"""
framework = "pytorch"
'''
    zoo = _zoo(tmp_path, {"object_detection/ssdlite": source})
    assert set(zoo_census.census(zoo)) == {"object_detection/ssdlite"}
    assert zoo_census.footprints(source, "ssdlite") == [
        "docstring says 'offline variant'",
        "says 'no weight file'",
        "gives a weights= upload instruction",
    ]


# --- the status derivation is the tool's tri-state, not a bool ---------------


def test_a_template_claiming_both_is_unclassified_here_too(tmp_path):
    source = '''\
"""A model.

Offline variant. Upload the matched ``{stem}_weights.pkl`` — but also there is
no weight file, upload with ``weights=False``.
"""
framework = "pytorch"
'''
    zoo = _zoo(tmp_path, {"object_detection/twinned": source})
    assert zoo_census.census(zoo) == {
        "object_detection/twinned": zoo_census.UNCLASSIFIED
    }


def test_a_migrated_template_that_says_nothing_is_unclassified_here_too(tmp_path):
    source = '''\
"""A model.

Offline variant: built from the inlined config, so it constructs anywhere.
"""
framework = "pytorch"
'''
    zoo = _zoo(tmp_path, {"object_detection/silent": source})
    assert zoo_census.census(zoo) == {
        "object_detection/silent": zoo_census.UNCLASSIFIED
    }


# --- PROSE_ONLY: the one literal left, and every way it can be wrong ---------
#
# The derivation alone lost one direction the old census literal had: a template
# whose ONLY footprint is its docstring drops out of BOTH sides when that
# paragraph is deleted. That mutation SURVIVED the first version of this module —
# measured on `image_classification/vit`, where the old literal went 61 -> 60 and
# the derivation stayed green. `zoo_census.PROSE_ONLY` closes it, and the four
# tests below are what stop the list itself becoming the next rotting census.


def test_every_prose_only_template_is_still_surveyed():
    """THE RESTORED DIRECTION. These eight have no `SEED_EXCLUDED_PREFIXES`, so
    deleting their migration paragraph would otherwise remove them from the
    census and from the tool at the same time, silently."""
    tool = _tool()
    surveyed = set(tool.survey(ROOT))
    census = zoo_census.census(ROOT)
    gone = [k for k in zoo_census.PROSE_ONLY if k not in surveyed or k not in census]
    assert not gone, (
        "template(s) listed in zoo_census.PROSE_ONLY are no longer surveyed by "
        "tools/check_dump_coverage.py. Their docstring is the only record that "
        "they were migrated, so an edit to it removes them from every gate at "
        f"once — restore the `Offline variant` paragraph: {gone}"
    )


def test_the_prose_only_list_names_real_files():
    """An entry for a file that does not exist protects nothing, and would keep
    protecting nothing after a rename."""
    missing = [
        k for k in zoo_census.PROSE_ONLY if not zoo_census.path_for(ROOT, k).is_file()
    ]
    assert not missing, (
        "zoo_census.PROSE_ONLY names template(s) that do not exist — if they were "
        f"renamed, rename the entries; if deleted, drop them: {missing}"
    )


def test_the_prose_only_list_holds_nothing_that_has_a_code_footprint():
    """Self-policing downward. Once a template gains `SEED_EXCLUDED_PREFIXES` the
    derivation covers it and the entry is dead weight — dead weight in a list
    like this is how the old census rotted."""
    unnecessary = [
        k
        for k in zoo_census.PROSE_ONLY
        if zoo_census.path_for(ROOT, k).is_file()
        and zoo_census.has_code_footprint(
            zoo_census.path_for(ROOT, k).read_text(encoding="utf-8")
        )
    ]
    assert not unnecessary, (
        "zoo_census.PROSE_ONLY entries whose template now declares "
        "SEED_EXCLUDED_PREFIXES — the derivation covers those without a literal. "
        f"Delete these entries: {unnecessary}"
    )


def test_the_prose_only_list_is_complete():
    """Self-policing upward, and the reason this list cannot silently fall behind
    the tree the way the census literal did.

    A NEW HEADLESS TEMPLATE fails here, by name, in its own PR. Every template
    with a task head — every detector on backend#2982's roster included — carries
    `SEED_EXCLUDED_PREFIXES` and never reaches this list.
    """
    candidates = zoo_census.prose_only_candidates(ROOT)
    unlisted = sorted(set(candidates) - set(zoo_census.PROSE_ONLY))
    assert not unlisted, (
        "migrated template(s) whose only footprint is their docstring, and which "
        "no tripwire covers: deleting that paragraph would remove them from the "
        "census and from tools/check_dump_coverage.py together. Add them to "
        f"zoo_census.PROSE_ONLY: {unlisted}"
    )


def test_the_prose_only_candidate_scan_discriminates(tmp_path):
    """Guard the guard above: a candidate scan that returned everything would
    demand the whole zoo be listed, and one that returned nothing would make the
    completeness test pass while the list decayed to noise."""
    zoo = _zoo(
        tmp_path,
        {
            "text_classification/headless": SEED_EXPECTING,
            "text_classification/with_head": SEED_EXPECTING
            + '\nSEED_EXCLUDED_PREFIXES = ("classifier.",)\n',
            "image_classification/simple_cnn": UNMIGRATED,
        },
    )
    assert zoo_census.prose_only_candidates(zoo) == ["text_classification/headless"]
