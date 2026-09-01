"""Tests for tools/check_dump_coverage.py — the complement gate (backend#2659).

The gate exists because four separate checks all asked "are the dumps we have
good?" and none asked "do we have a dump for every template that needs one?", so
a migrated-but-never-prepped template passed all four BY BEING ABSENT. On the
real zoo it found two: ``object_detection/faster_rcnn_resnet`` and
``text_classification/bert_base_uncased``.

A gate whose value is going red therefore has to be tested for going red. Every
verdict here is exercised against throwaway synthetic templates in a temp dir —
no torch, no transformers, because the classification is pure text and the
coverage arithmetic is pure set logic.

The tests that matter most are the UNCLASSIFIED ones. They are what makes it
safe to derive "expects a seed" from a docstring: prose rots, and the gate must
name a template that has gone silent rather than guess on its behalf.
"""

import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent
TOOL = ROOT / "tools" / "check_dump_coverage.py"

SEED_EXPECTING = '''\
"""A model.

Offline variant: built from the inlined config. The pretrained tensors are
delivered from the tracebloc model store as the training seed: upload the
matched ``{stem}_weights.pkl`` via ``weights=True``.
"""
framework = "pytorch"
'''

NO_SEED = '''\
"""A model.

Offline variant: built from the inlined config. A scratch template
random-initializes by design, so there is no weight file: upload with
``weights=False``.
"""
framework = "pytorch"
'''

SILENT = '''\
"""A model.

Offline variant: built from the inlined config, so it constructs anywhere.
"""
framework = "pytorch"
'''

CONTRADICTORY = '''\
"""A model.

Offline variant. Upload the matched ``{stem}_weights.pkl`` — but also there is
no weight file, upload with ``weights=False``.
"""
framework = "pytorch"
'''

UNMIGRATED = '''\
"""A model that still fetches from the hub."""
framework = "pytorch"
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
    """``templates`` is ``{"<category>/<stem>": <source>}``."""
    root = tmp_path / "model_zoo"
    for key, source in templates.items():
        category, stem = key.split("/")
        directory = root / category / "pytorch"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{stem}.py").write_text(source.format(stem=stem), "utf-8")
    return tmp_path


def _dumps(tmp_path, names):
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    for name in names:
        (weights / name).mkdir(exist_ok=True)
    return weights


# --- classification ---------------------------------------------------------

def test_a_seed_expecting_template_is_recognised(tmp_path):
    tool = _tool()
    zoo = _zoo(tmp_path, {"image_classification/resnet": SEED_EXPECTING})
    found = tool.survey(zoo)
    assert found["image_classification/resnet"]["status"] == tool.EXPECTS_SEED


def test_a_declared_no_seed_template_is_recognised(tmp_path):
    tool = _tool()
    zoo = _zoo(tmp_path, {"text_classification/scratch": NO_SEED})
    found = tool.survey(zoo)
    assert found["text_classification/scratch"]["status"] == tool.NO_SEED


def test_an_unmigrated_template_is_not_surveyed(tmp_path):
    """The gate is scoped to #1499's migration; it must not indict the rest."""
    tool = _tool()
    zoo = _zoo(tmp_path, {"image_classification/legacy": UNMIGRATED})
    assert tool.survey(zoo) == {}


def test_a_migrated_template_that_says_nothing_is_unclassified(tmp_path):
    """THE LOAD-BEARING TEST. A docstring edit that drops the seed sentence must
    turn the gate red by name — not silently reclassify the template as needing
    no dump, which is the exact shape of the bug this gate closes."""
    tool = _tool()
    zoo = _zoo(tmp_path, {"object_detection/silent": SILENT})
    found = tool.survey(zoo)
    assert found["object_detection/silent"]["status"] == tool.UNCLASSIFIED
    assert "says nothing" in found["object_detection/silent"]["detail"]


def test_a_template_claiming_both_is_unclassified(tmp_path):
    tool = _tool()
    zoo = _zoo(tmp_path, {"object_detection/both": CONTRADICTORY})
    found = tool.survey(zoo)
    assert found["object_detection/both"]["status"] == tool.UNCLASSIFIED
    assert "stale" in found["object_detection/both"]["detail"]


def test_an_unclassified_template_alone_fails_the_gate(tmp_path):
    zoo = _zoo(tmp_path, {"object_detection/silent": SILENT})
    assert _tool().main(["--zoo", str(zoo)]) == 1


# --- coverage arithmetic ----------------------------------------------------

def test_a_seed_expecting_template_with_no_dump_fails(tmp_path):
    """The `faster_rcnn_resnet` case, reduced."""
    zoo = _zoo(tmp_path, {"object_detection/faster_rcnn_resnet": SEED_EXPECTING})
    dumps = _dumps(tmp_path, [])
    assert _tool().main(["--zoo", str(zoo), "--dumps-dir", str(dumps)]) == 1


def test_a_seed_expecting_template_with_its_dump_passes(tmp_path):
    zoo = _zoo(tmp_path, {"object_detection/fcos": SEED_EXPECTING})
    dumps = _dumps(tmp_path, ["fcos"])
    assert _tool().main(["--zoo", str(zoo), "--dumps-dir", str(dumps)]) == 0


def test_a_declared_no_seed_template_needs_no_dump(tmp_path):
    zoo = _zoo(tmp_path, {"text_classification/scratch": NO_SEED})
    assert _tool().main(["--zoo", str(zoo), "--dumps-dir", str(_dumps(tmp_path, []))]) == 0


def test_an_orphaned_dump_fails(tmp_path):
    """A rename that strands its dump, or a dump for an un-migrated template.
    Fail-closed in BOTH directions or the gate only half-works."""
    zoo = _zoo(tmp_path, {"object_detection/fcos": SEED_EXPECTING})
    dumps = _dumps(tmp_path, ["fcos", "renamed_away"])
    assert _tool().main(["--zoo", str(zoo), "--dumps-dir", str(dumps)]) == 1


def test_a_colliding_stem_resolves_to_its_prefixed_dump(tmp_path):
    """``bert_base_uncased`` ships in two categories with a dump apiece; the
    sentence-pair one is filed under a category prefix. Reversing
    ``seed_index``'s map wrongly would report a false NO DUMP for one and a
    false ORPHAN for the other — which is how the real gap read at first."""
    zoo = _zoo(
        tmp_path,
        {
            "text_classification/bert_base_uncased": SEED_EXPECTING,
            "sentence_pair_classification/bert_base_uncased": SEED_EXPECTING,
        },
    )
    dumps = _dumps(tmp_path, ["bert_base_uncased", "sentence_pair_bert_base_uncased"])
    assert _tool().main(["--zoo", str(zoo), "--dumps-dir", str(dumps)]) == 0


def test_the_sentence_pair_dump_alone_does_not_cover_the_text_one(tmp_path):
    """The SECOND real gap this gate found. Both templates expect their own
    seed, so one shared dump is not coverage."""
    zoo = _zoo(
        tmp_path,
        {
            "text_classification/bert_base_uncased": SEED_EXPECTING,
            "sentence_pair_classification/bert_base_uncased": SEED_EXPECTING,
        },
    )
    dumps = _dumps(tmp_path, ["sentence_pair_bert_base_uncased"])
    assert _tool().main(["--zoo", str(zoo), "--dumps-dir", str(dumps)]) == 1


# --- manifest as the inventory ---------------------------------------------

def test_a_manifest_counts_as_the_inventory(tmp_path):
    zoo = _zoo(tmp_path, {"object_detection/fcos": SEED_EXPECTING})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": 2, "dumps": [{"name": "fcos"}]}), "utf-8")
    assert _tool().main(["--zoo", str(zoo), "--manifest", str(manifest)]) == 0


def test_a_manifest_missing_an_entry_fails(tmp_path):
    zoo = _zoo(tmp_path, {"object_detection/fcos": SEED_EXPECTING})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": 2, "dumps": []}), "utf-8")
    assert _tool().main(["--zoo", str(zoo), "--manifest", str(manifest)]) == 1


# --- the armed no-op, and its off switch ----------------------------------

def test_without_an_inventory_the_gate_reports_classification_only(tmp_path):
    """Mirrors verify_dumps_against_engine_pin: green-with-no-work while nothing
    is hosted yet (backend#2659), rather than red on a state nobody can fix."""
    zoo = _zoo(tmp_path, {"object_detection/fcos": SEED_EXPECTING})
    assert _tool().main(["--zoo", str(zoo)]) == 0


def test_require_dumps_makes_the_missing_inventory_itself_red(tmp_path):
    zoo = _zoo(tmp_path, {"object_detection/fcos": SEED_EXPECTING})
    assert _tool().main(["--zoo", str(zoo), "--require-dumps"]) == 1


def test_a_bad_dumps_dir_is_an_error_not_a_green(tmp_path):
    zoo = _zoo(tmp_path, {"object_detection/fcos": SEED_EXPECTING})
    assert _tool().main(["--zoo", str(zoo), "--dumps-dir", str(tmp_path / "nope")]) == 1


def test_an_empty_zoo_is_an_error_not_a_green(tmp_path):
    """A --zoo typo must not read as "nothing to check, all good"."""
    (tmp_path / "model_zoo").mkdir()
    assert _tool().main(["--zoo", str(tmp_path)]) == 1


# --- against the real zoo --------------------------------------------------

def test_the_real_zoo_classifies_every_migrated_template(tmp_path):
    """No template in the shipped zoo may be UNCLASSIFIED. This is the test that
    keeps the derivation honest as templates are edited."""
    tool = _tool()
    found = tool.survey(ROOT)
    unclassified = {k: v["detail"] for k, v in found.items()
                    if v["status"] == tool.UNCLASSIFIED}
    assert not unclassified, f"unclassified migrated templates: {unclassified}"
    # 56 = #1499's 57 minus mask_rcnn, deleted as unusable in backend#2988.
    assert len(found) == 56, f"expected 56 migrated templates, got {len(found)}"


def test_the_real_zoo_writes_a_survey(tmp_path):
    out = tmp_path / "survey.json"
    assert _tool().main(["--zoo", str(ROOT), "--out", str(out)]) == 0
    survey = json.loads(out.read_text("utf-8"))
    assert survey["inventory_checked"] is False
    assert len(survey["templates"]) == 56


# --- unmapped collisions ----------------------------------------------------
#
# `seed_index.resolve` already settled this question in the forward direction:
# AMBIGUITY IS AN ERROR, NOT A PICK — it raises rather than `setdefault`-ing,
# because a tool that strips head keys using the wrong template's declaration
# produces a seed that is wrong in a way nobody can see.
#
# The reverse mapping here quietly chose "pick". `DUMP_DIR_CATEGORY.get(stem,
# category) == category` is ALWAYS TRUE for a stem absent from that map, so two
# seed-expecting templates sharing an unmapped stem both claimed the same bare
# dump and coverage went green with one of them unseeded (Bugbot) — the exact
# gap this gate exists to close, reintroduced inside the gate.

def test_two_unmapped_colliding_templates_do_not_share_one_dump(tmp_path):
    """THE REGRESSION. One bare dump, two seed-expecting templates, nothing in
    seed_index to tell them apart: this must be red, not green."""
    zoo = _zoo(
        tmp_path,
        {
            "object_detection/twinned": SEED_EXPECTING,
            "semantic_segmentation/twinned": SEED_EXPECTING,
        },
    )
    dumps = _dumps(tmp_path, ["twinned"])
    assert _tool().main(["--zoo", str(zoo), "--dumps-dir", str(dumps)]) == 1


def test_an_unmapped_collision_is_named_as_ambiguous_not_as_a_missing_dump(tmp_path):
    """A distinct status, and the message carries the fix. "NO DUMP" would be a
    lie — the dump may exist; what is missing is a rule saying whose it is."""
    tool = _tool()
    zoo = _zoo(
        tmp_path,
        {
            "object_detection/twinned": SEED_EXPECTING,
            "semantic_segmentation/twinned": SEED_EXPECTING,
        },
    )
    found = tool.survey(zoo)
    assert found["object_detection/twinned"]["status"] == tool.AMBIGUOUS
    assert found["semantic_segmentation/twinned"]["status"] == tool.AMBIGUOUS
    assert "seed_index" in found["object_detection/twinned"]["detail"]


def test_an_unmapped_collision_is_red_even_with_no_inventory(tmp_path):
    """The classification half must catch it too — a collision is a defect in
    the naming rules, and does not need any dump to exist to be one."""
    zoo = _zoo(
        tmp_path,
        {
            "object_detection/twinned": SEED_EXPECTING,
            "semantic_segmentation/twinned": SEED_EXPECTING,
        },
    )
    assert _tool().main(["--zoo", str(zoo)]) == 1


def test_a_stem_shared_with_a_NO_SEED_template_is_not_a_collision(tmp_path):
    """Only seed-expecting templates compete for a dump. A scratch sibling that
    declares it needs none must not turn its neighbour ambiguous."""
    tool = _tool()
    zoo = _zoo(
        tmp_path,
        {
            "object_detection/twinned": SEED_EXPECTING,
            "text_classification/twinned": NO_SEED,
        },
    )
    found = tool.survey(zoo)
    assert found["object_detection/twinned"]["status"] == tool.EXPECTS_SEED
    assert found["object_detection/twinned"]["candidates"] == ["twinned"]
    assert _tool().main(["--zoo", str(zoo), "--dumps-dir", str(_dumps(tmp_path, ["twinned"]))]) == 0


def test_the_mapped_collision_still_resolves_both_ways(tmp_path):
    """The real `bert_base_uncased` case must be unaffected: one side files
    under its category prefix, the other owns the bare name."""
    tool = _tool()
    zoo = _zoo(
        tmp_path,
        {
            "text_classification/bert_base_uncased": SEED_EXPECTING,
            "sentence_pair_classification/bert_base_uncased": SEED_EXPECTING,
        },
    )
    found = tool.survey(zoo)
    assert found["text_classification/bert_base_uncased"]["candidates"] == [
        "bert_base_uncased"
    ]
    assert found["sentence_pair_classification/bert_base_uncased"]["candidates"] == [
        "sentence_pair_bert_base_uncased"
    ]


def test_the_real_zoo_has_no_ambiguous_template(tmp_path):
    """Today's only seed-expecting collision (`bert_base_uncased`) is fully
    disambiguated, so this bug was latent rather than firing. This test is what
    makes the NEXT collision loud instead of silent."""
    tool = _tool()
    found = tool.survey(ROOT)
    ambiguous = {k: v["detail"] for k, v in found.items()
                 if v["status"] == tool.AMBIGUOUS}
    assert not ambiguous, f"ambiguous templates: {ambiguous}"


# --- the prefix applies to collisions, not to a whole category --------------
#
# `seed_index` states the convention outright: "Dump directories are named
# `<stem>` except where a stem collides, in which case the category is
# prefixed." An earlier revision returned the category prefix for EVERY template
# in a prefixed category, so a unique-stem template in
# `sentence_pair_classification` resolved to a name nothing is filed under
# (Bugbot). That direction fails LOUD — a false "NO DUMP" — unlike the collision
# bug above, which failed silent. Both are worth closing; a gate that cries wolf
# gets switched off.

def test_a_unique_stem_in_a_prefixed_category_uses_the_bare_name(tmp_path):
    """THE REGRESSION. Only one `solo` template exists, so its dump is filed
    bare — the category prefix must not be imposed on it."""
    zoo = _zoo(tmp_path, {"sentence_pair_classification/solo": SEED_EXPECTING})
    dumps = _dumps(tmp_path, ["solo"])
    assert _tool().main(["--zoo", str(zoo), "--dumps-dir", str(dumps)]) == 0


def test_a_unique_stem_also_accepts_the_prefixed_form(tmp_path):
    """A prefixed dump for a unique stem still resolves forward through
    `seed_index`, so it is findable rather than missing."""
    zoo = _zoo(tmp_path, {"sentence_pair_classification/solo": SEED_EXPECTING})
    dumps = _dumps(tmp_path, ["sentence_pair_solo"])
    assert _tool().main(["--zoo", str(zoo), "--dumps-dir", str(dumps)]) == 0


def test_the_bare_name_is_preferred_for_a_unique_stem(tmp_path):
    """Order matters for orphan detection: the convention's name comes first."""
    tool = _tool()
    zoo = _zoo(tmp_path, {"sentence_pair_classification/solo": SEED_EXPECTING})
    candidates = tool.survey(zoo)["sentence_pair_classification/solo"]["candidates"]
    assert candidates[0] == "solo"
    assert "sentence_pair_solo" in candidates


def test_a_colliding_stem_in_the_prefixed_category_still_uses_the_prefix(tmp_path):
    """The collision case is unchanged — this is the real `bert_base_uncased`,
    and the prefix is exactly what disambiguates it."""
    tool = _tool()
    zoo = _zoo(
        tmp_path,
        {
            "text_classification/bert_base_uncased": SEED_EXPECTING,
            "sentence_pair_classification/bert_base_uncased": SEED_EXPECTING,
        },
    )
    found = tool.survey(zoo)
    assert found["sentence_pair_classification/bert_base_uncased"]["candidates"] == [
        "sentence_pair_bert_base_uncased"
    ]
    assert found["text_classification/bert_base_uncased"]["candidates"] == [
        "bert_base_uncased"
    ]
