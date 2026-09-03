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
import re
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

# The census both tests below pin. It is an EXACT count, not a floor: the point
# is that a template appearing or disappearing is a deliberate edit here, not a
# silent drift in what the gate surveys.
#
# 57 (#1499's migrated set) -> 50: backend#2973 deleted the seven DETR templates
# (detr, rt_detr, rt_detr_v2, deformable_detr, d_fine, owlv2, grounding_dino)
# when the hf_transformer family was retired — the platform stopped supporting
# HuggingFace models and the engine handler those templates routed to never
# trained anything. Their seven dumps are NOT deleted by that change: under
# backend#2985 they are RETAINED and marked `"status": "retired"` in the
# manifest, because they were prepped under the stack pinned at
# `scripts/.tracebloc-engine-ref` and a deletion is irreversible in practice
# while a flag is one line. The orphan half of this gate is now armed (the CI
# job passes --manifest, pinned by the tests at the bottom of this file) and
# reports them as RETIRED rather than failing on them.
#
# This paragraph previously said the orphan half was "unarmed (CI passes no
# --manifest / --dumps-dir), so it will surface them by name on the day hosting
# arms it". Both halves of that are now false, and the second was never right:
# hosting (backend#2659) does not supply the manifest — a cross-repo read does.
#
# 50 -> 49: backend#2988 deleted mask_rcnn (unusable — its mask head needs a
# masks target key the OD path never supplies). Its dump was the one deletion
# taken under #2985: trivially regenerable from torchvision, and re-homing Mask
# R-CNN under backend#795 would need a seed built against a different module
# tree, so retaining it pre-paid for nothing. Manifest entry removed by
# backend#2996, blob deleted under #2985 — gone from both sides.
#
# 49 -> 55: backend#2982 Tier 0 added the six torchvision_detection roster
# templates the zoo never wrapped (faster_rcnn_resnet_v2, retinanet_v2,
# faster_rcnn_mobilenet, faster_rcnn_mobilenet_320, ssd_vgg16,
# ssdlite_mobilenet). All six classify NO_SEED — they build with weights=None,
# so they genuinely random-initialise and stage no dump. That is what makes them
# invisible to the newly-armed orphan half: no dump, and no expectation of one.
#
# 55 -> 59: backend#2982 Tier 1 added four modern-backbone detectors —
# faster_rcnn_convnext_small, faster_rcnn_swin_t, fcos_convnext_small,
# fcos_swin_t — assembled via detection.backbone_utils.BackboneWithFPN. All
# four classify NO_SEED for the same reason.
#
# 59 -> 60: backend#2982 Tier 2 added atss_resnet — RetinaNet's backbone and
# head with Adaptive Training Sample Selection replacing the fixed-IoU matcher.
# NO_SEED for the same reason.
#
# 60 -> 61: backend#2982 Tier 2 added gfl_resnet — Generalized Focal Loss over
# that same ATSS-assigned skeleton. NO_SEED for the same reason.
#
# 61 -> 62: backend#2982 Tier 2 added sparse_rcnn — 100 learned proposal boxes
# and features, six stages of dynamic instance interaction, and Hungarian set
# prediction with no RPN and no NMS. NO_SEED for the same reason.
#
# ⚠️ EACH NUMBER HERE IS A RUNNING TOTAL, NOT ITS BRANCH'S ARITHMETIC. Tier 0
# had to unlearn a plausible-looking `57 + 6 = 63` — wrong by the eight
# templates #2973 and #2988 deleted. Whoever rebases onto a moved develop takes
# what `tools/check_dump_coverage.py --zoo .` reports against the merged tree.
#
# ⚠️ SIX Tier 2 templates are in flight against this one number, and every one
# of them can honestly claim 62 from its own branch: centernet_resnet (#236),
# yolox_s + rtmdet_s (#237, +2), tood (#238), vfnet (#239), and this PR's two
# siblings cascade_rcnn (#242) and efficientdet_d0 (#244). The first to land is
# 62; after that the number is whatever the tool reports against the merged
# tree. RE-READ IT, DO NOT ADD — a rebase that keeps this branch's literal goes
# green locally and red on develop.
MIGRATED_TEMPLATE_CENSUS = 62


def test_the_real_zoo_classifies_every_migrated_template(tmp_path):
    """No template in the shipped zoo may be UNCLASSIFIED. This is the test that
    keeps the derivation honest as templates are edited."""
    tool = _tool()
    found = tool.survey(ROOT)
    unclassified = {k: v["detail"] for k, v in found.items()
                    if v["status"] == tool.UNCLASSIFIED}
    assert not unclassified, f"unclassified migrated templates: {unclassified}"
    assert len(found) == MIGRATED_TEMPLATE_CENSUS, (
        f"expected {MIGRATED_TEMPLATE_CENSUS} migrated templates, got {len(found)}"
    )


def test_the_real_zoo_writes_a_survey(tmp_path):
    out = tmp_path / "survey.json"
    assert _tool().main(["--zoo", str(ROOT), "--out", str(out)]) == 0
    survey = json.loads(out.read_text("utf-8"))
    assert survey["inventory_checked"] is False
    assert len(survey["templates"]) == MIGRATED_TEMPLATE_CENSUS


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


# --- a retirement is a decision, not an orphan (backend#2985) --------------
#
# The seven DETR seeds are RETAINED rather than deleted: they were prepped under
# the stack pinned at `scripts/.tracebloc-engine-ref` and regenerating them
# means standing that environment back up, so a deletion is irreversible in
# practice while a status flag is one line. That makes "no template claims this
# dump" two findings — a drift nobody noticed, and a decision somebody took —
# and this gate has to tell them apart without becoming a way to switch itself
# off. Each narrowing below is a test, and each was seen to fail before it
# passed.


def _manifest(tmp_path, entries, shape="entries"):
    """A schema-2 manifest in either of the two shapes that call themselves 2."""
    path = tmp_path / "manifest.json"
    if shape == "entries":
        body = {"schema": 2, "prefix": "zoo-weights", "entries": entries}
    else:
        body = {
            "schema": 2,
            "dumps": [{"name": name, **record} for name, record in entries.items()],
        }
    path.write_text(json.dumps(body), "utf-8")
    return path


LIVE = {"file": "x_weights.pkl", "sha256": "a" * 64, "size_bytes": 1}
RETIRED_ENTRY = {**LIVE, "status": "retired"}


def test_a_retired_dump_no_template_claims_is_not_an_orphan(tmp_path):
    """The state backend#2985 settles: seven kept dumps, zero live templates for
    them, and a gate that stays usable instead of permanently red."""
    zoo = _zoo(tmp_path, {"object_detection/fcos": SEED_EXPECTING})
    manifest = _manifest(tmp_path, {"fcos": LIVE, "detr": RETIRED_ENTRY})
    assert _tool().main(["--zoo", str(zoo), "--manifest", str(manifest)]) == 0


def test_a_retired_dump_is_still_reported_every_run(tmp_path, capsys):
    """Retaining them visibly is the point. A retirement nobody is reminded of
    is a deletion with extra steps, so this must not go quiet."""
    zoo = _zoo(tmp_path, {"object_detection/fcos": SEED_EXPECTING})
    manifest = _manifest(tmp_path, {"fcos": LIVE, "detr": RETIRED_ENTRY})
    _tool().main(["--zoo", str(zoo), "--manifest", str(manifest)])
    out = capsys.readouterr().out
    assert "RETIRED DUMP  detr" in out
    assert "1 retired" in out


def test_an_orphan_with_no_status_is_STILL_RED(tmp_path):
    """THE TEST THAT KEEPS THE EXEMPTION HONEST. If `retired` were read
    permissively — any status, or a missing one — this gate would go green over
    the exact eight dumps it failed to catch, and #2985 would recur silently."""
    zoo = _zoo(tmp_path, {"object_detection/fcos": SEED_EXPECTING})
    manifest = _manifest(tmp_path, {"fcos": LIVE, "stranded": LIVE})
    assert _tool().main(["--zoo", str(zoo), "--manifest", str(manifest)]) == 1


def test_an_unknown_status_is_named_rather_than_assumed_live(tmp_path, capsys):
    """A typo'd `retried` must not send someone hunting for a stranded blob that
    is really a misspelling — the same refusal-to-guess as UNCLASSIFIED.

    The bad entry is one a template DOES claim, on purpose: an unclaimed one
    goes red as an orphan anyway, so it would pass this test with the
    unknown-status verdict deleted. Here the verdict is the only thing red."""
    zoo = _zoo(tmp_path, {"object_detection/fcos": SEED_EXPECTING})
    manifest = _manifest(tmp_path, {"fcos": {**LIVE, "status": "retried"}})
    assert _tool().main(["--zoo", str(zoo), "--manifest", str(manifest)]) == 1
    assert "BAD STATUS    fcos: 'retried'" in capsys.readouterr().out


def test_a_retired_dump_a_template_still_claims_is_red(tmp_path, capsys):
    """The direction that would ship a retired seed into a live training run.
    Two unrelated edits produce it — retiring an entry whose template is still
    there, or re-adding a template whose seed was retired — and neither author
    is looking at the other half.

    A second, live template is present so that the inventory still holds a live
    dump. Without it the all-retired fail-closed guard goes red too, and this
    test would pass with the retired-in-use verdict deleted."""
    zoo = _zoo(
        tmp_path,
        {
            "object_detection/fcos": SEED_EXPECTING,
            "object_detection/retinanet": SEED_EXPECTING,
        },
    )
    manifest = _manifest(tmp_path, {"fcos": RETIRED_ENTRY, "retinanet": LIVE})
    assert _tool().main(["--zoo", str(zoo), "--manifest", str(manifest)]) == 1
    assert "RETIRED IN USE fcos" in capsys.readouterr().out


def test_a_manifest_of_only_retired_dumps_fails_closed(tmp_path, capsys):
    """"Every dump is exempt" must not be the quiet way to switch this gate off.
    An all-retired inventory protects exactly as little as an empty one, which
    the gate already refuses. No template is missing a seed here and nothing is
    orphaned, so this guard is the only thing standing between the arrangement
    and a green run."""
    zoo = _zoo(tmp_path, {"image_classification/vit": NO_SEED})
    manifest = _manifest(tmp_path, {"detr": RETIRED_ENTRY})
    assert _tool().main(["--zoo", str(zoo), "--manifest", str(manifest)]) == 1
    assert "names no live dumps" in capsys.readouterr().out


def test_retirement_is_honoured_in_the_dumps_list_shape_too(tmp_path):
    """Two shapes both call themselves schema 2. Wiring the exemption to the
    `entries` dict alone would mean a later switch to the `dumps` list silently
    un-retires all seven — a green gate over a decision it had stopped
    honouring."""
    zoo = _zoo(tmp_path, {"object_detection/fcos": SEED_EXPECTING})
    manifest = _manifest(
        tmp_path, {"fcos": LIVE, "detr": RETIRED_ENTRY}, shape="dumps"
    )
    assert _tool().main(["--zoo", str(zoo), "--manifest", str(manifest)]) == 0


def test_a_staging_dir_cannot_launder_an_unlisted_blob_into_a_retired_one(tmp_path):
    """Only the manifest grants the exemption. A staging directory is a listing
    of folders with no statuses in it, so a blob present there and absent from
    the manifest is an orphan — which is precisely the direction backend#2996
    created for `mask_rcnn` and the reason store state cannot be inferred from
    manifest state."""
    zoo = _zoo(tmp_path, {"object_detection/fcos": SEED_EXPECTING})
    dumps = _dumps(tmp_path, ["fcos", "detr", "unlisted"])
    manifest = _manifest(tmp_path, {"fcos": LIVE, "detr": RETIRED_ENTRY})
    assert (
        _tool().main(
            ["--zoo", str(zoo), "--manifest", str(manifest), "--dumps-dir", str(dumps)]
        )
        == 1
    )


def test_the_survey_json_records_the_retirements(tmp_path):
    """`--out` is what a human reads after a red run; a retirement invisible in
    it would look like a dump that had simply vanished from the report."""
    zoo = _zoo(tmp_path, {"object_detection/fcos": SEED_EXPECTING})
    manifest = _manifest(tmp_path, {"fcos": LIVE, "detr": RETIRED_ENTRY})
    out = tmp_path / "survey.json"
    _tool().main(
        ["--zoo", str(zoo), "--manifest", str(manifest), "--out", str(out)]
    )
    survey = json.loads(out.read_text())
    assert survey["retired"] == ["detr"]
    assert survey["orphans"] == []


def test_the_real_manifest_shape_is_read_by_status(tmp_path):
    """Against the real entry shape, not a reduced fixture: the seven retired
    DETR seeds carry `file`/`sha256`/`size_bytes` alongside their status, and a
    reader that only tolerated a two-key record would pass this suite and fail
    on the artifact."""
    tool = _tool()
    inventory = tool.manifest_names(
        {
            "schema": 2,
            "prefix": "zoo-weights",
            "entries": {
                "detr": {
                    "file": "detr_weights.pkl",
                    "sha256": "be50c162032ad57c17ca28285029a0f1c4ac6784e1716b69b5bf8d8317e2ffcb",
                    "size_bytes": 166639507,
                    "status": "retired",
                },
                "fcos": {"file": "fcos_weights.pkl", "sha256": "b" * 64, "size_bytes": 2},
            },
        }
    )
    assert inventory.retired == {"detr"}
    assert inventory.names == {"detr", "fcos"}
    assert inventory.bad_status == []


# --------------------------------------------------------------------------
# backend#2985 — the orphan half must STAY armed in CI.
#
# The gate was fail-closed both ways in code and half-executed in practice: the
# workflow passed neither `--manifest` nor `--dumps-dir`, so `have_inventory`
# was false and the orphan branch was dead. Eight orphaned dumps accumulated
# behind a green check, and a person found them, not the guard.
#
# So the arming itself now has a test. Dropping the flag again is a one-line
# YAML edit that changes no Python, breaks no other test, and leaves this
# workflow green — which is exactly the class of regression that produced #2985.
# --------------------------------------------------------------------------

WORKFLOW = ROOT / ".github/workflows/verify-dumps-engine-pin.yml"


def _job_block(job="dump-coverage"):
    """One job's YAML, as text. NO PyYAML.

    This suite runs in the template test environments (`test-sklearn`,
    `test-survival`, ...), and PyYAML is not installed in them -- the first
    version of these four tests imported it and failed two jobs on exactly
    that. The assertions here are about text a YAML parser would only
    round-trip anyway, so the parser was never earning its dependency.

    Sliced from the job key to the next top-level key at the same indent, so a
    job added after it cannot leak into the block.
    """
    text = WORKFLOW.read_text()
    key = f"  {job}:\n"
    start = text.index("\n" + key) + 1
    rest = text[start + len(key) :]
    end = len(rest)
    for line_start, line in _lines_with_offsets(rest):
        if line.strip() and not line.startswith("   ") and not line.startswith("\t"):
            end = line_start
            break
    block = rest[:end]
    # Join shell line continuations so a wrapped invocation reads as one line.
    return block.replace("\\\n", " ")


def _coverage_job_block():
    return _job_block("dump-coverage")


def _lines_with_offsets(text):
    offset = 0
    for line in text.splitlines(keepends=True):
        yield offset, line
        offset += len(line)


def _coverage_invocation():
    return "\n".join(
        line
        for line in _coverage_job_block().splitlines()
        if "check_dump_coverage.py" in line
    )


def test_the_block_slicer_finds_the_job_and_stops_at_the_next_one():
    """The extractor itself, because every test below trusts its boundaries."""
    block = _coverage_job_block()
    assert "check_dump_coverage.py" in block
    assert "dump-coverage" not in block, "the slice re-included its own job key"
    assert "\n  verify-dumps:" not in block, "the slice ran into a sibling job"


def test_ci_invokes_the_coverage_tool_at_all():
    assert "check_dump_coverage.py" in _coverage_invocation()


def test_CI_ARMS_AN_INVENTORY_SO_THE_ORPHAN_BRANCH_EXECUTES():
    """Without one of these flags the orphan half is dead code in CI."""
    invocation = _coverage_invocation()
    assert (
        "--manifest" in invocation or "--dumps-dir" in invocation
    ), "the coverage job passes no inventory: the orphan half is unarmed again"


def test_the_manifest_it_arms_is_the_one_a_step_actually_puts_there():
    """The path passed to `--manifest` must be one a step actually populates.

    A stale path would make the tool exit 1 on `--manifest does not exist` --
    loud, so survivable -- but a path NOTHING writes to is how a cross-repo read
    rots silently when the sibling repo moves its file. The path now comes from
    a `download-artifact` rather than a checkout (see the token-split tests
    below), so this reads `path:` wherever the job declares it.
    """
    block = _coverage_job_block()
    paths = re.findall(r"^\s*path:\s*(\S+)\s*$", block, re.MULTILINE)
    invocation = _coverage_invocation()
    assert paths, "no step puts a manifest anywhere for the checker to read"
    assert any(
        f"{path}/" in invocation for path in paths
    ), f"--manifest path is not under any populated path {paths}"


def test_the_manifest_presence_is_asserted_rather_than_assumed():
    """A sparse checkout that matches nothing must fail, not fall through.

    `check_dump_coverage.py` exits 1 on a missing `--manifest`, so this is
    belt-and-braces -- but the braces are what stop a silent revert to the
    classification-only run that #2985 was invisible behind.
    """
    block = _coverage_job_block()
    assert "manifest.json" in block and ("test -s" in block or "test -f" in block)


# --------------------------------------------------------------------------
# backend#2985 — THE TOKEN MUST STAY IN ITS OWN JOB.
#
# Arming the manifest read put a `backend`-scoped App token in the same job as
# the PR-controlled checker. `persist-credentials: false` keeps it out of
# `_backend/.git/config` but NOT out of the runner's step-output files, so a PR
# that edited `tools/check_dump_coverage.py` could have read private `backend`.
# Cursor Bugbot caught it at high severity on a learned rule naming this repo's
# own convention, and the convention was already in this very workflow:
# `fetch-engine-pin` holds the engine token and runs no PR code,
# `engine-pin-drift-guard` runs the PR code and holds no token.
#
# The fix is a two-job split, so these tests pin the SPLIT rather than the flag.
# Re-collapsing the jobs is a small, plausible YAML tidy-up that would restore
# the vulnerability while leaving every other test here green.
# --------------------------------------------------------------------------

FETCH_JOB = "fetch-dump-manifest"

TOKEN_MARKERS = ("create-github-app-token", "secrets.")


def test_the_slicer_finds_the_fetch_job_too():
    block = _job_block(FETCH_JOB)
    assert "create-github-app-token" in block
    assert FETCH_JOB not in block, "the slice re-included its own job key"
    assert "\n  dump-coverage:" not in block, "the slice ran into the next job"


def test_THE_COVERAGE_JOB_HOLDS_NO_TOKEN():
    """The security property, stated directly. The job that runs PR-controlled
    code must not be able to mint or see a cross-repo credential."""
    block = _job_block("dump-coverage")
    for marker in TOKEN_MARKERS:
        assert marker not in block, (
            f"the coverage job references {marker!r}: a PR that edits "
            "check_dump_coverage.py could reach private backend"
        )


def test_the_fetch_job_runs_no_pr_controlled_code():
    """The other half of the isolation. A token-holding job that also ran a
    checked-in script would be the same hazard with the jobs merely renamed."""
    block = _job_block(FETCH_JOB)
    assert "python3 tools/" not in block
    assert "check_dump_coverage.py" not in block


def test_the_manifest_crosses_as_an_artifact_not_as_a_checkout():
    """`needs:` plus a download is what makes the split load-bearing rather than
    decorative -- without the dependency the coverage job would race a manifest
    that is not there yet."""
    coverage = _job_block("dump-coverage")
    assert f"needs: {FETCH_JOB}" in coverage
    assert "download-artifact" in coverage
    assert "upload-artifact" in _job_block(FETCH_JOB)


def test_only_the_manifest_crosses_the_boundary():
    """Never `_backend/.git`. The UPLOAD's path must name the one JSON file, so
    a widened glob cannot carry the token's credential store across.

    Scoped to the upload step rather than to every `path:` in the job. A first
    version read them all and excused `_backend` because the CHECKOUT legitimately
    uses it -- which excused exactly the widening this test exists to catch, and
    the mutation (`path: _backend` on the upload) passed. Read the one step whose
    path decides what crosses.
    """
    block = _job_block(FETCH_JOB)
    upload = block[block.index("upload-artifact") :]
    uploaded = re.findall(r"^\s*path:\s*(\S+)\s*$", upload, re.MULTILINE)
    assert uploaded == ["_backend/tools/offline_weights/manifest.json"], uploaded


def test_both_jobs_refuse_an_absent_manifest():
    """Fail-closed on each side of the hand-off: a sparse checkout that matched
    nothing, and an artifact that arrived empty, are different failures and
    neither may fall through to the classification-only run."""
    for job in (FETCH_JOB, "dump-coverage"):
        block = _job_block(job)
        assert "test -s" in block or "test -f" in block, job
