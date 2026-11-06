# Vendored contract: `ingest_categories.v1.json`

The platform's list of task categories: the `category` enum of the ingestor's
published ingest schema (`properties/category/enum` of `ingest.v1.json`), which
is what an ingest is validated against. A dataset whose category is not in it
cannot be ingested, so a template in a category outside it can never train.

- **Upstream:** the ingestor, a private repository (`PRODUCER_REPO` in
  `tools/ingest_category_contract.py`), at the newest of `CANDIDATE_PATHS`
  that exists in the release this file pins.
- **Pinned ref:** `generated_from.ref_tag` in the JSON, a producer release tag
  (`RELEASE_TAG`), with the commit it named (`ref`) and `ref_date`. A tag,
  never a branch, so a producer push cannot change a pull request's verdict.
- **Derived, not copied.** Like `../tracebloc_backend/` and
  `../tracebloc_sdk/`, only the one field this repo depends on is vendored,
  written by a tool rather than transcribed. Vendoring the whole schema would
  make every unrelated upstream edit a re-vendoring chore, and would carry
  upstream's prose into a public repo.

## What reads it

`tests/test_zoo_category_contract.py` and ci.yml, in three layers, each its own
check so a red names which one broke:

| layer | asserts | runs in |
|---|---|---|
| tree ↔ this file | every category here has a `model_zoo/` directory, and every directory is a category here -- **two tests, one per direction** | every `test-*` job and `category-contract` |
| module ↔ tree | each model module's `category = "..."` equals the directory it sits in (read from the AST, so a module this job cannot import is still checked) | same |
| this file ↔ producer | this file equals the enum of the producer release it pins, so a stale copy cannot pass the first layer by agreeing with itself | `ingest-schema-guard`, which `category-contract` requires; the log gives only a verdict |
| this file ↔ producer's `develop` | the same comparison against the moving branch: its red is the cue to re-vendor from a newer release, never a broken pull request | `category-drift.yml`, daily, required nowhere; a scheduled red files one issue |

`tests/test_model_contract.py` also reads it, in place of the sixteen-entry
literal it used to carry.

## Refreshing this copy

With `gh` logged in with read access to the producer:

```bash
python3 tools/ingest_category_contract.py                       # compare with the pinned release; exit 1 on drift
python3 tools/ingest_category_contract.py --ref develop         # what the daily drift run asks
python3 tools/ingest_category_contract.py --write --ref vX.Y.Z  # re-pin to a release and rewrite this file
```

`--write` takes a release tag only, and refuses a branch or a commit before it
reads anything.

The fetch tries the candidate paths newest first: a 404/410 selects the next,
any other failure is fatal naming the candidate that raised it, and all absent
is a refusal naming every path. After `--write` it prints any category the
tree now disagrees with; reconcile `model_zoo/` and commit the lot in one PR.

## The cost of this gate

Adding or retiring a category is a change in more than one repo: the producer
releases it, this file is re-pinned to that release, and a
`model_zoo/<category>/` directory with at least one template lands, in the same
pull request as the re-pin. Until then the daily drift run is red and its issue
stays open; pull requests that have nothing to do with categories stay green,
because the required check reads the pinned release, not the moving branch.
