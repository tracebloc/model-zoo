# Vendored contract: `ingest_categories.v1.json`

The platform's list of task categories: the `category` enum of the ingestor's
published ingest schema (`properties/category/enum` of `ingest.v1.json`), which
is what an ingest is validated against. A dataset whose category is not in it
cannot be ingested, so a template in a category outside it can never train.

- **Upstream:** the ingestor, a private repository (`PRODUCER_REPO` in
  `tools/ingest_category_contract.py`), at the newest of `CANDIDATE_PATHS`
  that exists on its `develop`.
- **Pinned ref:** `generated_from.ref` in the JSON, with `ref_branch` and
  `ref_date`.
- **Derived, not copied.** Like `../tracebloc_backend/` and
  `../tracebloc_sdk/`, only the one field this repo depends on is vendored,
  written by a tool rather than transcribed. Vendoring the whole schema would
  make every unrelated upstream edit a re-vendoring chore, and would carry
  upstream's prose into a public repo.

## What reads it

`tests/test_zoo_category_contract.py`, in three layers, each its own test so a
red names which one broke:

| layer | asserts | runs in |
|---|---|---|
| tree ↔ this file | every category here has a `model_zoo/` directory, and every directory is a category here -- **two tests, one per direction** | every `test-*` job and `category-contract` |
| module ↔ tree | each model module's `category = "..."` equals the directory it sits in (read from the AST, so a module this job cannot import is still checked) | same |
| this file ↔ producer | this file equals the producer's `develop` enum, so a stale copy cannot pass the first layer by agreeing with itself | `category-contract` only (the `ingest_producer` marker) |

`tests/test_model_contract.py` also reads it, in place of the sixteen-entry
literal it used to carry.

## Refreshing this copy

With `gh` logged in with read access to the producer:

```bash
python3 tools/ingest_category_contract.py           # compare only; exit 1 on drift
python3 tools/ingest_category_contract.py --write   # rewrite this file from develop
```

The fetch tries the candidate paths newest first: a 404/410 selects the next,
any other failure is fatal naming the candidate that raised it, and all absent
is a refusal naming every path. After `--write` it prints any category the
tree now disagrees with; reconcile `model_zoo/` and commit the lot in one PR.

## The cost of this gate

Adding or retiring a category is now a change in more than one repo that fails
loudly until every repo agrees: the producer publishes it, this file is
refreshed, and a `model_zoo/<category>/` directory with at least one template
lands. Until all three agree, the `category-contract` job is red on every zoo
run -- including runs of pull requests that have nothing to do with categories.
