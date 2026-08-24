# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Pre-built ML model templates for the tracebloc platform. Each `.py` file is a standalone model that the `tracebloc_package` SDK uploads via `user.uploadModel(path)`. Training runs inside customer Kubernetes environments — the zoo itself does not ship trained weights.

The `start-training` notebook clones this repo at runtime and hardcodes one example path. When that path needs to change, update the notebook — not this repo.

## Directory structure

```
model_zoo/
  causal_language_modeling/       pytorch/
  embeddings/                     pytorch/
  image_classification/           pytorch/
  keypoint_detection/             pytorch/
  masked_language_modeling/       pytorch/
  object_detection/               pytorch/
  semantic_segmentation/          pytorch/
  sentence_pair_classification/   pytorch/
  seq2seq/                        pytorch/
  tabular_classification/         pytorch/, sklearn/
  tabular_regression/             pytorch/, sklearn/
  text_classification/            pytorch/
  time_series_classification/     pytorch/
  time_series_forecasting/        pytorch/
  time_to_event_prediction/       lifelines/, pytorch/, scikit_survival/
  token_classification/           pytorch/
```

## Module-level metadata contract

Every model file defines:

- `framework`: one of `"pytorch"`, `"sklearn"`, `"lifelines"`, `"scikit_survival"`
- `main_class` OR `main_method`: the symbol the SDK loads (class for `nn.Module` subclasses, function for factory-style models)
- `category`: must match the task directory name
- `batch_size`, `output_classes`, plus task-specific fields (`image_size`, `num_feature_points`, `sequence_length`, `forecast_horizon`, etc.)
- `license` (recommended for new files): SPDX-style string such as `"Apache-2.0"`, `"MIT"`, `"AGPL-3.0"`, or `"non-commercial"`. Lets downstream tooling filter models by license — important since some pretrained weights ship under restrictive terms.

## Federated averaging conventions

The averaging service averages model parameters per-tensor across clients. New pretrained models should be authored with this in mind:

- **BatchNorm running stats** (`running_mean` / `running_var`) average poorly across non-IID clients. Either freeze BN layers (`eval()` + `requires_grad=False`) or replace with `GroupNorm` / `LayerNorm`.
- **EMA buffers** (some detectors, Mamba SSMs) are not trained parameters — strip them or document the workaround.
- **Foundation models** (Mitra, Chronos, ModernBERT-large, etc.) should be fine-tuned **LoRA-only** via `peft`. Freeze the base model and only the small adapter tensors get averaged. This is the only tractable path for >100M-param backbones over federated clients.

## File naming convention

- All lowercase `snake_case`. No PascalCase, no hyphens, no spaces — filenames must be importable as Python modules.
- Drop framework-implementation prefixes (no `sequential_api_`, `functional_api_`).
- Use canonical library names for sklearn-family models, but suffix with the task so the filename does not shadow the package it imports from: `xgboost_classifier.py` / `xgboost_regressor.py`, `lightgbm_classifier.py` / `lightgbm_regressor.py`, `catboost_classifier.py` (not `xgb`, `lgbm`, `cboost`, and not bare `xgboost.py` — that shadows `import xgboost`).
- Include the variant when there is more than one in the zoo: `resnet_18.py`, `resnet_50.py`, `densenet_121.py`.
- Drop redundant "net" suffixes where they aren't canonical: `vgg_16.py`, not `vggnet_16.py`.

## Weight file convention

If a user wants to ship pretrained weights alongside `mymodel.py`, name them `mymodel_weights.pkl` and place them in the same directory. The zoo itself does not bundle weight files.

## Tokenizer convention (NLP models)

Every NLP model (`text_classification`, `token_classification`, `masked_language_modeling`, `causal_language_modeling`) must declare a tokenizer — it is the federation's single source of truth, distributed to every client (issue #805). The rule depends on whether the model is a HuggingFace model (exposes `.config`) or a plain `nn.Module`:

- **HuggingFace models** (factory returns an `AutoModelFor…`, or the class subclasses an HF model like `BertForMaskedLM`) declare a module-level `tokenizer_id` — the HF repo id of the matching tokenizer, normally equal to `model_id`. Do **not** ship a `tokenizer.json` for these; the client loads the tokenizer from the Hub.
- **Custom (non-HF) models** (a bare `nn.Module`, including thin wrappers that hold an HF model in an attribute — those do *not* expose `.config`) must ship a `tokenizer.json` (a HuggingFace `tokenizers` file). It must contain the required special tokens (`[PAD]`/`[CLS]`/`[SEP]`/`[UNK]` for classification; `[MASK]`/`[PAD]` for MLM; `[PAD]` + an end-of-text/eos token for causal LM, where the client sets `pad_token = eos_token`) and its max token id must fit the model's embedding table.

The SDK auto-detects a `tokenizer.json` sitting next to the model file and ships it — which means it is also picked up by **any other model in the same directory**, overriding that model's `tokenizer_id`. So a bare `tokenizer.json` is only safe in a directory where it is correct for every model (e.g. `masked_language_modeling/pytorch/`, which is all bert-vocab). When a non-HF model shares a directory with HF models that use different tokenizers, give it a distinct, non-auto-detected name (`<model>_tokenizer.json`, e.g. `simple_text_tokenizer.json`) and upload it explicitly: `user.upload_model("simple_text", tokenizer="simple_text_tokenizer.json")`.

## How to add a new model

1. Create a `.py` file under `model_zoo/<task>/<framework>/` following the naming convention above.
2. Define the metadata contract (`framework`, `main_class`/`main_method`, `category`, etc.).
3. The entry point must construct with no arguments — `tests/test_model_contract.py::test_model_instantiates` calls it. Give every `__init__`/factory parameter a default.
4. Architecture names passed to `timm.create_model` / `torchvision.models` are lookups into a third-party registry, not checked by any linter. Confirm the exact string resolves in the pinned library version before committing (`timm.list_models("*name*")`, `timm.list_pretrained("*name*")`) — a plausible-looking name that does not exist raises only at construction time (backend#1859).
5. Full model structure requirements: https://docs.tracebloc.io/join-use-case/model-optimization

## Uploading a model via the SDK

```python
from tracebloc_package import User
user = User()
user.uploadModel("model_zoo/image_classification/pytorch/resnet_18.py")
```

## Branches

Work lands on `develop` (the default branch); the release train promotes
`develop -> staging -> main`. `master` no longer exists — it was renamed to
`main` under backend#1428 Change 2. This section previously declared `master`
the default, which misled a reviewer into flagging correct CI triggers as
wrong (model-zoo#120).

<!-- org-standards:begin -->
## tracebloc engineering standards (org-wide)

<!-- Canonical source: tracebloc/.github/org-standards.md.
     Synced into every repo's CLAUDE.md between org-standards markers — never
     edit it inside a consuming repo; open a PR against tracebloc/.github.
     Meta-rule: the moment a rule below becomes mechanically enforced (a lint
     rule, a house-rules grep, a required check), delete the sentence here and
     let the check carry it. Prose is only for what tooling can't judge. -->

### Branches & PRs

- Branch model: `develop → staging → main`. Branch off `develop`; every PR targets `develop`. Never open PRs to `staging` or `main` — promotions are the release train's job. (Sole exception: the `docs` repo may target `main`.)
- Before starting any task: `git fetch` and branch from the current tip of `develop` — never build on a stale checkout. A branch that lives more than a day gets `develop` merged back in before review. We move fast; stale starts mean silent divergence and duplicated work.
- One self-contained change per PR. A few hundred changed lines reviews well; at 1000+ split it. Refactors ship in separate PRs from behavior changes.
- Branches are short-lived (aim to merge within a day or two), single-author, and based on `develop` — no stacked PRs on top of other open PRs.
- Names and commits: `feat/ fix/ docs/ sec/ ci/ chore/` + issue number + short slug (`fix/1234-ingest-timeout`); commit subjects `type(scope): summary`, referencing the ticket (`backend#1234`).
- When you open a PR: assign yourself and request exactly one reviewer immediately — a PR without a reviewer stalls by construction. You pick the reviewer: whoever knows the code best. There is no per-repo default, and no automation assigns one — branch protection just refuses to merge without a review.
- When you are the reviewer: first response within one business day.

### Quality bar

- Before every push: run the linter and the tests that cover your change. Never push a branch you believe is red — CI is the backstop, not the first run.
- Read the full diff before opening the PR. You own every line you ship, whoever — or whatever — wrote it.
- AI sessions end with evidence, not assertion: run the relevant check (tests, build, lint) and show the output. A change that could not be verified does not ship.
- After opening or pushing to a PR, stay on it: poll CI and Bugbot on the current head and triage every finding the same day — fix it, or reply on the thread saying why not. No silent dismissals. Unresolved threads block the merge and stall the release train's settle stage; cheap now beats expensive later.
- A finding that recurs across PRs becomes a rule: add it to `.cursor/BUGBOT.md`, and if it is grep-expressible, to code-quality's house-rules — then stop re-arguing it in comments.
- Style and naming rules live in tooling (black/ruff, eslint/prettier, house-rules), never in prose. If a rule matters, encode it; do not restate linter rules in CLAUDE.md files.
- Never commit secrets, tokens, or customer data — not in code, config, tests, issues, or commit messages. gitleaks catches secrets in **code**. Nothing scans PR titles, descriptions or commit messages: the public PII gate that did was retired on 2026-08-06 (backend#1409), so keeping customer names out of PR prose on public repos is on you, not on a check.

### Engineer kanban

- Every ticket on the board carries a `Status` — no card sits at "No Status". New tickets start in `Backlog`. **Bugs are the exception:** label them `work-type:bug` (the Bug template does it) and put them straight into `Ready` — defects don't wait for refinement.
- Picking up work: the team coordinates. `Ready` is the refined queue — bugs excepted, per the line above — and the first choice when it's stocked; pulling from `Backlog` is normal when refinement hasn't caught up — say what you're taking.
- Merging to `develop` moves the card to `On dev` automatically; there is no dev-side review.
- Functional review happens once, on staging: when it passes, comment `/fr-pass` on the PR or drag the card to `Ready for prod`. Self-signoff is allowed.
- `fr-gate` is a required check on promotions. If it blocks, the board or the work isn't ready — fix that. `skip-fr-gate` is audited, for emergencies only.

### Releases & publishing

- The release train is the only path to `staging`, `main`, and every package registry. Never hand-cut a `v*` tag, hand-bump a version file, or publish an artifact — every legal publish path is inventoried in release-train's `PUBLISH-PATHS.md`.
- Findings on a promotion PR are fixed on the source branch (`develop`/`staging`), then the train re-prepares. Never push fixes onto a promotion PR — every push re-rolls its review.

### Filing issues

- Internal work — planning, epics, security findings, infrastructure, anything mentioning a customer — is filed in `backend` (the private catch-all), never in a public repo. When in doubt: `backend`.
- Public repos (`cli`, `client`, `docs`, `data-ingestors`, `model-zoo`, `start-training`, `.github`) only get issues a stranger could act on: about the public artifact itself, with no customer names, internal URLs, or internal paths.

### AI-assisted sessions (Claude Code, etc.)

- An AI session may open PRs and push its own branches. It never: merges a PR, closes another person's PR, deletes another person's branch, or force-pushes — each of those needs an explicit instruction from the human running it.
- If your change makes a statement in any CLAUDE.md, BUGBOT.md, or runbook false, update that file in the same PR.
<!-- org-standards:end -->
