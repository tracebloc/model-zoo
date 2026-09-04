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
- `SEED_EXCLUDED_PREFIXES` (models with a task head): the state_dict key prefixes whose shape depends on `output_classes` — i.e. the head. A hosted pretrained seed carries the **backbone only** and the head initialises fresh, because the linked dataset decides `output_classes` and a seed carrying the head fits only the one class count it was built with (backend#2642). **Do not write it by hand:** it is derived by `tools/derive_seed_excluded.py` (build the template twice at different class counts, diff the state_dict shapes) and applied by `tools/seed_contract.py apply`. CI re-derives and fails on drift — see the weight-file section.
- `license` (recommended for new files): SPDX-style string such as `"Apache-2.0"`, `"MIT"`, `"AGPL-3.0"`, or `"non-commercial"`. Lets downstream tooling filter models by license — important since some pretrained weights ship under restrictive terms.

## Federated averaging conventions

The averaging service averages model parameters per-tensor across clients. New pretrained models should be authored with this in mind:

- **BatchNorm running stats** (`running_mean` / `running_var`) average poorly across non-IID clients. Replace BN with `GroupNorm` / `LayerNorm`, which normalise per sample and hold no running statistics at all.
- **Freezing BN is not an equivalent alternative to that, and this bullet used to offer it as one.** Freezing answers the averaging problem and creates a worse one whenever the template builds from scratch: `FrozenBatchNorm2d` at construction holds `weight=1`, `bias=0`, `running_mean=0`, `running_var=1`, so it computes `(x - 0) / sqrt(1 + eps)` — the input, unchanged. Those buffers are only meaningful once a pretrained checkpoint loads real statistics into them, which is what the layer is *for*; on a `weights=None` backbone there is nothing to freeze and the layer is a no-op. Twelve shipped OD templates took this advice and trained with **no backbone normalisation at all** (backend#3093; activations reach sigma ~= 24 at the ROI head against ~= 3 with a live BN, while the loss stays finite and decreasing). All twelve moved to GroupNorm in model-zoo#259. Freeze BN only where a seed is guaranteed to have loaded real statistics first — and since every template in this repo builds `weights=None` and seeds are still blocked on backend#2659, that is nowhere today. `tests/test_od_norm_layers_normalise.py` enforces this for the object-detection roster; **the two `keypoint_detection` templates that build the same `resnet50(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d)` line are still un-fixed and un-guarded** (`faster_rcnn_sppe.py`, `keypoint_rcnn.py`) — the guard is driven by the OD directory scan and does not see them.
- **FrozenBN -> GroupNorm is not the same arithmetic as BN -> GroupNorm, and mixing them up misreports a fix.** Frozen BN holds `weight`/`bias` as **buffers**; live BN and GroupNorm hold them as **parameters**. So `BatchNorm2d -> GroupNorm` (model-zoo#237, yolox_s/rtmdet_s) leaves the parameter count identical and drops the running-statistic buffers, while `FrozenBatchNorm2d -> GroupNorm` (model-zoo#259, the twelve OD templates) **adds** 2 parameters per normalised channel and drops 4 buffers per channel — on ResNet-50's 26,560 normalised channels that is +53,120 parameters and -106,240 buffer elements per template. Any published-parameter-count anchor has to be recomputed for the second migration, not carried over from the first.
- **EMA buffers** (some detectors, Mamba SSMs) are not trained parameters — strip them or document the workaround.
- **Foundation models** (Mitra, Chronos, ModernBERT-large, etc.) should be fine-tuned **LoRA-only**, so only the small adapter tensors get averaged — the only tractable path for >100M-param backbones over federated clients. LoRA is selected in the **training plan** (the platform applies the adapter via the experiment configuration); it is **never** bundled into the model file. A model file that imports `peft` / builds its own `get_peft_model` wrapper is rejected by the server model-checker (its validation environment has no `peft`), and it changes the module tree the seed-weight dump has to strict-load against. The model file ships the plain backbone.

## File naming convention

- All lowercase `snake_case`. No PascalCase, no hyphens, no spaces — filenames must be importable as Python modules.
- Drop framework-implementation prefixes (no `sequential_api_`, `functional_api_`).
- Use canonical library names for sklearn-family models, but suffix with the task so the filename does not shadow the package it imports from: `xgboost_classifier.py` / `xgboost_regressor.py`, `lightgbm_classifier.py` / `lightgbm_regressor.py`, `catboost_classifier.py` (not `xgb`, `lgbm`, `cboost`, and not bare `xgboost.py` — that shadows `import xgboost`).
- Include the variant when there is more than one in the zoo: `resnet_18.py`, `resnet_50.py`, `densenet_121.py`.
- Drop redundant "net" suffixes where they aren't canonical: `vgg_16.py`, not `vggnet_16.py`.

## Weight file convention

If a user wants to ship pretrained weights alongside `mymodel.py`, name them `mymodel_weights.pkl` and place them in the same directory. The zoo itself does not bundle weight files.

### Prepping offline-weight dumps is pinned to the engine

A prepped `<base>_weights.pkl` state_dict's key layout is fixed by the transformers/timm/torchvision version that built the module tree, and the engine strict-loads seeds — so a dump built under a different version than the engine pins is a silent, edge-only training abort (backend#2641). Prep AND verify therefore run against one pinned environment, `tools/requirements-engine-pin.txt`, which mirrors the engine's `use_cases/requirements.txt` (the single source of truth). Install that file before running `tools/prep_offline_weights.py`, and record the versions in `manifest.json`'s schema-2 `built_with` block. CI (`.github/workflows/verify-dumps-engine-pin.yml`) enforces both: `tools/check_engine_pin_drift.py` fails if a mirror drifts from the live engine pin, and `tools/verify_dumps_against_engine_pin.py` rebuilds each shipped template under that pin and strict-loads every staged dump — so an engine `transformers` bump turns the gate red instead of stranding hosted seeds (backend#2658).

**There are TWO mirrors of the engine's pin, and the drift guard checks both.** `tools/requirements-engine-pin.txt` is the prep/verify environment; `.github/requirements/pytorch.txt` is the environment ci.yml's required `test-pytorch` job installs, and it carries the same rule in its own header ("never ahead of them"). It was unguarded on both legs until model-zoo#229 — the workflow's `paths:` did not name `.github/requirements/**`, so a lone bump there fired no job, and the guard step had a single `--mirror` pointed at the tools/ copy — which is how model-zoo#227 sat two minors ahead of the engine and went all green. If you add a CI requirement set that exact-pins any of `torch`/`torchvision`/`transformers`/`timm`/`peft`, add it to the guard step's mirror list; `tests/test_engine_pin_drift_guard.py` fails if you do not.

### A seed carries the backbone, not the head

`SEED_EXCLUDED_PREFIXES` (above) names each template's head, and the tooling reads that one declaration on both sides: `tools/seed_contract.py strip` removes exactly those keys when building a seed, and the loaders allow exactly those keys to be missing. Neither side restates the other, so they cannot drift apart.

The declaration is a literal, so CI re-derives it rather than trusting it: `seed-contract-drift` in `.github/workflows/verify-dumps-engine-pin.yml` rebuilds the templates a PR touches (the scheduled run does all of them) under the engine's pinned stack and fails when a declared head disagrees with the derived one — or when a template that *has* a head declares nothing. Re-run `tools/derive_seed_excluded.py` and re-apply rather than editing a constant by hand.

`tools/verify_backbone_seeds.py` checks the other half: each seed must load into a model built at a class count no dump ever had, with an unexpected key failing, a missing key failing unless the template declared it, and a shape mismatch failing.

## Tokenizer convention (NLP models)

Every NLP model (`text_classification`, `token_classification`, `masked_language_modeling`, `causal_language_modeling`) must declare a tokenizer — it is the federation's single source of truth, distributed to every client (issue #805). The rule depends on whether the model is a HuggingFace model (exposes `.config`) or a plain `nn.Module`:

- **HuggingFace models** (factory returns an `AutoModelFor…`, or the class subclasses an HF model like `BertForMaskedLM`) declare a module-level `tokenizer_id` — the HF repo id of the matching tokenizer, normally equal to `model_id`. Do **not** ship a `tokenizer.json` for these; the client loads the tokenizer from the Hub.
- **Offline-migrated templates** (issue #156: architecture built from an inlined config, pretrained weights loaded from the tracebloc model store) declare **no** `model_id`/`tokenizer_id` — the template must build with no hub lookup at all. They follow the custom-model rule instead: a distinctly named `<model>_tokenizer.json` sibling, uploaded explicitly (see `text_classification/pytorch/bert_base_uncased.py`).
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

- Branch model, **for a repo on the release train**: `develop → staging → main`. Branch off `develop`; every PR targets `develop`. Never open PRs to `staging` or `main` — promotions are the train's job.
- **For a repo not on the train, do not infer the branch model from this file — read `repo-inventory.yml`.** `release_train:` says whether the model above applies at all, and the per-branch `exempt:` anchors record which branches actually exist. This bullet used to enumerate the exceptions by name and **drifted from the inventory on every one of them**: `docs` was called `main`-only while it had been on the train since 2026-08-04 (`release_train: true`, `develop: required`, staging present), and `rfcs` was called `main`-only while it had a `develop` taking merges (measured 2026-08-22, backend#2242 / .github#306). Restating the authority is the defect; pointing at it is the fix.
- **Trap, recorded in the inventory and caught by no check:** a `develop` created on a non-train repo and left **unprotected** is invisible to the guards — that is the `develop_unprotected_non_train` anchor, and the inventory notes "a `develop` created and left UNPROTECTED is not flagged … no check was going to surface it." So creating one to satisfy the first bullet **forks the repo silently**: PRs split between the new branch and the repo's existing convention, nothing promotes between them, and the two heads diverge until someone reconciles by hand. If a repo appears to lack a `develop`, that is a fact to verify in the inventory, not a gap to fill.
- Before starting any task: `git fetch` and branch from the current tip of `develop` — never build on a stale checkout. A branch that lives more than a day gets `develop` merged back in before review. We move fast; stale starts mean silent divergence and duplicated work.
- One self-contained change per PR. A few hundred changed lines reviews well; at 1000+ split it. Refactors ship in separate PRs from behavior changes.
- Branches are short-lived (aim to merge within a day or two), single-author, and based on `develop` — no stacked PRs on top of other open PRs.
- Your branches are yours to clean up. Merged ones now delete themselves server-side, so this is about the rest: run `git reap` (from `tracebloc/.github/scripts/git-reap`) in your checkouts now and then. It is dry-run by default and only proposes a branch when it can prove the work landed. Nobody else can do this for you — you are the only one who knows whether an *unmerged* branch of yours still matters, and `git branch --merged` will not tell you, because we squash-merge and a squashed branch is not an ancestor of `develop`.
- **"Yours" is the branch you opened the PR for, never the branch whose last commit is yours.** Pushing a review fixup onto someone else's branch makes you its tip-commit author and changes nothing about whose work it is — so a "my branches" list built from `%(authorname)`, or from the tip author in any form, aims your cleanup at other people's work. Measured: two of Shujaat's `client` branches showed up on such a list and were one confirmation step away from `--delete` (backend#2365). If you are building any list that reasons about ownership, call `tracebloc/.github/scripts/branch_owner.py` rather than re-deriving it; a branch it cannot attribute comes back as `unattributable`, which is the answer to act on, not to fill in.
- Names and commits: `feat/ fix/ docs/ sec/ ci/ chore/` + issue number + short slug (`fix/1234-ingest-timeout`); commit subjects `type(scope): summary`, referencing the ticket (`backend#1234`).
- When you open a PR: assign yourself and request exactly one reviewer immediately — a PR without a reviewer stalls by construction. You pick the reviewer: whoever knows the code best. There is no per-repo default, and no automation assigns one — branch protection just refuses to merge without a review.
- When you are the reviewer: first response within one business day.

### Quality bar

- Before every push: run the linter and the tests that cover your change. Never push a branch you believe is red — CI is the backstop, not the first run.
- Read the full diff before opening the PR. You own every line you ship, whoever — or whatever — wrote it.
- AI sessions end with evidence, not assertion: run the relevant check (tests, build, lint) and show the output. A change that could not be verified does not ship.
- Fix the class, not the instance. The bug you just fixed is a member of a class; check the rest of the class before you push. Two shapes, and aiming at only the first catches half of them: **other call sites** — grep the symbol or pattern you changed — and **other inputs to the same guard** — what else reaches this branch? If the class can't be cheaply enumerated, say so in the PR rather than leaving it implied that you covered it.
- After opening or pushing to a PR, stay on it: poll CI and Bugbot on the current head and triage every finding the same day — fix it, or reply on the thread saying why not. No silent dismissals. Unresolved threads block the merge and stall the release train's settle stage; cheap now beats expensive later.
- A finding that recurs across PRs becomes a rule: add it to `.cursor/BUGBOT.md`, and if it is grep-expressible, to code-quality's house-rules — then stop re-arguing it in comments.
- Style and naming rules live in tooling (black/ruff, eslint/prettier, house-rules), never in prose. If a rule matters, encode it; do not restate linter rules in CLAUDE.md files.
- Never commit secrets, tokens, or customer data — not in code, config, tests, issues, or commit messages. gitleaks catches secrets in **code**. Nothing scans PR titles, descriptions or commit messages: the public PII gate that did was retired on 2026-08-06 (backend#1409), so keeping customer names out of PR prose on public repos is on you, not on a check.

### Engineer kanban

- Every ticket on the board carries a `Status` — no card sits at "No Status". New tickets start in `Backlog`. **Bugs are the exception:** label them `work-type:bug` (the Bug template does it) and automation moves the card straight into `Ready` — defects don't wait for refinement. Three repos aren't wired for the label trigger yet (`.github`, `release-train`, `rfcs`); move the card yourself there.
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
