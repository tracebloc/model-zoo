# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Pre-built ML model templates for the tracebloc platform. Each `.py` file is a standalone model that the `tracebloc_package` SDK uploads via `user.uploadModel(path)`. Training runs inside customer Kubernetes environments — the zoo itself does not ship trained weights.

The `start-training` notebook clones this repo at runtime and hardcodes one example path. When that path needs to change, update the notebook — not this repo.

## Directory structure

```
model_zoo/
  image_classification/       pytorch/, tensorflow/
  object_detection/           pytorch/
  text_classification/        pytorch/
  token_classification/       pytorch/
  semantic_segmentation/      pytorch/
  keypoint_detection/         pytorch/
  tabular_classification/     pytorch/, sklearn/, tensorflow/
  tabular_regression/         pytorch/, sklearn/, tensorflow/
  time_series_forecasting/    pytorch/
  time_to_event_prediction/   lifelines/, pytorch/, scikit-survival/
```

## Module-level metadata contract

Every model file defines:

- `framework`: one of `"pytorch"`, `"tensorflow"`, `"sklearn"`, `"lifelines"`, `"scikit_survival"`
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

Every NLP model (`text_classification`, `token_classification`, `masked_language_modeling`) must declare a tokenizer — it is the federation's single source of truth, distributed to every client (issue #805). The rule depends on whether the model is a HuggingFace model (exposes `.config`) or a plain `nn.Module`:

- **HuggingFace models** (factory returns an `AutoModelFor…`, or the class subclasses an HF model like `BertForMaskedLM`) declare a module-level `tokenizer_id` — the HF repo id of the matching tokenizer, normally equal to `model_id`. Do **not** ship a `tokenizer.json` for these; the client loads the tokenizer from the Hub.
- **Custom (non-HF) models** (a bare `nn.Module`, including thin wrappers that hold an HF model in an attribute — those do *not* expose `.config`) must ship a `tokenizer.json` (a HuggingFace `tokenizers` file). It must contain the required special tokens (`[PAD]`/`[CLS]`/`[SEP]`/`[UNK]` for classification; `[MASK]`/`[PAD]` for MLM) and its max token id must fit the model's embedding table.

The SDK auto-detects a `tokenizer.json` sitting next to the model file and ships it — which means it is also picked up by **any other model in the same directory**, overriding that model's `tokenizer_id`. So a bare `tokenizer.json` is only safe in a directory where it is correct for every model (e.g. `masked_language_modeling/pytorch/`, which is all bert-vocab). When a non-HF model shares a directory with HF models that use different tokenizers, give it a distinct, non-auto-detected name (`<model>_tokenizer.json`, e.g. `simple_text_tokenizer.json`) and upload it explicitly: `user.upload_model("simple_text", tokenizer="simple_text_tokenizer.json")`.

## How to add a new model

1. Create a `.py` file under `model_zoo/<task>/<framework>/` following the naming convention above.
2. Define the metadata contract (`framework`, `main_class`/`main_method`, `category`, etc.).
3. Full model structure requirements: https://docs.tracebloc.io/join-use-case/model-optimization

## Uploading a model via the SDK

```python
from tracebloc_package import User
user = User()
user.uploadModel("model_zoo/image_classification/pytorch/resnet_18.py")
```

## Default branch

`master`. Not `main`, not `develop`.
