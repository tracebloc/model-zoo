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
  semantic_segmentation/      pytorch/
  keypoint_detection/         pytorch/
  tabular_classification/     pytorch/, sklearn/, tensorflow/
  tabular_regression/         pytorch/, sklearn/
  time_series_forecasting/    pytorch/
  time_to_event_prediction/   lifelines/, pytorch/, scikit-survival/
```

## Module-level metadata contract

Every model file defines:

- `framework`: one of `"pytorch"`, `"tensorflow"`, `"sklearn"`, `"lifelines"`, `"scikit_survival"`
- `main_class` OR `main_method`: the symbol the SDK loads (class for `nn.Module` subclasses, function for factory-style models)
- `category`: must match the task directory name
- `batch_size`, `output_classes`, plus task-specific fields (`image_size`, `num_feature_points`, `sequence_length`, `forecast_horizon`, etc.)

## File naming convention

- All lowercase `snake_case`. No PascalCase, no hyphens, no spaces — filenames must be importable as Python modules.
- Drop framework-implementation prefixes (no `sequential_api_`, `functional_api_`).
- Use canonical library names for sklearn-family models: `xgboost.py`, `lightgbm.py`, `catboost.py` (not `xgb`, `lgbm`, `cboost`).
- Include the variant when there is more than one in the zoo: `resnet_18.py`, `resnet_50.py`, `densenet_121.py`.
- Drop redundant "net" suffixes where they aren't canonical: `vgg_16.py`, not `vggnet_16.py`.

## Weight file convention

If a user wants to ship pretrained weights alongside `mymodel.py`, name them `mymodel_weights.pkl` and place them in the same directory. The zoo itself does not bundle weight files.

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
