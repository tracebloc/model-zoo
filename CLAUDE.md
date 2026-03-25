# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo does

Pre-built ML model templates for the tracebloc platform. Users pick a model, upload it via the `tracebloc_package` Python SDK, and start training experiments. Models run inside customers' secure Kubernetes environments.

## Directory structure

```
model_zoo/
  image_classification/    pytorch/, tensorflow/
  object_detection/        pytorch/
  text_classification/     pytorch/
  semantic_segmentation/   pytorch/
  keypoint_detection/      pytorch/
  tabular_classification/  pytorch/, sklearn/
  tabular_regression/      pytorch/, sklearn/
  time_series_forecasting/ pytorch/
  time_to_event_prediction/
```

## Supported task types

Image classification, object detection, text classification, semantic segmentation, keypoint detection, tabular classification, tabular regression, time series forecasting, time-to-event prediction.

## Supported frameworks

PyTorch, TensorFlow, scikit-learn (sklearn).

## Weight file convention

If your model is `mymodel.py`, name the weights `mymodel_weights.pkl` and place them in the same directory.

## How to add a new model

1. Create a Python file under `model_zoo/<task_type>/<framework>/`
2. Follow the model structure requirements documented at https://docs.tracebloc.io/join-use-case/model-optimization
3. If including pretrained weights, follow the naming convention above

## Uploading a model

```python
from tracebloc_package import User
user = User()
user.uploadModel("model_zoo/image_classification/pytorch/resnet_18.py")
```
