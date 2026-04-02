[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org) [![Platform](https://img.shields.io/badge/platform-tracebloc-00C9A7.svg)](https://ai.tracebloc.io)

# tracebloc Model Zoo 🧠

Pre-built ML models you can train and benchmark on the [tracebloc](https://tracebloc.io/) platform — or use as a starting point for your own.

Every model in this repo is compatible with tracebloc's secure training environment. Pick one, upload it, and start an experiment in minutes. No setup, no configuration, no boilerplate.

## Models

| Task | Framework | Path | Use when you need to… |
|---|---|---|---|
| Image classification | PyTorch, TensorFlow | `model_zoo/image_classification/` | Categorize images into labels |
| Object detection | PyTorch | `model_zoo/object_detection/pytorch/` | Locate and classify objects in images |
| Text classification | PyTorch | `model_zoo/text_classification/pytorch/` | Classify documents, reviews, tickets |
| Semantic segmentation | PyTorch | `model_zoo/semantic_segmentation/pytorch/` | Pixel-level image labeling |
| Keypoint detection | PyTorch | `model_zoo/keypoint_detection/pytorch/` | Detect landmarks on objects or bodies |
| Tabular classification | — | `model_zoo/tabular_classification/` | Classify structured/tabular data |
| Tabular regression | — | `model_zoo/tabular_regression/` | Predict continuous values from tables |
| Time series forecasting | PyTorch | `model_zoo/time_series_forecasting/pytorch/` | Forecast future values from sequences |
| Time-to-event prediction | PyTorch | `model_zoo/time_to_event_prediction/` | Predict when an event will occur |

## Quick start

```bash
git clone https://github.com/tracebloc/model-zoo.git
```

Then upload a model to tracebloc:

```python
from tracebloc_package import User

user = User()  # log in with your tracebloc email + password
user.uploadModel("model-zoo/model_zoo/image_classification/densenet.py")
```

Full walkthrough → [Open the training notebook in Colab](https://colab.research.google.com/drive/1N00idtpoaq1lk9OJE6g4bMqd8o-Qex2C?usp=sharing)

## Bring your own model

Not limited to this zoo. Any model that follows the [model structure requirements](https://docs.tracebloc.io/join-use-case/model-optimization) works — PyTorch, TensorFlow, or custom containers.

**Weight file convention:** if your model is `mymodel.py`, name the weights `mymodel_weights.pkl` and place them in the same directory.

## Links

[Platform](https://ai.tracebloc.io/) · [Docs](https://docs.tracebloc.io/) · [Training notebook](https://colab.research.google.com/drive/1N00idtpoaq1lk9OJE6g4bMqd8o-Qex2C?usp=sharing) · [PyPI package](https://pypi.org/project/tracebloc-package/) · [Discord](https://discord.gg/tracebloc)

## License

Apache 2.0 — see [LICENSE](LICENSE).

**Questions?** [support@tracebloc.io](mailto:support@tracebloc.io) or [open an issue](https://github.com/tracebloc/model-zoo/issues).
