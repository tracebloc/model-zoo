"""Build ``model_zoo/index.v1.json``: one row per template, for pickers that must
list the zoo without importing it.

A consumer that shows "which templates fit this dataset" (a model picker, a
catalogue page) needs a handful of facts per template -- its task, a label, how
big it is, the batch size it starts from, and whether it can be trained across
more than one edge. Reading them means importing every template, and for the
PyTorch ones building them, which is not something a web tier should do. So they
are computed HERE, once, and committed as JSON. ``tests/test_zoo_index.py``
regenerates the index and fails on any difference, so a template change that
moves one of these facts has to carry the index change with it.

What counts as a template is the zoo's existing rule
(``tests/test_model_contract.py``): a ``.py`` under ``model_zoo/`` with a
module-level ``framework = "..."``. Everything else is a support file.

Each row::

    id           the template's path under model_zoo/, without ".py"
                 (e.g. "image_classification/pytorch/resnet_18"). Stems are not
                 unique -- mlp.py and cox_ph.py exist under two frameworks in one
                 task, and packaged templates are all called model.py -- so the
                 path is the only id that is.
    upload       the file under model_zoo/ to pass to ``upload_model``: the
                 template's own ``.py``, or for a packaged template
                 (``<name>/model.py``) the flat ``<name>.zip`` beside its folder
                 (``tools/build_folder_zips.py``), e.g.
                 "object_detection/pytorch/yolo_v8.zip".
    category     the declared ``category`` (the task).
    framework    the declared ``framework``.
    model_type   the declared ``model_type``, or null when the template has none.
    label        derived from the template's name (``label_for``): the stem, or
                 for a packaged template (``<name>/model.py``) its directory.
    batch        the module-level ``batch_size`` literal, read by AST.
    params       PyTorch only: total ``numel`` over ``parameters()``, counted on
                 the ``meta`` device so nothing is allocated (the multi-billion
                 parameter templates would not fit a CI runner otherwise). Null
                 for every other framework, which has no parameter tensor.
    averageable  whether the federated merge can combine this model across
                 edges (``is_averageable``). Always true for PyTorch (weights are
                 averaged); for the pickle frameworks it depends on the final
                 estimator, and a false row is still listed -- it trains on a
                 single-edge dataset -- so a picker can flag it on a multi-edge
                 one.

The static fields need nothing but the standard library. ``params`` needs torch
and ``averageable`` needs the template's own framework, and CI runs one
framework per job -- so a run computes what its environment can and CARRIES the
rest from the committed index (``--write`` says how many rows it carried).
Across the three CI jobs every field is recomputed by one of them.

Usage::

    python tools/build_index.py            # rewrite model_zoo/index.v1.json
    python tools/build_index.py --check    # exit 1 if it is out of date
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import importlib.util
import io
import json
import os
import pathlib
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "model_zoo"
INDEX_PATH = MODEL_ROOT / "index.v1.json"
INDEX_VERSION = 1

#: The import each framework needs, the same map ``test_model_contract.py`` uses.
FRAMEWORK_IMPORT_NAME = {
    "pytorch": "torch",
    "sklearn": "sklearn",
    "lifelines": "lifelines",
    "scikit_survival": "sksurv",
}

#: Third-party libraries a template may import on top of its framework. A
#: template importing one that is absent cannot be built here, so its computed
#: fields are carried rather than recomputed.
OPTIONAL_THIRD_PARTY = {"xgboost", "lightgbm", "catboost", "interpret", "peft", "timm"}

# --- averageability ----------------------------------------------------------
#
# The federated merge is ALLOWLIST-default: it combines the estimator families
# below and refuses everything else (a single decision tree, sklearn's own
# boosting, stacking, kernel SVMs, nearest neighbours, EBM, ...). Listed as
# (module, class) and matched with ``isinstance`` after unwrapping a Pipeline,
# so a subclass counts and a wrapped estimator is judged by what it wraps. An
# allowlist fails closed: an estimator nobody listed reads as not averageable,
# which a picker shows as a flag, instead of a green row that fails at merge.
AVERAGEABLE_ESTIMATORS = {
    "sklearn": (
        # coefficient averaging
        ("sklearn.linear_model", "LogisticRegression"),
        ("sklearn.linear_model", "LogisticRegressionCV"),
        ("sklearn.linear_model", "RidgeClassifier"),
        ("sklearn.linear_model", "RidgeClassifierCV"),
        ("sklearn.linear_model", "SGDClassifier"),
        ("sklearn.linear_model", "Perceptron"),
        ("sklearn.linear_model", "PassiveAggressiveClassifier"),
        ("sklearn.svm", "LinearSVC"),
        ("sklearn.discriminant_analysis", "LinearDiscriminantAnalysis"),
        ("sklearn.linear_model", "LinearRegression"),
        ("sklearn.linear_model", "Ridge"),
        ("sklearn.linear_model", "RidgeCV"),
        ("sklearn.linear_model", "Lasso"),
        ("sklearn.linear_model", "LassoCV"),
        ("sklearn.linear_model", "ElasticNet"),
        ("sklearn.linear_model", "ElasticNetCV"),
        ("sklearn.linear_model", "SGDRegressor"),
        ("sklearn.linear_model", "PassiveAggressiveRegressor"),
        ("sklearn.linear_model", "HuberRegressor"),
        ("sklearn.linear_model", "BayesianRidge"),
        ("sklearn.linear_model", "ARDRegression"),
        ("sklearn.svm", "LinearSVR"),
        # bagging: the edges' trees are pooled
        ("sklearn.ensemble", "RandomForestClassifier"),
        ("sklearn.ensemble", "RandomForestRegressor"),
        ("sklearn.ensemble", "ExtraTreesClassifier"),
        ("sklearn.ensemble", "ExtraTreesRegressor"),
        ("sklearn.ensemble", "BaggingClassifier"),
        ("sklearn.ensemble", "BaggingRegressor"),
        # the gradient-boosting libraries: a weighted vote over the edges' boosters
        ("xgboost", "XGBClassifier"),
        ("xgboost", "XGBRegressor"),
        ("lightgbm", "LGBMClassifier"),
        ("lightgbm", "LGBMRegressor"),
        ("catboost", "CatBoostClassifier"),
        ("catboost", "CatBoostRegressor"),
        # neural networks, naive Bayes, discriminant models, clustering
        ("sklearn.neural_network", "MLPClassifier"),
        ("sklearn.neural_network", "MLPRegressor"),
        ("sklearn.naive_bayes", "GaussianNB"),
        ("sklearn.naive_bayes", "MultinomialNB"),
        ("sklearn.naive_bayes", "BernoulliNB"),
        ("sklearn.naive_bayes", "CategoricalNB"),
        ("sklearn.neighbors", "NearestCentroid"),
        ("sklearn.discriminant_analysis", "QuadraticDiscriminantAnalysis"),
        ("sklearn.cluster", "KMeans"),
        ("sklearn.cluster", "MiniBatchKMeans"),
    ),
    "lifelines": (
        ("lifelines", "CoxPHFitter"),
        ("lifelines", "WeibullAFTFitter"),
        ("lifelines", "LogNormalAFTFitter"),
        ("lifelines", "LogLogisticAFTFitter"),
    ),
    "scikit_survival": (
        ("sksurv.linear_model", "CoxPHSurvivalAnalysis"),
        ("sksurv.ensemble", "RandomSurvivalForest"),
        ("sksurv.ensemble", "ExtraSurvivalTrees"),
        ("sksurv.ensemble", "GradientBoostingSurvivalAnalysis"),
        ("sksurv.ensemble", "ComponentwiseGradientBoostingSurvivalAnalysis"),
    ),
}

# --- labels ------------------------------------------------------------------
#
# A label is derived from the template's name, one ``_``-separated word at a
# time: a word in LABEL_WORDS is written as listed, any other alphabetic word is
# capitalised, and anything with a digit is upper-cased ("b0" -> "B0"). The map
# only holds words plain capitalisation gets wrong (acronyms and brand casing).
LABEL_WORDS = {
    "aft": "AFT",
    "aimv2": "AIMv2",
    "alphapose": "AlphaPose",
    "atss": "ATSS",
    "bert": "BERT",
    "catboost": "CatBoost",
    "centernet": "CenterNet",
    "cnn": "CNN",
    "convnext": "ConvNeXt",
    "cpm": "CPM",
    "cpn": "CPN",
    "deberta": "DeBERTa",
    "deephit": "DeepHit",
    "deeplab": "DeepLab",
    "deeppose": "DeepPose",
    "deepsurv": "DeepSurv",
    "dinov3": "DINOv3",
    "distilbert": "DistilBERT",
    "distilgpt2": "DistilGPT2",
    "dpt": "DPT",
    "dsnt": "DSNT",
    "ebm": "EBM",
    "efficientdet": "EfficientDet",
    "electra": "ELECTRA",
    "eva": "EVA",
    "fastvit": "FastViT",
    "fcn": "FCN",
    "fcos": "FCOS",
    "ft": "FT",
    "gbm": "GBM",
    "gfl": "GFL",
    "gru": "GRU",
    "gte": "GTE",
    "hrnet": "HRNet",
    "itransformer": "iTransformer",
    "knn": "KNN",
    "lenet": "LeNet",
    "lightgbm": "LightGBM",
    "lm": "LM",
    "lstm": "LSTM",
    "mambaout": "MambaOut",
    "mask2former": "Mask2Former",
    "maxvit": "MaxViT",
    "minilm": "MiniLM",
    "mlm": "MLM",
    "mlp": "MLP",
    "mobilenet": "MobileNet",
    "modernbert": "ModernBERT",
    "nanogpt": "NanoGPT",
    "netmedgpt": "NetMedGPT",
    "oneformer": "OneFormer",
    "openpose": "OpenPose",
    "ph": "PH",
    "prenorm": "PreNorm",
    "qwen2": "Qwen2",
    "rcnn": "R-CNN",
    "resnet": "ResNet",
    "retinanet": "RetinaNet",
    "rnn": "RNN",
    "roberta": "RoBERTa",
    "rtmdet": "RTMDet",
    "rtmpose": "RTMPose",
    "sam2": "SAM2",
    "segformer": "SegFormer",
    "segnet": "SegNet",
    "seq2seq": "Seq2Seq",
    "shg": "SHG",
    "siglip2": "SigLIP2",
    "sppe": "SPPE",
    "squeezenet": "SqueezeNet",
    "ssd": "SSD",
    "ssdlite": "SSDLite",
    "svm": "SVM",
    "t5": "T5",
    "tabm": "TabM",
    "tcn": "TCN",
    "tft": "TFT",
    "timesfm": "TimesFM",
    "tood": "TOOD",
    "tsmixer": "TSMixer",
    "tst": "TST",
    "unet": "U-Net",
    "upernet": "UperNet",
    "vfnet": "VFNet",
    "vgg": "VGG",
    "vgg16": "VGG16",
    "vit": "ViT",
    "vitpose": "ViTPose",
    "xgboost": "XGBoost",
    "xlm": "XLM",
    "yolo": "YOLO",
    "yolo11": "YOLO11",
    "yolov8": "YOLOv8",
    "yolov9": "YOLOv9",
    "yolov10": "YOLOv10",
    "yolov12": "YOLOv12",
    "yolox": "YOLOX",
}

#: Whole names the word rule cannot spell, because the stem had to drop a "."
#: (a version number) to be a Python module name.
LABEL_NAMES = {
    "qwen2_5_0_5b": "Qwen2.5 0.5B",
}


def label_for(name: str) -> str:
    """``resnet_18`` -> ``ResNet 18``; ``faster_rcnn_mobilenet_320`` ->
    ``Faster R-CNN MobileNet 320``."""
    if name in LABEL_NAMES:
        return LABEL_NAMES[name]
    words = []
    for word in name.split("_"):
        if not word:
            continue
        if word in LABEL_WORDS:
            words.append(LABEL_WORDS[word])
        elif word.isalpha():
            words.append(word.capitalize())
        else:
            words.append(word.upper())
    return " ".join(words)


# --- discovery (standard library only) ---------------------------------------


def _module_constants(path: pathlib.Path) -> dict:
    """Module-level literal assignments, by AST. Text matching would read a
    declaration quoted in a comment as a declaration."""
    out: dict = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:  # not a literal
                out[node.targets[0].id] = None
    return out


def template_name(rel: pathlib.PurePosixPath) -> str:
    """The name a label is derived from: the stem, or the package directory
    for a packaged template (``object_detection/pytorch/yolo_v8/model.py``)."""
    # <category>/<framework dir>/<stem>.py, or <category>/<framework dir>/<pkg>/<file>.py
    parts = rel.parts
    return parts[2] if len(parts) > 3 else rel.stem


#: Directories the SDK's ``upload_model()`` leaves beside a template, holding a
#: verbatim copy of it. Gitignored (``tmpmodel_*/``), so absent from CI and
#: present in many checkouts; indexing them would add rows no clone has.
SCRATCH_DIR_PREFIX = "tmpmodel_"


def discover(root: pathlib.Path = MODEL_ROOT) -> list[tuple[pathlib.Path, dict]]:
    """Every template under model_zoo/, in id order, with its module constants."""
    found = []
    for path in sorted(root.rglob("*.py")):
        rel_dirs = path.relative_to(root).parts[:-1]
        if any(part.startswith(SCRATCH_DIR_PREFIX) for part in rel_dirs):
            continue
        consts = _module_constants(path)
        if isinstance(consts.get("framework"), str):
            found.append((path, consts))
    return found


def upload_path(rel: pathlib.PurePosixPath) -> str:
    """The file under model_zoo/ a user passes to ``upload_model``.

    A one-file template is uploaded as itself. A folder template
    (``<category>/<framework>/<name>/model.py``) is uploaded as the flat zip
    ``tools/build_folder_zips.py`` builds next to its folder,
    ``<category>/<framework>/<name>.zip``: its ``model.py`` alone lacks the
    ``loss.py`` the upload needs."""
    if len(rel.parts) == 4:
        return str(rel.parent.with_name(rel.parent.name + ".zip"))
    return str(rel)


def static_row(path: pathlib.Path, consts: dict) -> dict:
    """Everything the AST can say. Raises ValueError on a template the index
    cannot describe, naming the file and the field."""
    rel = pathlib.PurePosixPath(path.relative_to(MODEL_ROOT).as_posix())
    framework = consts["framework"]
    if framework not in FRAMEWORK_IMPORT_NAME:
        raise ValueError(f"{rel}: framework {framework!r} is not one the zoo knows")
    if not isinstance(consts.get("category"), str):
        raise ValueError(f"{rel}: `category` must be a string literal")
    # Not every template declares a model_type (most PyTorch ones do not need
    # one); absent is null, but a declared non-string is a mistake.
    model_type = consts.get("model_type")
    if "model_type" in consts and not isinstance(model_type, str):
        raise ValueError(f"{rel}: `model_type` must be a string literal")
    batch = consts.get("batch_size")
    # bool is an int subclass; `batch_size = True` is not a batch size.
    if type(batch) is not int or batch < 1:
        raise ValueError(
            f"{rel}: `batch_size` must be a positive int literal, got {batch!r}"
        )
    return {
        "id": str(rel.with_suffix("")),
        "upload": upload_path(rel),
        "category": consts["category"],
        "framework": framework,
        "model_type": model_type,
        "label": label_for(template_name(rel)),
        "batch": batch,
    }


# --- computed fields (need the template's framework) -------------------------


def _importable(name: str) -> bool:
    """Imports, not merely installed: a library whose native part cannot load
    (xgboost without an OpenMP runtime) cannot build its templates either."""
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


def can_build(path: pathlib.Path, framework: str) -> bool:
    """Whether this environment has the framework and every optional library
    the template imports."""
    if not _importable(FRAMEWORK_IMPORT_NAME[framework]):
        return False
    text = path.read_text(encoding="utf-8")
    imported = set(re.findall(r"^\s*(?:from|import)\s+(\w+)", text, re.MULTILINE))
    return all(_importable(mod) for mod in imported & OPTIONAL_THIRD_PARTY)


def _load(path: pathlib.Path):
    name = "zoo_index_" + re.sub(r"\W", "_", str(path.relative_to(MODEL_ROOT)))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Loaded the way tests/test_model_contract.py loads it. Templates print
    # while they build; the index output should not.
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    entry = getattr(module, "main_class", None) or getattr(module, "main_method", None)
    if not entry or not hasattr(module, entry):
        raise ValueError(f"{path.relative_to(MODEL_ROOT)}: no main_class/main_method")
    return getattr(module, entry)


def count_params(path: pathlib.Path) -> int:
    """Total ``numel`` over ``parameters()``, built on the meta device: shapes
    without storage, so a 2.6B-parameter template costs nothing to count."""
    import torch

    entry = _load(path)
    with torch.device("meta"), contextlib.redirect_stdout(io.StringIO()):
        model = entry()
    return int(sum(p.numel() for p in model.parameters()))


def _final_estimator(model):
    """A Pipeline is judged by its last step (nested ones too)."""
    for _ in range(10):
        steps = getattr(model, "steps", None)
        if not (isinstance(steps, list) and steps):
            break
        model = steps[-1][1]
    return model


def is_averageable(path: pathlib.Path, framework: str) -> bool:
    if framework == "pytorch":
        return True
    allowed = []
    for module_name, cls_name in AVERAGEABLE_ESTIMATORS[framework]:
        if not _importable(module_name):
            continue
        cls = getattr(importlib.import_module(module_name), cls_name, None)
        if cls is not None:
            allowed.append(cls)
    with contextlib.redirect_stdout(io.StringIO()):
        estimator = _final_estimator(_load(path)())
    return isinstance(estimator, tuple(allowed))


# --- the index ---------------------------------------------------------------


def _offline() -> None:
    """Build with the hub and the network shut, the way the test suite does
    (``tests/conftest.py``): a template that fetched at construction would
    otherwise download while being indexed. The environment is TAKEN from
    ``tools/prep_offline_weights.py``'s ``_offline_env`` rather than restated,
    so the two cannot drift; the datasets flag goes on top, as in conftest."""
    spec = importlib.util.spec_from_file_location(
        "prep_offline_weights", ROOT / "tools" / "prep_offline_weights.py"
    )
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    os.environ.update(prep._offline_env(tempfile.mkdtemp(prefix="model-zoo-index-cache-")))
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    prep._block_network()


def load_committed() -> dict[str, dict]:
    if not INDEX_PATH.is_file():
        return {}
    return {row["id"]: row for row in json.loads(INDEX_PATH.read_text())["templates"]}


def build(committed: dict[str, dict] | None = None) -> tuple[dict, list[str]]:
    """The index, and the ids whose computed fields were carried from
    ``committed`` because this environment cannot build them."""
    committed = load_committed() if committed is None else committed
    rows, carried = [], []
    for path, consts in discover():
        row = static_row(path, consts)
        if can_build(path, row["framework"]):
            try:
                row["params"] = (
                    count_params(path) if row["framework"] == "pytorch" else None
                )
                row["averageable"] = is_averageable(path, row["framework"])
            except Exception as exc:
                raise RuntimeError(f"{row['id']}: could not be indexed: {exc!r}") from exc
        elif row["id"] in committed:
            row["params"] = committed[row["id"]].get("params")
            row["averageable"] = committed[row["id"]].get("averageable")
            carried.append(row["id"])
        else:
            raise RuntimeError(
                f"{row['id']} is new and needs {row['framework']} (and every library "
                "it imports) installed to be indexed -- see "
                ".github/requirements/<framework>.txt"
            )
        rows.append(row)
    return {"version": INDEX_VERSION, "templates": rows}, carried


def render(index: dict) -> str:
    return json.dumps(index, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if the committed index is stale"
    )
    args = parser.parse_args(argv)
    _offline()
    index, carried = build()
    text = render(index)
    if carried:
        print(
            f"note: {len(carried)} row(s) carried from the committed index "
            "(their framework is not installed here); CI recomputes them.",
            file=sys.stderr,
        )
    if args.check:
        if INDEX_PATH.is_file() and INDEX_PATH.read_text() == text:
            print(f"{INDEX_PATH.relative_to(ROOT)} is up to date.")
            return 0
        print(
            f"{INDEX_PATH.relative_to(ROOT)} is stale. Run: python tools/build_index.py",
            file=sys.stderr,
        )
        return 1
    INDEX_PATH.write_text(text)
    print(f"wrote {INDEX_PATH.relative_to(ROOT)} ({len(index['templates'])} templates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
