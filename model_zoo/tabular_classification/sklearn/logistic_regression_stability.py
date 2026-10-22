"""Stability selection with L1-penalised logistic regression.

For biomarker discovery: many candidate features and few rows, where per-feature
testing finds markers by chance and a single lasso picks a different set on every
resample. ``fit`` draws ``n_subsamples`` random halves of the training rows and
fits an L1-penalised logistic regression on each half at every penalty of a
path of ``n_penalties`` values, from ``penalty_range[0]`` to ``penalty_range[1]``
times the smallest C that selects anything. A feature's stability is the largest
share of fits, over the path, that kept it; the features at or above
``threshold`` form the stable set. One logistic regression, refit on the stable
set, makes every prediction.

The Meinshausen-Buehlmann bound is reported beside the result: at most
q**2 / ((2 * threshold - 1) * p) features of the stable set are expected to be
false selections, where q is the mean number of features a half-sample selects
anywhere on the path and p is the number of features.

Features are standardised on the training rows before any fit. Run the
experiment with 1 epoch and 1 cycle: the resampling happens inside one ``fit``,
on one CPU core.

The fitted model is saved as a plain scikit-learn ``Pipeline``
(``StandardScaler`` then ``LogisticRegression``) whose coefficients are zero off
the stable set, so a run's weights artifact loads without this file. The report
rides on it as ``stability_report_``::

    import pickle

    with open("trained_weights.pkl", "rb") as f:
        model = pickle.load(f)
    report = model.stability_report_
    report["stable_set"]       # feature, index, stability, coefficient and odds ratio per SD
    report["stability"]        # every feature's stability and share per penalty, highest first
    report["expected_false_selections_bound"]

A feature is named by its column when ``fit`` is given a DataFrame, else ``x<i>``
for the i-th column of the training table.
"""

import math

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import l1_min_c

framework = "sklearn"
model_type = "linear"
main_method = "MyModel"
batch_size = 4
output_classes = 2
category = "tabular_classification"
num_feature_points = 12
license = "MIT"


def _l1_logistic(c, random_state):
    """scikit-learn 1.8 deprecated ``penalty`` and reads the L1 penalty from ``l1_ratio``."""
    version = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
    if version >= (1, 8):
        return LogisticRegression(
            l1_ratio=1.0, C=c, solver="liblinear", random_state=random_state
        )
    return LogisticRegression(
        penalty="l1", C=c, solver="liblinear", random_state=random_state
    )


class StabilitySelectionClassifier(ClassifierMixin, BaseEstimator):
    """Stability selection over a penalty path, then a refit on the stable set.

    Parameters
    ----------
    n_subsamples : int
        Random halves of the training rows, each fitted at every penalty.
    n_penalties : int
        Penalties on the path, spaced evenly on a log scale.
    penalty_range : tuple of two floats
        The path's first and last C, as multiples of the smallest C at which
        an L1 fit on all rows selects any feature.
    threshold : float
        The stability, in (0.5, 1], at which a feature joins the stable set.
    refit_C : float
        Inverse L2 strength of the refit on the stable set.
    random_state : int
        Seeds the half-samples and the solver.
    """

    def __init__(
        self,
        n_subsamples=1000,
        n_penalties=10,
        penalty_range=(1.5, 15.0),
        threshold=0.75,
        refit_C=1.0,
        random_state=7,
    ):
        self.n_subsamples = n_subsamples
        self.n_penalties = n_penalties
        self.penalty_range = penalty_range
        self.threshold = threshold
        self.refit_C = refit_C
        self.random_state = random_state

    def fit(self, X, y):
        names = (
            [str(c) for c in X.columns]
            if hasattr(X, "columns")
            else [f"x{i}" for i in range(np.shape(X)[1])]
        )
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if not 0.5 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0.5, 1], got {self.threshold}")
        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("Stability selection needs at least two classes in y.")

        scaler = StandardScaler().fit(X)
        z = scaler.transform(X)
        n_rows, n_features = z.shape
        steps = np.logspace(
            math.log10(self.penalty_range[0]),
            math.log10(self.penalty_range[1]),
            self.n_penalties,
        )
        path = l1_min_c(z, y, loss="log") * steps

        rng = np.random.default_rng(self.random_state)
        half = n_rows // 2
        counts = np.zeros((len(path), n_features))
        union_sizes = []
        for _ in range(self.n_subsamples):
            rows = rng.permutation(n_rows)[:half]
            # A half holding one class has nothing to fit; it counts as no draw.
            if np.unique(y[rows]).size < 2:
                continue
            union = np.zeros(n_features, dtype=bool)
            for k, c in enumerate(path):
                fit = _l1_logistic(c, self.random_state).fit(z[rows], y[rows])
                chosen = np.any(fit.coef_ != 0, axis=0)
                counts[k] += chosen
                union |= chosen
            union_sizes.append(int(union.sum()))

        drawn = len(union_sizes)
        shares = counts / max(drawn, 1)
        stability = shares.max(axis=0)
        stable = np.flatnonzero(stability >= self.threshold)
        q = float(np.mean(union_sizes)) if drawn else 0.0
        bound = q**2 / ((2 * self.threshold - 1) * n_features)

        refit = LogisticRegression(C=self.refit_C, max_iter=5000)
        if stable.size:
            refit.fit(z[:, stable], y)
            coef = np.zeros((refit.coef_.shape[0], n_features))
            coef[:, stable] = refit.coef_
        else:
            # Nothing is stable: the refit predicts the training base rate.
            refit.fit(np.zeros((n_rows, 1)), y)
            coef = np.zeros((refit.coef_.shape[0], n_features))
        refit.coef_ = coef
        refit.n_features_in_ = n_features

        self.model_ = Pipeline([("scale", scaler), ("model", refit)])
        self.model_.stability_report_ = {
            "method": {
                "kind": "stability selection, L1-penalised logistic regression",
                "subsamples": drawn,
                "subsample_size": half,
                "penalty_path_c": [float(c) for c in path],
                "penalty_path_steps_over_l1_min_c": [float(s) for s in steps],
                "threshold": float(self.threshold),
                "features": n_features,
                "random_state": self.random_state,
            },
            "stable_set": [
                {
                    "feature": names[i],
                    "index": int(i),
                    "stability": float(stability[i]),
                    "coefficient_per_sd": float(coef[-1, i]),
                    "odds_ratio_per_sd": float(math.exp(coef[-1, i])),
                }
                for i in sorted(stable, key=lambda i: -stability[i])
            ],
            "stability": [
                {
                    "feature": names[i],
                    "index": int(i),
                    "stability": float(stability[i]),
                    "share_by_penalty": [float(share) for share in shares[:, i]],
                }
                for i in np.argsort(-stability, kind="stable")
            ],
            "mean_selected_q": q,
            "expected_false_selections_bound": float(bound),
            "refit": {"C": float(self.refit_C), "intercept": float(refit.intercept_[-1])},
        }
        self.classes_ = self.model_.classes_
        self.n_features_in_ = n_features
        return self

    def __reduce_ex__(self, protocol):
        # The platform reloads weights with a loader that admits scikit-learn and
        # numpy classes only, never this file's: a fitted selector pickles as its Pipeline.
        if hasattr(self, "model_"):
            return (Pipeline, (self.model_.steps,), self.model_.__getstate__())
        return super().__reduce_ex__(protocol)

    def get_stability_report(self):
        return self.model_.stability_report_

    def decision_function(self, X):
        return self.model_.decision_function(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))


def MyModel():
    return StabilitySelectionClassifier()
