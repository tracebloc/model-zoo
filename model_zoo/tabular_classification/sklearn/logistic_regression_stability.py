"""LASSO logistic regression with stability-based feature selection.

A drop-in tabular-classification model whose ``fit`` runs repeated stratified
cross-validation and records, per fold, which features the model keeps
(non-zero L1 coefficients) and the held-out AUROC. After fit it behaves like
any sklearn classifier — ``predict`` / ``predict_proba`` come from a final
model trained on all rows — while also exposing a stability report:

    selection_frequency_   fraction of folds in which each feature was kept
    mean_coefficient_      mean signed coefficient per feature (direction of effect)
    fold_aurocs_           held-out AUROC for every fold

Call ``get_stability_report()`` for a ready-to-tabulate dict. The fitted model
is saved by the tracebloc client as the experiment's weights artifact; download
it, load the estimator, and read these attributes to build a selection-frequency
panel — no raw data leaves the environment.

Preprocessing (median imputation + standardisation) is fit INSIDE each CV fold
via an sklearn Pipeline, so per-fold statistics never leak across the split.
Class imbalance is handled with ``class_weight='balanced'``.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

framework = "sklearn"
model_type = "linear"
main_method = "MyModel"
batch_size = 4
output_classes = 2
category = "tabular_classification"
num_feature_points = 12
license = "MIT"


class StabilityFeatureSelector(ClassifierMixin, BaseEstimator):
    """Repeated stratified-CV feature-stability wrapper around a base classifier.

    ``fit`` runs ``n_splits`` x ``n_repeats`` cross-validation, fitting a fresh
    ``Pipeline(impute -> scale -> base_estimator)`` per fold so imputation and
    scaling are learned inside the fold (no leakage). It tracks how often each
    feature is selected and its mean effect, plus per-fold held-out AUROC, then
    trains one final pipeline on all rows for ``predict`` / ``predict_proba``.
    """

    def __init__(
        self,
        base_estimator=None,
        n_splits=10,
        n_repeats=10,
        random_state=42,
        importance_threshold=None,
    ):
        self.base_estimator = base_estimator
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.importance_threshold = importance_threshold

    @staticmethod
    def _build_pipeline(estimator):
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("estimator", estimator),
            ]
        )

    def _selection_signal(self, fitted_estimator, n_features):
        """Return (selected_mask, signed_effect) for one fitted leaf estimator."""
        if hasattr(fitted_estimator, "coef_"):
            coef = np.atleast_2d(np.asarray(fitted_estimator.coef_, dtype=float))
            # Binary logreg has shape (1, n_features); multiclass has (n_classes, n_features).
            # Use the per-feature coefficient with the largest magnitude so both
            # the selected mask and the signed effect stay meaningful across classes.
            if coef.shape[1] != n_features:
                return np.zeros(n_features, dtype=bool), np.zeros(n_features)
            argmax_class = np.argmax(np.abs(coef), axis=0)
            signed = coef[argmax_class, np.arange(n_features)]
            return np.abs(signed) > 1e-8, signed
        if hasattr(fitted_estimator, "feature_importances_"):
            imp = np.asarray(fitted_estimator.feature_importances_, dtype=float)
            threshold = (
                self.importance_threshold
                if self.importance_threshold is not None
                else 1.0 / max(n_features, 1)
            )
            return imp > threshold, imp
        # No coef_/feature_importances_ — permutation importance is out of scope for V1.
        return np.zeros(n_features, dtype=bool), np.zeros(n_features)

    def fit(self, X, y):
        if self.base_estimator is None:
            raise ValueError("StabilityFeatureSelector requires a base_estimator.")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        n_features = X.shape[1]
        self.n_features_in_ = n_features
        self.classes_ = np.unique(y)

        # StratifiedKFold needs n_splits <= smallest class count; degrade gracefully
        # (e.g. the SDK's tiny dummy-data validation fit) instead of erroring.
        _, class_counts = np.unique(y, return_counts=True)
        min_class = int(class_counts.min()) if class_counts.size else 0
        eff_splits = min(self.n_splits, min_class)

        select_counts = np.zeros(n_features)
        effect_sums = np.zeros(n_features)
        fold_aurocs = []
        n_folds = 0

        # Stratified CV needs >=2 classes overall AND enough samples per class.
        if eff_splits >= 2 and self.classes_.size >= 2:
            splitter = RepeatedStratifiedKFold(
                n_splits=eff_splits,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
            )
            for train_idx, val_idx in splitter.split(X, y):
                pipe = self._build_pipeline(clone(self.base_estimator))
                pipe.fit(X[train_idx], y[train_idx])
                selected, effect = self._selection_signal(
                    pipe.named_steps["estimator"], n_features
                )
                select_counts += selected
                effect_sums += effect
                n_folds += 1
                y_val = y[val_idx]
                if np.unique(y_val).size >= 2 and hasattr(pipe, "predict_proba"):
                    try:
                        proba = pipe.predict_proba(X[val_idx])
                        if proba.shape[1] == 2:
                            score = roc_auc_score(y_val, proba[:, 1])
                        else:
                            score = roc_auc_score(
                                y_val, proba, multi_class="ovr", labels=self.classes_
                            )
                        fold_aurocs.append(float(score))
                    except Exception:
                        pass

        self.n_folds_total_ = n_folds
        self.selection_frequency_ = (
            select_counts / n_folds if n_folds else select_counts
        )
        self.mean_coefficient_ = effect_sums / n_folds if n_folds else effect_sums
        self.fold_aurocs_ = fold_aurocs

        # Final model on all rows — used for predict/predict_proba and weight export.
        self.final_estimator_ = self._build_pipeline(
            clone(self.base_estimator)
        ).fit(X, y)
        return self

    def predict(self, X):
        return self.final_estimator_.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X):
        return self.final_estimator_.predict_proba(np.asarray(X, dtype=float))

    def get_stability_report(self):
        """Plain dict of the stability results, ready to tabulate or store as JSON."""
        freq = np.asarray(getattr(self, "selection_frequency_", []))
        eff = np.asarray(getattr(self, "mean_coefficient_", []))
        features = [
            {
                "feature_index": int(i),
                "selection_frequency": float(freq[i]),
                "mean_effect": float(eff[i]),
                "direction": int(np.sign(eff[i])),
            }
            for i in range(len(freq))
        ]
        features.sort(key=lambda d: d["selection_frequency"], reverse=True)
        aurocs = [float(a) for a in getattr(self, "fold_aurocs_", [])]
        summary = {}
        if aurocs:
            summary = {
                "auroc_mean": float(np.mean(aurocs)),
                "auroc_std": float(np.std(aurocs)),
                "auroc_min": float(np.min(aurocs)),
                "auroc_max": float(np.max(aurocs)),
                "n_folds": int(getattr(self, "n_folds_total_", 0)),
            }
        return {"features": features, "fold_aurocs": aurocs, "summary": summary}


def MyModel():
    return StabilityFeatureSelector(
        base_estimator=LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
        ),
        n_splits=10,
        n_repeats=10,
        random_state=42,
    )
