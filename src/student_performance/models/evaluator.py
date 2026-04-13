"""Model evaluation utilities."""
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from typing import Optional


class Evaluator:
    """Evaluates classification and regression models."""

    def classification_report(self, y_true, y_pred, y_proba=None) -> dict:
        """Return dict with accuracy, precision, recall, f1, and optionally roc_auc."""
        report = {
            "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
            "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        }
        if y_proba is not None:
            proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            report["roc_auc"] = float(metrics.roc_auc_score(y_true, proba))
        return report

    def regression_report(self, y_true, y_pred) -> dict:
        """Return dict with mse, rmse, mae, r2."""
        mse = float(metrics.mean_squared_error(y_true, y_pred))
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
            "r2": float(metrics.r2_score(y_true, y_pred)),
        }

    def cross_validate_classifier(self, model, X, y, cv: int = 5) -> dict:
        """Return mean/std of cross-validated accuracy scores."""
        scores = cross_val_score(model._pipeline, X, y, cv=cv, scoring="accuracy")
        return {"mean_accuracy": float(scores.mean()), "std_accuracy": float(scores.std())}

    def cross_validate_regressor(self, model, X, y, cv: int = 5) -> dict:
        """Return mean/std of cross-validated R² scores."""
        scores = cross_val_score(model._pipeline, X, y, cv=cv, scoring="r2")
        return {"mean_r2": float(scores.mean()), "std_r2": float(scores.std())}
