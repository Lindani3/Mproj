"""Classification models for student pass/fail prediction."""
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class StudentClassifier:
    """Classifies whether a student will pass or fail."""

    SUPPORTED_MODELS = ["random_forest", "logistic_regression", "svm"]

    def __init__(self, model_type: str = "random_forest"):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {self.SUPPORTED_MODELS}")
        self.model_type = model_type
        self._pipeline: Pipeline = self._build_pipeline()
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _build_pipeline(self) -> Pipeline:
        if self.model_type == "random_forest":
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "logistic_regression":
            clf = LogisticRegression(max_iter=1000, random_state=42)
        else:
            clf = SVC(probability=True, random_state=42)
        return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])

    def fit(self, X, y) -> None:
        self._pipeline.fit(X, y)
        self._fitted = True

    def predict(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        return self._pipeline.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        return self._pipeline.predict_proba(X)

    def save(self, filepath: str) -> None:
        joblib.dump(self._pipeline, filepath)

    def load(self, filepath: str) -> None:
        self._pipeline = joblib.load(filepath)
        self._fitted = True
