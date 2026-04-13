"""Regression models for student grade prediction."""
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge


class StudentRegressor:
    """Predicts a student's final grade (continuous value)."""

    SUPPORTED_MODELS = ["random_forest", "linear_regression", "ridge"]

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
            reg = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == "linear_regression":
            reg = LinearRegression()
        else:
            reg = Ridge(random_state=42)
        return Pipeline([("scaler", StandardScaler()), ("regressor", reg)])

    def fit(self, X, y) -> None:
        self._pipeline.fit(X, y)
        self._fitted = True

    def predict(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        return self._pipeline.predict(X)

    def save(self, filepath: str) -> None:
        joblib.dump(self._pipeline, filepath)

    def load(self, filepath: str) -> None:
        self._pipeline = joblib.load(filepath)
        self._fitted = True
