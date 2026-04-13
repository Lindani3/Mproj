"""Feature engineering for student performance data."""
import pandas as pd


class FeatureEngineer:
    """Creates derived features from preprocessed student data."""

    def __init__(self):
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the DataFrame."""
        self._fitted = True
        return self._create_features(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted state."""
        if not self._fitted:
            raise RuntimeError("FeatureEngineer must be fitted before calling transform.")
        return self._create_features(df)

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "study_hours" in df.columns and "attendance_rate" in df.columns:
            df["study_attendance_interaction"] = df["study_hours"] * df["attendance_rate"]
        if "final_grade" in df.columns and "previous_grade" in df.columns:
            df["grade_improvement"] = df["final_grade"] - df["previous_grade"]
        else:
            df["grade_improvement"] = 0
        if "previous_grade" in df.columns:
            df["high_performer"] = (df["previous_grade"] >= 75).astype(int)
        return df
