"""Tests for feature engineering."""
import pytest
import pandas as pd
import numpy as np

from student_performance.data.loader import DataLoader
from student_performance.data.preprocessor import Preprocessor
from student_performance.features.engineering import FeatureEngineer


@pytest.fixture
def processed_df():
    loader = DataLoader()
    df = loader.load_sample_data()
    prep = Preprocessor()
    return prep.fit_transform(df)


def test_feature_engineer_fit_transform_creates_columns(processed_df):
    fe = FeatureEngineer()
    df_eng = fe.fit_transform(processed_df)
    assert "study_attendance_interaction" in df_eng.columns
    assert "grade_improvement" in df_eng.columns
    assert "high_performer" in df_eng.columns


def test_study_attendance_interaction_values(processed_df):
    fe = FeatureEngineer()
    df_eng = fe.fit_transform(processed_df)
    expected = processed_df["study_hours"] * processed_df["attendance_rate"]
    pd.testing.assert_series_equal(
        df_eng["study_attendance_interaction"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_high_performer_binary(processed_df):
    fe = FeatureEngineer()
    df_eng = fe.fit_transform(processed_df)
    assert set(df_eng["high_performer"].unique()).issubset({0, 1})


def test_feature_engineer_is_fitted(processed_df):
    fe = FeatureEngineer()
    assert not fe.is_fitted
    fe.fit_transform(processed_df)
    assert fe.is_fitted


def test_feature_engineer_transform_without_fit(processed_df):
    fe = FeatureEngineer()
    with pytest.raises(RuntimeError):
        fe.transform(processed_df)


def test_feature_engineer_transform_after_fit(processed_df):
    fe = FeatureEngineer()
    fe.fit_transform(processed_df)
    df_eng2 = fe.transform(processed_df)
    assert "study_attendance_interaction" in df_eng2.columns
