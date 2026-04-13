"""Tests for data loading and preprocessing."""
import pytest
import pandas as pd
import numpy as np

from student_performance.data.loader import DataLoader
from student_performance.data.preprocessor import Preprocessor


@pytest.fixture
def sample_df():
    loader = DataLoader()
    return loader.load_sample_data()


def test_load_sample_data_shape(sample_df):
    assert sample_df.shape[0] == 500
    assert "final_grade" in sample_df.columns
    assert "passed" in sample_df.columns


def test_load_sample_data_columns(sample_df):
    expected = ["student_id", "age", "gender", "study_hours", "attendance_rate",
                "previous_grade", "extracurricular", "internet_access",
                "parent_education", "final_grade", "passed"]
    for col in expected:
        assert col in sample_df.columns


def test_load_sample_data_passed_binary(sample_df):
    assert set(sample_df["passed"].unique()).issubset({0, 1})


def test_validate_data_valid(sample_df):
    loader = DataLoader()
    is_valid, issues = loader.validate_data(sample_df)
    assert is_valid
    assert issues == []


def test_validate_data_missing_column(sample_df):
    loader = DataLoader()
    df_missing = sample_df.drop(columns=["final_grade"])
    is_valid, issues = loader.validate_data(df_missing)
    assert not is_valid
    assert any("final_grade" in issue for issue in issues)


def test_validate_data_all_null_column(sample_df):
    loader = DataLoader()
    df_null = sample_df.copy()
    df_null["age"] = np.nan
    is_valid, issues = loader.validate_data(df_null)
    assert not is_valid


def test_load_csv_file_not_found():
    loader = DataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_csv("nonexistent_file.csv")


def test_preprocessor_fit_transform(sample_df):
    prep = Preprocessor()
    df_proc = prep.fit_transform(sample_df)
    assert prep.is_fitted
    assert "student_id" not in df_proc.columns
    assert df_proc.isnull().sum().sum() == 0


def test_preprocessor_fit_transform_shape(sample_df):
    prep = Preprocessor()
    df_proc = prep.fit_transform(sample_df)
    assert df_proc.shape[0] == 500


def test_preprocessor_transform_without_fit(sample_df):
    prep = Preprocessor()
    with pytest.raises(RuntimeError):
        prep.transform(sample_df)


def test_preprocessor_get_feature_names(sample_df):
    prep = Preprocessor()
    prep.fit_transform(sample_df)
    feature_names = prep.get_feature_names()
    assert isinstance(feature_names, list)
    assert "final_grade" not in feature_names
    assert "passed" not in feature_names


def test_preprocessor_transform_consistency(sample_df):
    prep = Preprocessor()
    df_proc1 = prep.fit_transform(sample_df)
    df_proc2 = prep.transform(sample_df)
    # Both should have same shape
    assert df_proc1.shape == df_proc2.shape
