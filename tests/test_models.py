"""Tests for classification, regression, and evaluation models."""
import pytest
import numpy as np

from student_performance.data.loader import DataLoader
from student_performance.data.preprocessor import Preprocessor
from student_performance.features.engineering import FeatureEngineer
from student_performance.models.classifier import StudentClassifier
from student_performance.models.regressor import StudentRegressor
from student_performance.models.evaluator import Evaluator


@pytest.fixture
def prepared_data():
    loader = DataLoader()
    df = loader.load_sample_data()
    prep = Preprocessor()
    df_proc = prep.fit_transform(df)
    fe = FeatureEngineer()
    df_eng = fe.fit_transform(df_proc)
    feature_cols = [c for c in df_eng.columns if c not in ("final_grade", "passed")]
    X = df_eng[feature_cols].values
    y_cls = df_eng["passed"].values
    y_reg = df_eng["final_grade"].values
    return X, y_cls, y_reg


def test_classifier_random_forest_fit_predict(prepared_data):
    X, y_cls, _ = prepared_data
    clf = StudentClassifier(model_type="random_forest")
    assert not clf.is_fitted
    clf.fit(X, y_cls)
    assert clf.is_fitted
    preds = clf.predict(X)
    assert preds.shape == (500,)
    assert set(preds).issubset({0, 1})


def test_classifier_predict_proba(prepared_data):
    X, y_cls, _ = prepared_data
    clf = StudentClassifier(model_type="random_forest")
    clf.fit(X, y_cls)
    proba = clf.predict_proba(X)
    assert proba.shape == (500, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_classifier_logistic_regression(prepared_data):
    X, y_cls, _ = prepared_data
    clf = StudentClassifier(model_type="logistic_regression")
    clf.fit(X, y_cls)
    preds = clf.predict(X)
    assert preds.shape[0] == 500


def test_classifier_unsupported_type():
    with pytest.raises(ValueError):
        StudentClassifier(model_type="unsupported")


def test_classifier_predict_without_fit(prepared_data):
    X, _, _ = prepared_data
    clf = StudentClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(X)


def test_regressor_random_forest_fit_predict(prepared_data):
    X, _, y_reg = prepared_data
    reg = StudentRegressor(model_type="random_forest")
    assert not reg.is_fitted
    reg.fit(X, y_reg)
    assert reg.is_fitted
    preds = reg.predict(X)
    assert preds.shape == (500,)


def test_regressor_linear_regression(prepared_data):
    X, _, y_reg = prepared_data
    reg = StudentRegressor(model_type="linear_regression")
    reg.fit(X, y_reg)
    preds = reg.predict(X)
    assert preds.shape[0] == 500


def test_regressor_unsupported_type():
    with pytest.raises(ValueError):
        StudentRegressor(model_type="unsupported")


def test_regressor_predict_without_fit(prepared_data):
    X, _, _ = prepared_data
    reg = StudentRegressor()
    with pytest.raises(RuntimeError):
        reg.predict(X)


def test_evaluator_classification_report(prepared_data):
    X, y_cls, _ = prepared_data
    clf = StudentClassifier(model_type="random_forest")
    clf.fit(X, y_cls)
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)
    evaluator = Evaluator()
    report = evaluator.classification_report(y_cls, y_pred, y_proba)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in report
        assert 0.0 <= report[key] <= 1.0


def test_evaluator_classification_report_no_proba(prepared_data):
    X, y_cls, _ = prepared_data
    clf = StudentClassifier(model_type="random_forest")
    clf.fit(X, y_cls)
    y_pred = clf.predict(X)
    evaluator = Evaluator()
    report = evaluator.classification_report(y_cls, y_pred)
    assert "roc_auc" not in report
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in report


def test_evaluator_regression_report(prepared_data):
    X, _, y_reg = prepared_data
    reg = StudentRegressor(model_type="random_forest")
    reg.fit(X, y_reg)
    y_pred = reg.predict(X)
    evaluator = Evaluator()
    report = evaluator.regression_report(y_reg, y_pred)
    for key in ("mse", "rmse", "mae", "r2"):
        assert key in report
    assert report["rmse"] >= 0
    assert report["mae"] >= 0
