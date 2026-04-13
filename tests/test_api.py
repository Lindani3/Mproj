"""Tests for the Flask REST API."""
import json
import pytest

from student_performance.api.app import create_app

VALID_FEATURES = {
    "age": 20,
    "gender": "M",
    "study_hours": 10.0,
    "attendance_rate": 0.85,
    "previous_grade": 65.0,
    "extracurricular": 1,
    "internet_access": 1,
    "parent_education": "secondary",
}


@pytest.fixture(scope="module")
def client():
    app = create_app({"TESTING": True})
    with app.test_client() as c:
        yield c


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_status_ok(client):
    response = client.get("/health")
    data = json.loads(response.data)
    assert data["status"] == "ok"
    assert "version" in data


def test_predict_pass_valid_returns_200(client):
    response = client.post(
        "/predict/pass",
        data=json.dumps({"features": VALID_FEATURES}),
        content_type="application/json",
    )
    assert response.status_code == 200


def test_predict_pass_valid_returns_prediction(client):
    response = client.post(
        "/predict/pass",
        data=json.dumps({"features": VALID_FEATURES}),
        content_type="application/json",
    )
    data = json.loads(response.data)
    assert "prediction" in data
    assert data["prediction"] in (0, 1)
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_grade_valid_returns_200(client):
    response = client.post(
        "/predict/grade",
        data=json.dumps({"features": VALID_FEATURES}),
        content_type="application/json",
    )
    assert response.status_code == 200


def test_predict_grade_valid_returns_grade(client):
    response = client.post(
        "/predict/grade",
        data=json.dumps({"features": VALID_FEATURES}),
        content_type="application/json",
    )
    data = json.loads(response.data)
    assert "predicted_grade" in data
    assert 0.0 <= data["predicted_grade"] <= 100.0


def test_predict_pass_missing_features_returns_400(client):
    incomplete = {"age": 20, "gender": "M"}
    response = client.post(
        "/predict/pass",
        data=json.dumps({"features": incomplete}),
        content_type="application/json",
    )
    assert response.status_code == 400


def test_predict_grade_missing_features_returns_400(client):
    incomplete = {"age": 20}
    response = client.post(
        "/predict/grade",
        data=json.dumps({"features": incomplete}),
        content_type="application/json",
    )
    assert response.status_code == 400


def test_predict_pass_no_body_returns_400(client):
    response = client.post("/predict/pass", content_type="application/json")
    assert response.status_code == 400


def test_model_info_returns_200(client):
    response = client.get("/model/info")
    assert response.status_code == 200


def test_model_info_content(client):
    response = client.get("/model/info")
    data = json.loads(response.data)
    assert "classifier" in data
    assert "regressor" in data
    assert data["classifier"]["is_fitted"] is True
    assert data["regressor"]["is_fitted"] is True
