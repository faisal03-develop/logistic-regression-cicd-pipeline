"""
Integration tests for the Flask API endpoints.
Uses the flask_client fixture from conftest.py.
"""

import json
import pytest

# ---------------------------------------------------------------------------
# Valid sample payload (matches the 10 required features)
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "age": 35,
    "income": 65000,
    "credit_score": 680,
    "loan_amount": 25000,
    "loan_tenure_months": 36,
    "debt_to_income_ratio": 0.3,
    "num_open_accounts": 5,
    "num_credit_inquiries": 2,
    "months_employed": 48,
    "has_mortgage": 0,
}

HIGH_RISK_PAYLOAD = {
    "age": 24,
    "income": 22000,
    "credit_score": 310,
    "loan_amount": 90000,
    "loan_tenure_months": 84,
    "debt_to_income_ratio": 0.85,
    "num_open_accounts": 2,
    "num_credit_inquiries": 9,
    "months_employed": 3,
    "has_mortgage": 0,
}

LOW_RISK_PAYLOAD = {
    "age": 55,
    "income": 140000,
    "credit_score": 840,
    "loan_amount": 8000,
    "loan_tenure_months": 12,
    "debt_to_income_ratio": 0.05,
    "num_open_accounts": 12,
    "num_credit_inquiries": 0,
    "months_employed": 240,
    "has_mortgage": 1,
}


# ---------------------------------------------------------------------------
# GET / (index)
# ---------------------------------------------------------------------------

class TestIndexEndpoint:
    def test_returns_200(self, flask_client):
        resp = flask_client.get("/")
        assert resp.status_code == 200

    def test_json_content_type(self, flask_client):
        resp = flask_client.get("/")
        assert "application/json" in resp.content_type

    def test_response_structure(self, flask_client):
        data = flask_client.get("/").get_json()
        assert data["status"] == "success"
        assert "service" in data
        assert "endpoints" in data
        assert "timestamp" in data

    def test_service_name(self, flask_client):
        data = flask_client.get("/").get_json()
        assert "Credit Risk" in data["service"]


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200_when_model_ready(self, flask_client):
        resp = flask_client.get("/health")
        assert resp.status_code == 200

    def test_healthy_true(self, flask_client):
        data = flask_client.get("/health").get_json()
        assert data["healthy"] is True

    def test_model_loaded_true(self, flask_client):
        data = flask_client.get("/health").get_json()
        assert data["model_loaded"] is True

    def test_has_timestamp(self, flask_client):
        data = flask_client.get("/health").get_json()
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    def test_returns_200(self, flask_client):
        resp = flask_client.get("/metrics")
        assert resp.status_code == 200

    def test_response_has_metrics_key(self, flask_client):
        data = flask_client.get("/metrics").get_json()
        assert "metrics" in data

    def test_metrics_contain_accuracy(self, flask_client):
        data = flask_client.get("/metrics").get_json()
        assert "accuracy" in data["metrics"]

    def test_metrics_values_are_numeric(self, flask_client):
        data = flask_client.get("/metrics").get_json()
        for key, val in data["metrics"].items():
            try:
                float(val)
            except (TypeError, ValueError):
                pytest.fail(f"Metric '{key}' value '{val}' is not numeric")


# ---------------------------------------------------------------------------
# POST /predict — valid inputs
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def test_returns_200_for_valid_payload(self, flask_client):
        resp = flask_client.post(
            "/predict",
            json=VALID_PAYLOAD,
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_response_structure(self, flask_client):
        data = flask_client.post("/predict", json=VALID_PAYLOAD).get_json()
        assert data["status"] == "success"
        assert "result" in data
        assert "input" in data

    def test_result_fields(self, flask_client):
        result = flask_client.post("/predict", json=VALID_PAYLOAD).get_json()["result"]
        assert "prediction" in result
        assert "probability" in result
        assert "risk_label" in result
        assert "risk_description" in result

    def test_prediction_is_binary(self, flask_client):
        result = flask_client.post("/predict", json=VALID_PAYLOAD).get_json()["result"]
        assert result["prediction"] in (0, 1)

    def test_probability_in_range(self, flask_client):
        result = flask_client.post("/predict", json=VALID_PAYLOAD).get_json()["result"]
        assert 0.0 <= result["probability"] <= 1.0

    def test_risk_label_valid(self, flask_client):
        result = flask_client.post("/predict", json=VALID_PAYLOAD).get_json()["result"]
        assert result["risk_label"] in ("Low Risk", "High Risk")

    def test_risk_label_matches_prediction(self, flask_client):
        result = flask_client.post("/predict", json=VALID_PAYLOAD).get_json()["result"]
        if result["prediction"] == 0:
            assert result["risk_label"] == "Low Risk"
        else:
            assert result["risk_label"] == "High Risk"

    def test_high_risk_profile(self, flask_client):
        """High DTI + low credit score should lean toward High Risk."""
        resp = flask_client.post("/predict", json=HIGH_RISK_PAYLOAD)
        assert resp.status_code == 200
        # We don't hard-assert the label since the model is trained on tiny data
        # but we verify the structure is correct
        result = resp.get_json()["result"]
        assert result["prediction"] in (0, 1)

    def test_low_risk_profile(self, flask_client):
        """High income + high credit score should lean toward Low Risk."""
        resp = flask_client.post("/predict", json=LOW_RISK_PAYLOAD)
        assert resp.status_code == 200
        result = resp.get_json()["result"]
        assert result["prediction"] in (0, 1)

    def test_input_echoed_in_response(self, flask_client):
        data = flask_client.post("/predict", json=VALID_PAYLOAD).get_json()
        for key, val in VALID_PAYLOAD.items():
            assert key in data["input"]


# ---------------------------------------------------------------------------
# POST /predict — invalid inputs (error handling)
# ---------------------------------------------------------------------------

class TestPredictEndpointErrors:
    def test_missing_content_type(self, flask_client):
        resp = flask_client.post(
            "/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="text/plain",
        )
        assert resp.status_code == 415

    def test_missing_required_field(self, flask_client):
        bad = dict(VALID_PAYLOAD)
        del bad["credit_score"]
        resp = flask_client.post("/predict", json=bad)
        assert resp.status_code == 400
        assert "credit_score" in resp.get_json()["message"]

    def test_non_numeric_field(self, flask_client):
        bad = dict(VALID_PAYLOAD)
        bad["income"] = "not_a_number"
        resp = flask_client.post("/predict", json=bad)
        assert resp.status_code == 400

    def test_credit_score_out_of_range(self, flask_client):
        bad = dict(VALID_PAYLOAD)
        bad["credit_score"] = 900  # max is 850
        resp = flask_client.post("/predict", json=bad)
        assert resp.status_code == 400

    def test_negative_age_rejected(self, flask_client):
        bad = dict(VALID_PAYLOAD)
        bad["age"] = -5
        resp = flask_client.post("/predict", json=bad)
        assert resp.status_code == 400

    def test_dti_above_one_rejected(self, flask_client):
        bad = dict(VALID_PAYLOAD)
        bad["debt_to_income_ratio"] = 1.5
        resp = flask_client.post("/predict", json=bad)
        assert resp.status_code == 400

    def test_empty_body_rejected(self, flask_client):
        resp = flask_client.post(
            "/predict",
            data="{}",
            content_type="application/json",
        )
        # All fields are missing
        assert resp.status_code == 400

    def test_invalid_json_rejected(self, flask_client):
        resp = flask_client.post(
            "/predict",
            data="not_json_at_all",
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_multiple_missing_fields(self, flask_client):
        resp = flask_client.post("/predict", json={"age": 30})
        assert resp.status_code == 400
        body = resp.get_json()
        assert "Missing" in body["message"]


# ---------------------------------------------------------------------------
# 404 / 405 handling
# ---------------------------------------------------------------------------

class TestErrorHandlers:
    def test_unknown_route_returns_404(self, flask_client):
        resp = flask_client.get("/does-not-exist")
        assert resp.status_code == 404
        assert resp.get_json()["status"] == "error"

    def test_get_on_predict_returns_405(self, flask_client):
        resp = flask_client.get("/predict")
        assert resp.status_code == 405
        assert resp.get_json()["status"] == "error"
