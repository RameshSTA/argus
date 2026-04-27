"""Integration tests for FastAPI endpoints."""
import os
import pytest
from fastapi.testclient import TestClient

HAS_ANTHROPIC = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
skip_if_no_anthropic = pytest.mark.skipif(
    not HAS_ANTHROPIC,
    reason="ANTHROPIC_API_KEY not set — skipping LLM-dependent tests",
)


@pytest.fixture(scope="module")
def client():
    from backend.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_score_valid(client):
    payload = {
        "transaction_amt": 5000.0, "card_type": "prepaid", "device_type": "mobile",
        "hour_of_day": 2, "transaction_velocity": 7.0, "account_age_days": 30,
        "address_match": False, "email_risk_score": 0.88,
        "distance_from_home_km": 900.0, "prior_claims_count": 1,
    }
    r = client.post("/api/score", json=payload)
    assert r.status_code == 200
    d = r.json()
    assert 0 <= d["fraud_probability"] <= 1
    assert d["risk_label"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert isinstance(d["shap_features"], list)
    assert d["claim_id"].startswith("TXN-")


def test_score_invalid_missing_field(client):
    r = client.post("/api/score", json={"transaction_amt": 100.0})
    assert r.status_code == 422


def test_score_invalid_hour(client):
    payload = {
        "transaction_amt": 100.0, "card_type": "credit", "device_type": "mobile",
        "hour_of_day": 30, "transaction_velocity": 1.0, "account_age_days": 100,
        "address_match": True, "email_risk_score": 0.1,
        "distance_from_home_km": 5.0, "prior_claims_count": 0,
    }
    r = client.post("/api/score", json=payload)
    assert r.status_code == 422


def test_index_status(client):
    r = client.get("/api/index/status")
    assert r.status_code == 200
    assert "status" in r.json()


@skip_if_no_anthropic
def test_query_valid(client):
    r = client.post("/api/query", json={"question": "Does storm damage cover parked vehicles?", "top_k": 3})
    assert r.status_code == 200
    d = r.json()
    assert isinstance(d["answer"], str)
    assert len(d["answer"]) > 10
    assert d["confidence"] in ("HIGH", "MEDIUM", "LOW")


@skip_if_no_anthropic
def test_agent_valid(client):
    r = client.post("/api/agent", json={
        "claim_description": "Vehicle damaged in hailstorm. Parked in open driveway. Claiming $2,800."
    })
    assert r.status_code == 200
    d = r.json()
    assert d["claim_id"].startswith("CLM-")
    assert isinstance(d["final_recommendation"], str)
    assert isinstance(d["tool_calls"], list)
    assert d["processing_time_ms"] >= 0
