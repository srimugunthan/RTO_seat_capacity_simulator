"""Tests for POST /simulate endpoint."""
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
FIXTURES = Path(__file__).parent / "fixtures"


def test_basic_simulate_returns_200():
    payload = json.loads((FIXTURES / "basic_request.json").read_text())
    response = client.post("/simulate", json=payload)
    assert response.status_code == 200


def test_response_schema():
    payload = json.loads((FIXTURES / "basic_request.json").read_text())
    data = client.post("/simulate", json=payload).json()
    assert "model_name" in data
    assert "daily_results" in data
    assert "summary" in data


def test_daily_results_count():
    """One week Mon–Fri should return 5 daily results."""
    payload = {
        "params": {
            "total_employees": 100,
            "total_seats": 80,
            "wfh_days_per_week": 2,
            "num_simulation_runs": 100,
            "start_date": "2026-03-02",
            "end_date": "2026-03-06",
        },
        "backend": "binomial",
    }
    data = client.post("/simulate", json=payload).json()
    assert len(data["daily_results"]) == 5


def test_day_result_fields():
    payload = json.loads((FIXTURES / "basic_request.json").read_text())
    data = client.post("/simulate", json=payload).json()
    day = data["daily_results"][0]
    required = {
        "date", "day_of_week", "expected_occupancy", "std_dev",
        "overflow_probability", "percentile_5", "percentile_95",
        "effective_capacity", "team_breakdown"
    }
    assert required.issubset(day.keys())


def test_summary_fields():
    payload = json.loads((FIXTURES / "basic_request.json").read_text())
    data = client.post("/simulate", json=payload).json()
    summary = data["summary"]
    assert "avg_utilization" in summary
    assert "peak_occupancy" in summary
    assert "overflow_days_count" in summary
    assert "overflow_days_pct" in summary
    assert "avg_overflow_magnitude" in summary


def test_engineering_mandates_fixture():
    payload = json.loads((FIXTURES / "engineering_mandates.json").read_text())
    response = client.post("/simulate", json=payload)
    assert response.status_code == 200


def test_invalid_backend_returns_400():
    payload = {
        "params": {"total_employees": 100, "total_seats": 80},
        "backend": "nonexistent_model",
    }
    response = client.post("/simulate", json=payload)
    assert response.status_code == 400


def test_end_date_before_start_date_returns_422():
    payload = {
        "params": {
            "total_employees": 100,
            "total_seats": 80,
            "start_date": "2026-03-31",
            "end_date": "2026-03-01",
        },
        "backend": "binomial",
    }
    response = client.post("/simulate", json=payload)
    assert response.status_code == 422


def test_both_backends_return_results():
    base = {
        "params": {
            "total_employees": 200,
            "total_seats": 150,
            "wfh_days_per_week": 2,
            "num_simulation_runs": 200,
            "start_date": "2026-03-02",
            "end_date": "2026-03-06",
        }
    }
    for backend_id in ["binomial", "monte_carlo"]:
        payload = {**base, "backend": backend_id}
        response = client.post("/simulate", json=payload)
        assert response.status_code == 200, f"Failed for backend: {backend_id}"
