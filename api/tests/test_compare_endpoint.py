"""Tests for POST /compare endpoint."""
import pytest
from fastapi.testclient import TestClient
from api.main import app, _simulation_cache

client = TestClient(app)

_BASE_PARAMS = {
    "total_employees": 200,
    "total_seats": 160,
    "wfh_days_per_week": 2,
    "seat_reduction_pct": 0.0,
    "mandatory_office_days": {},
    "day_of_week_weights": {
        "Monday": 0.85, "Tuesday": 1.00, "Wednesday": 1.00,
        "Thursday": 0.95, "Friday": 0.65,
    },
    "compliance_rate": 0.9,
    "teams": [],
    "num_simulation_runs": 500,
    "start_date": "2026-03-02",
    "end_date": "2026-03-06",
}


def _compare_payload(wfh_a: int = 2, wfh_b: int = 3):
    params_a = {**_BASE_PARAMS, "wfh_days_per_week": wfh_a}
    params_b = {**_BASE_PARAMS, "wfh_days_per_week": wfh_b}
    return {
        "scenario_a": {"params": params_a, "backend": "binomial"},
        "scenario_b": {"params": params_b, "backend": "binomial"},
        "label_a": f"{wfh_a}-day WFH",
        "label_b": f"{wfh_b}-day WFH",
    }


# ── Basic contract ─────────────────────────────────────────────────────────────

def test_compare_returns_200():
    response = client.post("/compare", json=_compare_payload())
    assert response.status_code == 200


def test_compare_response_has_both_scenarios():
    data = client.post("/compare", json=_compare_payload()).json()
    assert "scenario_a" in data
    assert "scenario_b" in data


def test_compare_labels_passed_through():
    data = client.post("/compare", json=_compare_payload(2, 3)).json()
    assert data["scenario_a"]["label"] == "2-day WFH"
    assert data["scenario_b"]["label"] == "3-day WFH"


def test_compare_each_scenario_has_simulation_result():
    data = client.post("/compare", json=_compare_payload()).json()
    for key in ("scenario_a", "scenario_b"):
        sim = data[key]["simulation_result"]
        assert "daily_results" in sim
        assert "summary" in sim
        assert "model_name" in sim


def test_compare_daily_results_non_empty():
    data = client.post("/compare", json=_compare_payload()).json()
    assert len(data["scenario_a"]["simulation_result"]["daily_results"]) == 5
    assert len(data["scenario_b"]["simulation_result"]["daily_results"]) == 5


# ── Scenario differentiation ──────────────────────────────────────────────────

def test_more_wfh_lowers_occupancy():
    """Scenario B (3-day WFH) should have lower avg occupancy than A (2-day WFH)."""
    data = client.post("/compare", json=_compare_payload(2, 3)).json()
    util_a = data["scenario_a"]["simulation_result"]["summary"]["avg_utilization"]
    util_b = data["scenario_b"]["simulation_result"]["summary"]["avg_utilization"]
    assert util_b < util_a, (
        f"Expected B ({util_b:.3f}) < A ({util_a:.3f}) with more WFH days"
    )


def test_compare_with_different_backends():
    params = {**_BASE_PARAMS}
    payload = {
        "scenario_a": {"params": params, "backend": "binomial"},
        "scenario_b": {"params": params, "backend": "monte_carlo"},
        "label_a": "Binomial",
        "label_b": "Monte Carlo",
    }
    response = client.post("/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["scenario_a"]["simulation_result"]["model_name"] != \
           data["scenario_b"]["simulation_result"]["model_name"]


# ── Error handling ─────────────────────────────────────────────────────────────

def test_compare_invalid_backend_returns_error():
    params = {**_BASE_PARAMS}
    payload = {
        "scenario_a": {"params": params, "backend": "bad_backend"},
        "scenario_b": {"params": params, "backend": "binomial"},
    }
    response = client.post("/compare", json=payload)
    assert response.status_code == 400


def test_compare_default_labels():
    """When label_a/label_b are omitted, defaults are used."""
    params = {**_BASE_PARAMS}
    payload = {
        "scenario_a": {"params": params, "backend": "binomial"},
        "scenario_b": {"params": params, "backend": "binomial"},
    }
    data = client.post("/compare", json=payload).json()
    assert data["scenario_a"]["label"] == "Scenario A"
    assert data["scenario_b"]["label"] == "Scenario B"


# ── Caching ────────────────────────────────────────────────────────────────────

def test_simulate_result_is_cached():
    """After /simulate, the same request via /compare should hit the cache."""
    params = {**_BASE_PARAMS, "wfh_days_per_week": 4}
    # First call: populates cache
    client.post("/simulate", json={"params": params, "backend": "binomial"})
    cache_size_before = len(_simulation_cache)
    # Second identical call: should NOT grow the cache
    client.post("/simulate", json={"params": params, "backend": "binomial"})
    assert len(_simulation_cache) == cache_size_before


def test_health_reports_cache_size():
    data = client.get("/health").json()
    assert "cache_size" in data
    assert isinstance(data["cache_size"], int)
