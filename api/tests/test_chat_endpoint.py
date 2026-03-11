"""
Tests for POST /chat endpoint.

All LLM calls are mocked via pytest monkeypatch — no API key required.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app
from api.query_parser import ParamDelta

client = TestClient(app)


def mock_parse_query(delta: ParamDelta):
    """Patch api.main.parse_query to return a fixed delta."""
    return patch("api.main.parse_query", return_value=delta)


def mock_get_llm():
    """Patch api.main.get_llm to return a dummy object (not called directly in chat)."""
    return patch("api.main.get_llm", return_value=MagicMock())


def test_chat_creates_session():
    delta = ParamDelta(requires_simulation=False, explanation="Hello!")
    with mock_get_llm(), mock_parse_query(delta):
        response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["session_id"] is not None


def test_chat_returns_explanation():
    delta = ParamDelta(requires_simulation=False, explanation="No changes.")
    with mock_get_llm(), mock_parse_query(delta):
        response = client.post("/chat", json={"message": "Thanks"})
    assert response.json()["explanation"] == "No changes."


def test_chat_applies_param_delta():
    delta = ParamDelta(
        total_employees=300,
        total_seats=250,
        requires_simulation=True,
        explanation="Set 300 employees, 250 seats.",
    )
    with mock_get_llm(), mock_parse_query(delta):
        response = client.post("/chat", json={"message": "300 employees, 250 seats"})
    data = response.json()
    assert data["current_params"]["total_employees"] == 300
    assert data["current_params"]["total_seats"] == 250


def test_chat_runs_simulation_when_required():
    delta = ParamDelta(
        wfh_days_per_week=3,
        requires_simulation=True,
        explanation="3 days WFH.",
    )
    with mock_get_llm(), mock_parse_query(delta):
        response = client.post("/chat", json={"message": "3 days WFH"})
    data = response.json()
    assert data["requires_simulation"] is True
    assert data["simulation_result"] is not None
    assert "daily_results" in data["simulation_result"]
    assert "summary" in data["simulation_result"]


def test_chat_no_simulation_when_not_required():
    delta = ParamDelta(requires_simulation=False, explanation="Just chatting.")
    with mock_get_llm(), mock_parse_query(delta):
        response = client.post("/chat", json={"message": "Hello"})
    data = response.json()
    assert data["simulation_result"] is None


def test_chat_switches_backend():
    delta = ParamDelta(backend="binomial", requires_simulation=False, explanation="Switched to Binomial.")
    with mock_get_llm(), mock_parse_query(delta):
        response = client.post("/chat", json={"message": "switch to binomial"})
    assert response.json()["active_backend"] == "binomial"


def test_chat_reset_restores_defaults():
    from api.conversation import DEFAULT_PARAMS

    # First, change some params
    setup_delta = ParamDelta(total_employees=999, requires_simulation=True, explanation="setup")
    with mock_get_llm(), mock_parse_query(setup_delta):
        r1 = client.post("/chat", json={"message": "999 employees"})
    session_id = r1.json()["session_id"]
    assert r1.json()["current_params"]["total_employees"] == 999

    # Now reset
    reset_delta = ParamDelta(reset=True, requires_simulation=False, explanation="Reset done.")
    with mock_get_llm(), mock_parse_query(reset_delta):
        r2 = client.post("/chat", json={"message": "reset", "session_id": session_id})
    assert r2.json()["current_params"]["total_employees"] == DEFAULT_PARAMS["total_employees"]


def test_chat_maintains_session_across_turns():
    """Params accumulate correctly across multiple messages in the same session."""
    # Turn 1: set employees and seats
    delta1 = ParamDelta(total_employees=400, total_seats=300,
                        requires_simulation=True, explanation="400 emp, 300 seats")
    with mock_get_llm(), mock_parse_query(delta1):
        r1 = client.post("/chat", json={"message": "400 employees, 300 seats"})
    session_id = r1.json()["session_id"]

    # Turn 2: change only WFH days — employees and seats should be preserved
    delta2 = ParamDelta(wfh_days_per_week=4, requires_simulation=True, explanation="4 days WFH")
    with mock_get_llm(), mock_parse_query(delta2):
        r2 = client.post("/chat", json={"message": "4 days WFH", "session_id": session_id})

    params = r2.json()["current_params"]
    assert params["total_employees"] == 400   # from turn 1
    assert params["total_seats"] == 300       # from turn 1
    assert params["wfh_days_per_week"] == 4   # from turn 2


def test_chat_returns_current_params_schema():
    delta = ParamDelta(requires_simulation=False, explanation="ok")
    with mock_get_llm(), mock_parse_query(delta):
        response = client.post("/chat", json={"message": "hi"})
    params = response.json()["current_params"]
    required_keys = {
        "total_employees", "total_seats", "wfh_days_per_week",
        "compliance_rate", "start_date", "end_date",
    }
    assert required_keys.issubset(params.keys())
