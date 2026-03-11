"""Tests for the API client — httpx calls are mocked, no server needed."""
import pytest
from unittest.mock import MagicMock, patch
import httpx

from streamlit_app.api_client import get_backends, simulate, chat, APIError


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status_code
    mock.json.return_value = json_data
    if status_code >= 400:
        mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=mock
        )
    else:
        mock.raise_for_status.return_value = None
    return mock


# ── get_backends ──────────────────────────────────────────────────────────────

def test_get_backends_returns_data():
    payload = {
        "available": [
            {"id": "binomial", "name": "Binomial", "description": "..."},
            {"id": "monte_carlo", "name": "Monte Carlo", "description": "..."},
        ],
        "active": "monte_carlo",
    }
    with patch("httpx.get", return_value=_mock_response(payload)):
        result = get_backends()
    assert len(result["available"]) == 2
    assert result["active"] == "monte_carlo"


def test_get_backends_raises_api_error_on_http_failure():
    with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(APIError, match="Failed to fetch backends"):
            get_backends()


# ── simulate ──────────────────────────────────────────────────────────────────

def test_simulate_returns_result():
    payload = {
        "model_name": "Binomial/Poisson Analytical",
        "daily_results": [
            {
                "date": "2026-03-02",
                "day_of_week": "Monday",
                "expected_occupancy": 229.5,
                "std_dev": 11.1,
                "overflow_probability": 0.0,
                "percentile_5": 210.0,
                "percentile_95": 249.0,
                "effective_capacity": 400,
                "team_breakdown": {},
            }
        ],
        "summary": {
            "avg_utilization": 0.72,
            "peak_occupancy": 342.0,
            "overflow_days_count": 0,
            "overflow_days_pct": 0.0,
            "avg_overflow_magnitude": 0.0,
        },
    }
    with patch("httpx.post", return_value=_mock_response(payload)):
        result = simulate({"total_employees": 500, "total_seats": 400}, "binomial")
    assert result["model_name"] == "Binomial/Poisson Analytical"
    assert len(result["daily_results"]) == 1


def test_simulate_sends_backend_in_payload():
    payload = {"model_name": "mc", "daily_results": [], "summary": {
        "avg_utilization": 0, "peak_occupancy": 0,
        "overflow_days_count": 0, "overflow_days_pct": 0, "avg_overflow_magnitude": 0,
    }}
    with patch("httpx.post", return_value=_mock_response(payload)) as mock_post:
        simulate({"total_employees": 100, "total_seats": 80}, "monte_carlo")
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["backend"] == "monte_carlo"


def test_simulate_raises_api_error_on_400():
    error_response = _mock_response({"detail": "Unknown backend"}, status_code=400)
    with patch("httpx.post", return_value=error_response):
        with pytest.raises(APIError, match="Simulation failed"):
            simulate({}, "bad_backend")


def test_simulate_raises_api_error_on_connection_failure():
    with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(APIError, match="Simulation failed"):
            simulate({}, "binomial")


# ── chat ──────────────────────────────────────────────────────────────────────

def test_chat_returns_response():
    payload = {
        "session_id": "abc-123",
        "param_delta": {"wfh_days_per_week": 3},
        "explanation": "Changed WFH to 3 days.",
        "requires_simulation": True,
        "current_params": {"total_employees": 500, "total_seats": 400},
        "active_backend": "monte_carlo",
        "simulation_result": None,
    }
    with patch("httpx.post", return_value=_mock_response(payload)):
        result = chat("3 days WFH")
    assert result["session_id"] == "abc-123"
    assert result["explanation"] == "Changed WFH to 3 days."


def test_chat_sends_session_id_when_provided():
    payload = {
        "session_id": "existing-session",
        "param_delta": {},
        "explanation": "ok",
        "requires_simulation": False,
        "current_params": {},
        "active_backend": "monte_carlo",
        "simulation_result": None,
    }
    with patch("httpx.post", return_value=_mock_response(payload)) as mock_post:
        chat("hello", session_id="existing-session")
    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["session_id"] == "existing-session"


def test_chat_omits_session_id_when_none():
    payload = {
        "session_id": "new-session",
        "param_delta": {}, "explanation": "ok",
        "requires_simulation": False,
        "current_params": {}, "active_backend": "monte_carlo",
        "simulation_result": None,
    }
    with patch("httpx.post", return_value=_mock_response(payload)) as mock_post:
        chat("hello", session_id=None)
    sent_payload = mock_post.call_args.kwargs["json"]
    assert "session_id" not in sent_payload


def test_chat_raises_api_error_on_503():
    error_response = _mock_response({"detail": "LLM unavailable"}, status_code=503)
    with patch("httpx.post", return_value=error_response):
        with pytest.raises(APIError, match="Chat failed"):
            chat("hello")


def test_chat_raises_api_error_on_connection_failure():
    with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(APIError, match="Chat failed"):
            chat("hello")
