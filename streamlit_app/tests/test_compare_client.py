"""Tests for the compare() function in api_client."""
import pytest
from unittest.mock import MagicMock, patch
import httpx

from streamlit_app.api_client import compare, APIError


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


_SCENARIO_RESULT = {
    "model_name": "Binomial/Poisson Analytical",
    "daily_results": [],
    "summary": {
        "avg_utilization": 0.6, "peak_occupancy": 250.0,
        "overflow_days_count": 0, "overflow_days_pct": 0.0,
        "avg_overflow_magnitude": 0.0,
    },
}

_COMPARE_PAYLOAD = {
    "scenario_a": {"label": "2-day WFH", "simulation_result": _SCENARIO_RESULT},
    "scenario_b": {"label": "3-day WFH", "simulation_result": _SCENARIO_RESULT},
}


def test_compare_returns_data():
    with patch("httpx.post", return_value=_mock_response(_COMPARE_PAYLOAD)):
        result = compare({"total_employees": 500}, "binomial",
                         {"total_employees": 500}, "binomial",
                         "2-day WFH", "3-day WFH")
    assert "scenario_a" in result
    assert "scenario_b" in result


def test_compare_sends_correct_payload():
    with patch("httpx.post", return_value=_mock_response(_COMPARE_PAYLOAD)) as mock_post:
        compare({"wfh": 2}, "binomial", {"wfh": 3}, "monte_carlo", "A", "B")
    sent = mock_post.call_args.kwargs["json"]
    assert sent["scenario_a"]["backend"] == "binomial"
    assert sent["scenario_b"]["backend"] == "monte_carlo"
    assert sent["label_a"] == "A"
    assert sent["label_b"] == "B"


def test_compare_raises_api_error_on_400():
    err = _mock_response({"detail": "bad backend"}, status_code=400)
    with patch("httpx.post", return_value=err):
        with pytest.raises(APIError, match="Comparison failed"):
            compare({}, "bad", {}, "bad")


def test_compare_raises_api_error_on_connection_failure():
    with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(APIError, match="Comparison failed"):
            compare({}, "binomial", {}, "binomial")
