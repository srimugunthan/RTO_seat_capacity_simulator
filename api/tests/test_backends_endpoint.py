"""Tests for GET /backends endpoint."""
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.main import app
from api.query_parser import ParamDelta

client = TestClient(app)


def test_get_backends_returns_200():
    response = client.get("/backends")
    assert response.status_code == 200


def test_both_builtin_backends_listed():
    data = client.get("/backends").json()
    ids = {b["id"] for b in data["available"]}
    assert "binomial" in ids
    assert "monte_carlo" in ids


def test_backend_entry_has_required_fields():
    data = client.get("/backends").json()
    for entry in data["available"]:
        assert "id" in entry
        assert "name" in entry
        assert "description" in entry


def test_active_backend_is_valid():
    data = client.get("/backends").json()
    ids = {b["id"] for b in data["available"]}
    assert data["active"] in ids


def test_parse_query_returns_200():
    delta = ParamDelta(seat_reduction_pct=10.0, requires_simulation=True,
                       explanation="Reducing seats by 10%.")
    with patch("api.main.get_llm", return_value=MagicMock()), \
         patch("api.main.parse_query", return_value=delta):
        response = client.post("/parse-query", json={
            "message": "What if we reduce seats by 10%?",
            "current_params": None,
        })
    assert response.status_code == 200


def test_parse_query_response_schema():
    delta = ParamDelta(requires_simulation=False, explanation="No change.")
    with patch("api.main.get_llm", return_value=MagicMock()), \
         patch("api.main.parse_query", return_value=delta):
        data = client.post("/parse-query", json={"message": "hello"}).json()
    assert "param_delta" in data
    assert "explanation" in data
    assert "requires_simulation" in data
