"""Tests for the backend registry."""
import pytest
from backend.registry import register_backend, get_backend, list_backends
from backend.models import SimulationBackend, SimulationInput, SimulationResult


def test_builtin_backends_registered():
    backends = {b["id"] for b in list_backends()}
    assert "binomial" in backends
    assert "monte_carlo" in backends


def test_get_known_backend():
    b = get_backend("binomial")
    assert b is not None
    assert b.name() == "Binomial/Poisson Analytical"


def test_get_unknown_backend_raises():
    with pytest.raises(KeyError, match="unknown_model"):
        get_backend("unknown_model")


def test_list_backends_schema():
    for entry in list_backends():
        assert "id" in entry
        assert "name" in entry
        assert "description" in entry


def test_register_custom_backend():
    class DummyBackend(SimulationBackend):
        def name(self): return "Dummy"
        def description(self): return "Test backend"
        def run(self, params: SimulationInput) -> SimulationResult:
            return SimulationResult(
                model_name=self.name(),
                parameters_used=params,
                daily_results=[],
                summary={
                    "avg_utilization": 0.0, "peak_occupancy": 0.0,
                    "overflow_days_count": 0, "overflow_days_pct": 0.0,
                    "avg_overflow_magnitude": 0.0,
                }
            )

    register_backend("dummy", DummyBackend())
    assert get_backend("dummy").name() == "Dummy"
    ids = {b["id"] for b in list_backends()}
    assert "dummy" in ids
