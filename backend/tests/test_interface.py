"""Tests that both backends correctly implement the SimulationBackend interface."""
import pytest
from backend.models import SimulationBackend, SimulationInput, SimulationResult, DayResult
from backend.binomial_backend import BinomialBackend
from backend.monte_carlo_backend import MonteCarloBackend

BACKENDS = [BinomialBackend(), MonteCarloBackend()]

BASE_PARAMS = SimulationInput(
    total_employees=100,
    total_seats=80,
    wfh_days_per_week=2,
    start_date="2026-03-02",
    end_date="2026-03-06",  # one week
)


@pytest.mark.parametrize("backend", BACKENDS, ids=["binomial", "monte_carlo"])
def test_returns_simulation_result(backend):
    result = backend.run(BASE_PARAMS)
    assert isinstance(result, SimulationResult)


@pytest.mark.parametrize("backend", BACKENDS, ids=["binomial", "monte_carlo"])
def test_name_is_string(backend):
    assert isinstance(backend.name(), str)
    assert len(backend.name()) > 0


@pytest.mark.parametrize("backend", BACKENDS, ids=["binomial", "monte_carlo"])
def test_description_is_string(backend):
    assert isinstance(backend.description(), str)
    assert len(backend.description()) > 0


@pytest.mark.parametrize("backend", BACKENDS, ids=["binomial", "monte_carlo"])
def test_daily_results_count(backend):
    """One week (Mon–Fri) should produce exactly 5 DayResult entries."""
    result = backend.run(BASE_PARAMS)
    assert len(result.daily_results) == 5


@pytest.mark.parametrize("backend", BACKENDS, ids=["binomial", "monte_carlo"])
def test_day_result_schema(backend):
    result = backend.run(BASE_PARAMS)
    for day in result.daily_results:
        assert isinstance(day, DayResult)
        assert day.date
        assert day.day_of_week in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        assert day.expected_occupancy >= 0
        assert day.std_dev >= 0
        assert 0.0 <= day.overflow_probability <= 1.0
        assert day.percentile_5 <= day.percentile_95
        assert day.effective_capacity > 0
        assert isinstance(day.team_breakdown, dict)


@pytest.mark.parametrize("backend", BACKENDS, ids=["binomial", "monte_carlo"])
def test_summary_keys(backend):
    result = backend.run(BASE_PARAMS)
    required_keys = {
        "avg_utilization", "peak_occupancy",
        "overflow_days_count", "overflow_days_pct", "avg_overflow_magnitude"
    }
    assert required_keys.issubset(result.summary.keys())


@pytest.mark.parametrize("backend", BACKENDS, ids=["binomial", "monte_carlo"])
def test_parameters_preserved(backend):
    result = backend.run(BASE_PARAMS)
    assert result.parameters_used is BASE_PARAMS
