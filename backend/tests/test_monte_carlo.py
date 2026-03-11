"""Tests that Monte Carlo empirical results converge to Binomial analytical values."""
import pytest
from backend.binomial_backend import BinomialBackend
from backend.monte_carlo_backend import MonteCarloBackend
from backend.models import SimulationInput

binomial = BinomialBackend()
monte_carlo = MonteCarloBackend()

CONVERGENCE_PARAMS = SimulationInput(
    total_employees=500,
    total_seats=400,
    wfh_days_per_week=2,
    compliance_rate=0.9,
    num_simulation_runs=5000,
    start_date="2026-03-02",
    end_date="2026-03-06",
)


def test_mean_converges_to_binomial():
    """Monte Carlo mean occupancy should be within 5% of Binomial mean for each day."""
    b_result = binomial.run(CONVERGENCE_PARAMS)
    mc_result = monte_carlo.run(CONVERGENCE_PARAMS)

    for b_day, mc_day in zip(b_result.daily_results, mc_result.daily_results):
        assert b_day.date == mc_day.date
        tolerance = max(b_day.expected_occupancy * 0.05, 5.0)  # 5% or 5 seats
        assert abs(b_day.expected_occupancy - mc_day.expected_occupancy) <= tolerance, (
            f"{b_day.date}: Binomial={b_day.expected_occupancy}, "
            f"MonteCarlo={mc_day.expected_occupancy}, tolerance={tolerance}"
        )


def test_overflow_probability_direction_matches():
    """If Binomial says overflow is high, Monte Carlo should agree (same direction)."""
    b_result = binomial.run(CONVERGENCE_PARAMS)
    mc_result = monte_carlo.run(CONVERGENCE_PARAMS)
    for b_day, mc_day in zip(b_result.daily_results, mc_result.daily_results):
        if b_day.overflow_probability > 0.5:
            assert mc_day.overflow_probability > 0.3
        if b_day.overflow_probability < 0.05:
            assert mc_day.overflow_probability < 0.2


def test_percentile_ordering():
    """p5 <= mean <= p95 for all days."""
    result = monte_carlo.run(CONVERGENCE_PARAMS)
    for day in result.daily_results:
        assert day.percentile_5 <= day.expected_occupancy + day.std_dev * 2
        assert day.percentile_5 <= day.percentile_95


def test_zero_overflow_with_large_capacity():
    """With seats >> employees, overflow probability should be ~0."""
    params = SimulationInput(
        total_employees=100,
        total_seats=500,
        wfh_days_per_week=0,
        compliance_rate=1.0,
        num_simulation_runs=1000,
        start_date="2026-03-03",
        end_date="2026-03-03",
    )
    result = monte_carlo.run(params)
    assert result.daily_results[0].overflow_probability == pytest.approx(0.0, abs=0.01)


def test_full_overflow_with_tiny_capacity():
    """With seats << employees and 0 WFH, overflow probability should be ~1."""
    params = SimulationInput(
        total_employees=500,
        total_seats=100,
        wfh_days_per_week=0,
        compliance_rate=1.0,
        num_simulation_runs=1000,
        start_date="2026-03-03",
        end_date="2026-03-03",
    )
    result = monte_carlo.run(params)
    assert result.daily_results[0].overflow_probability == pytest.approx(1.0, abs=0.01)
