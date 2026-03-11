"""
Performance benchmarks.

Success criterion from PRD: Monte Carlo simulation with 500 employees,
1 month, 5000 runs must complete in under 2 seconds.
"""
import time
from backend.models import SimulationInput
from backend.registry import get_backend


def _standard_input(num_runs: int = 5000) -> SimulationInput:
    return SimulationInput(
        total_employees=500,
        total_seats=400,
        wfh_days_per_week=2,
        seat_reduction_pct=0.0,
        mandatory_office_days={},
        day_of_week_weights={
            "Monday": 0.85, "Tuesday": 1.00, "Wednesday": 1.00,
            "Thursday": 0.95, "Friday": 0.65,
        },
        compliance_rate=0.9,
        teams=[],
        num_simulation_runs=num_runs,
        start_date="2026-03-01",
        end_date="2026-03-31",
    )


def test_monte_carlo_5000_runs_under_2_seconds():
    """PRD success metric: 500 employees, 1 month, 5000 MC runs < 2s."""
    backend = get_backend("monte_carlo")
    inp = _standard_input(num_runs=5000)

    start = time.perf_counter()
    result = backend.run(inp)
    elapsed = time.perf_counter() - start

    assert len(result.daily_results) == 22, "Expected 23 working days in March 2026"
    assert elapsed < 2.0, f"Monte Carlo took {elapsed:.2f}s — exceeds 2s limit"


def test_binomial_1_month_under_100ms():
    """Binomial model should be nearly instantaneous."""
    backend = get_backend("binomial")
    inp = _standard_input()

    start = time.perf_counter()
    result = backend.run(inp)
    elapsed = time.perf_counter() - start

    assert len(result.daily_results) == 22
    assert elapsed < 0.1, f"Binomial took {elapsed:.3f}s — expected < 100ms"


def test_monte_carlo_with_teams_under_2_seconds():
    """Monte Carlo with team configuration should still be under 2 seconds."""
    backend = get_backend("monte_carlo")
    inp = SimulationInput(
        total_employees=500,
        total_seats=400,
        wfh_days_per_week=2,
        seat_reduction_pct=0.0,
        mandatory_office_days={"Engineering": ["Tuesday", "Thursday"]},
        day_of_week_weights={
            "Monday": 0.85, "Tuesday": 1.00, "Wednesday": 1.00,
            "Thursday": 0.95, "Friday": 0.65,
        },
        compliance_rate=0.9,
        teams=[
            {"name": "Engineering", "size": 200},
            {"name": "Sales", "size": 150},
            {"name": "Operations", "size": 150},
        ],
        num_simulation_runs=5000,
        start_date="2026-03-01",
        end_date="2026-03-31",
    )

    start = time.perf_counter()
    result = backend.run(inp)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"Monte Carlo with teams took {elapsed:.2f}s — exceeds 2s limit"
    assert len(result.daily_results) == 22
