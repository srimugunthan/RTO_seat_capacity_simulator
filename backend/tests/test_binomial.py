"""Known-answer tests for the Binomial analytical backend."""
import pytest
from backend.binomial_backend import BinomialBackend
from backend.models import SimulationInput

backend = BinomialBackend()


def make_params(**kwargs) -> SimulationInput:
    defaults = dict(
        total_employees=500,
        total_seats=500,
        wfh_days_per_week=0,
        compliance_rate=1.0,
        start_date="2026-03-02",
        end_date="2026-03-06",
    )
    defaults.update(kwargs)
    return SimulationInput(**defaults)


def test_no_overflow_when_seats_equal_employees_no_wfh():
    """500 employees, 500 seats, 0 WFH → occupancy ≈ seats, very low overflow."""
    params = make_params(total_employees=500, total_seats=500, wfh_days_per_week=0)
    result = backend.run(params)
    # With day-of-week weights < 1 (e.g. Monday 0.85, Friday 0.65), mean < 500
    # so overflow_days_count should be 0
    assert result.summary["overflow_days_count"] == 0


def test_high_overflow_when_seats_far_below_employees():
    """500 employees, 200 seats, 0 WFH, compliance=1 → all days overflow."""
    params = make_params(total_employees=500, total_seats=200, wfh_days_per_week=0, compliance_rate=1.0)
    result = backend.run(params)
    assert result.summary["overflow_days_count"] == len(result.daily_results)
    for day in result.daily_results:
        assert day.overflow_probability > 0.99


def test_zero_occupancy_with_full_wfh():
    """5 WFH days → nobody comes in → expected occupancy ≈ 0."""
    params = make_params(wfh_days_per_week=5, compliance_rate=1.0)
    result = backend.run(params)
    for day in result.daily_results:
        assert day.expected_occupancy == pytest.approx(0.0, abs=1.0)


def test_seat_reduction_applied():
    """10% seat reduction on 500 seats → effective capacity = 450."""
    params = make_params(total_seats=500, seat_reduction_pct=10.0)
    result = backend.run(params)
    for day in result.daily_results:
        assert day.effective_capacity == 450


def test_mandatory_office_days_raises_attendance():
    """Team with mandatory Tuesday should show higher p on Tuesday than a non-mandatory team."""
    params = SimulationInput(
        total_employees=200,
        total_seats=300,
        wfh_days_per_week=4,          # normally almost nobody comes in
        compliance_rate=0.9,
        mandatory_office_days={"Engineering": ["Tuesday"]},
        teams=[{"name": "Engineering", "size": 100}, {"name": "General", "size": 100}],
        start_date="2026-03-03",       # Tuesday only
        end_date="2026-03-03",
    )
    result = backend.run(params)
    day = result.daily_results[0]
    assert day.day_of_week == "Tuesday"
    # Engineering expected ≈ 90 (100 * 0.9), General expected ≈ 100 * 0.2 * 1.0 * 0.9 = 18
    assert day.team_breakdown["Engineering"] > day.team_breakdown["General"]


def test_summary_utilization_range():
    params = make_params(total_employees=300, total_seats=400, wfh_days_per_week=2)
    result = backend.run(params)
    assert 0.0 <= result.summary["avg_utilization"] <= 1.5  # allow slight overflow


def test_weekends_excluded():
    """Date range covering a full week including weekend should only have 5 days."""
    params = make_params(start_date="2026-03-02", end_date="2026-03-08")  # Mon–Sun
    result = backend.run(params)
    assert len(result.daily_results) == 5
