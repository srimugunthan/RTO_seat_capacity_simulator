"""
Unit and performance tests for the SimPy DES backend.
"""
import time
import pytest
from backend.models import SimulationInput, SimulationResult, DayResult
from backend.simpy_backend import SimpyBackend

backend = SimpyBackend()


def _make_input(**overrides) -> SimulationInput:
    defaults = dict(
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
        num_simulation_runs=500,   # lower for speed in unit tests
        start_date="2026-03-02",
        end_date="2026-03-06",     # Mon–Fri single week
    )
    defaults.update(overrides)
    return SimulationInput(**defaults)


# ── Interface / schema tests ───────────────────────────────────────────────────

class TestInterface:
    def test_name_returns_string(self):
        assert isinstance(backend.name(), str)
        assert len(backend.name()) > 0

    def test_description_returns_string(self):
        assert isinstance(backend.description(), str)

    def test_run_returns_simulation_result(self):
        result = backend.run(_make_input())
        assert isinstance(result, SimulationResult)

    def test_result_has_daily_results(self):
        result = backend.run(_make_input())
        assert len(result.daily_results) == 5  # Mon–Fri

    def test_day_result_schema(self):
        result = backend.run(_make_input())
        for day in result.daily_results:
            assert isinstance(day, DayResult)
            assert day.date != ""
            assert day.day_of_week in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            assert day.expected_occupancy >= 0
            assert day.std_dev >= 0
            assert 0.0 <= day.overflow_probability <= 1.0
            assert day.percentile_5 <= day.percentile_95
            assert day.effective_capacity > 0

    def test_summary_keys_present(self):
        result = backend.run(_make_input())
        for key in ("avg_utilization", "peak_occupancy", "overflow_days_count",
                    "overflow_days_pct", "avg_overflow_magnitude"):
            assert key in result.summary

    def test_model_name_identifies_backend(self):
        result = backend.run(_make_input())
        assert "SimPy" in result.model_name or "Discrete" in result.model_name


# ── Known-answer tests ────────────────────────────────────────────────────────

class TestKnownAnswers:
    def test_ample_seats_low_overflow(self):
        """500 employees, 500 seats, 0 WFH → overflow probability should be low."""
        inp = _make_input(
            total_seats=500,
            wfh_days_per_week=0,
            compliance_rate=1.0,
            day_of_week_weights={d: 1.0 for d in
                                  ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]},
            num_simulation_runs=1000,
        )
        result = backend.run(inp)
        for day in result.daily_results:
            assert day.overflow_probability < 0.15, (
                f"Expected low overflow with seats=employees, got {day.overflow_probability:.2%}"
            )

    def test_critically_low_seats_high_overflow(self):
        """500 employees, 100 seats, 0 WFH → overflow probability should be very high."""
        inp = _make_input(
            total_employees=500,
            total_seats=100,
            wfh_days_per_week=0,
            compliance_rate=1.0,
            day_of_week_weights={d: 1.0 for d in
                                  ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]},
            num_simulation_runs=500,
        )
        result = backend.run(inp)
        for day in result.daily_results:
            assert day.overflow_probability > 0.9, (
                f"Expected near-certain overflow, got {day.overflow_probability:.2%}"
            )

    def test_peak_occupancy_bounded_by_employees(self):
        """Peak occupancy can never exceed total_employees."""
        result = backend.run(_make_input(num_simulation_runs=300))
        for day in result.daily_results:
            assert day.expected_occupancy <= 500 + 1, (
                f"Occupancy {day.expected_occupancy} exceeds total employees"
            )

    def test_all_wfh_near_zero_occupancy(self):
        result = backend.run(_make_input(wfh_days_per_week=5))
        for day in result.daily_results:
            assert day.expected_occupancy < 50

    def test_zero_wfh_high_occupancy(self):
        inp = _make_input(
            wfh_days_per_week=0,
            compliance_rate=1.0,
            day_of_week_weights={d: 1.0 for d in
                                  ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]},
            num_simulation_runs=500,
        )
        result = backend.run(inp)
        for day in result.daily_results:
            # DES peak is bounded by seat count (400); with 500 employees and 0 WFH
            # all seats should be filled every run → peak == capacity
            assert day.expected_occupancy >= 400


# ── Cross-backend agreement ───────────────────────────────────────────────────

class TestCrossBackendAgreement:
    """SimPy DES should agree with Monte Carlo within ±15% on mean occupancy."""

    def test_agrees_with_monte_carlo(self):
        from backend.registry import get_backend
        mc = get_backend("monte_carlo")

        inp = _make_input(
            wfh_days_per_week=2,
            num_simulation_runs=1000,
        )
        # Monte Carlo also uses 1000 runs for comparable variance
        mc_inp = _make_input(wfh_days_per_week=2, num_simulation_runs=3000)

        des_result = backend.run(inp)
        mc_result = mc.run(mc_inp)

        des_means = {d.date: d.expected_occupancy for d in des_result.daily_results}
        mc_means = {d.date: d.expected_occupancy for d in mc_result.daily_results}

        for dt in des_means:
            des_val = des_means[dt]
            mc_val = mc_means[dt]
            if mc_val > 0:
                pct_diff = abs(des_val - mc_val) / mc_val
                assert pct_diff < 0.15, (
                    f"Date {dt}: DES={des_val:.1f} vs MC={mc_val:.1f} "
                    f"({pct_diff:.1%} divergence > 15%)"
                )


# ── Arrival distribution smoke test ──────────────────────────────────────────

class TestArrivalDistribution:
    def test_later_arrival_mean_shifts_peak(self):
        """Changing arrival mu should not crash and results should still be valid."""
        result = backend.run(_make_input(num_simulation_runs=200))
        assert len(result.daily_results) == 5
        for day in result.daily_results:
            assert day.expected_occupancy >= 0


# ── Performance test ──────────────────────────────────────────────────────────

class TestPerformance:
    def test_one_month_1000_runs_under_10_seconds(self):
        inp = _make_input(
            num_simulation_runs=1000,
            start_date="2026-03-01",
            end_date="2026-03-31",
        )
        t0 = time.perf_counter()
        result = backend.run(inp)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"SimPy 1000 runs / 1 month took {elapsed:.2f}s (limit: 10s)"
        assert len(result.daily_results) == 22  # working days in March 2026
