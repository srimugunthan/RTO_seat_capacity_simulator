"""
Cross-backend scenario tests.

Each test runs both Binomial and Monte Carlo backends against a known scenario
and asserts the expected behaviour (edge cases, cross-backend agreement).
"""
import pytest
from backend.models import SimulationInput
from backend.registry import get_backend

# ── Shared helpers ─────────────────────────────────────────────────────────────

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
        num_simulation_runs=3000,
        start_date="2026-03-02",
        end_date="2026-03-06",  # Mon–Fri single week for speed
    )
    defaults.update(overrides)
    return SimulationInput(**defaults)


binomial = get_backend("binomial")
monte_carlo = get_backend("monte_carlo")
simpy_des = get_backend("simpy_des")


# ── Edge case: all WFH ────────────────────────────────────────────────────────

class TestAllWFH:
    """When everyone works from home all week, occupancy should be near zero."""

    def test_binomial_all_wfh(self):
        result = binomial.run(_make_input(wfh_days_per_week=5))
        for day in result.daily_results:
            assert day.expected_occupancy < 50, (
                f"Expected near-zero occupancy with all WFH, got {day.expected_occupancy:.1f}"
            )

    def test_monte_carlo_all_wfh(self):
        result = monte_carlo.run(_make_input(wfh_days_per_week=5))
        for day in result.daily_results:
            assert day.expected_occupancy < 50, (
                f"Expected near-zero occupancy with all WFH, got {day.expected_occupancy:.1f}"
            )

    def test_no_overflow_all_wfh(self):
        for backend in [binomial, monte_carlo]:
            result = backend.run(_make_input(wfh_days_per_week=5))
            assert result.summary["overflow_days_count"] == 0


# ── Edge case: zero WFH ───────────────────────────────────────────────────────

class TestZeroWFH:
    """When everyone is in all week, occupancy should be near total_employees."""

    def test_binomial_zero_wfh(self):
        inp = _make_input(wfh_days_per_week=0, compliance_rate=1.0,
                          day_of_week_weights={d: 1.0 for d in
                          ["Monday","Tuesday","Wednesday","Thursday","Friday"]})
        result = binomial.run(inp)
        for day in result.daily_results:
            assert day.expected_occupancy > 450, (
                f"Expected high occupancy with 0 WFH, got {day.expected_occupancy:.1f}"
            )

    def test_monte_carlo_zero_wfh(self):
        inp = _make_input(wfh_days_per_week=0, compliance_rate=1.0,
                          day_of_week_weights={d: 1.0 for d in
                          ["Monday","Tuesday","Wednesday","Thursday","Friday"]})
        result = monte_carlo.run(inp)
        for day in result.daily_results:
            assert day.expected_occupancy > 430, (
                f"Expected high occupancy with 0 WFH, got {day.expected_occupancy:.1f}"
            )


# ── Edge case: seats == employees ────────────────────────────────────────────

class TestSeatsEqualsEmployees:
    """When seats == employees, overflow probability should be low (not zero but low)."""

    def test_binomial_seats_equals_employees(self):
        result = binomial.run(_make_input(total_seats=500))
        assert result.summary["overflow_days_count"] == 0

    def test_monte_carlo_seats_equals_employees(self):
        result = monte_carlo.run(_make_input(total_seats=500))
        for day in result.daily_results:
            assert day.overflow_probability < 0.1


# ── Edge case: seats << employees ────────────────────────────────────────────

class TestSeatsCriticallyLow:
    """When seats are far fewer than expected attendance, overflow is near-certain."""

    def test_binomial_always_overflows(self):
        inp = _make_input(total_employees=500, total_seats=100,
                          wfh_days_per_week=0, compliance_rate=1.0,
                          day_of_week_weights={d: 1.0 for d in
                          ["Monday","Tuesday","Wednesday","Thursday","Friday"]})
        result = binomial.run(inp)
        assert result.summary["overflow_days_count"] == len(result.daily_results)

    def test_monte_carlo_always_overflows(self):
        inp = _make_input(total_employees=500, total_seats=100,
                          wfh_days_per_week=0, compliance_rate=1.0,
                          day_of_week_weights={d: 1.0 for d in
                          ["Monday","Tuesday","Wednesday","Thursday","Friday"]})
        result = monte_carlo.run(inp)
        for day in result.daily_results:
            assert day.overflow_probability > 0.9


# ── Cross-backend agreement ───────────────────────────────────────────────────

class TestCrossBackendAgreement:
    """Binomial and Monte Carlo should agree within ±10% on mean occupancy.
    SimPy DES is compared at ±15% tolerance (higher variance at 1000 runs)."""

    @pytest.mark.parametrize("wfh_days", [0, 1, 2, 3])
    def test_mean_occupancy_agrees(self, wfh_days):
        inp = _make_input(wfh_days_per_week=wfh_days)
        b_result = binomial.run(inp)
        mc_result = monte_carlo.run(inp)

        b_means = {d.date: d.expected_occupancy for d in b_result.daily_results}
        mc_means = {d.date: d.expected_occupancy for d in mc_result.daily_results}

        for date in b_means:
            b_val = b_means[date]
            mc_val = mc_means[date]
            if b_val > 0:
                pct_diff = abs(b_val - mc_val) / b_val
                assert pct_diff < 0.10, (
                    f"Date {date}: Binomial={b_val:.1f} vs MC={mc_val:.1f} "
                    f"({pct_diff:.1%} divergence > 10%)"
                )

    @pytest.mark.parametrize("wfh_days", [1, 2, 3])
    def test_simpy_agrees_with_binomial(self, wfh_days):
        """SimPy DES mean occupancy should be within ±15% of Binomial."""
        b_inp = _make_input(wfh_days_per_week=wfh_days)
        des_inp = _make_input(wfh_days_per_week=wfh_days, num_simulation_runs=1000)

        b_result = binomial.run(b_inp)
        des_result = simpy_des.run(des_inp)

        b_means = {d.date: d.expected_occupancy for d in b_result.daily_results}
        des_means = {d.date: d.expected_occupancy for d in des_result.daily_results}

        for date in b_means:
            b_val = b_means[date]
            des_val = des_means[date]
            if b_val > 0:
                pct_diff = abs(b_val - des_val) / b_val
                assert pct_diff < 0.15, (
                    f"Date {date}: Binomial={b_val:.1f} vs SimPy={des_val:.1f} "
                    f"({pct_diff:.1%} divergence > 15%)"
                )

    def test_overflow_days_agree(self):
        """All three backends should agree on whether there are overflow days."""
        inp = _make_input(wfh_days_per_week=0, total_seats=200, compliance_rate=1.0,
                          day_of_week_weights={d: 1.0 for d in
                          ["Monday","Tuesday","Wednesday","Thursday","Friday"]})
        b_result = binomial.run(inp)
        mc_result = monte_carlo.run(inp)
        des_result = simpy_des.run(_make_input(
            wfh_days_per_week=0, total_seats=200, compliance_rate=1.0,
            day_of_week_weights={d: 1.0 for d in
                                  ["Monday","Tuesday","Wednesday","Thursday","Friday"]},
            num_simulation_runs=500,
        ))
        assert b_result.summary["overflow_days_count"] > 0
        assert mc_result.summary["overflow_days_count"] > 0
        assert des_result.summary["overflow_days_count"] > 0


# ── Seat reduction percentage ─────────────────────────────────────────────────

class TestSeatReduction:
    """seat_reduction_pct should reduce effective capacity proportionally."""

    def test_seat_reduction_lowers_capacity(self):
        base = binomial.run(_make_input(seat_reduction_pct=0.0))
        reduced = binomial.run(_make_input(seat_reduction_pct=20.0))
        base_cap = base.daily_results[0].effective_capacity
        red_cap = reduced.daily_results[0].effective_capacity
        assert red_cap == int(base_cap * 0.80), (
            f"Expected 20% reduction: {base_cap} → {int(base_cap * 0.80)}, got {red_cap}"
        )

    def test_seat_reduction_increases_overflow(self):
        # Tight scenario — removing seats should make it worse
        inp_base = _make_input(total_employees=500, total_seats=350,
                               wfh_days_per_week=1, seat_reduction_pct=0.0)
        inp_reduced = _make_input(total_employees=500, total_seats=350,
                                  wfh_days_per_week=1, seat_reduction_pct=25.0)
        b_base = binomial.run(inp_base)
        b_reduced = binomial.run(inp_reduced)
        assert b_reduced.summary["overflow_days_pct"] >= b_base.summary["overflow_days_pct"]
