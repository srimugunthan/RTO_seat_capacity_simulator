"""
SimPy Discrete-Event Simulation backend.

Implements DES semantics: each employee is an agent that arrives at a random
time, claims a seat (first-come-first-served), works, then departs. Overflow
= employees turned away when all seats are occupied.

The hot-path uses a heapq-based sweep-line scheduler rather than SimPy
generator processes to meet the performance budget (1000 runs × 1 month < 10s).
SimPy is used as the conceptual DES framework; heapq replaces the Python
generator overhead that made coroutine-per-employee O(n) in interpreter time.
"""
import heapq
import simpy  # DES framework — kept as package dependency and conceptual reference
import numpy as np
from datetime import date, timedelta

from .binomial_backend import WEEKDAYS, _effective_capacity, _compute_p_effective
from .models import DayResult, SimulationBackend, SimulationInput, SimulationResult


def _working_days(start_date: str, end_date: str) -> list[date]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _run_one_day(rng: np.random.Generator, seats: int, n_employees: int,
                 arrival_mu: float, arrival_sigma: float,
                 work_mu: float, work_sigma: float) -> tuple[int, int]:
    """
    DES for a single day using a heapq sweep-line scheduler.

    Each employee arrives at a sampled time, takes a seat if available
    (first-come-first-served), works for a sampled duration, then departs.
    Employees who find no free seat are turned away immediately.

    Returns (peak_concurrent_occupancy, turned_away_count).
    """
    arrivals = rng.normal(arrival_mu, arrival_sigma, n_employees)
    np.clip(arrivals, 7 * 60, 12 * 60, out=arrivals)   # 7 am – 12 pm

    durations = rng.normal(work_mu, work_sigma, n_employees)
    np.clip(durations, 4 * 60, 10 * 60, out=durations)  # 4 h – 10 h

    departures = arrivals + durations

    # Sort employees by arrival time (DES event ordering)
    order = np.argsort(arrivals)
    arrivals = arrivals[order]
    departures = departures[order]

    # heap holds departure times of currently occupied seats (min-heap)
    heap: list[float] = []
    peak = 0
    turned_away = 0

    for i in range(n_employees):
        arr = arrivals[i]
        dep = departures[i]

        # Free seats whose occupant has already left
        while heap and heap[0] <= arr:
            heapq.heappop(heap)

        if len(heap) < seats:
            heapq.heappush(heap, dep)
            occupied = len(heap)
            if occupied > peak:
                peak = occupied
        else:
            turned_away += 1

    return peak, turned_away


class SimpyBackend(SimulationBackend):
    """
    Discrete-event simulation backend.

    Simulates intraday office dynamics with first-come-first-served seat
    contention. Captures peak-hour crowding that daily-average models miss.

    Arrival distribution: Normal(μ=9 h, σ=1 h), clipped to [7 h, 12 h]
    Work duration:        Normal(μ=8 h, σ=1 h), clipped to [4 h, 10 h]
    """

    ARRIVAL_MU    = 9 * 60   # minutes from midnight
    ARRIVAL_SIGMA = 60
    WORK_MU       = 8 * 60
    WORK_SIGMA    = 60

    def name(self) -> str:
        return "SimPy Discrete-Event Simulation"

    def description(self) -> str:
        return (
            "Agent-based discrete-event simulation with first-come-first-served "
            "seat queuing. Each employee arrives at a random time, claims a physical "
            "seat, works, then departs. Overflow = employees turned away when all "
            "seats are occupied. Captures intraday crowding that daily averages hide."
        )

    def run(self, params: SimulationInput) -> SimulationResult:
        capacity = _effective_capacity(params.total_seats, params.seat_reduction_pct)
        working_days = _working_days(params.start_date, params.end_date)
        n_runs = params.num_simulation_runs
        rng = np.random.default_rng()

        teams = params.teams if params.teams else [
            {"name": "General", "size": params.total_employees}
        ]

        daily_results: list[DayResult] = []
        overflow_magnitudes: list[float] = []

        for d in working_days:
            day_name = WEEKDAYS[d.weekday()]

            # Per-team: Binomial draws to decide how many come in each run
            team_breakdown: dict[str, float] = {}
            n_in_office_per_run = np.zeros(n_runs, dtype=np.int32)

            for team in teams:
                p = _compute_p_effective(
                    day_name,
                    team["name"],
                    params.wfh_days_per_week,
                    params.mandatory_office_days,
                    params.day_of_week_weights,
                    params.compliance_rate,
                )
                counts = rng.binomial(team["size"], p, size=n_runs)
                n_in_office_per_run += counts
                team_breakdown[team["name"]] = round(float(counts.mean()), 2)

            # DES replications
            peaks = np.zeros(n_runs, dtype=np.int32)
            turned_away_counts = np.zeros(n_runs, dtype=np.int32)

            for run_idx in range(n_runs):
                n_emp = int(n_in_office_per_run[run_idx])
                if n_emp == 0:
                    continue
                peak, turned_away = _run_one_day(
                    rng, capacity, n_emp,
                    self.ARRIVAL_MU, self.ARRIVAL_SIGMA,
                    self.WORK_MU, self.WORK_SIGMA,
                )
                peaks[run_idx] = peak
                turned_away_counts[run_idx] = turned_away

            mean_occ = float(peaks.mean())
            std_dev = float(peaks.std())
            # Overflow = any run where employees were turned away
            overflow_runs = int((turned_away_counts > 0).sum())
            overflow_prob = overflow_runs / n_runs

            if overflow_prob > 0:
                overflow_magnitudes.append(
                    float(turned_away_counts[turned_away_counts > 0].mean())
                )

            daily_results.append(DayResult(
                date=d.isoformat(),
                day_of_week=day_name,
                expected_occupancy=round(mean_occ, 2),
                std_dev=round(std_dev, 2),
                overflow_probability=round(overflow_prob, 4),
                percentile_5=round(float(np.percentile(peaks, 5)), 2),
                percentile_95=round(float(np.percentile(peaks, 95)), 2),
                effective_capacity=capacity,
                team_breakdown=team_breakdown,
            ))

        n_days = len(daily_results)
        # DES peak is bounded by seat count; use overflow_probability for overflow days
        overflow_days = [r for r in daily_results if r.overflow_probability > 0]
        peak_occ = max((r.expected_occupancy for r in daily_results), default=0.0)
        avg_util = (
            sum(r.expected_occupancy for r in daily_results) / (n_days * capacity)
            if n_days > 0 else 0.0
        )

        summary = {
            "avg_utilization": round(avg_util, 4),
            "peak_occupancy": round(peak_occ, 2),
            "overflow_days_count": len(overflow_days),
            "overflow_days_pct": round(len(overflow_days) / n_days, 4) if n_days > 0 else 0.0,
            "avg_overflow_magnitude": round(
                sum(overflow_magnitudes) / len(overflow_magnitudes), 2
            ) if overflow_magnitudes else 0.0,
        }

        return SimulationResult(
            model_name=self.name(),
            parameters_used=params,
            daily_results=daily_results,
            summary=summary,
        )
