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


class MonteCarloBackend(SimulationBackend):

    def name(self) -> str:
        return f"Monte Carlo Simulation (n={5000})"

    def description(self) -> str:
        return (
            "Stochastic sampling with team-level social correlation and individual "
            "reliability factors. Empirical percentiles from repeated simulation runs."
        )

    def run(self, params: SimulationInput) -> SimulationResult:
        capacity = _effective_capacity(params.total_seats, params.seat_reduction_pct)
        working_days = _working_days(params.start_date, params.end_date)
        n_runs = params.num_simulation_runs
        rng = np.random.default_rng()

        teams = params.teams if params.teams else [{"name": "General", "size": params.total_employees}]

        # Pre-sample individual reliability factors once per employee per team.
        # Beta(5, 1) has mean = 5/6 ≈ 0.833. Normalize to mean=1.0 so reliability
        # adds variance around p_base without biasing the mean downward.
        _reliability_mean = 5 / 6
        team_reliability: dict[str, np.ndarray] = {
            t["name"]: rng.beta(5, 1, size=(n_runs, t["size"])) / _reliability_mean
            for t in teams
        }

        daily_results: list[DayResult] = []
        overflow_magnitudes: list[float] = []

        for d in working_days:
            day_name = WEEKDAYS[d.weekday()]
            total_occupancy = np.zeros(n_runs, dtype=np.float64)
            team_breakdown: dict[str, float] = {}

            for team in teams:
                n = team["size"]
                base_p = _compute_p_effective(
                    day_name,
                    team["name"],
                    params.wfh_days_per_week,
                    params.mandatory_office_days,
                    params.day_of_week_weights,
                    params.compliance_rate,
                )

                # Team-level social factor: Beta(8, 2) has mean = 0.8. Normalize to
                # mean=1.0 so it adds team-level variance without reducing the mean.
                _social_mean = 8 / 10
                social_factor = rng.beta(8, 2, size=n_runs) / _social_mean  # shape (n_runs,)

                # Effective p per run per employee: base_p * social * reliability
                # reliability shape: (n_runs, n_employees)
                reliability = team_reliability[team["name"]]
                p_matrix = np.clip(base_p * social_factor[:, np.newaxis] * reliability, 0, 1)

                # Bernoulli draws: (n_runs, n_employees)
                attendance = rng.random(size=(n_runs, n)) < p_matrix
                team_counts = attendance.sum(axis=1)  # shape (n_runs,)

                total_occupancy += team_counts
                team_breakdown[team["name"]] = round(float(team_counts.mean()), 2)

            mean_occ = float(total_occupancy.mean())
            std_dev = float(total_occupancy.std())
            overflow_runs = (total_occupancy > capacity).sum()
            overflow_prob = float(overflow_runs) / n_runs

            if mean_occ > capacity:
                overflow_magnitudes.append(mean_occ - capacity)

            daily_results.append(DayResult(
                date=d.isoformat(),
                day_of_week=day_name,
                expected_occupancy=round(mean_occ, 2),
                std_dev=round(std_dev, 2),
                overflow_probability=round(overflow_prob, 4),
                percentile_5=round(float(np.percentile(total_occupancy, 5)), 2),
                percentile_95=round(float(np.percentile(total_occupancy, 95)), 2),
                effective_capacity=capacity,
                team_breakdown=team_breakdown,
            ))

        n_days = len(daily_results)
        overflow_days = [r for r in daily_results if r.expected_occupancy > capacity]
        peak = max((r.expected_occupancy for r in daily_results), default=0.0)
        avg_util = (
            sum(r.expected_occupancy for r in daily_results) / (n_days * capacity)
            if n_days > 0 else 0.0
        )

        summary = {
            "avg_utilization": round(avg_util, 4),
            "peak_occupancy": round(peak, 2),
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
