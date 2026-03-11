import math
from datetime import date, timedelta

from scipy.stats import norm

from .models import DayResult, SimulationBackend, SimulationInput, SimulationResult

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _working_days(start_date: str, end_date: str) -> list[date]:
    """Return all Mon–Fri dates in the range [start_date, end_date]."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # 0=Mon, 4=Fri
            days.append(current)
        current += timedelta(days=1)
    return days


def _effective_capacity(total_seats: int, seat_reduction_pct: float) -> int:
    return max(1, int(total_seats * (1 - seat_reduction_pct / 100)))


def _compute_p_effective(
    day_name: str,
    team_name: str,
    wfh_days_per_week: int,
    mandatory_office_days: dict,
    day_of_week_weights: dict,
    compliance_rate: float,
) -> float:
    """Compute attendance probability for one employee on a given day."""
    if team_name in mandatory_office_days and day_name in mandatory_office_days[team_name]:
        return compliance_rate
    base_p = (5 - wfh_days_per_week) / 5
    weight = day_of_week_weights.get(day_name, 1.0)
    return base_p * weight * compliance_rate


class BinomialBackend(SimulationBackend):

    def name(self) -> str:
        return "Binomial/Poisson Analytical"

    def description(self) -> str:
        return (
            "Closed-form probability model using normal approximation of the Binomial "
            "distribution. Instant results, assumes independence between employees."
        )

    def run(self, params: SimulationInput) -> SimulationResult:
        capacity = _effective_capacity(params.total_seats, params.seat_reduction_pct)
        working_days = _working_days(params.start_date, params.end_date)

        # Build team list: if none provided, treat all employees as one pool
        teams = params.teams if params.teams else [{"name": "General", "size": params.total_employees}]

        daily_results: list[DayResult] = []
        overflow_magnitudes: list[float] = []

        for d in working_days:
            day_name = WEEKDAYS[d.weekday()]

            # Per-team expected occupancy and variance
            team_breakdown: dict[str, float] = {}
            total_mean = 0.0
            total_variance = 0.0

            for team in teams:
                p = _compute_p_effective(
                    day_name,
                    team["name"],
                    params.wfh_days_per_week,
                    params.mandatory_office_days,
                    params.day_of_week_weights,
                    params.compliance_rate,
                )
                n = team["size"]
                mean = n * p
                variance = n * p * (1 - p)
                team_breakdown[team["name"]] = round(mean, 2)
                total_mean += mean
                total_variance += variance

            std_dev = math.sqrt(total_variance) if total_variance > 0 else 0.0

            # Overflow probability: P(X > capacity) using normal approximation
            if std_dev > 0:
                z = (capacity - total_mean) / std_dev
                overflow_prob = 1 - norm.cdf(z)
            else:
                overflow_prob = 1.0 if total_mean > capacity else 0.0

            # Confidence interval (5th and 95th percentile)
            p5 = max(0.0, norm.ppf(0.05, loc=total_mean, scale=std_dev) if std_dev > 0 else total_mean)
            p95 = norm.ppf(0.95, loc=total_mean, scale=std_dev) if std_dev > 0 else total_mean

            if total_mean > capacity:
                overflow_magnitudes.append(total_mean - capacity)

            daily_results.append(DayResult(
                date=d.isoformat(),
                day_of_week=day_name,
                expected_occupancy=round(total_mean, 2),
                std_dev=round(std_dev, 2),
                overflow_probability=round(overflow_prob, 4),
                percentile_5=round(p5, 2),
                percentile_95=round(p95, 2),
                effective_capacity=capacity,
                team_breakdown=team_breakdown,
            ))

        # Summary stats
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
