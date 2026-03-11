from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SimulationInput:
    total_employees: int
    total_seats: int
    wfh_days_per_week: int = 2
    seat_reduction_pct: float = 0.0
    mandatory_office_days: dict = field(default_factory=dict)   # {team_name: [day_names]}
    day_of_week_weights: dict = field(default_factory=lambda: {
        "Monday": 0.85,
        "Tuesday": 1.00,
        "Wednesday": 1.00,
        "Thursday": 0.95,
        "Friday": 0.65,
    })
    compliance_rate: float = 0.9
    teams: list = field(default_factory=list)                   # [{"name": str, "size": int}]
    num_simulation_runs: int = 5000
    start_date: str = "2026-03-01"                              # YYYY-MM-DD
    end_date: str = "2026-03-31"                                # YYYY-MM-DD


@dataclass
class DayResult:
    date: str                       # YYYY-MM-DD
    day_of_week: str
    expected_occupancy: float       # mean seats filled
    std_dev: float
    overflow_probability: float     # P(demand > capacity)
    percentile_5: float
    percentile_95: float
    effective_capacity: int         # seats available that day
    team_breakdown: dict            # {team_name: expected_count}


@dataclass
class SimulationResult:
    model_name: str
    parameters_used: SimulationInput
    daily_results: list             # List[DayResult]
    summary: dict
    # summary keys:
    #   avg_utilization: float
    #   peak_occupancy: float
    #   overflow_days_count: int
    #   overflow_days_pct: float
    #   avg_overflow_magnitude: float


class SimulationBackend(ABC):

    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        pass

    @abstractmethod
    def run(self, params: SimulationInput) -> SimulationResult:
        """Execute the simulation and return results."""
        pass

    @abstractmethod
    def description(self) -> str:
        """Describe the model's approach and assumptions."""
        pass
