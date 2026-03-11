from pydantic import BaseModel, Field, model_validator
from typing import Optional
from datetime import date


# ── Request schemas ──────────────────────────────────────────────────────────

class TeamSchema(BaseModel):
    name: str
    size: int = Field(gt=0)


class SimulationParamsSchema(BaseModel):
    total_employees: int = Field(default=500, gt=0)
    total_seats: int = Field(default=400, gt=0)
    wfh_days_per_week: int = Field(default=2, ge=0, le=5)
    seat_reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    mandatory_office_days: dict[str, list[str]] = Field(default_factory=dict)
    day_of_week_weights: dict[str, float] = Field(default_factory=lambda: {
        "Monday": 0.85, "Tuesday": 1.00, "Wednesday": 1.00,
        "Thursday": 0.95, "Friday": 0.65,
    })
    compliance_rate: float = Field(default=0.9, ge=0.0, le=1.0)
    teams: list[TeamSchema] = Field(default_factory=list)
    num_simulation_runs: int = Field(default=5000, ge=100, le=50000)
    start_date: str = Field(default="2026-03-01")
    end_date: str = Field(default="2026-03-31")

    @model_validator(mode="after")
    def validate_dates(self):
        start = date.fromisoformat(self.start_date)
        end = date.fromisoformat(self.end_date)
        if end < start:
            raise ValueError("end_date must be >= start_date")
        return self


class SimulateRequest(BaseModel):
    params: SimulationParamsSchema
    backend: str = "monte_carlo"


class ParseQueryRequest(BaseModel):
    message: str
    current_params: Optional[SimulationParamsSchema] = None


# ── Response schemas ──────────────────────────────────────────────────────────

class DayResultSchema(BaseModel):
    date: str
    day_of_week: str
    expected_occupancy: float
    std_dev: float
    overflow_probability: float
    percentile_5: float
    percentile_95: float
    effective_capacity: int
    team_breakdown: dict[str, float]


class SummarySchema(BaseModel):
    avg_utilization: float
    peak_occupancy: float
    overflow_days_count: int
    overflow_days_pct: float
    avg_overflow_magnitude: float


class SimulateResponse(BaseModel):
    model_name: str
    daily_results: list[DayResultSchema]
    summary: SummarySchema


class BackendInfoSchema(BaseModel):
    id: str
    name: str
    description: str


class BackendsResponse(BaseModel):
    available: list[BackendInfoSchema]
    active: str


class ParseQueryResponse(BaseModel):
    param_delta: dict
    explanation: str
    requires_simulation: bool


# ── Comparison schemas ────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    scenario_a: SimulateRequest
    scenario_b: SimulateRequest
    label_a: str = "Scenario A"
    label_b: str = "Scenario B"


class ScenarioResult(BaseModel):
    label: str
    simulation_result: SimulateResponse


class CompareResponse(BaseModel):
    scenario_a: ScenarioResult
    scenario_b: ScenarioResult
