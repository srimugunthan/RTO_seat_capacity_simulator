# RTO Capacity Simulator — Project Requirements Document

**Version:** 1.0  
**Author:** Srimugunthan  
**Date:** March 2026  
**Status:** Draft

---

## 1. Executive Summary

The RTO Capacity Simulator is a conversational, AI-powered tool that allows workspace planners, HR leaders, and operations teams to simulate office seat occupancy under various hybrid work policies. Users interact via a natural language chat interface — asking questions like *"What happens if we reduce seat capacity by 10%?"* — and receive visual, calendar-based occupancy projections highlighting overflow risk in real time.

The backend simulation engine is designed with a **pluggable architecture**, allowing different probabilistic models (Binomial, Monte Carlo, SimPy-based, ABM) to be swapped in without changing the frontend or chat interface.

---

## 2. Problem Statement

Organizations transitioning to hybrid work face a core question: **how many seats do we actually need?** Over-provisioning wastes real estate cost; under-provisioning leads to employees arriving with no place to sit. The answer depends on a complex interaction of policies (WFH days, mandatory office days), team structures, behavioral patterns, and day-of-week effects.

Current approaches rely on static spreadsheets or gut feel. This project provides a dynamic, simulation-backed tool where stakeholders can explore "what-if" scenarios conversationally and see results visually.

---

## 3. Goals and Non-Goals

### 3.1 Goals

- Provide a chat-based interface where users describe scenarios in natural language and receive simulation results.
- Display results on a **monthly calendar view** showing daily projected seat occupancy, with overflow days highlighted in red.
- Support multiple simulation backends via a pluggable model architecture.
- Ship with two initial backends: **Binomial/Poisson analytical model** and **Monte Carlo simulation**.
- Allow configuration of key parameters: total employees, total seats, WFH policy, team-specific mandates, and day-of-week effects.
- Make it easy to add new backends (SimPy, Mesa ABM, ML-calibrated models) in the future without modifying the frontend or orchestration layer.

### 3.2 Non-Goals (v1)

- Real-time integration with badge/access control systems.
- Individual employee-level tracking or recommendations.
- Seat booking or reservation functionality.
- Multi-floor or zone-level spatial simulation (v1 treats office as a single pool of seats).
- Production deployment with authentication and multi-tenancy.

---

## 4. User Personas

| Persona | Description | Key Questions |
|---|---|---|
| **Workspace Planner** | Facilities/real estate team member deciding how many desks to provision | "Can we cut 20% of seats and still be safe?" |
| **HR Policy Designer** | Designing hybrid work policies for the organization | "What if engineering has mandatory Tuesdays and Thursdays?" |
| **Finance/Ops Lead** | Evaluating cost savings from seat reduction | "What utilization rate do we hit with 3-day WFH?" |
| **Team Manager** | Understanding how their team's policy interacts with office capacity | "If my team of 40 all come in on Wednesday, do we overflow?" |

---

## 5. Functional Requirements

### 5.1 Chat Interface

| ID | Requirement | Priority |
|---|---|---|
| F-01 | User can type natural language queries describing scenarios | P0 |
| F-02 | System parses the query to extract simulation parameters (seat count changes, WFH policy, team mandates, etc.) | P0 |
| F-03 | System responds with a textual summary of the simulation result alongside the calendar visualization | P0 |
| F-04 | Conversation maintains context — follow-up queries like "now try 3 days WFH instead" modify the previous scenario | P1 |
| F-05 | User can reset to baseline configuration via chat command | P1 |
| F-06 | User can ask comparative questions: "Compare 2-day vs 3-day WFH" and see side-by-side calendars | P2 |

### 5.2 Calendar Visualization

| ID | Requirement | Priority |
|---|---|---|
| V-01 | Display a monthly calendar grid (Mon–Fri, excluding weekends) | P0 |
| V-02 | Each day cell shows: projected seat demand (mean), seat capacity, and utilization percentage | P0 |
| V-03 | Days where projected demand exceeds seat capacity are highlighted in **red** (overflow) | P0 |
| V-04 | Days with >85% utilization highlighted in **amber** (near-overflow warning) | P1 |
| V-05 | Days with <50% utilization highlighted in **blue/grey** (underutilization signal) | P2 |
| V-06 | Hovering/clicking a day cell shows a detailed breakdown: confidence interval, overflow probability, team-level occupancy | P1 |
| V-07 | Support toggling between months | P1 |
| V-08 | Support side-by-side calendar comparison for two scenarios | P2 |

### 5.3 Simulation Parameters

The following parameters must be configurable either via chat or via a settings panel:

| Parameter | Type | Default | Example |
|---|---|---|---|
| `total_employees` | Integer | 500 | "We have 500 employees" |
| `total_seats` | Integer | 400 | "Office has 400 seats" |
| `wfh_days_per_week` | Integer (0–5) | 2 | "Employees WFH 2 days a week" |
| `seat_reduction_pct` | Float (0–100) | 0 | "Reduce seats by 10%" |
| `mandatory_office_days` | Dict[team, List[day]] | {} | "Engineering must be in on Tue and Thu" |
| `day_of_week_weights` | Dict[day, float] | See below | "Fridays have 30% lower attendance" |
| `compliance_rate` | Float (0–1) | 0.9 | "Assume 90% actually follow the policy" |
| `num_teams` | Integer | 10 | "We have 10 teams" |
| `team_sizes` | List[int] | Equal split | "Engineering is 80, Sales is 60..." |
| `simulation_months` | Integer | 1 | "Simulate for 3 months" |

**Default day-of-week weights** (relative attendance probability):

| Day | Weight |
|---|---|
| Monday | 0.85 |
| Tuesday | 1.00 |
| Wednesday | 1.00 |
| Thursday | 0.95 |
| Friday | 0.65 |

### 5.4 Pluggable Backend

| ID | Requirement | Priority |
|---|---|---|
| B-01 | All simulation backends implement a common interface/abstract class | P0 |
| B-02 | The orchestration layer selects the backend based on configuration — no frontend changes needed to swap models | P0 |
| B-03 | Backend returns a standardized result schema (see Section 7) | P0 |
| B-04 | User can select the active backend via chat ("use Monte Carlo") or settings panel | P1 |
| B-05 | Adding a new backend requires only implementing the interface and registering it — no changes to existing code | P0 |

---

## 6. System Architecture

### 6.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Frontend (React)                    │
│  ┌────────────┐  ┌──────────────────────────────┐    │
│  │  Chat Panel │  │  Calendar Visualization       │    │
│  │  (NL Input) │  │  (Monthly Grid + Heatmap)     │    │
│  └─────┬──────┘  └──────────────▲───────────────┘    │
│        │                        │                     │
│        ▼                        │                     │
│  ┌─────────────────────────────────────────────┐     │
│  │         Query Parser / Orchestrator          │     │
│  │   (Extracts params, manages conversation     │     │
│  │    state, dispatches to backend)              │     │
│  └─────────────────┬───────────────────────────┘     │
│                    │                                  │
│                    ▼                                  │
│  ┌─────────────────────────────────────────────┐     │
│  │         Simulation Engine (Pluggable)         │     │
│  │  ┌───────────┐ ┌──────────┐ ┌────────────┐  │     │
│  │  │ Binomial/  │ │  Monte   │ │  SimPy /   │  │     │
│  │  │ Poisson    │ │  Carlo   │ │  Mesa ABM  │  │     │
│  │  └───────────┘ └──────────┘ └────────────┘  │     │
│  └─────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────┘
```

### 6.2 Component Breakdown

**Chat Panel** — Text input area where users type natural language queries. Displays conversation history with interleaved simulation results.

**Query Parser / Orchestrator** — Parses natural language into structured simulation parameters. For v1, this can be an LLM call (Claude API) with a system prompt that extracts parameters into JSON. Maintains conversation state so follow-up queries inherit previous parameters. Dispatches parsed parameters to the active simulation backend.

**Simulation Engine (Pluggable)** — Implements the `SimulationBackend` interface. Receives standardized input parameters, runs the simulation, and returns standardized results.

**Calendar Visualization** — Renders simulation results as a color-coded monthly calendar grid. Handles hover interactions for detailed breakdowns.

### 6.3 Backend Interface (Abstract Class)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class SimulationInput:
    total_employees: int
    total_seats: int
    wfh_days_per_week: int
    seat_reduction_pct: float
    mandatory_office_days: dict        # {team_name: [days]}
    day_of_week_weights: dict          # {day_name: float}
    compliance_rate: float
    teams: list                        # [{name, size}]
    num_simulation_runs: int           # for stochastic methods
    start_date: str                    # YYYY-MM-DD
    end_date: str                      # YYYY-MM-DD

@dataclass
class DayResult:
    date: str                          # YYYY-MM-DD
    day_of_week: str
    expected_occupancy: float          # mean seats filled
    std_dev: float                     # standard deviation
    overflow_probability: float        # P(demand > capacity)
    percentile_5: float                # lower bound
    percentile_95: float               # upper bound
    effective_capacity: int            # seats available that day
    team_breakdown: dict               # {team_name: expected_count}

@dataclass
class SimulationResult:
    model_name: str
    parameters_used: SimulationInput
    daily_results: list                # List[DayResult]
    summary: dict                      # aggregate stats
    # summary keys:
    #   avg_utilization: float
    #   peak_occupancy: float
    #   overflow_days_count: int
    #   overflow_days_pct: float
    #   avg_overflow_magnitude: float  (when overflow happens, by how many seats)

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
```

---

## 7. Simulation Backends — Detailed Specifications

### 7.1 Backend 1: Binomial/Poisson Analytical Model

**Approach:** Closed-form computation. For each working day, compute the attendance probability `p` for each employee based on their WFH policy, team mandates, day-of-week weight, and compliance rate. Total occupancy follows a Binomial(N, p_effective) distribution. Use normal approximation for large N.

**How p_effective is computed for a given day:**

```
For each employee e on day d:
  if e.team has mandatory_office_day on d:
      p_e = compliance_rate
  else:
      base_p = (5 - wfh_days_per_week) / 5
      p_e = base_p * day_of_week_weight[d] * compliance_rate

p_effective = mean(p_e across all employees)
Expected occupancy = sum(p_e for all employees)
Variance = sum(p_e * (1 - p_e) for all employees)
```

**Overflow probability:** `P(X > seats) = 1 - Φ((seats - μ) / σ)` using normal approximation, or exact Poisson CDF if using Poisson model.

**Strengths:** Instant computation, no sampling noise, good baseline.

**Limitations:** Assumes independence between employees, no behavioral feedback loops.

### 7.2 Backend 2: Monte Carlo Simulation

**Approach:** For each simulated day, draw attendance for each employee independently from Bernoulli(p_e), where p_e is computed as above. Repeat for `num_simulation_runs` (default: 5000). Aggregate results into empirical distributions.

**Enhancements over Binomial:**

- **Team correlation:** For each team on each day, first draw a team-level "social factor" from Beta(α, β), then use this as a multiplier on individual attendance probability. This captures the "if one person goes in, others follow" effect.
- **Individual variance:** Each employee has a personal reliability factor sampled once from Beta(α, β), representing how consistently they follow their stated schedule.
- **Temporal patterns:** Optionally model week-to-week autocorrelation (employees who WFH Monday one week are slightly more likely to WFH Monday the next week).

**Output:** Empirical percentiles (5th, 25th, 50th, 75th, 95th) of occupancy for each day, plus exact overflow probability as the fraction of runs where demand exceeded capacity.

**Strengths:** Flexible, handles correlations, easy to extend.

**Limitations:** Slower than analytical (but 5000 runs × 500 employees is still sub-second).

### 7.3 Future Backend: SimPy Discrete Event Simulation (Placeholder)

**Approach:** Model the office as a resource with `total_seats` capacity. Employees arrive as a stochastic process throughout the morning (not all at once). If seats are full, employees are "turned away" (or wait in a queue if modeling shared booking). Captures time-of-day dynamics and seat turnover (e.g., someone leaves at 2 PM, freeing a seat).

**When this adds value:** When the question shifts from "how many total people" to "what happens with staggered arrivals and flexible hours."

### 7.4 Future Backend: Mesa Agent-Based Model (Placeholder)

**Approach:** Each employee is an agent with memory and adaptive rules. Agents observe previous days' occupancy and adjust their attendance probability. Captures emergent equilibria — e.g., crowding on Tuesday causes people to shift to Wednesday organically.

**When this adds value:** When modeling policy adoption dynamics over time and second-order behavioral effects.

---

## 8. Query Parsing Specification

The Query Parser converts natural language into `SimulationInput` parameter updates. It operates in **delta mode** — each query modifies the current state rather than starting from scratch.

### 8.1 Example Query Mappings

| User Query | Parameter Change |
|---|---|
| "We have 500 employees and 400 seats" | `total_employees=500, total_seats=400` |
| "What if seats are reduced by 10%?" | `seat_reduction_pct=10` |
| "Employees work from home 2 days a week" | `wfh_days_per_week=2` |
| "Make it 3 days WFH instead" | `wfh_days_per_week=3` |
| "Engineering team (80 people) must be in on Tuesday and Thursday" | `mandatory_office_days={"Engineering": ["Tuesday", "Thursday"]}, teams=[{name: "Engineering", size: 80}]` |
| "What if compliance drops to 70%?" | `compliance_rate=0.7` |
| "Fridays are dead — assume 40% attendance" | `day_of_week_weights={"Friday": 0.4}` |
| "Reset everything" | Restore defaults |
| "Compare 2-day vs 3-day WFH" | Run two simulations, return side-by-side results |
| "Switch to Monte Carlo model" | Change active backend |

### 8.2 Parser Implementation (v1)

Use Claude API with a system prompt that instructs the model to extract parameter deltas as JSON. The parser receives the current parameter state plus the new user message and returns a JSON delta.

```
System prompt (condensed):
"You are a parameter extraction engine for an office occupancy simulator.
Given the current parameters and user's message, return ONLY a JSON object
with the parameters that should change. Return {} if the message is
conversational with no parameter change."
```

---

## 9. UI Wireframe Specification

```
┌─────────────────────────────────────────────────────────────────┐
│  RTO Capacity Simulator                          [Settings ⚙️]  │
├──────────────────────────┬──────────────────────────────────────┤
│                          │                                      │
│   Chat Panel             │   Results Panel                      │
│                          │                                      │
│   ┌──────────────────┐   │   ┌──────────────────────────────┐  │
│   │ System: Welcome!  │   │   │  March 2026                   │  │
│   │ Start by telling  │   │   │  ┌───┬───┬───┬───┬───┐      │  │
│   │ me your setup...  │   │   │  │Mon│Tue│Wed│Thu│Fri│      │  │
│   └──────────────────┘   │   │  ├───┼───┼───┼───┼───┤      │  │
│                          │   │  │285│342│338│310│195│ Wk1   │  │
│   ┌──────────────────┐   │   │  │71%│86%│85%│78%│49%│      │  │
│   │ User: We have 500 │   │   │  ├───┼───┼───┼───┼───┤      │  │
│   │ employees, 400    │   │   │  │290│[R]│340│315│190│ Wk2   │  │
│   │ seats, 2 days WFH │   │   │  │73%│410│85%│79%│48%│      │  │
│   └──────────────────┘   │   │  ├───┼───┼───┼───┼───┤      │  │
│                          │   │  │288│[R]│335│312│192│ Wk3   │  │
│   ┌──────────────────┐   │   │  │72%│405│84%│78%│48%│      │  │
│   │ Bot: Running sim. │   │   │  └───┴───┴───┴───┴───┘      │  │
│   │ With 2-day WFH,   │   │   │                               │  │
│   │ avg utilization    │   │   │  [R] = Red (overflow)         │  │
│   │ is 74%. Tuesdays   │   │   │  Capacity: 400 seats          │  │
│   │ overflow in 3 of   │   │   │  Avg Util: 74%                │  │
│   │ 4 weeks (see red)  │   │   │  Overflow Days: 3/20 (15%)    │  │
│   └──────────────────┘   │   │   │  Peak: 415 (Tue Wk2)         │  │
│                          │   │  └──────────────────────────────┘  │
│   ┌──────────────────┐   │   │                                      │
│   │ User: What if we  │   │   │  ┌──────────────────────────────┐  │
│   │ reduce seats by   │   │   │  │  Summary Stats               │  │
│   │ 10%?              │   │   │  │  ━━━━━━━━━━━━━━━━━━━━━━━━━  │  │
│   └──────────────────┘   │   │  │  Model: Monte Carlo (n=5000) │  │
│                          │   │  │  Overflow Risk: 45%           │  │
│   ┌──────────────────────┐   │  │  Avg Excess: 22 seats        │  │
│   │ 💬 Type a scenario... │   │  └──────────────────────────────┘  │
│   └──────────────────────┘   │                                      │
├──────────────────────────┴──────────────────────────────────────┤
│  Active Model: [Monte Carlo ▾]    Confidence: [90% ▾]          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Frontend | React + Tailwind CSS | Component-based UI, rapid styling |
| Visualization | Recharts or custom SVG calendar grid | Lightweight, embeddable in React |
| Chat Orchestration | Claude API (Sonnet) | NL parsing, conversational state |
| Simulation Engine | Python (NumPy, SciPy) | Numerical computation, scientific ecosystem |
| API Layer | FastAPI | Lightweight, async, Python-native |
| Future: DES Backend | SimPy | Standard Python DES library |
| Future: ABM Backend | Mesa | Standard Python ABM library |
| Future: ML Calibration | XGBoost / scikit-learn | Attendance prediction from data |

---

## 11. API Specification

### 11.1 POST /simulate

**Request:**

```json
{
  "params": {
    "total_employees": 500,
    "total_seats": 400,
    "wfh_days_per_week": 2,
    "seat_reduction_pct": 10,
    "mandatory_office_days": {
      "Engineering": ["Tuesday", "Thursday"]
    },
    "day_of_week_weights": {
      "Monday": 0.85,
      "Tuesday": 1.0,
      "Wednesday": 1.0,
      "Thursday": 0.95,
      "Friday": 0.65
    },
    "compliance_rate": 0.9,
    "teams": [
      {"name": "Engineering", "size": 80},
      {"name": "Sales", "size": 60},
      {"name": "General", "size": 360}
    ],
    "num_simulation_runs": 5000,
    "start_date": "2026-03-01",
    "end_date": "2026-03-31"
  },
  "backend": "monte_carlo"
}
```

**Response:**

```json
{
  "model_name": "Monte Carlo Simulation (n=5000)",
  "daily_results": [
    {
      "date": "2026-03-02",
      "day_of_week": "Monday",
      "expected_occupancy": 285.3,
      "std_dev": 12.4,
      "overflow_probability": 0.0,
      "percentile_5": 262,
      "percentile_95": 308,
      "effective_capacity": 360,
      "team_breakdown": {
        "Engineering": 45.2,
        "Sales": 34.1,
        "General": 206.0
      }
    }
  ],
  "summary": {
    "avg_utilization": 0.74,
    "peak_occupancy": 415,
    "overflow_days_count": 3,
    "overflow_days_pct": 0.15,
    "avg_overflow_magnitude": 22.5
  }
}
```

### 11.2 POST /parse-query

**Request:**

```json
{
  "message": "What if we reduce seats by 10%?",
  "current_params": { ... }
}
```

**Response:**

```json
{
  "param_delta": {
    "seat_reduction_pct": 10
  },
  "explanation": "Reducing seat capacity by 10% from 400 to 360 seats.",
  "requires_simulation": true
}
```

### 11.3 GET /backends

**Response:**

```json
{
  "available": [
    {
      "id": "binomial",
      "name": "Binomial/Poisson Analytical",
      "description": "Closed-form probability model. Instant results, assumes independence."
    },
    {
      "id": "monte_carlo",
      "name": "Monte Carlo Simulation",
      "description": "Stochastic sampling with team correlation. 5000 runs default."
    }
  ],
  "active": "monte_carlo"
}
```

---

## 12. Project Milestones

| Phase | Deliverable | Timeline |
|---|---|---|
| **Phase 1: Core Engine** | Binomial + Monte Carlo backends with pluggable interface, FastAPI endpoints, unit tests | Week 1–2 |
| **Phase 2: Chat + Calendar UI** | React frontend with chat panel, calendar grid visualization, Claude API integration for NL parsing | Week 3–4 |
| **Phase 3: Polish + Scenarios** | Conversation context management, hover details on calendar, comparison mode, parameter settings panel | Week 5–6 |
| **Phase 4: Extensions** | SimPy backend, Mesa ABM backend, ML-calibrated attendance probabilities (stretch) | Week 7–8 |

---

## 13. Testing Strategy

**Unit Tests:** Each backend is tested independently against known analytical results. For the Binomial model, verify against scipy.stats.binom. For Monte Carlo, verify that empirical distributions converge to analytical values within tolerance.

**Integration Tests:** End-to-end test from NL query through parsing, simulation, and result rendering. Verify that parameter deltas apply correctly in conversation context.

**Scenario Tests:** A suite of predefined scenarios with expected outcomes — e.g., "500 employees, 500 seats, 0 WFH days" should yield ~0% overflow; "500 employees, 200 seats, 0 WFH days" should yield ~100% overflow.

**Visual Regression:** Calendar rendering matches expected color coding for known simulation outputs.

---

## 14. Future Enhancements (Post-v1)

- **ML-Calibrated Attendance:** Train XGBoost on real badge data to replace assumed probabilities with learned ones.
- **Behavioral Clustering:** GMM/K-Means on attendance patterns to create employee archetypes feeding into simulation.
- **Optimization Mode:** "Find me the minimum seats needed for <5% overflow risk" using Bayesian optimization over the simulator.
- **Multi-Floor / Zone Simulation:** Model individual floors or zones with separate capacities.
- **Cost Module:** Translate seat counts into real estate cost estimates (cost per seat per month).
- **Collaboration Score:** Use team interaction data to measure in-person collaboration overlap and optimize mandatory days accordingly.
- **Real-Time Dashboard:** Connect to live badge/Wi-Fi data and show actual vs. predicted occupancy.

---

## 15. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| NL parsing misinterprets user intent | Wrong simulation parameters, confusing results | Show parsed parameters back to user for confirmation before running simulation |
| Monte Carlo too slow for large employee counts | Poor UX with loading delays | Cache results, use analytical model as fast fallback, optimize with vectorized NumPy |
| Assumed attendance probabilities unrealistic | Results don't match reality | Clearly label assumptions, provide calibration guidance, plan ML backend for v2 |
| Scope creep into spatial/booking simulation | Delays core delivery | Strict non-goals enforcement; spatial modeling deferred to v2+ |

---

## 16. Success Metrics

- User can go from zero configuration to first simulation result in **under 60 seconds** of chat interaction.
- Simulation runs in **under 2 seconds** for 500 employees, 1 month, 5000 Monte Carlo runs.
- Adding a new backend requires implementing **only the interface methods** (no changes to frontend, API, or orchestrator).
- Calendar visualization clearly communicates overflow risk — validated via user feedback that "red days" are immediately understood.
