# RTO Capacity Simulator

An AI-powered tool for simulating office seat occupancy under hybrid work policies. Users describe scenarios in natural language and receive visual, calendar-based occupancy projections highlighting overflow risk.

## Architecture

```
FastAPI (port 8000)  ←  Streamlit UI (port 8501)
                     ←  Future frontends (React, etc.)

Simulation backends: Binomial/Poisson | Monte Carlo
NL parser: LangChain (provider-agnostic: OpenAI, Anthropic, Groq, Ollama)
```

## Project Status

| Phase | Description | Status |
|---|---|---|
| 1 | Core simulation engine | ✓ Done |
| 2 | FastAPI REST layer | ✓ Done |
| 3 | LangChain NL query parser | ✓ Done |
| 4 | Streamlit frontend | ✓ Done |
| 5 | Polish, comparison mode, Docker | ✓ Done |

---

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone the repo
git clone <repo-url>
cd RTO_Capacity_simulator

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
uv pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set your LLM_PROVIDER and API key
```

`.env` example:
```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
DEFAULT_BACKEND=monte_carlo
```

Supported providers: `openai`, `anthropic`, `groq`, `ollama` (set `LLM_PROVIDER` accordingly).

---

## Running the App

### Option A — Local (two terminals)

```bash
# Terminal 1: start the API
source .venv/bin/activate
uvicorn api.main:app --reload
# API running at http://localhost:8000
# Swagger docs at http://localhost:8000/docs

# Terminal 2: start the UI
source .venv/bin/activate
streamlit run streamlit_app/app.py
# UI running at http://localhost:8501
```

### Option B — Docker Compose (one command)

```bash
cp .env.example .env   # fill in your API key
docker compose up --build
# API: http://localhost:8000
# UI:  http://localhost:8501
```

---

## Running the Tests

```bash
source .venv/bin/activate

# All tests (147)
pytest backend/tests/ api/tests/ streamlit_app/tests/ tests/scenarios/ -v

# By layer
pytest backend/tests/ -v          # simulation engine (31 tests)
pytest api/tests/ -v              # FastAPI endpoints (68 tests)
pytest streamlit_app/tests/ -v    # Streamlit client (33 tests)
pytest tests/scenarios/ -v        # edge cases + performance (15 tests)
```

---

## API Reference

### GET /health

```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "cache_size": 0}
```

---

### GET /backends — List simulation models

```bash
curl http://localhost:8000/backends
```
```json
{
  "available": [
    {"id": "binomial", "name": "Binomial/Poisson Analytical", "description": "..."},
    {"id": "monte_carlo", "name": "Monte Carlo Simulation", "description": "..."}
  ],
  "active": "monte_carlo"
}
```

---

### POST /simulate — Run a simulation

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "params": {
      "total_employees": 500,
      "total_seats": 400,
      "wfh_days_per_week": 2,
      "compliance_rate": 0.9,
      "num_simulation_runs": 1000,
      "start_date": "2026-03-02",
      "end_date": "2026-03-06"
    },
    "backend": "monte_carlo"
  }'
```

With teams and mandatory office days:

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "params": {
      "total_employees": 500,
      "total_seats": 400,
      "wfh_days_per_week": 3,
      "seat_reduction_pct": 10,
      "mandatory_office_days": {"Engineering": ["Tuesday", "Thursday"]},
      "teams": [
        {"name": "Engineering", "size": 80},
        {"name": "Sales", "size": 60},
        {"name": "General", "size": 360}
      ],
      "start_date": "2026-03-01",
      "end_date": "2026-03-31"
    },
    "backend": "monte_carlo"
  }'
```

Response shape:
```json
{
  "model_name": "Monte Carlo Simulation",
  "daily_results": [
    {
      "date": "2026-03-02",
      "day_of_week": "Monday",
      "expected_occupancy": 229.5,
      "std_dev": 12.3,
      "overflow_probability": 0.0,
      "percentile_5": 205.0,
      "percentile_95": 253.0,
      "effective_capacity": 360,
      "team_breakdown": {"Engineering": 45.2, "Sales": 21.3, "General": 163.0}
    }
  ],
  "summary": {
    "avg_utilization": 0.72,
    "peak_occupancy": 312.4,
    "overflow_days_count": 0,
    "overflow_days_pct": 0.0,
    "avg_overflow_magnitude": 0.0
  }
}
```

---

### POST /compare — Side-by-side scenario comparison

Runs two simulations in parallel and returns both results.

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_a": {
      "params": {"total_employees": 500, "total_seats": 400, "wfh_days_per_week": 2},
      "backend": "binomial"
    },
    "scenario_b": {
      "params": {"total_employees": 500, "total_seats": 400, "wfh_days_per_week": 3},
      "backend": "binomial"
    },
    "label_a": "2-day WFH",
    "label_b": "3-day WFH"
  }'
```

Response shape:
```json
{
  "scenario_a": {"label": "2-day WFH", "simulation_result": {...}},
  "scenario_b": {"label": "3-day WFH", "simulation_result": {...}}
}
```

You can also trigger comparison from the chat UI by saying e.g. *"Compare 2 vs 3 WFH days"* — the LLM detects comparison intent and the UI renders two calendars side-by-side.

---

### POST /chat — NL chat (parse + simulate in one call)

Requires a running LLM (set `.env` first).

```bash
# First message — sets up a scenario
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "500 employees, 400 seats, 2 days WFH"}'

# Follow-up (include the session_id returned above)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "reduce seats by 10%", "session_id": "<session_id>"}'
```

Response includes `session_id`, `explanation`, `param_delta`, `current_params`, `simulation_result`, and optionally `comparison_result`.

---

### POST /parse-query — NL parsing only (no simulation)

```bash
curl -X POST http://localhost:8000/parse-query \
  -H "Content-Type: application/json" \
  -d '{"message": "3 days WFH and reduce seats by 10%"}'
```

```json
{
  "param_delta": {"wfh_days_per_week": 3, "seat_reduction_pct": 10.0},
  "explanation": "Changed WFH to 3 days and applied 10% seat reduction.",
  "requires_simulation": true
}
```

---

## Validation Errors

```bash
# Invalid backend → 400
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"params": {"total_employees": 100, "total_seats": 80}, "backend": "invalid"}'

# end_date before start_date → 422
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "params": {"total_employees": 100, "total_seats": 80,
               "start_date": "2026-03-31", "end_date": "2026-03-01"},
    "backend": "binomial"
  }'
```

---

## Project Structure

```
RTO_Capacity_Simulator/
├── backend/
│   ├── models.py              # SimulationInput, DayResult, SimulationResult, SimulationBackend
│   ├── binomial_backend.py    # Analytical closed-form model
│   ├── monte_carlo_backend.py # Stochastic sampling with team correlation
│   ├── registry.py            # Pluggable backend registry
│   └── tests/
├── api/
│   ├── main.py                # FastAPI app + caching + compare endpoint
│   ├── schemas.py             # Pydantic request/response models
│   ├── llm.py                 # LangChain provider factory
│   ├── query_parser.py        # Structured NL output via LangChain
│   ├── conversation.py        # Session state + delta accumulation
│   ├── config.py
│   └── tests/
│       └── fixtures/          # Sample request JSON files
├── streamlit_app/
│   ├── app.py                 # Main UI (chat + calendar + comparison view)
│   ├── api_client.py          # HTTP client (simulate, compare, chat)
│   ├── calendar.py            # HTML calendar renderer
│   ├── stats.py               # Summary metric tiles
│   └── tests/
├── tests/
│   └── scenarios/
│       ├── test_known_scenarios.py   # Edge cases + cross-backend agreement
│       └── test_performance.py       # MC < 2s, Binomial < 100ms
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── conftest.py
├── requirements.txt
├── .env.example
└── plan.md
```

## Adding a New Simulation Backend

1. Create `backend/your_backend.py` implementing `SimulationBackend`:
   ```python
   from backend.models import SimulationBackend, SimulationInput, SimulationResult

   class YourBackend(SimulationBackend):
       def name(self) -> str: return "Your Model Name"
       def description(self) -> str: return "What your model does."
       def run(self, params: SimulationInput) -> SimulationResult: ...
   ```

2. Register it in `backend/registry.py`:
   ```python
   from .your_backend import YourBackend
   register_backend("your_model", YourBackend())
   ```

3. It appears automatically in `GET /backends`, the Streamlit sidebar, and the `/compare` endpoint. No other files change.
