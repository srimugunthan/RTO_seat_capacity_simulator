# RTO Capacity Simulator — Implementation Plan

## Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **UI Framework** | Streamlit | Rapid development, pure Python, no JS/npm toolchain |
| **API Layer** | FastAPI | Kept so future frontends (React, mobile) can use the same API |
| **LLM Integration** | LangChain | Provider-agnostic; swap LLM via config with no code changes |
| **Default LLM** | Configurable via `.env` | Start with any cheap model (GPT-4o mini, Gemini Flash, Ollama) |
| **Simulation Engine** | Python (NumPy, SciPy) | Pluggable backends; shared by API and Streamlit directly |

### System Architecture

```
┌─────────────────────────────────────────────┐
│           FastAPI (port 8000)                │
│   /simulate  /chat  /backends  /parse-query  │
└────────────────────┬────────────────────────┘
                     │ HTTP
          ┌──────────┴──────────┐
          │                     │
   ┌──────▼──────┐      ┌───────▼──────┐
   │  Streamlit   │      │ React / other │
   │  (port 8501) │      │ (future)      │
   └─────────────┘      └──────────────┘
```

Streamlit is the first frontend client of the FastAPI — identical to how any future React frontend would consume it.

### LLM Provider Abstraction (LangChain)

```python
# Change provider by setting LLM_PROVIDER in .env — zero code changes
LLM_PROVIDER=openai          # or: anthropic, groq, ollama
LLM_MODEL=gpt-4o-mini        # or: claude-haiku-4-5, llama3.1, gemma2

# All providers share the same interface via LangChain BaseChatModel
llm = get_llm()  # returns ChatOpenAI / ChatAnthropic / ChatOllama / ChatGroq

# Structured output: returns ParamDelta Pydantic model directly
structured_llm = llm.with_structured_output(ParamDelta)
delta = structured_llm.invoke([SystemMessage(...), HumanMessage(...)])
```

---

## Phase 1: Core Simulation Engine (Backend Only)

**Goal:** Implement both simulation backends with the pluggable interface. No UI, no API — pure Python library testable via scripts and unit tests.

### Deliverables

1. **Project scaffolding**
   - `backend/` directory with `__init__.py`
   - `requirements.txt` (numpy, scipy, pytest)
   - Basic project structure

2. **Abstract base + data classes** (`backend/models.py`)
   - `SimulationInput` dataclass
   - `DayResult` dataclass
   - `SimulationResult` dataclass
   - `SimulationBackend` abstract class with `name()`, `run()`, `description()`

3. **Binomial/Poisson analytical backend** (`backend/binomial_backend.py`)
   - Per-employee `p_effective` computation (mandatory days, WFH days, day-of-week weights, compliance rate)
   - Normal approximation for occupancy distribution
   - Overflow probability via `1 - Φ((seats - μ) / σ)`
   - Returns full `SimulationResult` with daily stats

4. **Monte Carlo backend** (`backend/monte_carlo_backend.py`)
   - Bernoulli sampling per employee per day
   - Team-level social factor via Beta distribution
   - Individual reliability factor per employee
   - Empirical percentile computation (5th, 25th, 50th, 75th, 95th)
   - Exact overflow probability as fraction of overflowing runs

5. **Backend registry** (`backend/registry.py`)
   - `register_backend(id, backend_class)`
   - `get_backend(id) -> SimulationBackend`
   - `list_backends() -> list`
   - Pre-registered: `binomial`, `monte_carlo`

### Test Criteria

```
pytest backend/tests/
```

- `test_binomial.py`: Known-answer tests (500 employees, 500 seats, 0 WFH → ~0% overflow; 500 employees, 200 seats, 0 WFH → ~100% overflow)
- `test_monte_carlo.py`: Monte Carlo empirical mean converges to Binomial mean within ±5%
- `test_interface.py`: Both backends implement the full interface and return `SimulationResult` with correct schema
- `test_registry.py`: Registry correctly registers, retrieves, and lists backends

### Directory Structure After Phase 1

```
backend/
  models.py
  binomial_backend.py
  monte_carlo_backend.py
  registry.py
  tests/
    test_binomial.py
    test_monte_carlo.py
    test_interface.py
    test_registry.py
requirements.txt
```

---

## Phase 2: FastAPI REST Layer

**Goal:** Wrap the simulation engine in a REST API. Testable via curl, Postman, or automated API tests — no frontend needed.

### Deliverables

1. **FastAPI app** (`api/main.py`)
   - Dependency injection for backend registry
   - CORS middleware configured for Streamlit and future frontends

2. **`POST /simulate`** endpoint
   - Accepts `SimulationInput` params + `backend` id as JSON
   - Validates input (date range, employee counts > 0, etc.)
   - Returns `SimulationResult` serialized as JSON

3. **`POST /parse-query`** endpoint (stub for Phase 3)
   - Accepts `message` + `current_params`
   - Returns hardcoded `{ "param_delta": {}, "explanation": "stub", "requires_simulation": false }` in this phase
   - Replaced with real LLM call in Phase 3

4. **`GET /backends`** endpoint
   - Returns all registered backends with id, name, description
   - Returns currently active backend id

5. **Input validation and error responses**
   - Pydantic models for request/response schemas
   - Meaningful 422 errors for invalid parameters

### Test Criteria

```
pytest api/tests/
# or manually:
uvicorn api.main:app --reload
curl -X POST http://localhost:8000/simulate -H "Content-Type: application/json" -d @api/tests/fixtures/basic_request.json
```

- `test_simulate_endpoint.py`: Valid requests return 200 with correct schema; invalid requests return 422
- `test_backends_endpoint.py`: Returns both backends; switching backend changes results
- `test_parse_query_stub.py`: Stub returns expected shape
- Verify sub-2-second response for 500 employees, 1 month, 5000 Monte Carlo runs

### Directory Structure After Phase 2

```
api/
  main.py
  schemas.py        # Pydantic request/response models
  tests/
    test_simulate_endpoint.py
    test_backends_endpoint.py
    test_parse_query_stub.py
    fixtures/
      basic_request.json
      engineering_mandates.json
```

---

## Phase 3: NL Query Parser (LangChain Integration)

**Goal:** Replace the `/parse-query` stub with a real LLM integration via LangChain. Provider is configurable via `.env` — no code changes to swap models. Testable end-to-end via the API, still no frontend required.

### Deliverables

1. **LLM provider factory** (`api/llm.py`)
   - `get_llm() -> BaseChatModel` reads `LLM_PROVIDER` + `LLM_MODEL` from env
   - Supports: `openai` (ChatOpenAI), `anthropic` (ChatAnthropic), `groq` (ChatGroq), `ollama` (ChatOllama)
   - Used by query parser; easily extensible to new providers

2. **Query parser module** (`api/query_parser.py`)
   - `ParamDelta` Pydantic model with all optional `SimulationInput` fields
   - Uses `llm.with_structured_output(ParamDelta)` — returns typed object, no JSON parsing
   - System prompt instructs LLM to extract only changed parameters
   - Returns `param_delta` dict, `explanation` string, `requires_simulation` bool
   - Handles empty delta `{}` for conversational messages with no param change

3. **Conversation context manager** (`api/conversation.py`)
   - Maintains session state: current params + LangChain `ChatMessageHistory`
   - Applies `param_delta` to current params on each turn
   - Passes full message history to LLM for follow-up context
   - Supports reset to defaults

4. **`POST /parse-query`** endpoint (real implementation)
   - Wires up `query_parser.py` with session context
   - Returns parsed delta + updated full params

5. **`POST /chat`** convenience endpoint (combines parse + simulate)
   - Accepts message + session_id
   - Parses query, applies delta, runs simulation if needed
   - Returns `{ param_delta, explanation, simulation_result | null }`

6. **Environment config** (`api/config.py` + `.env.example`)
   ```
   LLM_PROVIDER=openai          # anthropic | groq | ollama
   LLM_MODEL=gpt-4o-mini
   DEFAULT_BACKEND=monte_carlo
   ```

### Test Criteria

```
pytest api/tests/test_parser.py
pytest api/tests/test_chat_endpoint.py
```

- `test_llm_factory.py`: Each provider config returns correct LangChain class; unknown provider raises clear error
- `test_parser.py`: Known query strings map to expected param deltas (table-driven, from PRD Section 8.1)
- `test_chat_endpoint.py`: Full conversation flow — initial setup → follow-up query → reset → verify params applied correctly
- `test_context_management.py`: Delta accumulation correct across multiple turns; reset restores defaults
- Manual test: multi-turn conversation via curl; verify provider swap by changing `.env`

### Directory Structure After Phase 3

```
api/
  main.py
  schemas.py
  llm.py            # LangChain provider factory
  query_parser.py   # uses LangChain structured output
  conversation.py   # ChatMessageHistory + param accumulation
  config.py
  tests/
    test_llm_factory.py
    test_parser.py
    test_chat_endpoint.py
    test_context_management.py
.env.example
```

---

## Phase 4: Streamlit Frontend

**Goal:** Build the full user-facing application in Streamlit, calling the FastAPI via HTTP. Delivers the complete chat + calendar experience in a single Python file. Testable end-to-end by running both services.

### Deliverables

1. **API client module** (`streamlit_app/api_client.py`)
   - `simulate(params, backend) -> SimulationResult`
   - `chat(message, session_id) -> ChatResponse`
   - `get_backends() -> list`
   - Calls FastAPI at `http://localhost:8000`; base URL configurable via env

2. **Calendar renderer** (`streamlit_app/calendar.py`)
   - Generates HTML table: Mon–Fri columns, weeks as rows
   - Color coding: red (overflow), amber (>85%), blue/grey (<50%), green (normal)
   - Each cell: occupancy, capacity, utilization %
   - Rendered via `st.markdown(html, unsafe_allow_html=True)`
   - Day detail shown in `st.expander` on click (confidence interval, overflow probability, team breakdown)

3. **Summary stats** (`streamlit_app/stats.py`)
   - `st.metric` tiles: avg utilization, peak occupancy, overflow days, avg overflow magnitude
   - Active model name displayed

4. **Main Streamlit app** (`streamlit_app/app.py`)
   - **Left panel (chat):** `st.chat_input` + `st.chat_message` conversation history; shows extracted params before running simulation
   - **Right panel (results):** calendar grid + summary stats; month navigation via `st.selectbox`
   - **Sidebar:**
     - `st.selectbox("Simulation Model", options)` — populated by calling `GET /backends` at startup; selecting a different model immediately re-runs the simulation with the new backend; new backends appear here automatically when registered in `registry.py`
     - `st.caption` showing the selected model's description (from `GET /backends` response)
     - LLM provider display (read-only, from env)
     - `st.selectbox("Confidence Interval", ["90%", "95%"])`
     - `st.button("Reset")` — clears conversation and restores default params
   - Session state: `st.session_state` holds session_id, current params, conversation history, last simulation result, active backend id

5. **Param confirmation display**
   - After NL parsing, shows `param_delta` as `st.info` box before running simulation
   - Addresses PRD risk: "NL parsing misinterprets user intent"

### Test Criteria

```
# Start both services:
uvicorn api.main:app --reload &
streamlit run streamlit_app/app.py

# Automated:
pytest streamlit_app/tests/test_calendar.py
pytest streamlit_app/tests/test_api_client.py
```

- `test_calendar.py`: Known `SimulationResult` fixture renders correct HTML — overflow days contain red styling, amber/blue thresholds applied correctly
- `test_api_client.py`: API client methods return correctly typed objects; handles API errors gracefully
- Manual E2E: Type "500 employees, 400 seats, 2 days WFH" → calendar renders with color coding → type "reduce seats by 10%" → calendar updates → backend dropdown switches model

### Directory Structure After Phase 4

```
streamlit_app/
  app.py            # main Streamlit entry point
  api_client.py     # HTTP client for FastAPI
  calendar.py       # HTML calendar renderer
  stats.py          # summary metric components
  tests/
    test_calendar.py
    test_api_client.py
```

---

## Phase 5: Polish, Comparison Mode, and Hardening

**Goal:** Deliver remaining P1/P2 features, performance hardening, and deployment setup.

### Deliverables

1. **Side-by-side scenario comparison** (PRD F-06, V-08)
   - Chat command: "Compare 2-day vs 3-day WFH"
   - API runs two simulations in parallel (asyncio)
   - Streamlit renders two calendars side by side using `st.columns(2)`
   - Summary stats shown for both scenarios

2. **Result caching** (PRD Risk: Monte Carlo slowness)
   - Cache simulation results in FastAPI keyed by hashed `SimulationInput`
   - Return cached result instantly if params unchanged
   - `st.cache_data` on Streamlit side for repeated renders

3. **Performance validation**
   - Benchmark: 500 employees, 1 month, 5000 MC runs < 2 seconds (PRD success metric)
   - Vectorize Monte Carlo with NumPy if needed
   - Add `GET /health` endpoint with response time stats

4. **Scenario test suite** (PRD Section 13)
   - `tests/scenarios/test_known_scenarios.py`
   - Edge cases: all WFH, zero WFH, seats = employees, seats << employees
   - Verify Binomial and Monte Carlo agree within ±5% tolerance

5. **Error handling**
   - API: LLM failure → clear error message asking user to rephrase; LangChain `.with_fallbacks()` for backup model
   - Streamlit: `st.error` for API failures; spinner during LLM + simulation calls
   - Validation: start_date < end_date, total team sizes ≤ total_employees

6. **Docker Compose setup**
   - `docker-compose.yml`: starts FastAPI + Streamlit together
   - Single command: `docker compose up`
   - `docker compose up && open http://localhost:8501`

### Test Criteria

```
pytest tests/scenarios/
pytest --benchmark
docker compose up && open http://localhost:8501  # smoke test
```

- All scenario tests pass
- Monte Carlo benchmark < 2 seconds
- Comparison mode renders two calendars correctly side by side
- LLM fallback triggers when primary provider fails
- Application starts cleanly via Docker Compose

---

## Phase 6: SimPy Discrete-Event Simulation Backend

**Goal:** Add a third simulation backend using SimPy's discrete-event simulation (DES) to model fine-grained office dynamics — arrival queues, seat contention, and intraday flows — that the probabilistic backends cannot capture. Plugs into the existing registry with zero changes to the API or frontend.

### Motivation

The Binomial and Monte Carlo backends model *daily headcounts* statistically. SimPy DES models *events in time*:

- Each employee is an agent who arrives at a random time, claims a seat (or waits/leaves if none available), works, then departs
- Overflow is detected as *concurrent occupancy* rather than daily totals
- Captures peak-hour crowding that daily averages hide
- Useful for "what if half the team arrives between 9–10 am?" scenarios

### Deliverables

1. **SimPy backend** (`backend/simpy_backend.py`)
   - Implements `SimulationBackend` ABC: `name()`, `description()`, `run(input) -> SimulationResult`
   - Per-day simulation loop driven by `simpy.Environment`
   - **Employee process:** each employee has
     - arrival time drawn from a configurable distribution (default: Normal(μ=9h, σ=1h), clipped to 7–12h)
     - work duration drawn from Normal(μ=8h, σ=1h), clipped to 4–10h
     - WFH decision gate: same `p_effective` logic as Binomial/Monte Carlo backends
   - **Seat resource:** `simpy.Resource(capacity=seats)` — employees `request()` a seat on arrival; if all seats taken they are counted as turned-away (overflow event) and leave immediately
   - **Statistics collected per day:**
     - peak concurrent occupancy (max in-office at any instant)
     - total arrivals vs turned-away count
     - seat utilisation over time (sampled every 30 min)
     - overflow flag: `peak_occupancy > seats`
   - **Output mapping:** fills standard `DayResult` fields so the rest of the stack works unchanged
     - `expected_occupancy` ← mean peak occupancy across N runs
     - `overflow_probability` ← fraction of runs where peak > seats
     - `p5/p25/p50/p75/p95` ← percentiles of peak occupancy distribution
   - Configurable: `num_runs` (default 1000), `time_step_minutes` (default 30), arrival distribution parameters

2. **Registry registration** (`backend/registry.py`)
   - One new line: `register_backend("simpy_des", SimpyBackend())`
   - Backend id `"simpy_des"` appears automatically in `GET /backends` and Streamlit sidebar

3. **Dependency** (`requirements.txt`)
   - Add `simpy>=4.1`

4. **Tests** (`backend/tests/test_simpy_backend.py`)
   - Implements full `SimulationBackend` interface (schema test)
   - Known-answer: 500 employees, 500 seats, 0 WFH → overflow probability < 10%
   - Known-answer: 500 employees, 100 seats, 0 WFH → overflow probability > 90%
   - Agrees with Monte Carlo on expected daily occupancy within ±15% (DES variance is higher at 1000 runs)
   - Peak occupancy ≤ total employees on office days
   - Arrival distribution parameter changes shift peak hour (smoke test)
   - Performance: 500 employees, 1 month, 1000 runs < 10 seconds

5. **Scenario tests** (`tests/scenarios/test_known_scenarios.py`)
   - Extend `TestCrossBackendAgreement` to include `simpy_des` in the parametrized backend list
   - Tolerance widened to ±15% for DES vs analytical backends

6. **API/UI** — zero changes required
   - `GET /backends` returns `simpy_des` automatically
   - Streamlit sidebar shows it; `POST /compare` can compare DES vs Monte Carlo

### SimPy Process Design

```python
def employee_process(env, seats_resource, arrival_time, work_duration, result_tracker):
    yield env.timeout(arrival_time)          # wait until arrival time
    with seats_resource.request() as req:
        result = yield req | env.timeout(0)  # try to claim seat immediately
        if req in result:
            result_tracker.record_arrival()
            yield env.timeout(work_duration) # occupy seat for work duration
            result_tracker.record_departure()
        else:
            result_tracker.record_turned_away()  # no seat available

def run_one_day(seats, employees_in_office, arrival_params, work_params):
    env = simpy.Environment()
    seats_resource = simpy.Resource(env, capacity=seats)
    tracker = DayTracker(env, seats_resource)
    for _ in range(employees_in_office):
        arrival = sample_arrival(arrival_params)
        duration = sample_work_duration(work_params)
        env.process(employee_process(env, seats_resource, arrival, duration, tracker))
    env.run(until=END_OF_DAY)
    return tracker.peak_occupancy(), tracker.turned_away_count()
```

### Test Criteria

```
pytest backend/tests/test_simpy_backend.py -v
pytest tests/scenarios/test_known_scenarios.py -v -k "cross_backend"
```

- All interface/schema tests pass
- Known-answer overflow tests pass
- Cross-backend agreement within ±15%
- Performance: 1 month simulation under 10 seconds
- `simpy_des` visible in `GET /backends` response

### Directory Structure After Phase 6

```
backend/
  models.py
  binomial_backend.py
  monte_carlo_backend.py
  simpy_backend.py          ← new
  registry.py               ← +1 line
  tests/
    test_binomial.py
    test_monte_carlo.py
    test_interface.py
    test_registry.py
    test_simpy_backend.py   ← new
requirements.txt            ← simpy>=4.1 added
```

---

## Summary: Phase-by-Phase Test Matrix

| Phase | What | Test Type | Tool | Passes When |
|---|---|---|---|---|
| 1 | Simulation engine | Unit | pytest | Known-answer tests pass; Binomial ≈ Monte Carlo |
| 2 | FastAPI layer | API | pytest + curl | `/simulate` returns valid JSON in < 2s |
| 3 | LangChain NL parser | Integration | pytest | NL queries → correct param deltas; multi-turn context works; provider swap works |
| 4 | Streamlit frontend | E2E manual | Browser | Full chat → calendar flow works end-to-end |
| 5 | Polish + hardening | System | All above + benchmark | All tests pass; Docker Compose starts cleanly |
| 6 | SimPy DES backend | Unit + integration | pytest | DES agrees with MC within ±15%; auto-appears in API + UI; perf < 10s |

---

## File Structure (Final)

```
RTO_Capacity_Simulator/
├── backend/
│   ├── models.py
│   ├── binomial_backend.py
│   ├── monte_carlo_backend.py
│   ├── simpy_backend.py        # Phase 6
│   ├── registry.py
│   └── tests/
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── llm.py              # LangChain provider factory
│   ├── query_parser.py     # structured output via LangChain
│   ├── conversation.py     # ChatMessageHistory + param state
│   ├── config.py
│   └── tests/
├── streamlit_app/
│   ├── app.py
│   ├── api_client.py
│   ├── calendar.py
│   ├── stats.py
│   └── tests/
├── tests/
│   └── scenarios/
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── plan.md
```

---

## Dependencies Between Phases

```
Phase 1 (Simulation Engine)
    └── Phase 2 (FastAPI)
            └── Phase 3 (LangChain NL Parser)
                    └── Phase 4 (Streamlit Frontend)
                            └── Phase 5 (Polish + Hardening)
                                    └── Phase 6 (SimPy DES Backend)
```

Phases 1–5 are sequential — each builds directly on the previous. Phase 6 is additive: it only touches `backend/` and `requirements.txt`; the API and frontend require zero changes.
