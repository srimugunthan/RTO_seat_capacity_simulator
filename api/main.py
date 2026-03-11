import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.models import SimulationInput
from backend.registry import get_backend, list_backends
from .conversation import conversation_manager
from .llm import get_llm
from .query_parser import parse_query
from .schemas import (
    BackendInfoSchema,
    BackendsResponse,
    CompareRequest,
    CompareResponse,
    DayResultSchema,
    ParseQueryRequest,
    ParseQueryResponse,
    ScenarioResult,
    SimulateRequest,
    SimulateResponse,
    SimulationParamsSchema,
    SummarySchema,
    TeamSchema,
)

# ── Simulation cache ───────────────────────────────────────────────────────────
_simulation_cache: dict[str, SimulateResponse] = {}
_executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(
    title="RTO Capacity Simulator API",
    description="Simulate office seat occupancy under hybrid work policies.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _params_schema_to_sim_input(p: SimulationParamsSchema) -> SimulationInput:
    return SimulationInput(
        total_employees=p.total_employees,
        total_seats=p.total_seats,
        wfh_days_per_week=p.wfh_days_per_week,
        seat_reduction_pct=p.seat_reduction_pct,
        mandatory_office_days=p.mandatory_office_days,
        day_of_week_weights=p.day_of_week_weights,
        compliance_rate=p.compliance_rate,
        teams=[{"name": t.name, "size": t.size} for t in p.teams],
        num_simulation_runs=p.num_simulation_runs,
        start_date=p.start_date,
        end_date=p.end_date,
    )


def _dict_to_params_schema(params: dict) -> SimulationParamsSchema:
    teams = [TeamSchema(**t) for t in params.get("teams", [])]
    return SimulationParamsSchema(**{**params, "teams": teams})


def _result_to_response(result) -> SimulateResponse:
    return SimulateResponse(
        model_name=result.model_name,
        daily_results=[DayResultSchema(**dr.__dict__) for dr in result.daily_results],
        summary=SummarySchema(**result.summary),
    )


# ── Chat request/response schemas ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    param_delta: dict
    explanation: str
    requires_simulation: bool
    current_params: SimulationParamsSchema
    active_backend: str
    simulation_result: Optional[SimulateResponse] = None
    comparison_result: Optional[CompareResponse] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "cache_size": len(_simulation_cache)}


@app.get("/backends", response_model=BackendsResponse)
def get_backends_endpoint():
    return BackendsResponse(
        available=[BackendInfoSchema(**b) for b in list_backends()],
        active="monte_carlo",
    )


def _cache_key(request: SimulateRequest) -> str:
    return hashlib.md5(request.model_dump_json().encode()).hexdigest()


def _run_simulation(request: SimulateRequest) -> SimulateResponse:
    """Core simulation runner with caching. Used by /simulate and /compare."""
    key = _cache_key(request)
    if key in _simulation_cache:
        return _simulation_cache[key]

    try:
        backend = get_backend(request.backend)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    sim_input = _params_schema_to_sim_input(request.params)
    result = backend.run(sim_input)
    response = _result_to_response(result)
    _simulation_cache[key] = response
    return response


@app.post("/simulate", response_model=SimulateResponse)
def simulate(request: SimulateRequest):
    return _run_simulation(request)


@app.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest):
    """Run two simulations in parallel and return both results."""
    loop = asyncio.get_event_loop()
    result_a, result_b = await asyncio.gather(
        loop.run_in_executor(_executor, _run_simulation, request.scenario_a),
        loop.run_in_executor(_executor, _run_simulation, request.scenario_b),
    )
    return CompareResponse(
        scenario_a=ScenarioResult(label=request.label_a, simulation_result=result_a),
        scenario_b=ScenarioResult(label=request.label_b, simulation_result=result_b),
    )


@app.post("/parse-query", response_model=ParseQueryResponse)
def parse_query_endpoint(request: ParseQueryRequest):
    current = request.current_params.model_dump() if request.current_params else {}
    try:
        llm = get_llm()
        delta = parse_query(request.message, current, llm)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")

    return ParseQueryResponse(
        param_delta=delta.model_dump(
            exclude={"explanation", "requires_simulation", "reset", "backend"},
            exclude_none=True,
        ),
        explanation=delta.explanation,
        requires_simulation=delta.requires_simulation,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # 1. Get or create session
    state = conversation_manager.get_or_create(request.session_id)

    # 2. Parse the message with LLM
    try:
        llm = get_llm()
        delta = parse_query(
            message=request.message,
            current_params=state.params,
            llm=llm,
            message_history=state.messages,
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"LLM unavailable: {e}. Check your LLM_PROVIDER and API key in .env",
        )

    # 3. Apply delta (handles reset, backend switch, and param updates)
    conversation_manager.apply_delta(state, delta)

    # 4. Run simulation if params changed
    simulation_result = None
    comparison_result = None
    if delta.requires_simulation:
        try:
            sim_request = SimulateRequest(
                params=_dict_to_params_schema(state.params),
                backend=state.active_backend,
            )
            simulation_result = _run_simulation(sim_request)

            # Comparison: if user asked "compare X vs Y"
            if delta.comparison_with:
                import copy as _copy
                b_params = _copy.deepcopy(state.params)
                b_backend = state.active_backend
                for k, v in delta.comparison_with.items():
                    if k == "backend":
                        b_backend = v
                    elif k in b_params and isinstance(b_params[k], dict) and isinstance(v, dict):
                        b_params[k].update(v)
                    else:
                        b_params[k] = v
                sim_b = SimulateRequest(
                    params=_dict_to_params_schema(b_params),
                    backend=b_backend,
                )
                result_b = _run_simulation(sim_b)
                comparison_result = CompareResponse(
                    scenario_a=ScenarioResult(label="Scenario A", simulation_result=simulation_result),
                    scenario_b=ScenarioResult(label="Scenario B", simulation_result=result_b),
                )
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 5. Record turn in conversation history
    conversation_manager.add_turn(state, request.message, delta.explanation)

    # 6. Build param_delta dict (only non-null changed fields)
    param_delta_dict = delta.model_dump(
        exclude={"explanation", "requires_simulation", "reset", "backend"},
        exclude_none=True,
    )

    return ChatResponse(
        session_id=state.session_id,
        param_delta=param_delta_dict,
        explanation=delta.explanation,
        requires_simulation=delta.requires_simulation,
        current_params=_dict_to_params_schema(state.params),
        active_backend=state.active_backend,
        simulation_result=simulation_result,
        comparison_result=comparison_result,
    )
