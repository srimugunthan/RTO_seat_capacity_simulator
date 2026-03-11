import os
import httpx

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT = 30.0


class APIError(Exception):
    pass


def get_backends() -> dict:
    """
    GET /backends → {"available": [...], "active": str}
    """
    try:
        response = httpx.get(f"{API_BASE_URL}/backends", timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise APIError(f"Failed to fetch backends: {e}")


def simulate(params: dict, backend: str = "monte_carlo") -> dict:
    """
    POST /simulate → simulation result dict.
    params is the SimulationParamsSchema dict (camelCase keys not needed — FastAPI uses snake_case).
    """
    try:
        response = httpx.post(
            f"{API_BASE_URL}/simulate",
            json={"params": params, "backend": backend},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e))
        raise APIError(f"Simulation failed: {detail}")
    except httpx.HTTPError as e:
        raise APIError(f"Simulation failed: {e}")


def compare(
    params_a: dict,
    backend_a: str,
    params_b: dict,
    backend_b: str,
    label_a: str = "Scenario A",
    label_b: str = "Scenario B",
) -> dict:
    """
    POST /compare → {"scenario_a": {...}, "scenario_b": {...}}
    """
    try:
        payload = {
            "scenario_a": {"params": params_a, "backend": backend_a},
            "scenario_b": {"params": params_b, "backend": backend_b},
            "label_a": label_a,
            "label_b": label_b,
        }
        response = httpx.post(
            f"{API_BASE_URL}/compare",
            json=payload,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e))
        raise APIError(f"Comparison failed: {detail}")
    except httpx.HTTPError as e:
        raise APIError(f"Comparison failed: {e}")


def chat(message: str, session_id: str | None = None) -> dict:
    """
    POST /chat → {"session_id", "param_delta", "explanation",
                  "requires_simulation", "current_params",
                  "active_backend", "simulation_result"}
    """
    try:
        payload: dict = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        response = httpx.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e))
        raise APIError(f"Chat failed: {detail}")
    except httpx.HTTPError as e:
        raise APIError(f"Chat failed: {e}")
