from .models import SimulationBackend
from .binomial_backend import BinomialBackend
from .monte_carlo_backend import MonteCarloBackend
from .simpy_backend import SimpyBackend

_registry: dict[str, SimulationBackend] = {}


def register_backend(backend_id: str, backend: SimulationBackend) -> None:
    _registry[backend_id] = backend


def get_backend(backend_id: str) -> SimulationBackend:
    if backend_id not in _registry:
        raise KeyError(f"Unknown backend '{backend_id}'. Available: {list(_registry.keys())}")
    return _registry[backend_id]


def list_backends() -> list[dict]:
    return [
        {"id": bid, "name": b.name(), "description": b.description()}
        for bid, b in _registry.items()
    ]


# Pre-register built-in backends
register_backend("binomial", BinomialBackend())
register_backend("monte_carlo", MonteCarloBackend())
register_backend("simpy_des", SimpyBackend())
