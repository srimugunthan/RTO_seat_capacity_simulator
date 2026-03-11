import copy
import uuid
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage

from .query_parser import ParamDelta

DEFAULT_PARAMS: dict = {
    "total_employees": 500,
    "total_seats": 400,
    "wfh_days_per_week": 2,
    "seat_reduction_pct": 0.0,
    "mandatory_office_days": {},
    "day_of_week_weights": {
        "Monday": 0.85,
        "Tuesday": 1.00,
        "Wednesday": 1.00,
        "Thursday": 0.95,
        "Friday": 0.65,
    },
    "compliance_rate": 0.9,
    "teams": [],
    "num_simulation_runs": 5000,
    "start_date": "2026-03-01",
    "end_date": "2026-03-31",
}


@dataclass
class ConversationState:
    session_id: str
    params: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_PARAMS))
    active_backend: str = "monte_carlo"
    messages: list = field(default_factory=list)  # HumanMessage / AIMessage history


class ConversationManager:
    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}

    def get_or_create(self, session_id: str | None) -> ConversationState:
        """Return existing session or create a new one. Returns the state."""
        if session_id is None or session_id not in self._sessions:
            sid = session_id or str(uuid.uuid4())
            self._sessions[sid] = ConversationState(session_id=sid)
        return self._sessions[session_id or list(self._sessions)[-1]]

    def apply_delta(self, state: ConversationState, delta: ParamDelta) -> None:
        """
        Merge delta into the session's current params.

        - Scalar fields: replace.
        - Dict fields (mandatory_office_days, day_of_week_weights): merge keys.
        - List fields (teams): replace entirely.
        - backend: update active_backend.
        - reset: restore all defaults.
        """
        if delta.reset:
            self.reset(state)
            return

        if delta.backend:
            state.active_backend = delta.backend

        scalar_and_list_fields = {
            "total_employees", "total_seats", "wfh_days_per_week",
            "seat_reduction_pct", "compliance_rate", "teams",
            "num_simulation_runs", "start_date", "end_date",
        }
        dict_fields = {"mandatory_office_days", "day_of_week_weights"}

        for field_name in scalar_and_list_fields:
            value = getattr(delta, field_name)
            if value is not None:
                state.params[field_name] = value

        for field_name in dict_fields:
            value = getattr(delta, field_name)
            if value is not None:
                state.params[field_name].update(value)

    def reset(self, state: ConversationState) -> None:
        """Restore params to defaults; preserve session_id and message history."""
        state.params = copy.deepcopy(DEFAULT_PARAMS)
        state.active_backend = "monte_carlo"

    def add_turn(
        self, state: ConversationState, user_message: str, ai_explanation: str
    ) -> None:
        """Append a conversation turn and cap history at 10 turns (20 messages)."""
        state.messages.append(HumanMessage(content=user_message))
        state.messages.append(AIMessage(content=ai_explanation))
        if len(state.messages) > 20:
            state.messages = state.messages[-20:]


# Module-level singleton — shared across all requests within one server process
conversation_manager = ConversationManager()
