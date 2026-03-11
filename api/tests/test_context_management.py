"""Tests for ConversationManager — no LLM calls, pure state logic."""
import pytest
from api.conversation import ConversationManager, DEFAULT_PARAMS
from api.query_parser import ParamDelta


@pytest.fixture
def manager():
    return ConversationManager()


def test_get_or_create_makes_new_session(manager):
    state = manager.get_or_create(None)
    assert state.session_id is not None
    assert state.params == DEFAULT_PARAMS


def test_get_or_create_returns_same_session(manager):
    state1 = manager.get_or_create(None)
    state2 = manager.get_or_create(state1.session_id)
    assert state1 is state2


def test_apply_delta_scalar_field(manager):
    state = manager.get_or_create(None)
    delta = ParamDelta(total_employees=300, requires_simulation=True, explanation="300 employees")
    manager.apply_delta(state, delta)
    assert state.params["total_employees"] == 300


def test_apply_delta_multiple_fields(manager):
    state = manager.get_or_create(None)
    delta = ParamDelta(
        total_employees=300,
        total_seats=250,
        wfh_days_per_week=3,
        requires_simulation=True,
        explanation="updated",
    )
    manager.apply_delta(state, delta)
    assert state.params["total_employees"] == 300
    assert state.params["total_seats"] == 250
    assert state.params["wfh_days_per_week"] == 3


def test_apply_delta_dict_field_merges(manager):
    """Updating day_of_week_weights should merge, not replace."""
    state = manager.get_or_create(None)
    original_monday = state.params["day_of_week_weights"]["Monday"]
    delta = ParamDelta(
        day_of_week_weights={"Friday": 0.4},
        requires_simulation=True,
        explanation="quiet Fridays",
    )
    manager.apply_delta(state, delta)
    assert state.params["day_of_week_weights"]["Friday"] == 0.4
    assert state.params["day_of_week_weights"]["Monday"] == original_monday  # untouched


def test_apply_delta_mandatory_office_days_merges(manager):
    state = manager.get_or_create(None)
    delta = ParamDelta(
        mandatory_office_days={"Engineering": ["Tuesday", "Thursday"]},
        requires_simulation=True,
        explanation="engineering mandates",
    )
    manager.apply_delta(state, delta)
    assert state.params["mandatory_office_days"]["Engineering"] == ["Tuesday", "Thursday"]

    # Apply a second team — should merge
    delta2 = ParamDelta(
        mandatory_office_days={"Sales": ["Wednesday"]},
        requires_simulation=True,
        explanation="sales mandates",
    )
    manager.apply_delta(state, delta2)
    assert "Engineering" in state.params["mandatory_office_days"]
    assert "Sales" in state.params["mandatory_office_days"]


def test_apply_delta_backend_switch(manager):
    state = manager.get_or_create(None)
    assert state.active_backend == "monte_carlo"
    delta = ParamDelta(backend="binomial", explanation="switching model", requires_simulation=False)
    manager.apply_delta(state, delta)
    assert state.active_backend == "binomial"


def test_apply_delta_reset_restores_defaults(manager):
    state = manager.get_or_create(None)
    # Modify params first
    delta = ParamDelta(total_employees=999, requires_simulation=True, explanation="")
    manager.apply_delta(state, delta)
    assert state.params["total_employees"] == 999

    # Now reset
    reset_delta = ParamDelta(reset=True, explanation="resetting", requires_simulation=False)
    manager.apply_delta(state, reset_delta)
    assert state.params["total_employees"] == DEFAULT_PARAMS["total_employees"]
    assert state.active_backend == "monte_carlo"


def test_delta_accumulation_across_turns(manager):
    """Each turn should build on the previous params (delta mode)."""
    state = manager.get_or_create(None)

    manager.apply_delta(state, ParamDelta(total_employees=500, total_seats=400,
                                          requires_simulation=True, explanation="setup"))
    manager.apply_delta(state, ParamDelta(wfh_days_per_week=3,
                                          requires_simulation=True, explanation="3 WFH"))
    manager.apply_delta(state, ParamDelta(compliance_rate=0.8,
                                          requires_simulation=True, explanation="80% compliance"))

    assert state.params["total_employees"] == 500
    assert state.params["total_seats"] == 400
    assert state.params["wfh_days_per_week"] == 3
    assert state.params["compliance_rate"] == 0.8


def test_add_turn_stores_messages(manager):
    state = manager.get_or_create(None)
    manager.add_turn(state, "500 employees", "Got it, 500 employees set.")
    assert len(state.messages) == 2


def test_add_turn_caps_at_20_messages(manager):
    state = manager.get_or_create(None)
    for i in range(15):
        manager.add_turn(state, f"message {i}", f"response {i}")
    assert len(state.messages) == 20  # capped at 20 (10 turns)
