"""
Tests for the query parser.

All LLM calls are mocked — no API key required.
The mock returns a predefined ParamDelta directly from structured_llm.invoke().
"""
import pytest
from unittest.mock import MagicMock, patch
from api.query_parser import parse_query, ParamDelta


def make_mock_llm(returned_delta: ParamDelta) -> MagicMock:
    """Return a mock LLM whose with_structured_output().invoke() returns the given delta."""
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = returned_delta
    llm = MagicMock()
    llm.with_structured_output.return_value = structured_llm
    return llm


CURRENT_PARAMS = {
    "total_employees": 500,
    "total_seats": 400,
    "wfh_days_per_week": 2,
    "seat_reduction_pct": 0.0,
    "compliance_rate": 0.9,
}


# Table-driven tests matching PRD Section 8.1 examples
@pytest.mark.parametrize("message, expected_delta", [
    (
        "We have 500 employees and 400 seats",
        ParamDelta(total_employees=500, total_seats=400,
                   requires_simulation=True, explanation="Set 500 employees and 400 seats."),
    ),
    (
        "What if seats are reduced by 10%?",
        ParamDelta(seat_reduction_pct=10.0,
                   requires_simulation=True, explanation="Reducing seats by 10%."),
    ),
    (
        "Employees work from home 2 days a week",
        ParamDelta(wfh_days_per_week=2,
                   requires_simulation=True, explanation="Set WFH to 2 days per week."),
    ),
    (
        "Make it 3 days WFH instead",
        ParamDelta(wfh_days_per_week=3,
                   requires_simulation=True, explanation="Changed WFH to 3 days."),
    ),
    (
        "What if compliance drops to 70%?",
        ParamDelta(compliance_rate=0.7,
                   requires_simulation=True, explanation="Compliance set to 70%."),
    ),
    (
        "Fridays are dead — assume 40% attendance",
        ParamDelta(day_of_week_weights={"Friday": 0.4},
                   requires_simulation=True, explanation="Friday attendance set to 40%."),
    ),
    (
        "Switch to Monte Carlo model",
        ParamDelta(backend="monte_carlo",
                   requires_simulation=False, explanation="Switched to Monte Carlo."),
    ),
    (
        "Reset everything",
        ParamDelta(reset=True,
                   requires_simulation=False, explanation="Resetting to defaults."),
    ),
    (
        "Thanks, looks good!",
        ParamDelta(requires_simulation=False, explanation="No changes needed."),
    ),
])
def test_parse_query_table(message, expected_delta):
    llm = make_mock_llm(expected_delta)
    result = parse_query(message, CURRENT_PARAMS, llm)

    # Verify the mock LLM was called with structured output
    llm.with_structured_output.assert_called_once_with(ParamDelta)

    # Verify the result is the delta we mocked
    assert result.requires_simulation == expected_delta.requires_simulation
    assert result.explanation == expected_delta.explanation


def test_parse_query_passes_current_params_in_system_prompt():
    """System prompt should contain the current params so the LLM has context."""
    delta = ParamDelta(wfh_days_per_week=3, requires_simulation=True, explanation="3 days WFH")
    llm = make_mock_llm(delta)

    parse_query("3 days WFH", CURRENT_PARAMS, llm)

    # Verify that invoke was called — get the messages passed
    structured_llm = llm.with_structured_output.return_value
    call_args = structured_llm.invoke.call_args[0][0]  # first positional arg = messages list

    # First message should be SystemMessage containing current params
    from langchain_core.messages import SystemMessage
    assert isinstance(call_args[0], SystemMessage)
    assert "total_employees" in call_args[0].content


def test_parse_query_includes_message_history():
    """Recent message history should be included in the prompt for follow-up context."""
    from langchain_core.messages import HumanMessage, AIMessage
    delta = ParamDelta(compliance_rate=0.8, requires_simulation=True, explanation="80% compliance")
    llm = make_mock_llm(delta)

    history = [
        HumanMessage(content="500 employees"),
        AIMessage(content="Set 500 employees."),
    ]
    parse_query("compliance at 80%", CURRENT_PARAMS, llm, message_history=history)

    structured_llm = llm.with_structured_output.return_value
    messages = structured_llm.invoke.call_args[0][0]
    # System + 2 history + 1 user = 4 messages
    assert len(messages) == 4


def test_parse_query_without_history():
    """No history → only system + user message."""
    delta = ParamDelta(total_seats=300, requires_simulation=True, explanation="300 seats")
    llm = make_mock_llm(delta)

    parse_query("300 seats", CURRENT_PARAMS, llm)

    structured_llm = llm.with_structured_output.return_value
    messages = structured_llm.invoke.call_args[0][0]
    assert len(messages) == 2  # system + user
