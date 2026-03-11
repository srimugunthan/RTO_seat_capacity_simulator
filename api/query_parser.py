import json
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel


class ParamDelta(BaseModel):
    """Fields that can change from a single user message. All optional — only set what changed."""
    total_employees: Optional[int] = None
    total_seats: Optional[int] = None
    wfh_days_per_week: Optional[int] = None
    seat_reduction_pct: Optional[float] = None
    mandatory_office_days: Optional[dict[str, list[str]]] = None
    day_of_week_weights: Optional[dict[str, float]] = None
    compliance_rate: Optional[float] = None
    teams: Optional[list[dict]] = None
    num_simulation_runs: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    backend: Optional[str] = None   # "binomial" or "monte_carlo"
    reset: Optional[bool] = None    # true → restore all defaults
    comparison_with: Optional[dict] = None  # second param delta for side-by-side compare
    explanation: str = ""
    requires_simulation: bool = False


_SYSTEM_PROMPT = """\
You are a parameter extraction engine for an office occupancy simulator.

Given the current simulation parameters (JSON) and the user's message, extract ONLY the parameters the user wants to change. Leave all other fields as null.

Current parameters:
{current_params}

Rules:
- Set only fields explicitly mentioned or clearly implied. Leave all others null.
- When teams are specified and their sizes don't add up to total_employees, add a "General" team for the remainder so no employees are silently dropped.
- Set requires_simulation=true if any simulation parameter changed.
- Set reset=true if the user says "reset", "start over", or "clear".
- For backend, use "binomial" or "monte_carlo" when the user wants to switch models.
- Write a short, friendly explanation of what you understood and changed.
- If the message is conversational with no parameter change, return all nulls and requires_simulation=false.

Examples of mappings:
- "reduce seats by 10%"                    → seat_reduction_pct=10
- "3 days WFH" / "work from home 3 days"  → wfh_days_per_week=3
- "500 employees"                          → total_employees=500
- "400 seats"                              → total_seats=400
- "compliance at 80%"                      → compliance_rate=0.8
- "Fridays are quiet, assume 40%"          → day_of_week_weights={{"Friday": 0.4}}
- "Engineering (80 people) in on Tue/Thu"  → mandatory_office_days={{"Engineering":["Tuesday","Thursday"]}}, teams=[{{"name":"Engineering","size":80}}]
- "500 employees total: 10 in team Alpha must be in on Tuesdays, remaining 490 are General"
  → teams=[{{"name":"Alpha","size":10}},{{"name":"General","size":490}}], mandatory_office_days={{"Alpha":["Tuesday"]}}, total_employees=500
- "Split into 2 teams: Sales (50 people) mandatory Mondays, rest (450) flexible"
  → teams=[{{"name":"Sales","size":50}},{{"name":"General","size":450}}], mandatory_office_days={{"Sales":["Monday"]}}
- "switch to binomial"                     → backend="binomial"
- "use Monte Carlo"                        → backend="monte_carlo"
- "reset everything"                       → reset=true
- "compare 2 vs 3 WFH days"               → wfh_days_per_week=2, comparison_with={{"wfh_days_per_week": 3}}, requires_simulation=true
- "compare binomial vs monte carlo"        → backend="binomial", comparison_with={{"backend": "monte_carlo"}}, requires_simulation=true
"""


def parse_query(
    message: str,
    current_params: dict,
    llm: BaseChatModel,
    message_history: list[HumanMessage | AIMessage] | None = None,
) -> ParamDelta:
    """
    Parse a natural language message and return the parameter delta.

    Args:
        message: The user's latest message.
        current_params: Current simulation parameters as a plain dict.
        llm: A LangChain BaseChatModel (provider-agnostic).
        message_history: Optional recent conversation turns for follow-up context.

    Returns:
        ParamDelta with only the changed fields set.
    """
    structured_llm = llm.with_structured_output(ParamDelta)

    system = SystemMessage(
        content=_SYSTEM_PROMPT.format(
            current_params=json.dumps(current_params, indent=2)
        )
    )

    messages: list = [system]
    if message_history:
        # Include recent turns so the LLM understands follow-up context
        messages.extend(message_history[-10:])  # last 5 turns
    messages.append(HumanMessage(content=message))

    return structured_llm.invoke(messages)
