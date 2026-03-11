import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so the streamlit_app package is
# importable whether this file is launched via `streamlit run` or pytest.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from streamlit_app.api_client import APIError, chat as chat_api, compare as compare_api, get_backends, simulate
from streamlit_app.calendar import render_calendar_html
from streamlit_app.stats import render_summary_stats

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RTO Capacity Simulator",
    page_icon="🏢",
    layout="wide",
)


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults: dict = {
        "session_id": None,
        "messages": [],
        "simulation_result": None,
        "comparison_result": None,
        "current_params": None,
        "active_backend": "monte_carlo",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# ── Load backends (cached) ────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def _load_backends() -> dict:
    try:
        return get_backends()
    except APIError:
        # Fallback so the UI still renders if API is not yet running
        return {
            "available": [
                {"id": "binomial", "name": "Binomial/Poisson Analytical",
                 "description": "Closed-form probability model. Instant results."},
                {"id": "monte_carlo", "name": "Monte Carlo Simulation",
                 "description": "Stochastic sampling with team correlation. 5000 runs."},
            ],
            "active": "monte_carlo",
        }


backends_data = _load_backends()
_backend_ids = [b["id"] for b in backends_data["available"]]
_backend_names = {b["id"]: b["name"] for b in backends_data["available"]}
_backend_descs = {b["id"]: b["description"] for b in backends_data["available"]}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    current_idx = (
        _backend_ids.index(st.session_state.active_backend)
        if st.session_state.active_backend in _backend_ids
        else 0
    )

    selected_backend = st.selectbox(
        "Simulation Model",
        options=_backend_ids,
        format_func=lambda x: _backend_names.get(x, x),
        index=current_idx,
    )
    st.caption(_backend_descs.get(selected_backend, ""))

    # Re-simulate immediately when the backend changes and we already have params
    if (
        selected_backend != st.session_state.active_backend
        and st.session_state.current_params is not None
    ):
        with st.spinner("Re-running with new model…"):
            try:
                result = simulate(st.session_state.current_params, selected_backend)
                st.session_state.simulation_result = result
                st.session_state.active_backend = selected_backend
            except APIError as e:
                st.error(str(e))
    elif selected_backend != st.session_state.active_backend:
        st.session_state.active_backend = selected_backend

    st.divider()

    st.text_input(
        "LLM Provider",
        value=os.getenv("LLM_PROVIDER", "openai"),
        disabled=True,
        help="Set LLM_PROVIDER in .env to change.",
    )

    st.divider()

    if st.button("🔄 Reset Conversation", use_container_width=True):
        for key in ["session_id", "messages", "simulation_result", "comparison_result", "current_params"]:
            st.session_state[key] = [] if key == "messages" else None
        st.session_state.active_backend = "monte_carlo"
        st.rerun()


# ── Page header ───────────────────────────────────────────────────────────────
st.title("🏢 RTO Capacity Simulator")
st.caption("Describe your office scenario in the chat to simulate seat occupancy.")

# ── Two-column layout ─────────────────────────────────────────────────────────
col_chat, col_results = st.columns([1, 2], gap="large")


# ── Chat panel ────────────────────────────────────────────────────────────────
with col_chat:
    # Replay message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("param_delta"):
                with st.expander("📋 Parameters updated", expanded=False):
                    st.json(msg["param_delta"])

    # Welcome message on first open
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.write(
                "👋 Welcome! Describe your office setup to get started.\n\n"
                "**Example:** *\"We have 500 employees, 400 seats, and employees "
                "work from home 2 days a week.\"*"
            )

    # Chat input
    if prompt := st.chat_input("Describe your scenario…"):
        # Show user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Call /chat endpoint
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    response = chat_api(prompt, st.session_state.session_id)

                    st.session_state.session_id = response["session_id"]
                    st.session_state.current_params = response["current_params"]
                    st.session_state.active_backend = response["active_backend"]

                    if response.get("comparison_result"):
                        st.session_state.comparison_result = response["comparison_result"]
                        st.session_state.simulation_result = None
                    elif response.get("simulation_result"):
                        st.session_state.simulation_result = response["simulation_result"]
                        st.session_state.comparison_result = None

                    explanation = response["explanation"]
                    param_delta = response.get("param_delta") or {}

                    st.write(explanation)

                    if param_delta:
                        with st.expander("📋 Parameters updated", expanded=False):
                            st.json(param_delta)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": explanation,
                        "param_delta": param_delta or None,
                    })

                except APIError as e:
                    msg = str(e)
                    st.error(f"⚠️ {msg}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Sorry, I ran into an error: {msg}",
                    })

        st.rerun()


def _render_scenario(result: dict, label: str) -> None:
    """Render a single scenario's calendar + stats (used in both normal and compare mode)."""
    daily_results = result["daily_results"]
    render_summary_stats(result["summary"], result["model_name"])
    st.divider()
    months = sorted({r["date"][:7] for r in daily_results})
    if len(months) > 1:
        selected_month = st.selectbox(
            "Month",
            months,
            key=f"month_{label}",
            format_func=lambda m: (
                __import__("datetime").date.fromisoformat(f"{m}-01").strftime("%B %Y")
            ),
        )
        month_results = [r for r in daily_results if r["date"].startswith(selected_month)]
    else:
        month_results = daily_results
    st.markdown(render_calendar_html(month_results), unsafe_allow_html=True)


_LEGEND_HTML = """
<div style="display:flex; gap:16px; margin-top:10px;
            font-size:0.82em; flex-wrap:wrap; font-family:sans-serif;">
  <span><span style="background:#FF6B6B;padding:1px 8px;
    border-radius:3px;">&nbsp;</span> Overflow</span>
  <span><span style="background:#FFA94D;padding:1px 8px;
    border-radius:3px;">&nbsp;</span> &gt;85% utilization</span>
  <span><span style="background:#8CE99A;padding:1px 8px;
    border-radius:3px;">&nbsp;</span> Normal</span>
  <span><span style="background:#AED6F1;padding:1px 8px;
    border-radius:3px;">&nbsp;</span> &lt;50% utilization</span>
</div>
"""


# ── Results panel ─────────────────────────────────────────────────────────────
with col_results:
    if st.session_state.comparison_result:
        # ── Comparison mode ───────────────────────────────────────────────
        cr = st.session_state.comparison_result
        sc_a = cr["scenario_a"]
        sc_b = cr["scenario_b"]
        st.subheader("🔀 Scenario Comparison")
        cmp_a, cmp_b = st.columns(2, gap="medium")
        with cmp_a:
            st.markdown(f"**{sc_a['label']}**")
            _render_scenario(sc_a["simulation_result"], "a")
        with cmp_b:
            st.markdown(f"**{sc_b['label']}**")
            _render_scenario(sc_b["simulation_result"], "b")
        st.markdown(_LEGEND_HTML, unsafe_allow_html=True)

    elif st.session_state.simulation_result:
        result = st.session_state.simulation_result
        _render_scenario(result, "main")
        st.markdown(_LEGEND_HTML, unsafe_allow_html=True)

        # Highest risk days table
        daily_results = result["daily_results"]
        months = sorted({r["date"][:7] for r in daily_results})
        if len(months) > 1:
            # reuse the month already selected by _render_scenario widget
            sel = st.session_state.get("month_main")
            month_results = [r for r in daily_results if sel and r["date"].startswith(sel)] or daily_results
        else:
            month_results = daily_results

        risk_days = sorted(
            [r for r in month_results if r["overflow_probability"] > 0.01],
            key=lambda r: r["overflow_probability"],
            reverse=True,
        )[:5]

        if risk_days:
            st.subheader("⚠️ Highest Risk Days")
            df = pd.DataFrame([
                {
                    "Date": r["date"],
                    "Day": r["day_of_week"],
                    "Expected": f"{r['expected_occupancy']:.0f}",
                    "Capacity": r["effective_capacity"],
                    "Overflow Risk": f"{r['overflow_probability']:.1%}",
                    "95th Pct": f"{r['percentile_95']:.0f}",
                }
                for r in risk_days
            ])
            st.dataframe(df, hide_index=True, use_container_width=True)

    else:
        st.info(
            "📊 Your simulation results will appear here.\n\n"
            "Start by describing your scenario in the chat panel on the left.\n\n"
            "**Try:** *\"500 employees, 400 seats, 2 days WFH\"*"
        )
