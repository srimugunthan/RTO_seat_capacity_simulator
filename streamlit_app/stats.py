import streamlit as st


def render_summary_stats(summary: dict, model_name: str) -> None:
    """Render four st.metric tiles summarising a SimulationResult summary dict."""
    st.caption(f"Model: **{model_name}**")

    col1, col2, col3, col4 = st.columns(4)

    avg_util = summary.get("avg_utilization", 0)
    peak_occ = summary.get("peak_occupancy", 0)
    overflow_count = summary.get("overflow_days_count", 0)
    overflow_pct = summary.get("overflow_days_pct", 0)
    avg_excess = summary.get("avg_overflow_magnitude", 0)

    with col1:
        st.metric(
            label="Avg Utilization",
            value=f"{avg_util:.1%}",
            help="Mean daily occupancy as a fraction of seat capacity.",
        )

    with col2:
        st.metric(
            label="Peak Occupancy",
            value=f"{peak_occ:.0f}",
            help="Highest single-day expected occupancy across the period.",
        )

    with col3:
        st.metric(
            label="Overflow Days",
            value=str(overflow_count),
            delta=f"{overflow_pct:.1%} of days" if overflow_count > 0 else "None",
            delta_color="inverse",
            help="Days where expected demand exceeds seat capacity.",
        )

    with col4:
        st.metric(
            label="Avg Overflow",
            value=f"{avg_excess:.0f} seats" if avg_excess > 0 else "—",
            help="On overflow days, how many seats short on average.",
        )
