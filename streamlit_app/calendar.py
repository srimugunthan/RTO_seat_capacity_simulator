"""
Renders simulation daily results as a color-coded HTML calendar table.

Color coding:
  Red    — overflow (expected_occupancy > effective_capacity)
  Amber  — >85% utilization (near-overflow warning)
  Green  — normal range (50%–85%)
  Blue   — <50% utilization (underutilized)
"""
from datetime import date, timedelta

_WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

# Colours
_COLOR_OVERFLOW = "#FF6B6B"
_COLOR_AMBER = "#FFA94D"
_COLOR_GREEN = "#8CE99A"
_COLOR_BLUE = "#AED6F1"
_COLOR_EMPTY = "#F8F9FA"

_HEADER_BG = "#E9ECEF"
_BORDER = "#DEE2E6"


def _cell_color(expected_occupancy: float, effective_capacity: int) -> str:
    if effective_capacity <= 0:
        return _COLOR_EMPTY
    utilization = expected_occupancy / effective_capacity
    if expected_occupancy > effective_capacity:
        return _COLOR_OVERFLOW
    if utilization >= 0.85:
        return _COLOR_AMBER
    if utilization < 0.50:
        return _COLOR_BLUE
    return _COLOR_GREEN


def _cell_html(result: dict | None) -> str:
    border = f"1px solid {_BORDER}"
    base_style = f"border: {border}; padding: 8px 6px; text-align: center; vertical-align: top; min-width: 88px;"

    if result is None:
        return f'<td style="{base_style} background-color: {_COLOR_EMPTY};"></td>'

    occ = result["expected_occupancy"]
    cap = result["effective_capacity"]
    util = occ / cap if cap > 0 else 0
    color = _cell_color(occ, cap)
    day_num = result["date"].split("-")[2].lstrip("0") or "0"
    overflow_p = result["overflow_probability"]

    overflow_badge = ""
    if overflow_p >= 0.01:
        badge_color = "#842029" if overflow_p >= 0.5 else "#6c4a00"
        overflow_badge = (
            f'<div style="font-size:0.65em; color:{badge_color}; margin-top:2px;">'
            f"⚠ {overflow_p:.0%} risk</div>"
        )

    return (
        f'<td style="{base_style} background-color: {color};">'
        f'<div style="font-size:0.7em; color:#6C757D; text-align:right;">{day_num}</div>'
        f'<div style="font-size:1.15em; font-weight:600; margin:2px 0;">{occ:.0f}</div>'
        f'<div style="font-size:0.78em; color:#495057;">{util:.0%}</div>'
        f"{overflow_badge}"
        f"</td>"
    )


def render_calendar_html(daily_results: list[dict]) -> str:
    """
    Render a list of DayResult dicts as an HTML calendar table.

    Groups results by week (Mon–Fri). Days outside the simulation range
    are rendered as empty grey cells.
    """
    if not daily_results:
        return "<p style='color:#888;'>No results to display.</p>"

    date_map: dict[str, dict] = {r["date"]: r for r in daily_results}
    dates = sorted(date.fromisoformat(r["date"]) for r in daily_results)

    # Expand to full Mon–Fri weeks
    first_monday = dates[0] - timedelta(days=dates[0].weekday())
    last_date = dates[-1]
    last_friday = last_date + timedelta(days=(4 - last_date.weekday()) % 7)

    # Month/year heading from the majority of dates
    month_year = dates[len(dates) // 2].strftime("%B %Y")

    # Table header row
    header_cells = "".join(
        f'<th style="padding:8px 6px; background:{_HEADER_BG}; border:1px solid {_BORDER};'
        f' text-align:center; min-width:88px; font-weight:600;">{d}</th>'
        for d in _WEEKDAY_LABELS
    )

    # Build week rows
    row_htmls: list[str] = []
    current = first_monday
    while current <= last_friday:
        cells = "".join(
            _cell_html(date_map.get((current + timedelta(days=i)).isoformat()))
            for i in range(5)
        )
        row_htmls.append(f"<tr>{cells}</tr>")
        current += timedelta(weeks=1)

    rows_html = "\n".join(row_htmls)

    return (
        f'<h4 style="margin:0 0 8px 0; font-family:sans-serif;">{month_year}</h4>'
        f'<table style="border-collapse:collapse; width:100%; font-family:sans-serif;">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>"
    )
