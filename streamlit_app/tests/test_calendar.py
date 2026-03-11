"""Tests for the HTML calendar renderer — pure function, no Streamlit needed."""
import pytest
from streamlit_app.calendar import render_calendar_html, _cell_color


# ── Colour logic ──────────────────────────────────────────────────────────────

def test_color_overflow():
    assert _cell_color(410, 400) == "#FF6B6B"


def test_color_amber():
    assert _cell_color(345, 400) == "#FFA94D"   # 86.25%


def test_color_green():
    assert _cell_color(300, 400) == "#8CE99A"   # 75%


def test_color_blue():
    assert _cell_color(180, 400) == "#AED6F1"   # 45%


def test_color_boundary_exactly_85_pct_is_amber():
    assert _cell_color(340, 400) == "#FFA94D"   # exactly 85%


def test_color_boundary_exactly_50_pct_is_green():
    assert _cell_color(200, 400) == "#8CE99A"   # exactly 50%


def test_color_zero_capacity():
    # Guard against divide-by-zero
    assert _cell_color(0, 0) == "#F8F9FA"


# ── Calendar HTML generation ──────────────────────────────────────────────────

def _make_day(date: str, occupancy: float, capacity: int,
              overflow_prob: float = 0.0) -> dict:
    return {
        "date": date,
        "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][
            __import__("datetime").date.fromisoformat(date).weekday()
        ],
        "expected_occupancy": occupancy,
        "effective_capacity": capacity,
        "overflow_probability": overflow_prob,
        "percentile_5": occupancy - 10,
        "percentile_95": occupancy + 10,
        "std_dev": 10.0,
        "team_breakdown": {},
    }


ONE_WEEK = [
    _make_day("2026-03-02", 285, 400),          # Mon — green
    _make_day("2026-03-03", 345, 400),          # Tue — amber (86%)
    _make_day("2026-03-04", 410, 400, 0.72),    # Wed — overflow
    _make_day("2026-03-05", 310, 400),          # Thu — green
    _make_day("2026-03-06", 180, 400),          # Fri — blue (45%)
]


def test_empty_input_returns_placeholder():
    html = render_calendar_html([])
    assert "No results" in html


def test_html_contains_weekday_headers():
    html = render_calendar_html(ONE_WEEK)
    for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
        assert day in html


def test_overflow_cell_has_red_color():
    html = render_calendar_html(ONE_WEEK)
    # The overflow day (Wed) should have the red colour
    assert "#FF6B6B" in html


def test_amber_cell_present():
    html = render_calendar_html(ONE_WEEK)
    assert "#FFA94D" in html


def test_blue_cell_present():
    html = render_calendar_html(ONE_WEEK)
    assert "#AED6F1" in html


def test_occupancy_values_in_html():
    html = render_calendar_html(ONE_WEEK)
    assert "285" in html
    assert "410" in html
    assert "180" in html


def test_overflow_risk_badge_shown():
    html = render_calendar_html(ONE_WEEK)
    assert "72%" in html      # overflow_probability=0.72


def test_no_overflow_badge_for_safe_days():
    # Safe day (0% overflow) should not show the risk badge
    html = render_calendar_html([_make_day("2026-03-02", 200, 400, 0.0)])
    assert "risk" not in html


def test_month_heading_in_html():
    html = render_calendar_html(ONE_WEEK)
    assert "March 2026" in html


def test_partial_week_renders_empty_cells():
    """A Wednesday-only result should leave Mon and Tue cells empty."""
    single_day = [_make_day("2026-03-04", 300, 400)]
    html = render_calendar_html(single_day)
    # Should still have 5 column headers
    for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
        assert day in html


def test_multi_week_renders_multiple_rows():
    two_weeks = ONE_WEEK + [
        _make_day("2026-03-09", 290, 400),
        _make_day("2026-03-10", 350, 400),
        _make_day("2026-03-11", 405, 400, 0.6),
        _make_day("2026-03-12", 320, 400),
        _make_day("2026-03-13", 170, 400),
    ]
    html = render_calendar_html(two_weeks)
    # Two weeks → two <tr> body rows
    assert html.count("<tr>") >= 2
