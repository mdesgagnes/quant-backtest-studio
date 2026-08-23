"""Brand palette -- the single place colour is defined.

Everything visual reads from here: the interface CSS, the Plotly charts, and
the exported tearsheet. Change a value in BRAND and the whole application
follows, including the PDF-ready report. Nothing else in the codebase should
contain a hex code.

National Bank refreshed its identity in September 2025 ("Build something.",
with LG2) to a red, black and sand palette. The exact values are not
published, so RED below is the documented figure from the previous
guidelines and BLACK and SAND are close readings of the new campaign work.
If you have the current brand guidelines, replace these three values and
the entire application restyles itself -- no other file needs touching.
"""
from __future__ import annotations

from typing import Dict

# ----------------------------------------------------------------------
# The three brand colours.
# ----------------------------------------------------------------------
RED = "#E41C23"        # signature red
BLACK = "#111111"      # near-black; true #000 is harsh on screen
SAND = "#E7DDCB"       # warm neutral ground

# Derived tones. These are tints and shades of the three above, not new
# colours, so correcting the three keeps the whole system coherent.
BRAND: Dict[str, str] = {
    "red": RED,
    "red_dark": "#B01419",
    "red_bright": "#FF3A41",     # lifts off a dark ground
    "red_wash": "#2A1416",       # tinted panel, not a pale wash
    "black": BLACK,
    "sand": SAND,
    "sand_light": "#F3ECDF",
    "sand_dark": "#CBBFA8",

    # Dark ground. Black is a brand colour, so the interface sits on it
    # rather than inverting into a generic grey. Neutrals are warmed toward
    # the sand so they belong to the same family.
    "bg": "#0F0F0E",
    "panel": "#181816",
    "panel_2": "#1E1E1B",
    "rule": "#2E2C27",
    "rule_soft": "#232320",
    "text": "#EFE9DE",           # sand-tinted, easier than pure white
    "muted": "#9A9389",
    "faint": "#6E6960",

    # Sign colours, tuned for a dark ground. Deliberately not the brand
    # red: if "loss" and "brand" share a colour the chart cannot be read,
    # and a brand-coloured "gain" flatters every result.
    "gain": "#4FB79A",
    "loss": "#E8705C",
}

# Light values for the exported tearsheet. A report is printed and shared,
# and a dark page wastes ink and reads badly on paper.
PRINT: Dict[str, str] = {
    "bg": "#FFFFFF",
    "panel": "#FAF8F4",
    "rule": "#DED8CC",
    "rule_soft": "#EDE8DF",
    "text": "#1A1A1A",
    "muted": "#6B655C",
    "sand": "#F5F0E7",
    "red": RED,
    "gain": "#1F6F5C",
    "loss": "#B3322B",
}

# Chart series. Red leads because the first series is the subject; the rest
# are context and stay in the sand and neutral family so red keeps meaning.
SERIES = [RED, "#C9B896", "#4FB79A", "#8C93A8",
          "#D08C4E", "#7E6BA8", "#5B9EC7"]

SERIES_PRINT = [RED, "#3F3A34", "#1F6F5C", "#C9B896",
                "#8C6D4F", "#5B7A8C", "#A8574E"]


def css_variables() -> str:
    """The palette as CSS custom properties."""
    pairs = [
        ("--nb-red", BRAND["red"]),
        ("--nb-red-dark", BRAND["red_dark"]),
        ("--nb-red-wash", BRAND["red_wash"]),
        ("--nb-black", BRAND["black"]),
        ("--nb-sand", BRAND["sand"]),
        ("--nb-sand-light", BRAND["sand_light"]),
        ("--nb-sand-dark", BRAND["sand_dark"]),
        ("--nb-red-bright", BRAND["red_bright"]),
        ("--panel-2", BRAND["panel_2"]),
        ("--bg", BRAND["bg"]),
        ("--panel", BRAND["panel"]),
        ("--rule", BRAND["rule"]),
        ("--rule-soft", BRAND["rule_soft"]),
        ("--ink", BRAND["text"]),
        ("--muted", BRAND["muted"]),
        ("--faint", BRAND["faint"]),
        ("--gain", BRAND["gain"]),
        ("--loss", BRAND["loss"]),
    ]
    return "\n  ".join(f"{k}:{v};" for k, v in pairs)


def config_toml() -> str:
    """Streamlit's own theme block, so its widgets match the CSS."""
    return (f'[theme]\nbase = "dark"\n'
            f'primaryColor = "{BRAND["red"]}"\n'
            f'backgroundColor = "{BRAND["bg"]}"\n'
            f'secondaryBackgroundColor = "{BRAND["panel"]}"\n'
            f'textColor = "{BRAND["text"]}"\n'
            f'font = "sans serif"\n')
