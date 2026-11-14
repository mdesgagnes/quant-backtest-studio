"""Tables that sort by value and look the same (qbt/sortable.py)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbt import sortable as S                                # noqa: E402
from pandas.io.formats.style import Styler                   # noqa: E402


def _shown(styler):
    """What each cell shows, as Streamlit reads it from a Styler."""
    body = styler._translate(False, False)["body"]
    return [[c["display_value"] for c in row if c.get("type") == "td"] for row in body]


SIDE = pd.DataFrame({
    "Series": ["SPY", "QQQ", "TLT", "GLD"],
    "From": ["1993-02-01", "1999-03-11", "2002-07-31", "2004-11-19"],
    "CAGR": ["10.59%", "10.35%", "3.36%", "9.75%"],
    "Sharpe": ["0.44", "0.31", "0.12", "n/a"],
    "Max Drawdown": ["-55.19%", "−82.96%", "-48.35%", "-45.56%"],
})


def test_to_number():
    assert S.to_number("10.59%") == pytest.approx(10.59)
    assert S.to_number("-55.19%") == pytest.approx(-55.19)
    assert S.to_number("−8.20%") == pytest.approx(-8.2)
    assert S.to_number("$1,234.50") == pytest.approx(1234.5)
    assert S.to_number("+0.5pp") == pytest.approx(0.5)
    assert S.to_number("12 bps") == pytest.approx(12)
    assert S.to_number("1.8x") == pytest.approx(1.8)
    assert np.isnan(S.to_number("n/a")) and np.isnan(S.to_number(None))
    with pytest.raises(ValueError):
        S.to_number("XIC.TO")


def test_side_by_side_sorts_by_value_and_shows_the_same_text():
    out = S.sortable(SIDE, index_shown=False)
    assert isinstance(out, Styler)
    assert list(out.data.sort_values("CAGR")["Series"]) == ["TLT", "GLD", "QQQ", "SPY"]
    assert out.data["Max Drawdown"].tolist() == pytest.approx([-55.19, -82.96, -48.35, -45.56])
    assert _shown(out) == SIDE.astype(str).values.tolist()          # every cell as before
    assert out.data["Series"].tolist() == SIDE["Series"].tolist()   # text left as text
    assert not pd.api.types.is_float_dtype(out.data["From"])          # dates are not figures


def test_plain_table_with_a_real_number_column_is_left_alone():
    df = SIDE.assign(Raw=[0.44, 1.5, 2.0, 3.0])      # a Styler would show 0.440000
    assert S.sortable(df, index_shown=False) is df


def test_shown_index_that_is_not_text_is_left_alone():
    df = SIDE.set_index(pd.to_datetime(SIDE["From"]))
    assert S.sortable(df, index_shown=True) is df
    assert isinstance(S.sortable(df, index_shown=False), Styler)


def test_columns_that_are_not_all_figures_stay_text():
    df = pd.DataFrame({"Metric": ["CAGR", "Sharpe"], "Value": ["10.59%", "Yes"]})
    assert S.sortable(df, False) is df
    one_number_two_spellings = pd.DataFrame({"A": ["x", "y"], "V": ["10.0%", "10.00%"]})
    assert S.sortable(one_number_two_spellings, False) is one_number_two_spellings


def test_styler_keeps_its_colours_and_text():
    def colour(v):
        return "color: red" if str(v).startswith("-") else "color: green"
    sty = SIDE.style.map(colour, subset=["CAGR", "Max Drawdown"]).set_properties(
        subset=["Series"], **{"font-weight": "600"})
    before_css = {k: list(v) for k, v in sty._compute().ctx.items()}
    before_text = _shown(sty)
    out = S.sortable(sty, index_shown=False)
    assert out is not sty and out.data["CAGR"].dtype == float
    after_css = {k: list(v) for k, v in out._compute().ctx.items()}
    assert after_css == before_css
    assert _shown(out) == before_text


def test_styler_keeps_formats_of_columns_it_does_not_touch():
    df = pd.DataFrame({"Name": ["a", "b"], "Ret": ["+1.20%", "-3.40%"], "Weight": [0.25, 0.75]})
    sty = df.style.format({"Weight": "{:.0%}"})
    out = S.sortable(sty, False)
    assert _shown(out) == [["a", "+1.20%", "25%"], ["b", "-3.40%", "75%"]]


def test_nothing_to_do_returns_the_input():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    assert S.sortable(df) is df
    assert S.sortable("not a table") == "not a table"
