"""Excel export of a Security review, in the style of the backtest workbook
(red table titles, black header row, real numbers with number formats,
tables stacked on one sheet per module).

Every table comes from `report.sections`, the same list the page renders.
"""
from __future__ import annotations

import io
from typing import List, Optional

import pandas as pd

from ..excel_export import _TITLE_FONT, _style_block, _widths
from . import report as RP
from .config import AllConfig
from .engine import Review


def write_sections(secs: List[RP.Section]) -> bytes:
    buf = io.BytesIO()
    order = [s for s in RP.SHEETS if any(x[0] == s for x in secs)]
    order += [s for s in dict.fromkeys(x[0] for x in secs) if s not in order]
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for sheet in order:
            r = 0
            titles = set()
            for sh, title, df, note in secs:
                if sh != sheet or df is None or df.empty:
                    continue
                df.to_excel(xw, sheet_name=sheet, startrow=r + 1, index=False)
                ws = xw.sheets[sheet]
                ws.cell(row=r + 1, column=1, value=title).font = _TITLE_FONT
                titles.add(r + 1)
                _style_block(ws, r + 2, len(df), df.shape[1], df, False, title)
                end = r + 2 + len(df)
                if note:
                    ws.cell(row=end + 1, column=1, value=note)
                    titles.add(end + 1)
                r = end + 3
            if sheet in xw.sheets:
                _widths(xw.sheets[sheet], titles)
                if sheet == "Series":
                    xw.sheets[sheet].freeze_panes = "B3"
    return buf.getvalue()


def workbook(rv: Review, cfg: AllConfig, sweep: Optional[pd.DataFrame] = None) -> bytes:
    return write_sections(RP.sections(rv, cfg, sweep))
