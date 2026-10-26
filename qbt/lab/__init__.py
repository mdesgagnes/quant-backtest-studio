"""qbt.lab -- the research layer on top of the backtest engine.

The engine answers "what would this rule have returned?". The lab answers the
questions that come before and after: is there anything here to trade
(factor research), is the result evidence or luck (validation), what is it
made of (risk x-ray), what did I already try (hypothesis ledger), and is it
still working (decay monitor). Everything is reachable three ways -- the
Streamlit "Research" workspace, `python -m qbt.lab` on the command line, and
an MCP server so Claude can drive it -- and all three call the same pure
functions in `tools.py`.

Submodules are imported explicitly (`from qbt.lab import stats`); this file
stays empty on purpose so that registering the factor strategy in
`qbt.strategies` cannot create an import cycle.

Nothing in this package modifies the engine or any existing module's output.
"""
LAB_VERSION = "1.0.0"
