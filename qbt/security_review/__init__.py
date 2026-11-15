"""Security review: does a security solve a real portfolio problem for a
given investor profile, and which candidate solves it best?

The package holds the UI-agnostic decision engine. Its building blocks --
return-stream loading, fund review and diversification -- stay at their
historical paths (`qbt.returns_input`, `qbt.fund_review`, `qbt.clustering`)
so every existing import keeps working, and are re-exported here so new code
has one entry point:

    from qbt.security_review import returns_input, fund_review, clustering
    from qbt.security_review import engine, config, bundle

See docs/DECISIONS.md ("Module placement") for the reasoning.
"""
from .. import clustering, fund_review, returns_input  # noqa: F401  (re-exports)
from . import config, data, portfolio, stats, scenarios, impact, verdict  # noqa: F401
from . import bundle, engine  # noqa: F401
from .engine import Request, Review, run, sweep  # noqa: F401
from .config import load_all  # noqa: F401
