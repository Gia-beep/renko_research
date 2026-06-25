"""Thin wrapper around tqcenter initialization.

Usage::

    from src.connection import tdx_session

    with tdx_session(__file__) as tq:
        df = tq.get_market_data(...)
"""
from __future__ import annotations

from contextlib import contextmanager

from config.settings import ensure_tqcenter_on_path


@contextmanager
def tdx_session(identifier: str | None = None):
    """Initialize tqcenter and yield the ``tq`` handle, closing on exit.

    ``identifier`` is the strategy's unique key passed to
    ``tq.initialize(__file__)``. Two instances sharing the same identifier
    cannot run simultaneously (ErrorId='12'). Pass ``__file__`` from the
    calling script so each entry point is distinct.
    """
    ensure_tqcenter_on_path()
    from tqcenter import tq  # imported only after sys.path is set up

    tq.initialize(identifier or __file__)
    try:
        yield tq
    finally:
        tq.close()
