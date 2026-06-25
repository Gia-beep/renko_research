"""vectorbt backtest wrapper.

Centralizes ``Portfolio.from_signals`` so every experiment uses the same
cost/fill assumptions from params.yaml's ``backtest`` block. Trades fill at
next-bar open by convention (signals are already shifted +1 in the strategy).
"""
from __future__ import annotations

import pandas as pd


def run_backtest(
    close: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    *,
    price: pd.DataFrame | None = None,
    bt_params: dict,
):
    """Build and return a vectorbt Portfolio.

    TODO:
        import vectorbt as vbt
        return vbt.Portfolio.from_signals(
            close=close, entries=entries, exits=exits,
            price=price if price is not None else close,
            init_cash=bt_params["init_cash"], fees=bt_params["fees"],
            slippage=bt_params.get("slippage", 0.0),
            freq=bt_params["freq"],
            size_granularity=bt_params["size_granularity"],  # A股整手
        )
    """
    raise NotImplementedError("wrap vbt.Portfolio.from_signals with bt_params")


def summarize(portfolio) -> pd.Series:
    """Return key metrics (total return, Sharpe, max drawdown, win rate).

    Start from ``portfolio.stats()``; add any custom momentum-specific metrics
    (e.g. average holding period in bricks) here.
    """
    raise NotImplementedError("return portfolio.stats() (+ custom metrics)")
