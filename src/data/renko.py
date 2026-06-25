"""Renko brick construction from a price series.

Renko ignores time: a new brick is drawn only when price moves at least
``brick_size`` from the last brick. Applying momentum indicators to the brick
series (instead of raw bars) is the core hypothesis of this project.

Two sizing modes:
  - fixed: a constant absolute brick size.
  - atr:   brick_size = ATR(window) * multiplier (sampled once here; a rolling
           variant is a natural extension — see TODO).
"""
from __future__ import annotations

import pandas as pd


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def build_renko(close: pd.Series, brick_size: float) -> pd.DataFrame:
    """Convert a close-price series into Renko bricks (fixed brick size).

    Returns a DataFrame indexed by the timestamp of the bar that *closed* each
    brick, with columns:
        price     - the brick's close level
        direction - +1 (up brick) or -1 (down brick)

    A single large move can close several bricks at once; each gets its own
    row stamped with the triggering bar's timestamp.
    """
    if brick_size <= 0:
        raise ValueError("brick_size must be positive")
    if close.empty:
        return pd.DataFrame(columns=["price", "direction"])

    prices: list[float] = []
    dirs: list[int] = []
    times: list = []
    last = float(close.iloc[0])

    for ts, raw in close.items():
        p = float(raw)
        move = p - last
        n_bricks = int(abs(move) // brick_size)
        if n_bricks >= 1:
            step = brick_size if move > 0 else -brick_size
            for _ in range(n_bricks):
                last += step
                prices.append(last)
                dirs.append(1 if step > 0 else -1)
                times.append(ts)

    return pd.DataFrame(
        {"price": prices, "direction": dirs},
        index=pd.Index(times, name=close.index.name),
    )


def renko_brick_size(high, low, close, mode: str, params: dict) -> float:
    """Resolve brick size from params for the given mode ('fixed' | 'atr').

    For 'atr', uses the last available ATR value * multiplier.
    TODO: support a rolling/recalibrated brick size for long backtests.
    """
    if mode == "fixed":
        return float(params["fixed_brick"])
    if mode == "atr":
        a = atr(high, low, close, window=int(params["atr_window"]))
        return float(a.dropna().iloc[-1] * params["atr_multiplier"])
    raise ValueError(f"unknown renko mode: {mode!r}")
