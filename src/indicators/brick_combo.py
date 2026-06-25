"""Dual-period brick signal rules.

The short-period brick (``shot`` in the user note) provides entries/exits, while
the long-period brick only filters entries when it is in a persistent green
sequence.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.indicators.brick_alpha import BrickAlphaResult, compute_brick_alpha


@dataclass(frozen=True)
class BrickComboSignals:
    """Container for the two brick indicators and the derived trading masks."""

    shot: BrickAlphaResult
    long: BrickAlphaResult
    long_green: pd.DataFrame
    entries: pd.DataFrame
    exits: pd.DataFrame


def consecutive_true(mask: pd.DataFrame, bars: int) -> pd.DataFrame:
    """Return True where ``mask`` has been True for ``bars`` consecutive rows."""
    if bars <= 0:
        raise ValueError("bars must be positive")
    mask = mask.fillna(False).astype(bool)
    if bars == 1:
        return mask
    streak = mask.rolling(bars, min_periods=bars).sum()
    return streak.eq(float(bars)).fillna(False).astype(bool)


def brick_combo_masks(
    shot: BrickAlphaResult,
    long: BrickAlphaResult,
    *,
    long_green_bars: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``long_green``, ``entries`` and ``exits`` from precomputed bricks."""
    long_green = consecutive_true(long.falling, long_green_bars)
    entries = (shot.turn_up & long_green).fillna(False).astype(bool)
    exits = shot.falling.fillna(False).astype(bool)
    return long_green, entries, exits


def yellow_line_filter(
    close: pd.DataFrame,
    *,
    window: int = 60,
    max_below: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter out signal days where price is too far below the yellow MA line.

    Passes when ``close >= MA(close, window) * (1 - max_below)``.  Warmup rows
    with no moving average compare False.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if max_below < 0:
        raise ValueError("max_below must be non-negative")
    yellow = close.rolling(window, min_periods=window).mean()
    mask = close.ge(yellow * (1.0 - max_below)).fillna(False).astype(bool)
    return yellow, mask


def compute_brick_combo_signals(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    *,
    shot_n: int = 7,
    shot_m: int = 3,
    long_n: int = 21,
    long_m: int = 28,
    long_green_bars: int = 2,
    trigger_level: float = 4.0,
) -> BrickComboSignals:
    """Compute the agreed dual-period brick entry/exit masks.

    Entry:
        ``shot.turn_up`` while ``long`` is green for ``long_green_bars`` bars.

    Exit:
        ``shot.falling``.  The long-period brick does not force exits.
    """
    shot = compute_brick_alpha(
        high=high,
        low=low,
        close=close,
        n=shot_n,
        m=shot_m,
        trigger_level=trigger_level,
    )
    long = compute_brick_alpha(
        high=high,
        low=low,
        close=close,
        n=long_n,
        m=long_m,
        trigger_level=trigger_level,
    )
    long_green, entries, exits = brick_combo_masks(
        shot,
        long,
        long_green_bars=long_green_bars,
    )
    return BrickComboSignals(
        shot=shot,
        long=long,
        long_green=long_green,
        entries=entries,
        exits=exits,
    )
