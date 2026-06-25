"""Momentum indicators.

All functions are pure (Series in, Series out) so they apply identically to
time-based bars and to Renko brick series (``src.data.renko.build_renko``'s
``price`` column).

For indicators not implemented here (MACD, KDJ, DMI/ADX), prefer calling the
native 通达信 formula via ``tq.formula_zb`` — but mind the SMA convergence
caveat: daily series need count >= ~250 for stable recursive averages.
"""
from __future__ import annotations

import pandas as pd


def roc(close: pd.Series, window: int = 12) -> pd.Series:
    """Rate of Change, in percent: (close / close[-window] - 1) * 100."""
    return (close / close.shift(window) - 1.0) * 100.0


def mtm(close: pd.Series, window: int = 12) -> pd.Series:
    """Momentum: close - close[-window]."""
    return close - close.shift(window)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Williams %R: position of close within the window's high-low range.

    Ranges from -100 (at window low) to 0 (at window high).
    """
    highest = high.rolling(window).max()
    lowest = low.rolling(window).min()
    return (highest - close) / (highest - lowest) * -100.0


def cross_sectional_rank(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank each row (date) across columns (stocks); 1.0 = strongest.

    Feed a wide DataFrame of a per-stock momentum score (e.g. ``roc`` applied
    column-wise) to get a cross-sectional relative-strength ranking for
    portfolio selection (hold the top quantile).
    """
    return scores.rank(axis=1, ascending=True, pct=True)
