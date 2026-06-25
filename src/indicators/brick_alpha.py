"""Red/green brick turn alpha translated from a Tongdaxin formula.

The source formula's ``SMA(X, N, M)`` is Tongdaxin's recursive moving average::

    SMA_t = (M * X_t + (N - M) * SMA_{t-1}) / N

That is an EWMA with ``alpha = M / N`` and the first valid observation as the
seed.  The main alpha is ``brick_value``; ``turn_up`` is the formula's ``XG``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BrickAlphaResult:
    """Container for the continuous factor and derived event signals."""

    brick_value: pd.DataFrame
    rising: pd.DataFrame
    falling: pd.DataFrame
    turn_up: pd.DataFrame
    turn_down: pd.DataFrame


def tdx_sma(data: pd.Series | pd.DataFrame, n: int, m: int = 1):
    """Return Tongdaxin ``SMA(data, n, m)`` for a Series or DataFrame."""
    if n <= 0:
        raise ValueError("n must be positive")
    if m <= 0 or m > n:
        raise ValueError("m must satisfy 0 < m <= n")
    return data.ewm(alpha=m / n, adjust=False, min_periods=1).mean()


def brick_value(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    *,
    n: int | None = None,
    m: int = 6,
    window: int | None = None,
    trigger_level: float = 4.0,
) -> pd.DataFrame:
    """Compute the continuous ``砖型图`` factor from OHLC wide DataFrames.

    Inputs are date x instrument.  A positive value means the hidden oscillator
    has exceeded ``trigger_level``; a rising value is plotted as a red brick in
    the original formula, and a falling value is plotted as green.

    ``n`` and ``m`` correspond to the Tongdaxin formula parameters:
    ``HHV/LLV(..., N)``, ``SMA(VAR1A, N, 1)``, and the two
    ``SMA(..., M, 1)`` calls.  ``window`` is kept as a backwards-compatible
    alias for ``n``.
    """
    if n is None:
        n = 4 if window is None else window
    elif window is not None and window != n:
        raise ValueError("window is a backwards-compatible alias for n; pass only one value")
    if n <= 0:
        raise ValueError("n must be positive")
    if m <= 0:
        raise ValueError("m must be positive")

    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    hhv = high.rolling(n, min_periods=n).max()
    llv = low.rolling(n, min_periods=n).min()
    spread = (hhv - llv).replace(0.0, np.nan)

    var1a = (hhv - close) / spread * 100.0 - 90.0
    var2a = tdx_sma(var1a, n, 1) + 100.0
    var3a = (close - llv) / spread * 100.0
    var4a = tdx_sma(var3a, m, 1)
    var5a = tdx_sma(var4a, m, 1) + 100.0
    var6a = var5a - var2a

    out = (var6a - trigger_level).where(var6a > trigger_level)
    out = out.mask(var6a.notna() & (var6a <= trigger_level), 0.0)
    return out


def brick_turn_signals(values: pd.DataFrame) -> BrickAlphaResult:
    """Derive red/green brick state and turn events from ``brick_value``.

    ``turn_up`` matches the Tongdaxin condition::

        AA := REF(砖型图, 1) < 砖型图
        XG := REF(AA, 1) = 0 AND AA = 1
    """
    clean = values.fillna(0.0)
    prev = clean.shift(1)
    rising = prev.lt(clean)
    falling = prev.gt(clean)
    turn_up = rising & ~rising.shift(1, fill_value=False)
    turn_down = falling & ~falling.shift(1, fill_value=False)
    return BrickAlphaResult(
        brick_value=values,
        rising=rising.astype(bool),
        falling=falling.astype(bool),
        turn_up=turn_up.astype(bool),
        turn_down=turn_down.astype(bool),
    )


def compute_brick_alpha(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    *,
    n: int | None = None,
    m: int = 6,
    window: int | None = None,
    trigger_level: float = 4.0,
) -> BrickAlphaResult:
    """Compute ``brick_value`` plus red/green turn signals in one call."""
    values = brick_value(
        high=high,
        low=low,
        close=close,
        n=n,
        m=m,
        window=window,
        trigger_level=trigger_level,
    )
    return brick_turn_signals(values)
