"""Market-regime gate derived from the AMV (活跃市值) index series.

``renko.md`` §3.1 sleeps the strategy in a bearish macro regime
(``Active_Market_Value <= -2.37``).  AMV is a proprietary Tongdaxin "活跃市值指数";
the ``-2.37`` is a level on its oscillator and is **not** recoverable from the daily
index data we have (``0AMV_daily_official.csv`` carries only OHLC + volume, whose
close is ~1.5e5).  So we reproduce the *intent* — "trade only when the broad market
is in an uptrend" — with the same strictly-causal trend gate the shipped
``zxt_brick`` regime filter uses (``PYPlugins/user/oos_regime_filter.py``):

    tradeable[t]  iff  AMV_close[t] >= SMA(AMV_close, window)[t]

The gate uses only data up to and including day *t*, so it carries no look-ahead;
a NaN-warmup bar passes (we never drop early dates merely for lack of MA history).
The per-date mask is broadcast across instruments so it can be ANDed with the
``date x instrument`` event matrices from :mod:`src.indicators.brick_alpha`.

Pure pandas/numpy (the only I/O is reading the AMV CSV), so it is unit-testable
without market data installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# AMV daily index maintained by the sibling PYPlugins project; external to this repo.
DEFAULT_AMV_CSV = Path("/mnt/c/new_tdx_test/PYPlugins/file/0AMV_daily_official.csv")


def load_amv_close(csv_path: str | Path = DEFAULT_AMV_CSV) -> pd.Series:
    """Load the AMV index daily close as a date-indexed Series (sorted, NaN-free).

    The CSV is ``date,open,high,low,close,metric_1,metric_2[,ret_1d]`` with a BOM;
    ``utf-8-sig`` strips the BOM so the date column parses cleanly.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"AMV CSV not found: {path}")
    raw = pd.read_csv(path, encoding="utf-8-sig", usecols=["date", "close"])
    close = pd.Series(
        pd.to_numeric(raw["close"], errors="coerce").to_numpy(),
        index=pd.to_datetime(raw["date"]),
        name="amv_close",
    )
    return close.dropna().sort_index()


def amv_regime_mask(amv_close: pd.Series, *, sma_window: int = 60) -> pd.Series:
    """Causal "tradeable regime" mask: ``close >= SMA(close, sma_window)``.

    NaN-warmup bars (the first ``sma_window-1`` dates) pass (``True``), and the gate
    uses only bars ``<= t``, so truncating the series after ``t`` never moves
    ``mask[t]`` (no look-ahead).  ``sma_window <= 0`` disables the gate (all True).
    """
    close = pd.Series(amv_close).astype(float)
    window = int(sma_window)
    if window <= 0:
        return pd.Series(True, index=close.index, name="amv_tradeable")
    sma = close.rolling(window, min_periods=window).mean()
    mask = (close >= sma).where(sma.notna(), True)  # NaN SMA (warmup) => pass
    mask.name = "amv_tradeable"
    return mask.astype(bool)


def broadcast_to_panel(
    mask: pd.Series,
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    """Broadcast a per-date regime mask to a ``date x instrument`` boolean frame.

    The mask is reindexed onto the panel's dates; a panel date absent from the mask
    passes (``True``), matching the "missing => pass" convention so partial AMV
    coverage never silently suppresses trading.  Every column gets the same per-date
    value, so ANDing with an event matrix gates all names by the market regime.
    """
    aligned = pd.Series(mask).reindex(index)
    aligned = aligned.where(aligned.notna(), True).astype(bool)  # missing date => pass
    values = np.broadcast_to(aligned.to_numpy()[:, None], (len(index), len(columns)))
    return pd.DataFrame(values.copy(), index=index, columns=columns)
