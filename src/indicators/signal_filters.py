"""Signal-enhancement filters for the brick turn alpha.

The raw ``turn_up`` / ``XG`` event fires for a large fraction of the universe
every day (≈130 of 300 CSI300 names in the 2018-2020 sample), so the proxy
strategy ends up ≈99% invested and tracks the market.  These filters gate or
rank that event to keep only higher-conviction entries.

Several families, all computable from OHLCV and composable by elementwise AND:

* **Volume confirmation** — :func:`volume_ratio` ≥ threshold (turn backed by an
  above-average-volume session).
* **Trend regime** — :func:`above_ma` / :func:`ma_uptrend` / :func:`dual_ma_bull`
  (only take turns while the name is in an uptrend / a bullish MA stack).
* **Candle quality** — :func:`strong_red_body` / :func:`upper_shadow_ratio` keep
  only tall-bodied bull bars with short upper shadows (renko.md §4.1 强红).
* **Cross-sectional selection** — :func:`cross_sectional_topk` keeps only the
  strongest K turning names each day, which directly caps daily breadth.

Every function takes wide ``date x instrument`` frames and returns a frame of
the same shape (a boolean mask or a float score), so they line up with the
``turn_up`` event matrix from :mod:`src.indicators.brick_alpha`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd



def moving_average(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Simple moving average (Tongdaxin ``MA``); NaN until ``window`` bars exist."""
    if window <= 0:
        raise ValueError("window must be positive")
    return frame.rolling(window, min_periods=window).mean()


def volume_ratio(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volume divided by its ``window``-bar average.

    ``> 1`` marks an above-average-volume session; warmup bars are NaN and a
    flat (zero-average) window yields NaN, so neither passes a ``>=`` threshold.
    """
    avg = moving_average(volume, window)
    return volume / avg.where(avg > 0)


def above_ma(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Boolean mask: close strictly above its ``window``-bar moving average.

    Warmup bars (NaN MA) and NaN closes compare False, so they are excluded.
    """
    ma = moving_average(close, window)
    return (close > ma).fillna(False)


def ma_uptrend(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Boolean mask: the ``window``-bar moving average is rising versus yesterday."""
    ma = moving_average(close, window)
    return (ma > ma.shift(1)).fillna(False)


def dual_ma_bull(
    close: pd.DataFrame,
    *,
    white: int = 20,
    yellow: int = 60,
) -> pd.DataFrame:
    """Boolean mask: standard bullish moving-average stack (renko.md 3.2).

    Passes when the close is above the mid-term (``yellow``) line *and* the
    short-term (``white``) line sits above the mid-term line:

        close > MA(yellow)   AND   MA(white) > MA(yellow)

    Note this implements the *standard* bull stack (short MA above mid MA), which
    is what the authoritative ``zxt_brick`` implementation uses (``白 > 黄``);
    ``renko.md`` §3.2 literally writes ``Yellow > White``, which is almost
    certainly a transcription error.  Warmup bars (NaN MA) compare False, so the
    earliest ``yellow`` bars are excluded.
    """
    ma_white = moving_average(close, white)
    ma_yellow = moving_average(close, yellow)
    return ((close > ma_yellow) & (ma_white > ma_yellow)).fillna(False)


def upper_shadow_ratio(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    """Upper-shadow fraction of the bar's range: ``(High - max(Close, Open)) / (High - Low)``.

    A doji/flat bar (``High == Low``) yields NaN (no meaningful shadow).
    """
    spread = (high - low).where(high > low)
    body_top = close.where(close >= open_, open_)  # elementwise max(close, open)
    return (high - body_top) / spread


def strong_red_body(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    *,
    body_ratio_min: float = 2.0 / 3.0,
    upper_shadow_max: float = 0.25,
) -> pd.DataFrame:
    """Boolean mask: a strong red (bullish) daily candle (renko.md 4.1).

    Passes when the candle is a tall-bodied bull bar with a short upper shadow:

        (Close - Open) / (High - Low) >= body_ratio_min
        (High - max(Close, Open)) / (High - Low) <= upper_shadow_max
        Close > Open

    Corresponds to renko.md's ``强红比例`` plus the upper-shadow constraint and the
    ``zxt_brick`` ``body_ratio`` gate.  Flat bars (``High == Low``) and NaN inputs
    compare False, so they are excluded.
    """
    spread = (high - low).where(high > low)
    body_ratio = (close - open_) / spread
    shadow = upper_shadow_ratio(open_, high, low, close)
    return (
        (body_ratio >= body_ratio_min)
        & (shadow <= upper_shadow_max)
        & (close > open_)
    ).fillna(False)


def roc(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rate of change over ``window`` bars: ``close / close[-window] - 1``."""
    if window <= 0:
        raise ValueError("window must be positive")
    return close / close.shift(window) - 1.0


def turn_strength(brick_value: pd.DataFrame) -> pd.DataFrame:
    """Day-over-day change in ``brick_value`` (the jump size that defines a turn).

    Uses the same zero-filled series as the turn-event definition in
    :func:`src.indicators.brick_alpha.brick_turn_signals`, so on a ``turn_up``
    day this score is positive and larger for sharper turns.
    """
    return brick_value.fillna(0.0).diff()


def cross_sectional_topk(
    score: pd.DataFrame,
    eligible: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """Keep, per date, the top-``k`` ``eligible`` names ranked by ``score``.

    Returns a boolean mask the shape of ``eligible``.  Ties break by column
    order (``rank(method="first")``); rows with ≤ ``k`` eligible names keep all
    of them.  Use this to cap daily breadth of an event signal.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    eligible = eligible.fillna(False).astype(bool)
    score = score.reindex(index=eligible.index, columns=eligible.columns)
    masked = score.where(eligible)
    ranks = masked.rank(axis=1, ascending=False, method="first")
    return (ranks.le(k) & eligible).fillna(False).astype(bool)


def bars_since_last_turn_down(turn_down: pd.DataFrame) -> pd.DataFrame:
    """Number of bars since the last turn_down event.

    Inclusive of the current bar (i.e. returns 0 on turn_down itself).
    Warmup bars prior to the first turn_down return NaN.
    """
    values = np.broadcast_to(np.arange(len(turn_down))[:, None], turn_down.shape)
    bar_idx = pd.DataFrame(
        values,
        index=turn_down.index,
        columns=turn_down.columns,
    )
    last_idx = bar_idx.where(turn_down).ffill()
    return bar_idx - last_idx


def preceding_downtrend_depth(
    close: pd.DataFrame,
    turn_down: pd.DataFrame,
    high: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate the relative price drawdown since the last turn_down event.

    By default, computes ``(close_at_last_down - close) / close_at_last_down``.
    If ``high`` is provided, it uses ``high`` at the last turn_down as the peak reference.
    """
    ref = high if high is not None else close
    ref_at_last_down = ref.where(turn_down).ffill()
    return (ref_at_last_down - close) / ref_at_last_down


def preceding_brick_drop(
    brick_value: pd.DataFrame,
    turn_down: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the absolute drop in brick_value since the last turn_down event.

    Computes ``brick_at_last_down - brick_value``.
    """
    brick_at_last_down = brick_value.where(turn_down).ffill()
    return brick_at_last_down - brick_value


def true_range(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate True Range (TR) from OHLC.

    TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|).
    """
    close_prev = close.shift(1).fillna(close)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    return np.maximum(np.maximum(tr1, tr2), tr3)


def atr(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """Calculate Average True Range (ATR) as SMA of True Range."""
    tr = true_range(high, low, close)
    return moving_average(tr, window)


def atr_ratio(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    short_window: int = 14,
    long_window: int = 50,
) -> pd.DataFrame:
    """Calculate the ratio of short-term ATR to long-term ATR.

    A ratio < 1.0 indicates volatility compression/convergence.
    """
    atr_short = atr(high, low, close, short_window)
    atr_long = atr(high, low, close, long_window)
    return atr_short / atr_long.where(atr_long > 0)


def close_above_vwap(close: pd.DataFrame, vwap: pd.DataFrame) -> pd.DataFrame:
    """Boolean mask: Close price is strictly above VWAP.

    This ensures the average intraday buyer is in profit.
    """
    return (close > vwap).fillna(False)


def bullish_divergence(
    price: pd.DataFrame,
    indicator: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Boolean mask: Price makes a new low, but the indicator does not.

    Specifically, passes when:
        Price == rolling_min(Price, window)
        AND
        Indicator > rolling_min(Indicator, window)
    """
    price_min = price.rolling(window, min_periods=window).min()
    ind_min = indicator.rolling(window, min_periods=window).min()
    price_at_min = price <= price_min
    ind_above_min = indicator > ind_min
    return (price_at_min & ind_above_min).fillna(False)


def market_breadth_filter(
    signal: pd.DataFrame,
    min_ratio: float = 0.05,
) -> pd.DataFrame:
    """Boolean mask: True on days when the fraction of active signals across the universe is >= min_ratio.

    Flags days with widespread systematic turn-ups (market-wide rebounds).
    """
    daily_active_count = signal.fillna(False).sum(axis=1)
    daily_total_count = signal.notna().sum(axis=1)
    daily_ratio = daily_active_count / daily_total_count.where(daily_total_count > 0)
    breadth_pass = (daily_ratio >= min_ratio).fillna(False)
    return pd.DataFrame({col: breadth_pass for col in signal.columns}, index=signal.index)


def strict_candle_filter(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    *,
    body_ratio_min: float = 2.0 / 3.0,
    upper_shadow_max: float = 0.20,
) -> pd.DataFrame:
    """Boolean mask: Strict candle filter requiring strong body and very short upper shadow.

    Combines Close > Open, body_ratio >= body_ratio_min, and upper_shadow_ratio <= upper_shadow_max.
    """
    return strong_red_body(
        open_=open_,
        high=high,
        low=low,
        close=close,
        body_ratio_min=body_ratio_min,
        upper_shadow_max=upper_shadow_max,
    )


def combine(*masks: pd.DataFrame) -> pd.DataFrame:
    """Elementwise AND of boolean masks, aligned on the union of their labels.

    Missing labels count as False, so a name must be present and True in every
    mask to survive.
    """
    if not masks:
        raise ValueError("combine requires at least one mask")
    index = masks[0].index
    columns = masks[0].columns
    for mask in masks[1:]:
        index = index.union(mask.index)
        columns = columns.union(mask.columns)
    result = pd.DataFrame(True, index=index, columns=columns)
    for mask in masks:
        aligned = mask.reindex(index=index, columns=columns, fill_value=False)
        result &= aligned.fillna(False).astype(bool)
    return result


def combine_any(*masks: pd.DataFrame) -> pd.DataFrame:
    """Elementwise OR of boolean masks, aligned on the union of their labels.

    Missing labels count as False, so the result is True when at least one mask is
    present and True for that date/name.  Use this for exit rules where independent
    stops should all be able to close a position.
    """
    if not masks:
        raise ValueError("combine_any requires at least one mask")
    index = masks[0].index
    columns = masks[0].columns
    for mask in masks[1:]:
        index = index.union(mask.index)
        columns = columns.union(mask.columns)
    result = pd.DataFrame(False, index=index, columns=columns)
    for mask in masks:
        aligned = mask.reindex(index=index, columns=columns, fill_value=False)
        result |= aligned.fillna(False).astype(bool)
    return result
