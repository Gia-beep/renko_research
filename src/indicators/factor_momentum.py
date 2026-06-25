"""Price-derived factor momentum filters for brick-alpha backtests.

The module keeps the factor-momentum layer independent from Qlib/TdxQuant data
loading.  Inputs are wide ``date x instrument`` OHLC frames; outputs are aligned
wide masks/scores that can gate the existing Renko ``turn_up`` event.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorMomentumResult:
    """Diagnostics and final stock-level filter produced by factor momentum."""

    factor_scores: dict[str, pd.DataFrame]
    factor_returns: pd.DataFrame
    factor_momentum: pd.DataFrame
    selected_factors: pd.DataFrame
    stock_score: pd.DataFrame
    stock_filter: pd.DataFrame
    low_uncertainty: pd.Series | None


def price_factor_scores(
    close: pd.DataFrame,
    *,
    high: pd.DataFrame | None = None,
    momentum_window: int = 60,
    high_window: int = 120,
    reversal_window: int = 5,
    volatility_window: int = 20,
) -> dict[str, pd.DataFrame]:
    """Build a compact price-only factor library, all oriented high-is-better.

    The factors map the user's research note to fields available in the current
    daily backtests:

    - ``price_momentum``: conventional medium-term price momentum.
    - ``high_to_current``: distance to the rolling high; values closer to zero
      mean less drawdown from the high and therefore stronger trend quality.
    - ``residual_momentum``: medium-term momentum after subtracting equal-weight
      market returns, a market-beta-light approximation of residual momentum.
    - ``short_reversal``: negative short-term return, used in high-uncertainty
      regimes where reversal tends to dominate.
    - ``low_volatility``: negative realized volatility.
    """
    if momentum_window <= 0:
        raise ValueError("momentum_window must be positive")
    if high_window <= 0:
        raise ValueError("high_window must be positive")
    if reversal_window <= 0:
        raise ValueError("reversal_window must be positive")
    if volatility_window <= 0:
        raise ValueError("volatility_window must be positive")

    close = close.astype(float)
    high = close if high is None else high.astype(float).reindex_like(close)
    returns = close.pct_change(fill_method=None)
    market_return = returns.mean(axis=1, skipna=True)
    residual_return = returns.sub(market_return, axis=0)
    rolling_high = high.rolling(high_window, min_periods=high_window).max()

    return {
        "price_momentum": close / close.shift(momentum_window) - 1.0,
        "high_to_current": close / rolling_high - 1.0,
        "residual_momentum": residual_return.rolling(
            momentum_window, min_periods=momentum_window
        ).sum(),
        "short_reversal": -(close / close.shift(reversal_window) - 1.0),
        "low_volatility": -returns.rolling(
            volatility_window, min_periods=volatility_window
        ).std(),
    }


def factor_spread_returns(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    *,
    quantile: float = 0.2,
    min_names: int = 20,
) -> pd.DataFrame:
    """Daily top-minus-bottom next-day returns for each stock-level factor.

    Scores at date ``t`` are matched to close-to-close returns from ``t`` to
    ``t+1``.  Consumers must shift these realized factor returns before using
    them as signals; :func:`factor_momentum_filter` does that internally.
    """
    if not 0 < quantile <= 0.5:
        raise ValueError("quantile must satisfy 0 < quantile <= 0.5")
    if min_names <= 0:
        raise ValueError("min_names must be positive")

    close = close.astype(float)
    next_ret = close.shift(-1) / close - 1.0
    out = pd.DataFrame(index=close.index, columns=list(factors), dtype=float)

    for name, score in factors.items():
        score = score.reindex(index=close.index, columns=close.columns)
        for dt in close.index:
            x = score.loc[dt]
            y = next_ret.loc[dt]
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            if n < min_names:
                continue
            tail = max(1, int(np.floor(n * quantile)))
            ranks = x[mask].rank(method="first", ascending=True)
            bottom = ranks <= tail
            top = ranks > n - tail
            if bool(top.any()) and bool(bottom.any()):
                out.at[dt, name] = float(y[mask][top].mean() - y[mask][bottom].mean())

    return out


def factor_momentum(
    factor_returns: pd.DataFrame,
    *,
    lookback: int = 20,
) -> pd.DataFrame:
    """Causal factor-momentum score: trailing factor return sum shifted one day."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    return factor_returns.rolling(lookback, min_periods=lookback).sum().shift(1)


def select_factors(
    momentum: pd.DataFrame,
    *,
    top_n: int = 2,
    require_positive: bool = True,
) -> pd.DataFrame:
    """Select the strongest factor-momentum columns for each date."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    ranks = momentum.rank(axis=1, ascending=False, method="first")
    selected = ranks.le(top_n)
    if require_positive:
        selected &= momentum.gt(0.0)
    return selected.fillna(False).astype(bool)


def low_uncertainty_regime(
    close: pd.DataFrame,
    *,
    volatility_window: int = 20,
    quantile_window: int = 252,
    threshold: float = 0.6,
) -> pd.Series:
    """Market-volatility regime: True when uncertainty is at/below threshold.

    The proxy is equal-weight market realized volatility.  Warmup dates are set
    to True so enabling the regime switch does not discard the early sample.
    """
    if volatility_window <= 0:
        raise ValueError("volatility_window must be positive")
    if quantile_window <= 0:
        raise ValueError("quantile_window must be positive")
    if not 0 < threshold < 1:
        raise ValueError("threshold must satisfy 0 < threshold < 1")

    market_return = (
        close.astype(float).pct_change(fill_method=None).mean(axis=1, skipna=True)
    )
    realized = market_return.rolling(
        volatility_window, min_periods=volatility_window
    ).std()
    min_periods = min(quantile_window, max(1, min(20, quantile_window // 3)))
    cutoff = realized.rolling(quantile_window, min_periods=min_periods).quantile(threshold)
    regime = realized <= cutoff
    regime = regime.mask(realized.isna() | cutoff.isna(), True)
    return regime.rename("low_uncertainty")


def _average_selected_factor_ranks(
    factor_scores: dict[str, pd.DataFrame],
    selected_factors: pd.DataFrame,
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    ranks = {
        name: score.reindex(index=index, columns=columns).rank(axis=1, pct=True)
        for name, score in factor_scores.items()
    }
    out = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    for dt in index:
        active = [name for name in selected_factors.columns if bool(selected_factors.at[dt, name])]
        if not active:
            continue
        out.loc[dt] = pd.concat([ranks[name].loc[dt] for name in active], axis=1).mean(axis=1)
    return out


def factor_momentum_filter(
    close: pd.DataFrame,
    *,
    high: pd.DataFrame | None = None,
    momentum_window: int = 60,
    high_window: int = 120,
    reversal_window: int = 5,
    volatility_window: int = 20,
    factor_lookback: int = 20,
    factor_top_n: int = 2,
    factor_spread_quantile: float = 0.2,
    stock_quantile: float = 0.2,
    min_names: int = 20,
    use_uncertainty_regime: bool = True,
    uncertainty_window: int = 20,
    uncertainty_quantile_window: int = 252,
    uncertainty_threshold: float = 0.6,
) -> FactorMomentumResult:
    """Return a stock-level filter driven by factor momentum and regime switching.

    In low-uncertainty regimes, stocks are ranked by the average percentile rank
    of the currently selected factor-momentum factors.  In high-uncertainty
    regimes, the ranking switches to ``short_reversal`` to reflect the empirical
    momentum/reversal regime split.
    """
    if not 0 < stock_quantile <= 1:
        raise ValueError("stock_quantile must satisfy 0 < stock_quantile <= 1")

    scores = price_factor_scores(
        close,
        high=high,
        momentum_window=momentum_window,
        high_window=high_window,
        reversal_window=reversal_window,
        volatility_window=volatility_window,
    )
    factor_returns = factor_spread_returns(
        scores,
        close,
        quantile=factor_spread_quantile,
        min_names=min_names,
    )
    fm = factor_momentum(factor_returns, lookback=factor_lookback)
    selected = select_factors(fm, top_n=factor_top_n)
    stock_score = _average_selected_factor_ranks(scores, selected, close.index, close.columns)

    low_uncertainty = None
    if use_uncertainty_regime:
        low_uncertainty = low_uncertainty_regime(
            close,
            volatility_window=uncertainty_window,
            quantile_window=uncertainty_quantile_window,
            threshold=uncertainty_threshold,
        )
        reversal_rank = scores["short_reversal"].rank(axis=1, pct=True)
        low_regime = low_uncertainty.reindex(close.index).fillna(True)
        low_panel = pd.DataFrame(
            {column: low_regime for column in close.columns},
            index=close.index,
        )
        stock_score = stock_score.where(low_panel, reversal_rank)

    rank = stock_score.rank(axis=1, pct=True)
    cutoff = 1.0 - stock_quantile
    stock_filter = rank.ge(cutoff).fillna(False).astype(bool)

    return FactorMomentumResult(
        factor_scores=scores,
        factor_returns=factor_returns,
        factor_momentum=fm,
        selected_factors=selected,
        stock_score=stock_score,
        stock_filter=stock_filter,
        low_uncertainty=low_uncertainty,
    )
