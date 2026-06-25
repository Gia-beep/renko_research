"""Tests for price-derived factor momentum filters."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research_renko_system_qlib import _build_variants
from src.indicators.factor_momentum import (
    factor_momentum,
    factor_momentum_filter,
    factor_spread_returns,
    low_uncertainty_regime,
    select_factors,
)


def _frame(values: dict[str, list[float | bool]], periods: int | None = None) -> pd.DataFrame:
    n = periods if periods is not None else len(next(iter(values.values())))
    return pd.DataFrame(values, index=pd.date_range("2020-01-01", periods=n))


def test_factor_spread_returns_uses_next_day_top_minus_bottom_return():
    idx = pd.date_range("2020-01-01", periods=3)
    close = pd.DataFrame(
        {
            "A": [10.0, 12.0, 12.0],
            "B": [10.0, 11.0, 11.0],
            "C": [10.0, 9.0, 9.0],
            "D": [10.0, 8.0, 8.0],
        },
        index=idx,
    )
    score = pd.DataFrame(
        {"A": [4.0, np.nan, np.nan], "B": [3.0, np.nan, np.nan],
         "C": [2.0, np.nan, np.nan], "D": [1.0, np.nan, np.nan]},
        index=idx,
    )

    out = factor_spread_returns({"demo": score}, close, quantile=0.25, min_names=4)

    assert out.loc[idx[0], "demo"] == pytest.approx(0.2 - (-0.2))
    assert np.isnan(out.loc[idx[1], "demo"])


def test_factor_momentum_is_shifted_to_avoid_lookahead():
    returns = pd.DataFrame({"factor_a": [1.0, 2.0, 3.0, 4.0]})

    out = factor_momentum(returns, lookback=2)

    assert np.isnan(out["factor_a"].iloc[0])
    assert np.isnan(out["factor_a"].iloc[1])
    assert out["factor_a"].iloc[2] == 3.0
    assert out["factor_a"].iloc[3] == 5.0


def test_select_factors_requires_positive_when_enabled():
    momentum = pd.DataFrame({"A": [2.0, -1.0], "B": [1.0, -2.0]})

    out = select_factors(momentum, top_n=1, require_positive=True)

    assert out.iloc[0].tolist() == [True, False]
    assert out.iloc[1].tolist() == [False, False]


def test_low_uncertainty_regime_warmup_passes():
    close = _frame(
        {
            "A": [10.0, 11.0, 12.0, 13.0, 14.0],
            "B": [10.0, 9.0, 8.0, 7.0, 6.0],
        }
    )

    out = low_uncertainty_regime(
        close,
        volatility_window=2,
        quantile_window=3,
        threshold=0.6,
    )

    assert bool(out.iloc[0]) is True
    assert out.index.equals(close.index)


def test_factor_momentum_filter_returns_aligned_boolean_mask():
    close = _frame(
        {
            "A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "B": [10.0, 10.5, 11.0, 11.5, 12.0, 12.5],
            "C": [10.0, 9.5, 9.0, 8.5, 8.0, 7.5],
            "D": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
        }
    )

    result = factor_momentum_filter(
        close,
        momentum_window=1,
        high_window=1,
        reversal_window=1,
        volatility_window=2,
        factor_lookback=1,
        factor_top_n=1,
        factor_spread_quantile=0.25,
        stock_quantile=0.5,
        min_names=4,
        use_uncertainty_regime=False,
    )

    assert result.stock_filter.shape == close.shape
    assert result.stock_filter.dtypes.eq(bool).all()
    assert result.factor_momentum.columns.tolist() == [
        "price_momentum",
        "high_to_current",
        "residual_momentum",
        "short_reversal",
        "low_volatility",
    ]


def test_renko_system_builds_factor_momentum_variants_when_mask_is_supplied():
    periods = 4
    close = _frame({"A": [10.0, 11.0, 12.0, 13.0]}, periods)
    open_ = close - 1.0
    high = close + 1.0
    low = close - 2.0
    volume = _frame({"A": [10.0, 10.0, 20.0, 20.0]}, periods)
    amv = _frame({"A": [True, True, True, True]}, periods)
    factor_mask = _frame({"A": [True, False, True, True]}, periods)
    alpha = type(
        "Alpha",
        (),
        {
            "turn_up": _frame({"A": [True, True, True, True]}, periods),
            "turn_down": _frame({"A": [False, False, False, False]}, periods),
            "rising": _frame({"A": [True, True, True, True]}, periods),
        },
    )()

    variants = _build_variants(
        alpha,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        amv=amv,
        body_ratio_min=0.0,
        upper_shadow_max=1.0,
        white=2,
        yellow=2,
        vol_window=2,
        max_hold=5,
        max_red_bricks=4,
        factor_momentum=factor_mask,
    )

    by_name = {variant.name: variant for variant in variants}
    assert "factor_momentum" in by_name
    assert "full_system_factor_momentum" in by_name
    assert by_name["factor_momentum"].entries["A"].tolist() == [True, False, True, True]
