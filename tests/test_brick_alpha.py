"""Tests for the Tongdaxin red/green brick alpha translation."""
from __future__ import annotations

import pandas as pd
import pytest

from src.indicators.brick_alpha import compute_brick_alpha, brick_turn_signals, brick_value, tdx_sma


def test_tdx_sma_matches_recursive_definition():
    data = pd.Series([10.0, 14.0, 18.0, 22.0])
    out = tdx_sma(data, n=4, m=1)

    expected = pd.Series([10.0])
    for value in data.iloc[1:]:
        expected.loc[len(expected)] = (value + 3 * expected.iloc[-1]) / 4

    pd.testing.assert_series_equal(out, expected)


def test_brick_turn_up_matches_formula_xg_transition():
    values = pd.DataFrame({"A": [0.0, 0.0, 1.0, 2.0, 1.5, 3.0]})
    alpha = brick_turn_signals(values)

    assert alpha.rising["A"].tolist() == [False, False, True, True, False, True]
    assert alpha.falling["A"].tolist() == [False, False, False, False, True, False]
    assert alpha.turn_up["A"].tolist() == [False, False, True, False, False, True]
    assert alpha.turn_down["A"].tolist() == [False, False, False, False, True, False]


def test_flat_price_window_does_not_create_turn_signal():
    idx = pd.date_range("2020-01-01", periods=8)
    frame = pd.DataFrame({"A": [10.0] * 8}, index=idx)

    values = brick_value(high=frame, low=frame, close=frame)
    alpha = brick_turn_signals(values)

    assert not alpha.turn_up.any().any()
    assert not alpha.turn_down.any().any()


def test_brick_value_is_non_negative_after_warmup():
    idx = pd.date_range("2020-01-01", periods=12)
    close = pd.DataFrame({"A": range(10, 22)}, index=idx, dtype=float)
    high = close + 0.5
    low = close - 0.5

    values = brick_value(high=high, low=low, close=close)

    arr = values.to_numpy()
    valid = arr[~pd.isna(arr)]
    assert valid.size > 0
    assert (valid >= 0).all()


def test_brick_value_uses_parameterized_tongdaxin_periods():
    idx = pd.date_range("2020-01-01", periods=8)
    close = pd.DataFrame({"A": [10.0, 11.0, 9.0, 12.0, 13.0, 11.0, 14.0, 15.0]}, index=idx)
    high = close + 0.7
    low = close - 0.5

    out = brick_value(high=high, low=low, close=close, n=3, m=2)

    hhv = high.rolling(3, min_periods=3).max()
    llv = low.rolling(3, min_periods=3).min()
    spread = hhv - llv
    var1a = (hhv - close) / spread * 100.0 - 90.0
    var2a = tdx_sma(var1a, 3, 1) + 100.0
    var3a = (close - llv) / spread * 100.0
    var4a = tdx_sma(var3a, 2, 1)
    var5a = tdx_sma(var4a, 2, 1) + 100.0
    var6a = var5a - var2a
    expected = (var6a - 4.0).where(var6a > 4.0)
    expected = expected.mask(var6a.notna() & (var6a <= 4.0), 0.0)

    pd.testing.assert_frame_equal(out, expected)


def test_brick_value_accepts_m_larger_than_n():
    idx = pd.date_range("2020-01-01", periods=12)
    close = pd.DataFrame({"A": range(10, 22)}, index=idx, dtype=float)

    out = brick_value(high=close + 1.0, low=close - 1.0, close=close, n=3, m=5)

    assert out.shape == close.shape


def test_window_alias_matches_n_and_rejects_conflict():
    idx = pd.date_range("2020-01-01", periods=8)
    close = pd.DataFrame({"A": range(10, 18)}, index=idx, dtype=float)
    high = close + 0.5
    low = close - 0.5

    pd.testing.assert_frame_equal(
        brick_value(high=high, low=low, close=close, n=3),
        brick_value(high=high, low=low, close=close, window=3),
    )
    with pytest.raises(ValueError, match="alias"):
        brick_value(high=high, low=low, close=close, n=3, window=4)


def test_compute_brick_alpha_keeps_legacy_default_equivalent():
    idx = pd.date_range("2020-01-01", periods=8)
    close = pd.DataFrame({"A": range(10, 18)}, index=idx, dtype=float)
    high = close + 0.5
    low = close - 0.5

    legacy = compute_brick_alpha(high=high, low=low, close=close)
    explicit = compute_brick_alpha(high=high, low=low, close=close, n=4, m=6)

    pd.testing.assert_frame_equal(legacy.brick_value, explicit.brick_value)
    pd.testing.assert_frame_equal(legacy.turn_up, explicit.turn_up)
