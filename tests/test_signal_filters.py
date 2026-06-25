"""Tests for the signal-enhancement filters."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.signal_filters import (
    above_ma,
    combine,
    combine_any,
    cross_sectional_topk,
    ma_uptrend,
    roc,
    turn_strength,
    volume_ratio,
    bars_since_last_turn_down,
    preceding_downtrend_depth,
    preceding_brick_drop,
    true_range,
    atr,
    atr_ratio,
    close_above_vwap,
    bullish_divergence,
    market_breadth_filter,
    strict_candle_filter,
)


def test_cross_sectional_topk_picks_top_eligible_by_score():
    idx = pd.RangeIndex(2)
    score = pd.DataFrame(
        {"A": [3.0, 1.0], "B": [1.0, 9.0], "C": [2.0, 0.0], "D": [5.0, 0.0]}, index=idx
    )
    eligible = pd.DataFrame(
        {"A": [True, True], "B": [True, False], "C": [True, True], "D": [False, False]},
        index=idx,
    )

    out = cross_sectional_topk(score, eligible, k=2)

    # row 0: among eligible A,B,C (D excluded), top-2 scores are A=3, C=2.
    assert out.loc[0].tolist() == [True, False, True, False]
    # row 1: only A,C eligible (<= k), so both kept regardless of score.
    assert out.loc[1].tolist() == [True, False, True, False]


def test_cross_sectional_topk_breaks_ties_by_column_order():
    score = pd.DataFrame({"A": [1.0], "B": [1.0], "C": [1.0]})
    eligible = pd.DataFrame({"A": [True], "B": [True], "C": [True]})

    out = cross_sectional_topk(score, eligible, k=2)

    # equal scores -> rank(method="first") keeps the first two columns.
    assert out.loc[0].tolist() == [True, True, False]


def test_cross_sectional_topk_empty_eligible_row_is_all_false():
    score = pd.DataFrame({"A": [1.0], "B": [2.0]})
    eligible = pd.DataFrame({"A": [False], "B": [False]})

    out = cross_sectional_topk(score, eligible, k=2)

    assert not out.to_numpy().any()


def test_volume_ratio_against_moving_average():
    vol = pd.DataFrame({"A": [10.0, 20.0, 30.0]})
    out = volume_ratio(vol, window=2)

    assert np.isnan(out["A"].iloc[0])  # warmup
    assert out["A"].iloc[1] == 20.0 / 15.0
    assert out["A"].iloc[2] == 30.0 / 25.0


def test_above_ma_excludes_warmup_and_compares_strictly():
    close = pd.DataFrame({"A": [10.0, 11.0, 9.0]})
    out = above_ma(close, window=2)

    # MA2 = [NaN, 10.5, 10.0]; only 11 > 10.5 is True, warmup is False.
    assert out["A"].tolist() == [False, True, False]


def test_ma_uptrend_flags_rising_average():
    close = pd.DataFrame({"A": [10.0, 12.0, 14.0, 16.0]})
    out = ma_uptrend(close, window=2)

    # MA2 = [NaN, 11, 13, 15]; rising from bar 2 onward.
    assert out["A"].tolist() == [False, False, True, True]


def test_roc_is_window_return():
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0]})
    out = roc(close, window=1)

    assert np.isnan(out["A"].iloc[0])
    assert out["A"].iloc[1] == 11.0 / 10.0 - 1.0
    assert out["A"].iloc[2] == 12.0 / 11.0 - 1.0


def test_turn_strength_matches_zero_filled_diff():
    brick = pd.DataFrame({"A": [np.nan, 0.0, 1.0, 3.0]})
    out = turn_strength(brick)

    assert np.isnan(out["A"].iloc[0])
    assert out["A"].tolist()[1:] == [0.0, 1.0, 2.0]


def test_combine_unions_labels_and_treats_missing_as_false():
    m1 = pd.DataFrame({"A": [True], "B": [True]})
    m2 = pd.DataFrame({"B": [True], "C": [True]})

    out = combine(m1, m2)

    # union columns A,B,C; only B is True in both (A absent in m2, C absent in m1).
    assert bool(out.loc[0, "A"]) is False
    assert bool(out.loc[0, "B"]) is True
    assert bool(out.loc[0, "C"]) is False


def test_combine_any_unions_labels_and_or_s_masks():
    m1 = pd.DataFrame({"A": [True], "B": [False]})
    m2 = pd.DataFrame({"B": [True], "C": [True]})

    out = combine_any(m1, m2)

    assert bool(out.loc[0, "A"]) is True
    assert bool(out.loc[0, "B"]) is True
    assert bool(out.loc[0, "C"]) is True


def test_bars_since_last_turn_down():
    turn_down = pd.DataFrame({"A": [False, True, False, False, True, False]})
    out = bars_since_last_turn_down(turn_down)
    # warmup (before first True) should be NaN
    assert np.isnan(out["A"].iloc[0])
    # turn_down on index 1 -> 0
    assert out["A"].iloc[1] == 0
    # subsequent elements count up
    assert out["A"].iloc[2] == 1
    assert out["A"].iloc[3] == 2
    # turn_down on index 4 -> 0
    assert out["A"].iloc[4] == 0
    assert out["A"].iloc[5] == 1


def test_preceding_downtrend_depth():
    close = pd.DataFrame({"A": [10.0, 10.0, 9.0, 8.0, 9.0]})
    turn_down = pd.DataFrame({"A": [False, True, False, False, False]})
    # turn_down close is 10.0
    out = preceding_downtrend_depth(close, turn_down)
    assert np.isnan(out["A"].iloc[0])
    assert out["A"].iloc[1] == 0.0
    assert out["A"].iloc[2] == 0.1  # (10 - 9) / 10
    assert out["A"].iloc[3] == 0.2  # (10 - 8) / 10
    assert out["A"].iloc[4] == 0.1  # (10 - 9) / 10


def test_preceding_brick_drop():
    brick = pd.DataFrame({"A": [5.0, 4.0, 2.0, 1.0, 3.0]})
    turn_down = pd.DataFrame({"A": [False, True, False, False, False]})
    out = preceding_brick_drop(brick, turn_down)
    assert np.isnan(out["A"].iloc[0])
    assert out["A"].iloc[1] == 0.0
    assert out["A"].iloc[2] == 4.0 - 2.0
    assert out["A"].iloc[3] == 4.0 - 1.0


def test_true_range_and_atr():
    high = pd.DataFrame({"A": [10.0, 12.0, 11.0]})
    low = pd.DataFrame({"A": [9.0, 10.0, 8.0]})
    close = pd.DataFrame({"A": [9.5, 11.0, 9.0]})
    # TR:
    # idx 0: 10 - 9 = 1.0
    # idx 1: max(12-10, |12-9.5|, |10-9.5|) = max(2.0, 2.5, 0.5) = 2.5
    # idx 2: max(11-8, |11-11|, |8-11|) = max(3.0, 0.0, 3.0) = 3.0
    tr = true_range(high, low, close)
    assert tr["A"].tolist()[1:] == [2.5, 3.0]
    
    # ATR with window=2: [NaN, (1.0+2.5)/2=1.75, (2.5+3.0)/2=2.75]
    out_atr = atr(high, low, close, window=2)
    assert np.isnan(out_atr["A"].iloc[0])
    assert out_atr["A"].iloc[1] == 1.75
    assert out_atr["A"].iloc[2] == 2.75


def test_atr_ratio():
    high = pd.DataFrame({"A": [10.0, 12.0, 11.0]})
    low = pd.DataFrame({"A": [9.0, 10.0, 8.0]})
    close = pd.DataFrame({"A": [9.5, 11.0, 9.0]})
    out = atr_ratio(high, low, close, short_window=1, long_window=2)
    # ATR1 = [1.0, 2.5, 3.0]
    # ATR2 = [NaN, 1.75, 2.75]
    # ATR1 / ATR2 = [NaN, 2.5/1.75, 3.0/2.75]
    assert np.isnan(out["A"].iloc[0])
    assert abs(out["A"].iloc[1] - 2.5 / 1.75) < 1e-7
    assert abs(out["A"].iloc[2] - 3.0 / 2.75) < 1e-7


def test_close_above_vwap():
    close = pd.DataFrame({"A": [10.0, 11.0, 9.0]})
    vwap = pd.DataFrame({"A": [9.5, 11.5, 8.5]})
    out = close_above_vwap(close, vwap)
    assert out["A"].tolist() == [True, False, True]


def test_bullish_divergence():
    price = pd.DataFrame({"A": [10.0, 9.0, 8.0, 7.0, 8.0]})
    indicator = pd.DataFrame({"A": [10.0, 9.0, 8.5, 9.0, 9.5]})
    # rolling min window = 3
    # price min (window=3): [NaN, NaN, 8.0, 7.0, 7.0]
    # indicator min (window=3): [NaN, NaN, 8.5, 8.5, 8.5]
    # idx 2: price=8.0 (<=8.0), indicator=8.5 (<=8.5 -> not >8.5) -> False
    # idx 3: price=7.0 (<=7.0), indicator=9.0 (>8.5) -> True
    # idx 4: price=8.0 (>7.0) -> False
    out = bullish_divergence(price, indicator, window=3)
    assert out["A"].tolist() == [False, False, False, True, False]


def test_market_breadth_filter():
    signal = pd.DataFrame({
        "A": [True, False, False],
        "B": [True, True, False],
        "C": [False, False, False],
    })
    # ratio of True:
    # idx 0: 2/3 = 0.667
    # idx 1: 1/3 = 0.333
    # idx 2: 0/3 = 0.0
    # min_ratio = 0.5
    out = market_breadth_filter(signal, min_ratio=0.5)
    # idx 0 passes (True for all columns), idx 1, 2 fail (False for all columns)
    assert out.iloc[0].tolist() == [True, True, True]
    assert out.iloc[1].tolist() == [False, False, False]
    assert out.iloc[2].tolist() == [False, False, False]


def test_strict_candle_filter():
    open_ = pd.DataFrame({"A": [10.0]})
    high = pd.DataFrame({"A": [14.0]})
    low = pd.DataFrame({"A": [10.0]})
    close = pd.DataFrame({"A": [13.5]})
    # body = (13.5 - 10) / 4 = 0.875
    # shadow = (14 - 13.5) / 4 = 0.125
    # body >= 2/3 and shadow <= 0.2 -> True
    out = strict_candle_filter(open_, high, low, close, upper_shadow_max=0.20)
    assert out["A"].tolist() == [True]
