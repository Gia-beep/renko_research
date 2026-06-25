"""Tests for the renko.md system pieces: candle-quality + dual-MA filters, the
AMV regime gate, and the 数四块砖 (4-red-brick) exit in ``event_positions``.

Synthetic pandas frames only — no Qlib, matching ``test_signal_filters.py`` /
``test_research_metrics.py`` style.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.research_renko_system_qlib import _build_variants
from src.indicators.signal_filters import (
    dual_ma_bull,
    strong_red_body,
    upper_shadow_ratio,
)
from src.research.amv_regime import amv_regime_mask, broadcast_to_panel, load_amv_close
from src.research.metrics import event_positions


def _frame(values: dict[str, list], periods: int) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=periods)
    return pd.DataFrame(values, index=idx)


# --------------------------------------------------------------------------- #
# candle quality (renko.md §4.1)
# --------------------------------------------------------------------------- #
# Four scenarios as rows of column "A":
#   0 strong red:  O10 H14.5 L10 C14 -> body 0.889, shadow 0.111, C>O  -> True
#   1 weak body:   O10 H15   L10 C11 -> body 0.20                       -> False
#   2 long shadow: O10 H17   L10 C15 -> body 0.714, shadow 0.286 > 0.25 -> False
#   3 flat bar:    O10 H10   L10 C10 -> High==Low -> NaN                -> False
_OPEN = _frame({"A": [10.0, 10.0, 10.0, 10.0]}, 4)
_HIGH = _frame({"A": [14.5, 15.0, 17.0, 10.0]}, 4)
_LOW = _frame({"A": [10.0, 10.0, 10.0, 10.0]}, 4)
_CLOSE = _frame({"A": [14.0, 11.0, 15.0, 10.0]}, 4)


def test_upper_shadow_ratio_and_flat_bar_is_nan():
    out = upper_shadow_ratio(_OPEN, _HIGH, _LOW, _CLOSE)["A"]
    assert out.iloc[0] == (14.5 - 14.0) / (14.5 - 10.0)
    assert out.iloc[1] == (15.0 - 11.0) / (15.0 - 10.0)
    assert out.iloc[2] == (17.0 - 15.0) / (17.0 - 10.0)
    assert np.isnan(out.iloc[3])  # High == Low -> no meaningful shadow


def test_strong_red_body_requires_body_short_shadow_and_bull():
    out = strong_red_body(_OPEN, _HIGH, _LOW, _CLOSE)["A"]
    assert out.tolist() == [True, False, False, False]


def test_strong_red_body_excludes_bearish_candle():
    # Tall body but Close < Open (a strong *green* bar) must not pass.
    open_ = _frame({"A": [15.0]}, 1)
    high = _frame({"A": [15.5]}, 1)
    low = _frame({"A": [10.0]}, 1)
    close = _frame({"A": [10.5]}, 1)
    assert strong_red_body(open_, high, low, close)["A"].tolist() == [False]


# --------------------------------------------------------------------------- #
# dual-MA bull stack (renko.md §3.2, implemented as the standard short>mid stack)
# --------------------------------------------------------------------------- #
def test_dual_ma_bull_passes_on_rising_stack_after_warmup():
    close = _frame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]}, 5)
    out = dual_ma_bull(close, white=2, yellow=3)["A"]
    # MA2=[N,10.5,11.5,12.5,13.5], MA3=[N,N,11,12,13]; pass when close>MA3 & MA2>MA3.
    assert out.tolist() == [False, False, True, True, True]


def test_dual_ma_bull_false_when_price_below_mid_line():
    close = _frame({"A": [14.0, 13.0, 12.0, 11.0, 10.0]}, 5)
    out = dual_ma_bull(close, white=2, yellow=3)["A"]
    assert not out.any()  # falling stack: close below mid MA every post-warmup bar


# --------------------------------------------------------------------------- #
# AMV regime gate
# --------------------------------------------------------------------------- #
def _series(values: list, periods: int) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=periods))


def test_amv_regime_mask_trend_cross_and_nan_warmup_passes():
    close = _series([10.0, 11.0, 12.0, 9.0, 8.0, 13.0], 6)
    mask = amv_regime_mask(close, sma_window=3)
    # SMA3=[N,N,11,10.667,9.667,10]; warmup NaN -> pass.
    assert mask.tolist() == [True, True, True, False, False, True]


def test_amv_regime_mask_truncation_invariance_no_lookahead():
    close = _series([10.0, 11.0, 12.0, 9.0, 8.0, 13.0], 6)
    full = amv_regime_mask(close, sma_window=3)
    trunc = amv_regime_mask(close.iloc[:4], sma_window=3)
    assert trunc.tolist() == full.iloc[:4].tolist()


def test_amv_regime_mask_disabled_when_window_nonpositive():
    close = _series([10.0, 5.0, 1.0], 3)
    assert amv_regime_mask(close, sma_window=0).all()


def test_broadcast_to_panel_missing_date_passes_and_fills_columns():
    idx = pd.date_range("2020-01-01", periods=3)
    mask = pd.Series([True, False], index=idx[:2])  # idx[2] missing
    out = broadcast_to_panel(mask, idx, pd.Index(["A", "B"]))
    assert out.shape == (3, 2)
    assert out.loc[idx[0]].tolist() == [True, True]
    assert out.loc[idx[1]].tolist() == [False, False]
    assert out.loc[idx[2]].tolist() == [True, True]  # missing date -> pass


def test_load_amv_close_strips_bom_and_sorts(tmp_path):
    path = tmp_path / "amv.csv"
    path.write_text(
        "﻿date,open,high,low,close,metric_1,metric_2,ret_1d\n"
        "2020-01-03,1,2,1,12.0,100,200,\n"
        "2020-01-02,1,2,1,11.0,100,200,0.1\n",
        encoding="utf-8",
    )
    s = load_amv_close(path)
    assert s.index.tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    assert s.tolist() == [11.0, 12.0]
    assert s.name == "amv_close"


# --------------------------------------------------------------------------- #
# 数四块砖 exit (renko.md §5.2) in event_positions
# --------------------------------------------------------------------------- #
def test_event_positions_exits_on_fourth_red_brick():
    # Enter day 0; red bricks on d0,d2,d3,d4 (d1 green, still held). Entry = brick #1,
    # so the 4th red brick is d4 -> exit *on* d4 (cumulative, not consecutive).
    entries = _frame({"A": [True, False, False, False, False, False]}, 6)
    exits = _frame({"A": [False] * 6}, 6)
    red = _frame({"A": [True, False, True, True, True, True]}, 6)
    pos = event_positions(entries, exits, max_hold=100, red_brick=red, max_red_bricks=4)
    assert pos["A"].tolist() == [True, True, True, True, False, False]


def test_event_positions_brick_exit_disabled_matches_legacy():
    # red_brick provided but max_red_bricks=None -> brick exit OFF -> pure max_hold.
    entries = _frame({"A": [True, False, False, False, False, False]}, 6)
    exits = _frame({"A": [False] * 6}, 6)
    red = _frame({"A": [True, True, True, True, True, True]}, 6)
    legacy = event_positions(entries, exits, max_hold=3)
    off = event_positions(entries, exits, max_hold=3, red_brick=red, max_red_bricks=None)
    assert off["A"].tolist() == legacy["A"].tolist() == [True, True, True, False, False, False]


def test_renko_system_trendbreak_exit_is_or_with_turn_down():
    close = _frame({"A": [10.0, 8.0, 9.0, 7.0]}, 4)
    open_ = close - 1.0
    high = close + 1.0
    low = close - 2.0
    volume = _frame({"A": [10.0, 10.0, 10.0, 10.0]}, 4)
    amv = _frame({"A": [True, True, True, True]}, 4)
    alpha = SimpleNamespace(
        turn_up=_frame({"A": [True, False, False, False]}, 4),
        turn_down=_frame({"A": [False, False, True, False]}, 4),
        rising=_frame({"A": [True, False, False, False]}, 4),
    )

    variants = {
        v.name: v
        for v in _build_variants(
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
        )
    }

    # close < MA2 on rows 1 and 3; turn_down on row 2.  The §5.1 exit is OR.
    assert variants["full_trendbreak"].exits["A"].tolist() == [False, True, True, True]
