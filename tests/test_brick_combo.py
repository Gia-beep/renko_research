"""Tests for the dual-period brick combo rules."""
from __future__ import annotations

import pandas as pd

from src.indicators.brick_alpha import BrickAlphaResult
from src.indicators.brick_combo import brick_combo_masks, consecutive_true, yellow_line_filter


def _frame(values: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({"A": values}, index=pd.date_range("2020-01-01", periods=len(values)))


def _alpha(
    *,
    rising: list[bool] | None = None,
    falling: list[bool] | None = None,
    turn_up: list[bool] | None = None,
    turn_down: list[bool] | None = None,
) -> BrickAlphaResult:
    periods = len(next(v for v in (rising, falling, turn_up, turn_down) if v is not None))
    idx = pd.date_range("2020-01-01", periods=periods)
    return BrickAlphaResult(
        brick_value=pd.DataFrame({"A": [0.0] * periods}, index=idx),
        rising=_frame(rising or [False] * periods),
        falling=_frame(falling or [False] * periods),
        turn_up=_frame(turn_up or [False] * periods),
        turn_down=_frame(turn_down or [False] * periods),
    )


def test_consecutive_true_requires_full_streak():
    mask = _frame([False, True, True, False, True, True])

    out = consecutive_true(mask, bars=2)

    assert out["A"].tolist() == [False, False, True, False, False, True]


def test_combo_enters_when_long_has_two_green_bars_and_shot_turns_up():
    shot = _alpha(turn_up=[False, False, True, False], falling=[False, False, False, False])
    long = _alpha(falling=[False, True, True, False], rising=[False, False, False, True])

    long_green, entries, exits = brick_combo_masks(shot, long, long_green_bars=2)

    assert long_green["A"].tolist() == [False, False, True, False]
    assert entries["A"].tolist() == [False, False, True, False]
    assert not exits["A"].any()


def test_combo_does_not_enter_on_single_long_green_bar():
    shot = _alpha(turn_up=[False, False, True, False], falling=[False, False, False, False])
    long = _alpha(falling=[False, False, True, False])

    _, entries, _ = brick_combo_masks(shot, long, long_green_bars=2)

    assert not entries["A"].any()


def test_combo_exits_only_on_shot_falling_not_long_turning_red():
    shot = _alpha(
        turn_up=[True, False, False, False],
        falling=[False, False, True, False],
    )
    long = _alpha(
        falling=[True, True, False, False],
        rising=[False, False, True, False],
    )

    _, entries, exits = brick_combo_masks(shot, long, long_green_bars=2)

    assert entries["A"].tolist() == [False, False, False, False]
    assert exits["A"].tolist() == [False, False, True, False]


def test_yellow_line_filter_drops_only_more_than_five_percent_below_ma():
    close = pd.DataFrame(
        {"A": [100.0, 100.0, 95.0, 94.0]},
        index=pd.date_range("2020-01-01", periods=4),
    )

    yellow, mask = yellow_line_filter(close, window=2, max_below=0.05)

    assert pd.isna(yellow["A"].iloc[0])
    assert yellow["A"].tolist()[1:] == [100.0, 97.5, 94.5]
    assert mask["A"].tolist() == [False, True, True, True]


def test_yellow_line_filter_rejects_price_below_allowed_band():
    close = pd.DataFrame(
        {"A": [100.0, 100.0, 89.0]},
        index=pd.date_range("2020-01-01", periods=3),
    )

    _, mask = yellow_line_filter(close, window=2, max_below=0.05)

    assert mask["A"].tolist() == [False, True, False]
