"""Tests for the shared factor-evaluation primitives."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.metrics import (
    benchmark_hedged_event_strategy,
    event_positions,
    event_strategy,
    forward_returns,
    long_short_event_strategy,
    max_drawdown,
)


def _frame(values: dict[str, list], periods: int) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=periods)
    return pd.DataFrame(values, index=idx)


def test_forward_returns_shift_and_tail_nan():
    close = _frame({"A": [10.0, 11.0, 12.0, 13.0]}, 4)
    out = forward_returns(close, horizon=1)

    assert out["A"].iloc[0] == 11.0 / 10.0 - 1.0
    assert out["A"].iloc[2] == 13.0 / 12.0 - 1.0
    assert np.isnan(out["A"].iloc[-1])


def test_event_positions_exits_after_max_hold():
    entries = _frame({"A": [True, False, False, False, False]}, 5)
    exits = _frame({"A": [False] * 5}, 5)

    pos = event_positions(entries, exits, max_hold=2)

    # enter day 0, hold day 1, forced out at day 2 (age reached max_hold).
    assert pos["A"].tolist() == [True, True, False, False, False]


def test_event_positions_max_hold_zero_disables_calendar_cap():
    entries = _frame({"A": [True, False, False, False, False]}, 5)
    exits = _frame({"A": [False] * 5}, 5)

    pos = event_positions(entries, exits, max_hold=0)

    assert pos["A"].tolist() == [True, True, True, True, True]


def test_event_positions_exits_on_exit_signal():
    entries = _frame({"A": [True, False, False, False]}, 4)
    exits = _frame({"A": [False, False, True, False]}, 4)

    pos = event_positions(entries, exits, max_hold=10)

    assert pos["A"].tolist() == [True, True, False, False]


def test_event_positions_allows_reentry_after_exit():
    entries = _frame({"A": [True, False, False, True]}, 4)
    exits = _frame({"A": [False, True, False, False]}, 4)

    pos = event_positions(entries, exits, max_hold=10)

    assert pos["A"].tolist() == [True, False, False, True]


def test_max_drawdown_is_worst_peak_to_trough():
    nav = pd.Series([1.0, 1.2, 0.9, 1.0])
    assert max_drawdown(nav) == 0.9 / 1.2 - 1.0


def test_event_strategy_equal_weights_active_names():
    # Two names; A enters and is held one day, B never enters.
    close = _frame({"A": [10.0, 11.0, 11.0], "B": [20.0, 20.0, 20.0]}, 3)
    entries = _frame({"A": [True, False, False], "B": [False, False, False]}, 3)
    exits = _frame({"A": [False, False, False], "B": [False] * 3}, 3)

    daily, summary = event_strategy(close, entries, exits, max_hold=1)

    # Day 0 holds A only: strategy return = A's next-day return = 11/10 - 1 = 0.1.
    assert daily["strategy_return"].iloc[0] == pytest.approx(0.1)
    assert daily["active_names"].iloc[0] == 1
    assert summary["exposure"] > 0.0


def test_benchmark_hedged_event_strategy_subtracts_active_day_benchmark():
    close = _frame({"A": [10.0, 11.0, 11.0], "B": [20.0, 20.0, 20.0]}, 3)
    entries = _frame({"A": [True, False, False], "B": [False, False, False]}, 3)
    exits = _frame({"A": [False, False, False], "B": [False, False, False]}, 3)

    daily, summary = benchmark_hedged_event_strategy(close, entries, exits, max_hold=1)

    # Active day 0: long A earns 10%; equal-weight benchmark earns 5%; excess = 5%.
    assert daily["excess_return"].iloc[0] == pytest.approx(0.05)
    # Inactive day 1 should not short the benchmark.
    assert daily["active_benchmark_return"].iloc[1] == 0.0
    assert summary["excess_total_return"] == pytest.approx(daily["excess_nav"].iloc[-1] - 1.0)


def test_long_short_event_strategy_uses_half_long_half_short_when_both_active():
    close = _frame({"A": [10.0, 11.0, 11.0], "B": [20.0, 18.0, 18.0]}, 3)
    long_entries = _frame({"A": [True, False, False], "B": [False, False, False]}, 3)
    short_entries = _frame({"A": [False, False, False], "B": [True, False, False]}, 3)
    exits = _frame({"A": [False, False, False], "B": [False, False, False]}, 3)

    daily, summary = long_short_event_strategy(
        close, long_entries, exits, short_entries, exits, max_hold=1
    )

    # Long A +10%; short B profits from B's -10%; 0.5 * 10% - 0.5 * (-10%) = 10%.
    assert daily["strategy_return"].iloc[0] == pytest.approx(0.10)
    assert daily["long_names"].iloc[0] == 1
    assert daily["short_names"].iloc[0] == 1
    assert summary["exposure"] > 0.0


def test_long_short_event_strategy_is_flat_without_both_sides_by_default():
    close = _frame({"A": [10.0, 11.0, 11.0], "B": [20.0, 18.0, 18.0]}, 3)
    long_entries = _frame({"A": [True, False, False], "B": [False, False, False]}, 3)
    no_short_entries = _frame({"A": [False, False, False], "B": [False, False, False]}, 3)
    exits = _frame({"A": [False, False, False], "B": [False, False, False]}, 3)

    daily, summary = long_short_event_strategy(
        close, long_entries, exits, no_short_entries, exits, max_hold=1
    )

    assert daily["strategy_return"].iloc[0] == 0.0
    assert summary["exposure"] == 0.0
