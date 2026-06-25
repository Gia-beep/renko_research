"""Tests for the Qlib long-only brick ML helpers."""
from __future__ import annotations

import pandas as pd

from scripts.qlib_long_only_brick_ml import _long_only_backtest, _select_topn


def _frame(values: dict[str, list], periods: int) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=periods)
    return pd.DataFrame(values, index=idx)


def test_select_topn_respects_eligible_mask_and_score_order():
    score = _frame({"A": [3.0], "B": [2.0], "C": [1.0]}, 1)
    eligible = _frame({"A": [False], "B": [True], "C": [True]}, 1)

    selected = _select_topn(score, eligible, top_n=1, min_score=None)

    assert selected.loc[score.index[0]].to_dict() == {"A": False, "B": True, "C": False}


def test_long_only_backtest_never_creates_short_weights():
    close = _frame({"A": [10.0, 11.0, 12.0], "B": [10.0, 9.0, 8.0]}, 3)
    score = _frame({"A": [2.0, 1.0], "B": [1.0, 2.0]}, 2)
    eligible = _frame({"A": [True, True], "B": [True, False]}, 2)

    daily, weights, summary = _long_only_backtest(
        close=close,
        score=score,
        eligible=eligible,
        top_n=1,
        min_score=None,
        entry_shift=0,
        hold_days=1,
        fee=0.0,
    )

    assert (weights >= 0.0).all().all()
    assert (weights.sum(axis=1) <= 1.0).all()
    assert weights.iloc[0].to_dict() == {"A": 1.0, "B": 0.0}
    assert weights.iloc[1].to_dict() == {"A": 1.0, "B": 0.0}
    assert daily["active_names"].tolist() == [1, 1]
    assert summary["exposure"] == 1.0
