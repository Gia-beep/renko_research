"""Tests for reversal signal research helpers."""
from __future__ import annotations

import pandas as pd

from scripts.research_reversal_signals_qlib import bottomk, delayed_forward_returns, topk


def test_bottomk_keeps_lowest_eligible_scores():
    score = pd.DataFrame({"A": [3.0], "B": [1.0], "C": [2.0]})
    eligible = pd.DataFrame({"A": [True], "B": [True], "C": [False]})

    out = bottomk(score, eligible, k=1)

    assert out.loc[0].to_dict() == {"A": False, "B": True, "C": False}


def test_topk_keeps_highest_eligible_scores():
    score = pd.DataFrame({"A": [3.0], "B": [1.0], "C": [2.0]})
    eligible = pd.DataFrame({"A": [True], "B": [True], "C": [False]})

    out = topk(score, eligible, k=1)

    assert out.loc[0].to_dict() == {"A": True, "B": False, "C": False}


def test_delayed_forward_returns_uses_next_bar_entry():
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 15.0]})

    out = delayed_forward_returns(close, horizon=2, entry_shift=1)

    assert out["A"].iloc[0] == 15.0 / 11.0 - 1.0
