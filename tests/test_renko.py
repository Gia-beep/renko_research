"""Tests for the Renko brick transform (src.data.renko)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.renko import build_renko


def test_steady_rise_makes_up_bricks():
    close = pd.Series([100, 101, 102, 103, 104], dtype=float)
    bricks = build_renko(close, brick_size=1.0)
    assert (bricks["direction"] == 1).all()
    assert bricks["price"].tolist() == [101, 102, 103, 104]


def test_small_moves_make_no_bricks():
    # moves under brick_size never close a brick
    close = pd.Series([100, 100.4, 100.8, 100.2], dtype=float)
    bricks = build_renko(close, brick_size=1.0)
    assert bricks.empty


def test_large_jump_closes_multiple_bricks():
    close = pd.Series([100, 103.5], dtype=float)  # +3.5 → 3 up bricks
    bricks = build_renko(close, brick_size=1.0)
    assert len(bricks) == 3
    assert bricks["price"].tolist() == [101, 102, 103]


def test_non_positive_brick_size_raises():
    with pytest.raises(ValueError):
        build_renko(pd.Series([1.0, 2.0]), brick_size=0)
