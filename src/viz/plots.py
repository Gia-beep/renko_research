"""Plotting helpers for results/ artifacts.

Keep matplotlib here so the rest of the library stays plot-free and testable.
"""
from __future__ import annotations

import pandas as pd


def plot_equity(portfolio, save_to: str | None = None):
    """Plot cumulative returns / equity curve. TODO: portfolio.plot()."""
    raise NotImplementedError


def plot_drawdown(portfolio, save_to: str | None = None):
    """Plot the underwater (drawdown) curve. TODO."""
    raise NotImplementedError


def plot_renko(bricks: pd.DataFrame, save_to: str | None = None):
    """Draw a Renko brick chart from ``src.data.renko.build_renko`` output.

    Up bricks and down bricks as stacked rectangles along an event axis
    (not time). TODO: render with matplotlib patches.
    """
    raise NotImplementedError
