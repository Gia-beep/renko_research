"""Strategy interface.

A strategy turns price/indicator data into boolean entry/exit signals that
feed straight into vectorbt's ``Portfolio.from_signals`` (see
``src.backtest.engine``). Signals must be aligned to the close series and
SHIFTED by one bar to avoid look-ahead.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Base class for all strategies.

    Subclasses implement :meth:`generate_signals`, returning aligned boolean
    ``(entries, exits)``. For cross-sectional strategies these are wide
    DataFrames (cols = stocks); for single-name, Series.
    """

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def generate_signals(
        self, close: pd.DataFrame, **fields: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(entries, exits)`` aligned to ``close``, shifted +1 bar.

        ``fields`` carries any extra inputs a strategy needs (e.g. ``high``,
        ``low`` for Williams %R, or precomputed renko series).
        """
        raise NotImplementedError
