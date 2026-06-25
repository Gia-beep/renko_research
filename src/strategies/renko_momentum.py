"""Example strategy: momentum signals on a Renko-denoised series.

Skeleton for the project's headline hypothesis — fill in per params.yaml's
``strategy`` block. Long when momentum is strong, exit when it fades.
"""
from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy


class RenkoMomentum(Strategy):
    """RSI-on-Renko long/flat strategy (single-name or cross-sectional).

    Intended flow (TODO implement):
      1. build renko bricks from close   -> src.data.renko.build_renko
      2. compute momentum on bricks       -> src.indicators.momentum.rsi/roc
      3. map brick-level signals back onto the bar timeline (forward-fill)
      4. entries = momentum crosses above long_threshold
         exits   = momentum crosses below short_threshold
      5. shift(1) both to avoid look-ahead
    """

    def generate_signals(
        self, close: pd.DataFrame, **fields: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError(
            "build renko -> momentum -> threshold-cross signals (shifted +1)"
        )
