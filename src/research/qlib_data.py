"""Qlib data-loading helpers shared by the research scripts.

Kept separate from :mod:`src.research.metrics` so the metric math stays free of
any Qlib import and remains unit-testable without market data installed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_FIELDS: tuple[str, ...] = ("$open", "$high", "$low", "$close")


def calendar_bounds(provider_uri: Path) -> tuple[str | None, str | None]:
    """First/last trading date in the provider's daily calendar (or ``None``)."""
    path = Path(provider_uri) / "calendars" / "day.txt"
    if not path.exists():
        return None, None
    dates = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not dates:
        return None, None
    return dates[0], dates[-1]


def warmup_start(provider_uri: Path, start: str, warmup_bars: int) -> str:
    """Trading date ``warmup_bars`` sessions before ``start`` (for recursive SMA)."""
    path = Path(provider_uri) / "calendars" / "day.txt"
    if warmup_bars <= 0 or not path.exists():
        return start
    dates = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not dates:
        return start
    start_ts = pd.Timestamp(start)
    first_idx = 0
    for i, date in enumerate(dates):
        if pd.Timestamp(date) >= start_ts:
            first_idx = i
            break
    else:
        return start
    return dates[max(0, first_idx - warmup_bars)]


def load_features(
    provider_uri: Path,
    market: str,
    start: str,
    end: str,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
) -> dict[str, pd.DataFrame]:
    """Load Qlib daily fields as wide ``date x instrument`` frames keyed by field.

    Example: ``load_features(...)["$close"]`` is the close-price matrix.
    """
    try:
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D
    except ImportError as exc:
        raise RuntimeError(
            "Qlib is not installed in this Python. Use the qlib venv, for example "
            "`/home/x1843/venvs/qlib/bin/python scripts/research_brick_alpha_qlib.py`."
        ) from exc

    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    instruments = D.instruments(market, start_time=start, end_time=end)
    raw = D.features(
        instruments,
        list(fields),
        start_time=start,
        end_time=end,
        freq="day",
    )
    if raw.empty:
        raise RuntimeError(f"Qlib returned no data for market={market}, {start=} {end=}")

    return {field: raw[field].unstack("instrument").sort_index() for field in fields}
