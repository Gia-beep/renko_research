"""Market-data loading with a local parquet cache.

Wraps ``tq.get_market_data`` so research code never hits the API twice for the
same request. A request is one price ``field`` over a (period, date-range,
dividend, code-set); the resulting wide table is cached under ``data/raw/``.

Constraints respected here:
  - get_market_data returns at most 24000 rows per call → minute periods are
    fetched in <=24000-bar windows and concatenated.
  - get_market_data returns ``{field: DataFrame(rows=time, cols=codes)}``;
    ``tq.price_df`` pulls one field into a clean wide DataFrame.
  - Dividend type is a STRING for market data ('none' | 'front' | 'back').

An empty ``end`` ("latest") is resolved to today's date so the cache is
date-stamped: re-running on the same day is a hit, the next day refetches.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

import pandas as pd

from config.settings import RAW_DIR

# Approx A-share intraday bars per trading day — used only to size minute
# batches. Daily and coarser periods always fit one 24000-row call (~95y),
# so they are absent here and fall through to a single request.
_BARS_PER_DAY = {"1m": 240, "5m": 48, "15m": 16, "30m": 8, "60m": 4, "1h": 4}
_MAX_ROWS = 24000


def _codes_digest(codes: list[str], dividend_type: str) -> str:
    """Stable short hash of the request's code-set + dividend type."""
    payload = "|".join(sorted(codes)) + "@" + dividend_type
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _cache_path(
    field: str, period: str, start: str, end: str, dividend_type: str, codes: list[str]
):
    """Build a deterministic parquet path for a request."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tag = _codes_digest(codes, dividend_type)
    return RAW_DIR / f"{field}_{period}_{start}_{end or 'latest'}_{tag}.parquet"


def _date_windows(start: str, end: str, period: str) -> list[tuple[str, str]]:
    """Split [start, end] (inclusive) into <=24000-bar windows.

    Sizes each minute window in *calendar* days; since trading days <= calendar
    days, a window never exceeds 24000 bars. Daily/coarser periods (and any
    period without a known bar/day rate) return a single original-bounds window.
    """
    bars = _BARS_PER_DAY.get(period)
    if bars is None:
        return [(start, end)]
    max_days = max(1, _MAX_ROWS // bars)
    start_d = datetime.strptime(start, "%Y%m%d").date()
    end_d = datetime.strptime(end, "%Y%m%d").date() if end else datetime.now().date()
    windows: list[tuple[str, str]] = []
    cur = start_d
    while cur <= end_d:
        w_end = min(cur + timedelta(days=max_days - 1), end_d)
        windows.append((cur.strftime("%Y%m%d"), w_end.strftime("%Y%m%d")))
        cur = w_end + timedelta(days=1)
    return windows


def load_prices(
    tq,
    codes: list[str],
    field: str = "Close",
    *,
    start: str = "20200101",
    end: str = "",
    period: str = "1d",
    dividend_type: str = "front",
    fill_data: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a wide DataFrame (rows=date, cols=code) for one price field.

    Reads the parquet cache when ``use_cache`` and a matching file exists;
    otherwise fetches via ``tq.get_market_data`` (minute periods batched to the
    24000-row limit), converts with ``tq.price_df``, caches, and returns. Only
    non-empty results are cached so a transient empty fetch can be retried.
    """
    codes = list(dict.fromkeys(codes))  # de-dup, preserve order
    if not codes:
        return pd.DataFrame()
    if not end:
        end = datetime.now().strftime("%Y%m%d")

    path = _cache_path(field, period, start, end, dividend_type, codes)
    if use_cache and path.exists():
        return pd.read_parquet(path)

    frames: list[pd.DataFrame] = []
    for w_start, w_end in _date_windows(start, end, period):
        raw = tq.get_market_data(
            field_list=[field],
            stock_list=codes,
            start_time=w_start,
            end_time=w_end,
            dividend_type=dividend_type,
            period=period,
            fill_data=fill_data,
        )
        wide = tq.price_df(raw, field, column_names=codes)
        if wide is not None and not wide.empty:
            frames.append(wide)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames) if len(frames) > 1 else frames[0]
    out = out.loc[~out.index.duplicated(keep="last")].sort_index()

    out.to_parquet(path)
    return out
