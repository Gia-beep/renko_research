"""Research the red/green brick turn alpha with Qlib data.

Example:
    /home/x1843/venvs/qlib/bin/python scripts/research_brick_alpha_qlib.py \
        --provider-uri /home/x1843/.qlib/qlib_data/cn_data \
        --market csi300 --start 2018-01-01 --end 2020-09-25

The translated Tongdaxin formula lives in ``src.indicators.brick_alpha`` and the
reusable evaluation primitives in ``src.research.metrics`` / ``qlib_data``.
This script evaluates:
  - RankIC of the continuous ``brick_value`` factor.
  - Forward returns by factor quantile.
  - Forward returns after the ``XG`` turn-up event.
  - A simple equal-weight event-holding proxy strategy.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indicators.brick_alpha import compute_brick_alpha
from src.research.metrics import (
    event_return_summary,
    event_strategy,
    forward_returns,
    quantile_returns,
    rank_ic,
)
from src.research.qlib_data import calendar_bounds, load_features, warmup_start

DEFAULT_PROVIDER_URI = Path(
    os.environ.get("QLIB_PROVIDER_URI", "/home/x1843/.qlib/qlib_data/cn_data")
)


def _recent_candidates(
    alpha,
    *,
    lookback: int,
    top_n: int,
) -> pd.DataFrame:
    values = alpha.brick_value
    if values.empty:
        return pd.DataFrame()
    recent_index = values.index[-lookback:]
    recent_turn = alpha.turn_up.loc[recent_index].any(axis=0)
    last_event_date = {}
    for code in values.columns[recent_turn]:
        hit_dates = alpha.turn_up.index[alpha.turn_up[code]]
        last_event_date[code] = hit_dates[-1] if len(hit_dates) else pd.NaT

    last_dt = values.index[-1]
    out = pd.DataFrame(
        {
            "brick_value": values.loc[last_dt],
            "rising": alpha.rising.loc[last_dt],
            "falling": alpha.falling.loc[last_dt],
            "turn_up_recent": recent_turn,
            "last_turn_up_date": pd.Series(last_event_date),
        }
    )
    out = out[out["turn_up_recent"] | out["rising"]].sort_values(
        ["turn_up_recent", "brick_value"], ascending=[False, False]
    )
    return out.head(top_n)


def _append_metric(
    rows: list[dict[str, float | int | str]],
    section: str,
    horizon: int | str,
    metric: str,
    value: float | int,
) -> None:
    rows.append(
        {
            "section": section,
            "horizon": horizon,
            "metric": metric,
            "value": value,
        }
    )


def parse_args() -> argparse.Namespace:
    cal_start, cal_end = calendar_bounds(DEFAULT_PROVIDER_URI)
    default_start = "2018-01-01"
    if cal_start and default_start < cal_start:
        default_start = cal_start
    return _parse_args(default_start, cal_end or "2020-09-25")


def _parse_args(default_start: str, default_end: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qlib research for brick turn alpha.")
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--start", default=default_start)
    parser.add_argument("--end", default=default_end)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--max-hold", type=int, default=5)
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=120,
        help="Extra trading bars fetched before --start to warm up recursive SMA.",
    )
    parser.add_argument("--latest-lookback", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "results" / "brick_alpha_qlib",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    load_start = warmup_start(args.provider_uri, args.start, args.warmup_bars)
    feats = load_features(args.provider_uri, args.market, load_start, args.end)
    high, low, close = feats["$high"], feats["$low"], feats["$close"]
    alpha = compute_brick_alpha(high=high, low=low, close=close)

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    keep = (close.index >= start_ts) & (close.index <= end_ts)
    close = close.loc[keep]
    alpha = replace(
        alpha,
        brick_value=alpha.brick_value.loc[keep],
        rising=alpha.rising.loc[keep],
        falling=alpha.falling.loc[keep],
        turn_up=alpha.turn_up.loc[keep],
        turn_down=alpha.turn_down.loc[keep],
    )

    metric_rows: list[dict[str, float | int | str]] = []
    event_rows: list[dict[str, float | int | str]] = []
    for horizon in args.horizons:
        fwd = forward_returns(close, horizon)

        ic = rank_ic(alpha.brick_value, fwd)
        _append_metric(metric_rows, "rank_ic", horizon, "mean", float(ic.mean()))
        _append_metric(metric_rows, "rank_ic", horizon, "std", float(ic.std(ddof=0)))
        _append_metric(
            metric_rows,
            "rank_ic",
            horizon,
            "positive_rate",
            float((ic > 0).mean()) if not ic.empty else np.nan,
        )
        _append_metric(metric_rows, "rank_ic", horizon, "days", int(ic.shape[0]))

        qret = quantile_returns(alpha.brick_value, fwd, quantiles=args.quantiles)
        if not qret.empty:
            qmean = qret.mean().rename(f"h{horizon}")
            for metric, value in qmean.items():
                _append_metric(metric_rows, "quantile_forward_return", horizon, metric, float(value))

        for label, signal in (
            ("turn_up_xg", alpha.turn_up),
            ("turn_down", alpha.turn_down),
            ("rising_red_brick", alpha.rising),
            ("falling_green_brick", alpha.falling),
        ):
            summary = event_return_summary(signal, fwd, label=label)
            summary["horizon"] = horizon
            event_rows.append(summary)

    daily, strategy_summary = event_strategy(
        close,
        alpha.turn_up,
        alpha.turn_down,
        max_hold=args.max_hold,
    )
    for metric, value in strategy_summary.items():
        _append_metric(metric_rows, "event_strategy", f"max_hold_{args.max_hold}", metric, value)

    candidates = _recent_candidates(
        alpha,
        lookback=args.latest_lookback,
        top_n=args.top_n,
    )

    metrics = pd.DataFrame(metric_rows)
    events = pd.DataFrame(event_rows)
    metrics_path = args.output_prefix.with_name(args.output_prefix.name + "_metrics.csv")
    events_path = args.output_prefix.with_name(args.output_prefix.name + "_events.csv")
    daily_path = args.output_prefix.with_name(args.output_prefix.name + "_daily_returns.csv")
    candidates_path = args.output_prefix.with_name(args.output_prefix.name + "_recent_candidates.csv")

    metrics.to_csv(metrics_path, index=False)
    events.to_csv(events_path, index=False)
    daily.to_csv(daily_path)
    candidates.to_csv(candidates_path)

    print(f"Loaded close matrix: {close.shape[0]} days x {close.shape[1]} names")
    print(f"Date range: {close.index.min().date()} -> {close.index.max().date()}")
    print(f"Market: {args.market}")
    print("\nRankIC summary:")
    print(metrics[metrics["section"].eq("rank_ic")].pivot(index="horizon", columns="metric", values="value"))
    print("\nTurn-up event summary:")
    print(events[events["event"].eq("turn_up_xg")].set_index("horizon"))
    print("\nEvent strategy proxy:")
    for key, value in strategy_summary.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    print("\nSaved:")
    print(f"  {metrics_path}")
    print(f"  {events_path}")
    print(f"  {daily_path}")
    print(f"  {candidates_path}")


if __name__ == "__main__":
    main()
