"""Backtest the dual-period brick rule on Qlib daily data.

Rules:
  - short/shot brick: default ``N=7, M=3``.
  - long brick: default ``N=21, M=28``.
  - enter when long is in a persistent green sequence and shot turns red.
  - exit when shot turns green.

This uses the repository's event-proxy backtest: equal-weight next-day returns
for active signal positions, not a Qlib account/executor simulation.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indicators.brick_combo import (
    BrickComboSignals,
    compute_brick_combo_signals,
    yellow_line_filter,
)
from src.research.metrics import event_return_summary, event_strategy, forward_returns
from src.research.qlib_data import calendar_bounds, load_features, warmup_start

DEFAULT_PROVIDER_URI = Path(
    os.environ.get("QLIB_PROVIDER_URI", "/home/x1843/.qlib/qlib_data/cn_data")
)


@dataclass(frozen=True)
class Variant:
    """One event-proxy backtest variant."""

    name: str
    entries: pd.DataFrame
    exits: pd.DataFrame


def _trim_combo(signals: BrickComboSignals, keep: pd.Series | np.ndarray) -> BrickComboSignals:
    def trim_alpha(alpha):
        return replace(
            alpha,
            brick_value=alpha.brick_value.loc[keep],
            rising=alpha.rising.loc[keep],
            falling=alpha.falling.loc[keep],
            turn_up=alpha.turn_up.loc[keep],
            turn_down=alpha.turn_down.loc[keep],
        )

    return replace(
        signals,
        shot=trim_alpha(signals.shot),
        long=trim_alpha(signals.long),
        long_green=signals.long_green.loc[keep],
        entries=signals.entries.loc[keep],
        exits=signals.exits.loc[keep],
    )


def _recent_candidates(
    signals: BrickComboSignals,
    *,
    entries: pd.DataFrame | None = None,
    yellow_line: pd.DataFrame | None = None,
    yellow_pass: pd.DataFrame | None = None,
    lookback: int,
    top_n: int,
) -> pd.DataFrame:
    values = signals.shot.brick_value
    if values.empty:
        return pd.DataFrame()

    entries = signals.entries if entries is None else entries
    recent_index = values.index[-lookback:]
    recent_entry = entries.loc[recent_index].any(axis=0)
    last_entry_date = {}
    for code in values.columns[recent_entry]:
        hit_dates = entries.index[entries[code]]
        last_entry_date[code] = hit_dates[-1] if len(hit_dates) else pd.NaT

    last_dt = values.index[-1]
    data = {
        "shot_brick_value": signals.shot.brick_value.loc[last_dt],
        "long_brick_value": signals.long.brick_value.loc[last_dt],
        "shot_rising": signals.shot.rising.loc[last_dt],
        "shot_falling": signals.shot.falling.loc[last_dt],
        "long_falling": signals.long.falling.loc[last_dt],
        "long_green": signals.long_green.loc[last_dt],
        "entry_recent": recent_entry,
        "last_entry_date": pd.Series(last_entry_date),
    }
    if yellow_line is not None:
        data["yellow_line"] = yellow_line.loc[last_dt]
    if yellow_pass is not None:
        data["yellow_pass"] = yellow_pass.loc[last_dt]
    out = pd.DataFrame(data)
    current_setup = out["long_green"] & out["shot_rising"]
    if "yellow_pass" in out:
        current_setup = current_setup & out["yellow_pass"]
    out = out[out["entry_recent"] | current_setup].sort_values(
        ["entry_recent", "shot_brick_value"], ascending=[False, False]
    )
    return out.head(top_n)


def parse_args() -> argparse.Namespace:
    cal_start, cal_end = calendar_bounds(DEFAULT_PROVIDER_URI)
    default_start = "2018-01-01"
    if cal_start and default_start < cal_start:
        default_start = cal_start
    default_end = cal_end or "2020-09-25"

    parser = argparse.ArgumentParser(description="Qlib backtest for shot/long brick combo.")
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--start", default=default_start)
    parser.add_argument("--end", default=default_end)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument(
        "--max-hold",
        type=int,
        default=0,
        help="Calendar holding cap. Use 0 to disable and exit only on shot green.",
    )
    parser.add_argument("--shot-n", "--short-n", dest="shot_n", type=int, default=7)
    parser.add_argument("--shot-m", "--short-m", dest="shot_m", type=int, default=3)
    parser.add_argument("--long-n", type=int, default=21)
    parser.add_argument("--long-m", type=int, default=28)
    parser.add_argument("--long-green-bars", type=int, default=2)
    parser.add_argument("--yellow-window", type=int, default=60)
    parser.add_argument(
        "--yellow-max-below",
        type=float,
        default=0.05,
        help="Drop signal-day entries when close is more than this below yellow MA.",
    )
    parser.add_argument("--trigger-level", type=float, default=4.0)
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=250,
        help="Extra trading bars fetched before --start to warm up recursive SMA.",
    )
    parser.add_argument("--latest-lookback", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "results" / "brick_combo_qlib",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    warmup_bars = max(
        args.warmup_bars,
        args.long_m * 3,
        args.long_n + args.long_green_bars + 5,
        args.yellow_window + 5,
    )
    load_start = warmup_start(args.provider_uri, args.start, warmup_bars)
    feats = load_features(args.provider_uri, args.market, load_start, args.end)
    high_f, low_f, close_f = feats["$high"], feats["$low"], feats["$close"]
    signals_full = compute_brick_combo_signals(
        high=high_f,
        low=low_f,
        close=close_f,
        shot_n=args.shot_n,
        shot_m=args.shot_m,
        long_n=args.long_n,
        long_m=args.long_m,
        long_green_bars=args.long_green_bars,
        trigger_level=args.trigger_level,
    )

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    keep = (close_f.index >= start_ts) & (close_f.index <= end_ts)
    close = close_f.loc[keep]
    signals = _trim_combo(signals_full, keep)
    yellow_full, yellow_pass_full = yellow_line_filter(
        close_f,
        window=args.yellow_window,
        max_below=args.yellow_max_below,
    )
    yellow_line = yellow_full.loc[keep]
    yellow_pass = yellow_pass_full.loc[keep]
    shot_turn_up_yellow = (signals.shot.turn_up & yellow_pass).fillna(False).astype(bool)
    long_green_yellow = (signals.entries & yellow_pass).fillna(False).astype(bool)

    variants = [
        Variant("shot_turn_up_baseline", signals.shot.turn_up, signals.exits),
        Variant("shot_turn_up_yellow_filter", shot_turn_up_yellow, signals.exits),
        Variant("long_green_shot_turn_up", signals.entries, signals.exits),
        Variant("long_green_yellow_shot_turn_up", long_green_yellow, signals.exits),
    ]
    fwd = {h: forward_returns(close, h) for h in args.horizons}

    comparison_rows: list[dict[str, float | int | str]] = []
    event_rows: list[dict[str, float | int | str]] = []
    daily_parts: list[pd.DataFrame] = []
    benchmark_nav: pd.Series | None = None
    benchmark_return: pd.Series | None = None

    for variant in variants:
        entries = variant.entries.reindex(index=close.index, columns=close.columns).fillna(False)
        exits = variant.exits.reindex(index=close.index, columns=close.columns).fillna(False)

        for horizon in args.horizons:
            summary = event_return_summary(entries, fwd[horizon], label=variant.name)
            summary["horizon"] = horizon
            event_rows.append(summary)

        daily, strat = event_strategy(close, entries, exits, max_hold=args.max_hold)
        if benchmark_nav is None:
            benchmark_nav = daily["benchmark_nav"]
            benchmark_return = daily["benchmark_return"]

        daily_parts.append(
            daily[["strategy_return", "strategy_nav", "active_names"]].rename(
                columns={
                    "strategy_return": f"{variant.name}_return",
                    "strategy_nav": f"{variant.name}_nav",
                    "active_names": f"{variant.name}_active_names",
                }
            )
        )

        h3 = next((r for r in event_rows if r["event"] == variant.name and r["horizon"] == 3), None)
        h5 = next((r for r in event_rows if r["event"] == variant.name and r["horizon"] == 5), None)
        comparison_rows.append(
            {
                "variant": variant.name,
                "total_entry_signals": int(entries.to_numpy().sum()),
                "avg_active_names": strat["avg_active_names"],
                "exposure": strat["exposure"],
                "total_return": strat["total_return"],
                "annualized_return": strat["annualized_return"],
                "sharpe": strat["sharpe"],
                "max_drawdown": strat["max_drawdown"],
                "excess_3d": (h3 or {}).get("excess_mean"),
                "winrate_3d": (h3 or {}).get("win_rate"),
                "excess_5d": (h5 or {}).get("excess_mean"),
                "winrate_5d": (h5 or {}).get("win_rate"),
            }
        )

    comparison = pd.DataFrame(comparison_rows).set_index("variant")
    events = pd.DataFrame(event_rows)
    daily_returns = pd.concat(daily_parts, axis=1)
    if benchmark_return is not None and benchmark_nav is not None:
        daily_returns.insert(0, "benchmark_return", benchmark_return)
        daily_returns.insert(1, "benchmark_nav", benchmark_nav)
    candidates = _recent_candidates(
        signals,
        entries=long_green_yellow,
        yellow_line=yellow_line,
        yellow_pass=yellow_pass,
        lookback=args.latest_lookback,
        top_n=args.top_n,
    )

    comparison_path = args.output_prefix.with_name(args.output_prefix.name + "_comparison.csv")
    events_path = args.output_prefix.with_name(args.output_prefix.name + "_events.csv")
    daily_path = args.output_prefix.with_name(args.output_prefix.name + "_daily_returns.csv")
    candidates_path = args.output_prefix.with_name(args.output_prefix.name + "_latest_candidates.csv")

    comparison.to_csv(comparison_path)
    events.to_csv(events_path, index=False)
    daily_returns.to_csv(daily_path)
    candidates.to_csv(candidates_path)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"Market: {args.market}  range: {close.index.min().date()} -> {close.index.max().date()}")
    print(
        "Params: "
        f"shot=({args.shot_n},{args.shot_m}) long=({args.long_n},{args.long_m}) "
        f"long_green_bars={args.long_green_bars} "
        f"yellow=MA{args.yellow_window}/max_below={args.yellow_max_below:.2%} "
        f"max_hold={args.max_hold}"
    )
    print(f"Panel: {close.shape[0]} days x {close.shape[1]} names")
    print("\nVariant comparison:")
    print(comparison.round(6).to_string())
    print("\nSaved:")
    print(f"  {comparison_path}")
    print(f"  {events_path}")
    print(f"  {daily_path}")
    print(f"  {candidates_path}")


if __name__ == "__main__":
    main()
