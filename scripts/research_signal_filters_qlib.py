"""Compare signal-enhancement filters on the brick turn-up event with Qlib data.

The baseline ``turn_up`` / ``XG`` event fires for a large share of the universe
every day, so its proxy strategy is ≈99% invested and tracks the market.  This
script gates or ranks that event with the filters in
``src.indicators.signal_filters`` (volume confirmation, trend regime,
cross-sectional top-K) and reports, per variant:

  - post-event excess forward return and win rate (1/3/5/10 day),
  - the equal-weight proxy strategy (total/annualized return, Sharpe, max
    drawdown, exposure, average daily breadth).

Entries are always a subset of ``turn_up``; the exit rule (``turn_down`` or
``max_hold``) is held fixed across variants so differences isolate the entry
filter.

Example:
    /home/x1843/venvs/qlib/bin/python scripts/research_signal_filters_qlib.py \
        --provider-uri /home/x1843/.qlib/qlib_data/cn_data \
        --market csi300 --start 2018-01-01 --end 2020-09-25
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indicators.brick_alpha import compute_brick_alpha
from src.indicators.signal_filters import (
    above_ma,
    combine,
    cross_sectional_topk,
    roc,
    turn_strength,
    volume_ratio,
)
from src.research.metrics import event_return_summary, event_strategy, forward_returns
from src.research.qlib_data import calendar_bounds, load_features, warmup_start

DEFAULT_PROVIDER_URI = Path(
    os.environ.get("QLIB_PROVIDER_URI", "/home/x1843/.qlib/qlib_data/cn_data")
)


def _build_variants(
    alpha,
    *,
    volume: pd.DataFrame,
    close: pd.DataFrame,
    vol_window: int,
    ma_short: int,
    ma_long: int,
    roc_window: int,
    topk: int,
) -> dict[str, pd.DataFrame]:
    """Map variant name -> boolean entry mask (each a subset of ``turn_up``)."""
    turn_up = alpha.turn_up

    vr = volume_ratio(volume, vol_window)
    vol_1_5 = vr >= 1.5
    vol_2_0 = vr >= 2.0
    above_short = above_ma(close, ma_short)
    above_long = above_ma(close, ma_long)
    r = roc(close, roc_window)
    ts = turn_strength(alpha.brick_value)

    return {
        # --- baseline ---
        "baseline_turn_up": turn_up,
        # --- volume confirmation ---
        "vol_ge_1.5x": combine(turn_up, vol_1_5),
        "vol_ge_2.0x": combine(turn_up, vol_2_0),
        # --- trend regime ---
        f"trend_above_ma{ma_short}": combine(turn_up, above_short),
        f"trend_above_ma{ma_long}": combine(turn_up, above_long),
        # --- volume + trend ---
        f"vol1.5_trend_ma{ma_short}": combine(turn_up, vol_1_5, above_short),
        # --- cross-sectional top-K (caps daily breadth) ---
        f"topk{topk}_by_brick": cross_sectional_topk(alpha.brick_value, turn_up, topk),
        f"topk{topk}_by_turnstrength": cross_sectional_topk(ts, turn_up, topk),
        f"topk{topk}_by_roc{roc_window}": cross_sectional_topk(r, turn_up, topk),
        f"topk{max(1, topk // 2)}_by_roc{roc_window}": cross_sectional_topk(
            r, turn_up, max(1, topk // 2)
        ),
        # --- combined gate + rank ---
        f"vol1.5_trend{ma_short}_topk{topk}_roc": cross_sectional_topk(
            r, combine(turn_up, vol_1_5, above_short), topk
        ),
    }


def parse_args() -> argparse.Namespace:
    cal_start, cal_end = calendar_bounds(DEFAULT_PROVIDER_URI)
    default_start = "2018-01-01"
    if cal_start and default_start < cal_start:
        default_start = cal_start
    default_end = cal_end or "2020-09-25"

    parser = argparse.ArgumentParser(description="Compare brick turn-up signal filters.")
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--start", default=default_start)
    parser.add_argument("--end", default=default_end)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--max-hold", type=int, default=5)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--ma-short", type=int, default=20)
    parser.add_argument("--ma-long", type=int, default=60)
    parser.add_argument("--roc-window", type=int, default=20)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=120,
        help="Extra trading bars fetched before --start to warm up recursive SMA / MAs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "results" / "signal_filter",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    load_start = warmup_start(args.provider_uri, args.start, args.warmup_bars)
    feats = load_features(
        args.provider_uri,
        args.market,
        load_start,
        args.end,
        fields=("$open", "$high", "$low", "$close", "$volume"),
    )
    high, low, close_full, volume_full = (
        feats["$high"],
        feats["$low"],
        feats["$close"],
        feats["$volume"],
    )
    alpha_full = compute_brick_alpha(high=high, low=low, close=close_full)

    # Warm up the recursive SMA / rolling MAs on pre-start bars, then trim.
    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    keep = (close_full.index >= start_ts) & (close_full.index <= end_ts)

    def trim(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[keep]

    close = trim(close_full)
    volume = trim(volume_full)
    alpha = replace(
        alpha_full,
        brick_value=trim(alpha_full.brick_value),
        rising=trim(alpha_full.rising),
        falling=trim(alpha_full.falling),
        turn_up=trim(alpha_full.turn_up),
        turn_down=trim(alpha_full.turn_down),
    )

    variants = _build_variants(
        alpha,
        volume=volume,
        close=close,
        vol_window=args.vol_window,
        ma_short=args.ma_short,
        ma_long=args.ma_long,
        roc_window=args.roc_window,
        topk=args.topk,
    )

    fwd = {h: forward_returns(close, h) for h in args.horizons}

    comparison_rows: list[dict[str, float | int | str]] = []
    event_rows: list[dict[str, float | int | str]] = []
    benchmark_row: dict[str, float] | None = None

    for name, entries in variants.items():
        entries = entries.reindex(index=close.index, columns=close.columns).fillna(False)

        for h in args.horizons:
            summary = event_return_summary(entries, fwd[h], label=name)
            summary["horizon"] = h
            event_rows.append(summary)

        daily, strat = event_strategy(close, entries, alpha.turn_down, max_hold=args.max_hold)

        if benchmark_row is None:
            bench_total = float(daily["benchmark_nav"].iloc[-1] - 1.0)
            bench_vol = float(daily["benchmark_return"].std(ddof=0))
            bench_sharpe = (
                float(daily["benchmark_return"].mean() / bench_vol * (252.0 ** 0.5))
                if bench_vol > 0
                else float("nan")
            )
            benchmark_row = {
                "variant": "benchmark_equal_weight",
                "total_return": bench_total,
                "sharpe": bench_sharpe,
            }

        h3 = next((r for r in event_rows if r["event"] == name and r["horizon"] == 3), None)
        h5 = next((r for r in event_rows if r["event"] == name and r["horizon"] == 5), None)
        comparison_rows.append(
            {
                "variant": name,
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
    comparison = comparison.sort_values("sharpe", ascending=False)
    events = pd.DataFrame(event_rows)

    comparison_path = args.output_prefix.with_name(args.output_prefix.name + "_comparison.csv")
    events_path = args.output_prefix.with_name(args.output_prefix.name + "_events.csv")
    comparison.to_csv(comparison_path)
    events.to_csv(events_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"Market: {args.market}  range: {close.index.min().date()} -> {close.index.max().date()}")
    print(f"Panel: {close.shape[0]} days x {close.shape[1]} names | max_hold={args.max_hold}")
    if benchmark_row is not None:
        print(
            f"Benchmark (equal-weight): total_return={benchmark_row['total_return']:.4f} "
            f"sharpe={benchmark_row['sharpe']:.4f}"
        )
    print("\nVariant comparison (sorted by Sharpe):")
    show = comparison[
        [
            "total_entry_signals",
            "avg_active_names",
            "exposure",
            "total_return",
            "sharpe",
            "max_drawdown",
            "excess_3d",
            "winrate_3d",
            "excess_5d",
        ]
    ].copy()
    print(show.round(6).to_string())
    print("\nSaved:")
    print(f"  {comparison_path}")
    print(f"  {events_path}")


if __name__ == "__main__":
    main()
