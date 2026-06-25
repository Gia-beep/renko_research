"""Research benchmark-hedged and long/short versions of the brick turn signal.

The earlier long-only studies showed that ``turn_up`` and especially
volume-confirmed ``turn_up`` have positive post-event excess returns, but the
proxy NAV is still dominated by broad-market beta.  This script asks whether the
same filters survive after removing that beta:

  - active-day benchmark hedge: long event book minus equal-weight universe return;
  - event long/short: 50% long ``turn_up`` variants and 50% short ``turn_down``
    variants, traded only when both sides are active by default.

Example:
    /home/x1843/venvs/qlib/bin/python scripts/research_hedged_brick_alpha_qlib.py \
        --market csi300 --start 2018-01-01 --end 2020-09-25 \
        --output-prefix results/hedged_brick_alpha_csi300_2018
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indicators.brick_alpha import compute_brick_alpha
from src.indicators.signal_filters import (
    above_ma,
    combine,
    dual_ma_bull,
    strong_red_body,
    volume_ratio,
)
from src.research.amv_regime import DEFAULT_AMV_CSV, amv_regime_mask, broadcast_to_panel, load_amv_close
from src.research.metrics import (
    benchmark_hedged_event_strategy,
    event_return_summary,
    forward_returns,
    long_short_event_strategy,
)
from src.research.qlib_data import calendar_bounds, load_features, warmup_start

DEFAULT_PROVIDER_URI = Path(
    os.environ.get("QLIB_PROVIDER_URI", "/home/x1843/.qlib/qlib_data/cn_data")
)


def _build_long_variants(
    alpha,
    *,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    amv: pd.DataFrame,
    vol_window: int,
    ma_short: int,
    ma_long: int,
    white: int,
    yellow: int,
    body_ratio_min: float,
    upper_shadow_max: float,
) -> dict[str, pd.DataFrame]:
    """Long-entry variants, each a subset of ``turn_up``."""
    turn_up = alpha.turn_up
    vr = volume_ratio(volume, vol_window)
    vol_1_5 = vr >= 1.5
    vol_2_0 = vr >= 2.0
    above_short = above_ma(close, ma_short)
    above_long = above_ma(close, ma_long)
    strong = strong_red_body(
        open_,
        high,
        low,
        close,
        body_ratio_min=body_ratio_min,
        upper_shadow_max=upper_shadow_max,
    )
    dual = dual_ma_bull(close, white=white, yellow=yellow)

    return {
        "baseline_turn_up": turn_up,
        "vol_ge_1.5x": combine(turn_up, vol_1_5),
        "vol_ge_2.0x": combine(turn_up, vol_2_0),
        f"trend_above_ma{ma_short}": combine(turn_up, above_short),
        f"trend_above_ma{ma_long}": combine(turn_up, above_long),
        "amv_regime": combine(turn_up, amv),
        "vol1.5_amv": combine(turn_up, vol_1_5, amv),
        f"vol1.5_trend_ma{ma_short}": combine(turn_up, vol_1_5, above_short),
        "strong_red": combine(turn_up, strong),
        "strong_red_amv": combine(turn_up, strong, amv),
        "vol1.5_strong_red": combine(turn_up, vol_1_5, strong),
        "vol1.5_strong_amv": combine(turn_up, vol_1_5, strong, amv),
        "renko_full_entry": combine(turn_up, strong, dual, amv),
        "renko_full_minus_dual": combine(turn_up, strong, amv),
        "renko_full_plus_vol": combine(turn_up, strong, dual, amv, vol_1_5),
    }


def _build_short_variants(
    alpha,
    *,
    volume: pd.DataFrame,
    amv: pd.DataFrame,
    vol_window: int,
) -> dict[str, pd.DataFrame]:
    """Short-entry variants, each a subset of ``turn_down``."""
    turn_down = alpha.turn_down
    vol_1_5 = volume_ratio(volume, vol_window) >= 1.5
    return {
        "turn_down": turn_down,
        "turn_down_vol_ge_1.5x": combine(turn_down, vol_1_5),
        "turn_down_amv": combine(turn_down, amv),
        "turn_down_vol1.5_amv": combine(turn_down, vol_1_5, amv),
    }


def _trim_variants(
    variants: dict[str, pd.DataFrame],
    index: pd.Index,
    columns: pd.Index,
) -> dict[str, pd.DataFrame]:
    return {
        name: mask.reindex(index=index, columns=columns).fillna(False).astype(bool)
        for name, mask in variants.items()
    }


def parse_args() -> argparse.Namespace:
    cal_start, cal_end = calendar_bounds(DEFAULT_PROVIDER_URI)
    default_start = "2018-01-01"
    if cal_start and default_start < cal_start:
        default_start = cal_start
    default_end = cal_end or "2020-09-25"

    parser = argparse.ArgumentParser(description="Hedged research for brick turn alpha.")
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--start", default=default_start)
    parser.add_argument("--end", default=default_end)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--max-hold", type=int, default=5)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--ma-short", type=int, default=20)
    parser.add_argument("--ma-long", type=int, default=60)
    parser.add_argument("--white", type=int, default=20)
    parser.add_argument("--yellow", type=int, default=60)
    parser.add_argument("--amv-csv", type=Path, default=DEFAULT_AMV_CSV)
    parser.add_argument("--amv-sma", type=int, default=60)
    parser.add_argument("--body-ratio-min", type=float, default=2.0 / 3.0)
    parser.add_argument("--upper-shadow-max", type=float, default=0.25)
    parser.add_argument(
        "--allow-one-sided",
        action="store_true",
        help="Allow long/short proxy to trade if only one side is active.",
    )
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=120,
        help="Extra trading bars fetched before --start to warm up recursive SMA / MAs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "results" / "hedged_brick_alpha",
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
    open_f, high_f, low_f, close_f, volume_f = (
        feats["$open"],
        feats["$high"],
        feats["$low"],
        feats["$close"],
        feats["$volume"],
    )
    alpha_full = compute_brick_alpha(high=high_f, low=low_f, close=close_f)
    amv_full = broadcast_to_panel(
        amv_regime_mask(load_amv_close(args.amv_csv), sma_window=args.amv_sma),
        close_f.index,
        close_f.columns,
    )

    long_full = _build_long_variants(
        alpha_full,
        open_=open_f,
        high=high_f,
        low=low_f,
        close=close_f,
        volume=volume_f,
        amv=amv_full,
        vol_window=args.vol_window,
        ma_short=args.ma_short,
        ma_long=args.ma_long,
        white=args.white,
        yellow=args.yellow,
        body_ratio_min=args.body_ratio_min,
        upper_shadow_max=args.upper_shadow_max,
    )
    short_full = _build_short_variants(
        alpha_full,
        volume=volume_f,
        amv=amv_full,
        vol_window=args.vol_window,
    )

    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)
    keep = (close_f.index >= start_ts) & (close_f.index <= end_ts)
    close = close_f.loc[keep]
    turn_down = alpha_full.turn_down.loc[keep]
    long_variants = _trim_variants(long_full, close.index, close.columns)
    short_variants = _trim_variants(short_full, close.index, close.columns)

    fwd = {h: forward_returns(close, h) for h in args.horizons}
    event_rows: list[dict[str, float | int | str]] = []
    for side, variants in (("long", long_variants), ("short", short_variants)):
        for name, signal in variants.items():
            for h in args.horizons:
                summary = event_return_summary(signal, fwd[h], label=name)
                summary["side"] = side
                summary["horizon"] = h
                event_rows.append(summary)

    comparison_rows: list[dict[str, float | int | str]] = []
    for name, entries in long_variants.items():
        _, strat = benchmark_hedged_event_strategy(
            close,
            entries,
            turn_down,
            max_hold=args.max_hold,
        )
        h3 = next((r for r in event_rows if r["event"] == name and r["horizon"] == 3), None)
        h5 = next((r for r in event_rows if r["event"] == name and r["horizon"] == 5), None)
        comparison_rows.append(
            {
                "variant": name,
                "total_entry_signals": int(entries.to_numpy().sum()),
                "avg_active_names": strat["avg_active_names"],
                "exposure": strat["exposure"],
                "long_total_return": strat["total_return"],
                "long_sharpe": strat["sharpe"],
                "long_max_drawdown": strat["max_drawdown"],
                "hedged_total_return": strat["excess_total_return"],
                "hedged_sharpe": strat["excess_sharpe"],
                "hedged_max_drawdown": strat["excess_max_drawdown"],
                "benchmark_beta": strat["benchmark_beta"],
                "benchmark_corr": strat["benchmark_corr"],
                "event_excess_3d": (h3 or {}).get("excess_mean"),
                "event_excess_5d": (h5 or {}).get("excess_mean"),
                "event_winrate_5d": (h5 or {}).get("win_rate"),
            }
        )

    pairs = {
        "baseline_vs_turn_down": ("baseline_turn_up", "turn_down"),
        "vol1.5_vs_vol1.5_turn_down": ("vol_ge_1.5x", "turn_down_vol_ge_1.5x"),
        "amv_vs_turn_down_amv": ("amv_regime", "turn_down_amv"),
        "vol1.5_amv_vs_turn_down_amv": ("vol1.5_amv", "turn_down_amv"),
        "vol1.5_amv_vs_vol1.5_turn_down_amv": (
            "vol1.5_amv",
            "turn_down_vol1.5_amv",
        ),
        "strong_amv_vs_turn_down_amv": ("strong_red_amv", "turn_down_amv"),
        "full_minus_dual_vs_turn_down_amv": ("renko_full_minus_dual", "turn_down_amv"),
        "full_entry_vs_turn_down_amv": ("renko_full_entry", "turn_down_amv"),
        "full_plus_vol_vs_vol1.5_turn_down_amv": (
            "renko_full_plus_vol",
            "turn_down_vol1.5_amv",
        ),
    }
    long_short_rows: list[dict[str, float | int | str]] = []
    for pair_name, (long_name, short_name) in pairs.items():
        long_entries = long_variants[long_name]
        short_entries = short_variants[short_name]
        _, strat = long_short_event_strategy(
            close,
            long_entries,
            turn_down,
            short_entries,
            long_variants["baseline_turn_up"],
            max_hold=args.max_hold,
            require_both_sides=not args.allow_one_sided,
        )
        long_short_rows.append(
            {
                "pair": pair_name,
                "long_variant": long_name,
                "short_variant": short_name,
                "long_entry_signals": int(long_entries.to_numpy().sum()),
                "short_entry_signals": int(short_entries.to_numpy().sum()),
                "total_return": strat["total_return"],
                "sharpe": strat["sharpe"],
                "max_drawdown": strat["max_drawdown"],
                "exposure": strat["exposure"],
                "long_exposure": strat["long_exposure"],
                "short_exposure": strat["short_exposure"],
                "avg_long_names": strat["avg_long_names"],
                "avg_short_names": strat["avg_short_names"],
                "benchmark_beta": strat["benchmark_beta"],
                "benchmark_corr": strat["benchmark_corr"],
            }
        )

    comparison = pd.DataFrame(comparison_rows).set_index("variant").sort_values(
        "hedged_sharpe", ascending=False
    )
    long_short = pd.DataFrame(long_short_rows).set_index("pair").sort_values(
        "sharpe", ascending=False
    )
    events = pd.DataFrame(event_rows)

    comparison_path = args.output_prefix.with_name(args.output_prefix.name + "_comparison.csv")
    long_short_path = args.output_prefix.with_name(args.output_prefix.name + "_long_short.csv")
    events_path = args.output_prefix.with_name(args.output_prefix.name + "_events.csv")
    comparison.to_csv(comparison_path)
    long_short.to_csv(long_short_path)
    events.to_csv(events_path, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 24)
    print(f"Market: {args.market}  range: {close.index.min().date()} -> {close.index.max().date()}")
    print(
        f"Panel: {close.shape[0]} days x {close.shape[1]} names | "
        f"max_hold={args.max_hold} require_both_sides={not args.allow_one_sided}"
    )
    print("\nBenchmark-hedged long variants (sorted by hedged Sharpe):")
    show_long = comparison[
        [
            "total_entry_signals",
            "avg_active_names",
            "exposure",
            "long_sharpe",
            "hedged_total_return",
            "hedged_sharpe",
            "hedged_max_drawdown",
            "benchmark_beta",
            "event_excess_3d",
            "event_excess_5d",
        ]
    ].copy()
    print(show_long.round(6).to_string())

    print("\nEvent long/short pairs (sorted by Sharpe):")
    show_ls = long_short[
        [
            "long_variant",
            "short_variant",
            "total_return",
            "sharpe",
            "max_drawdown",
            "exposure",
            "avg_long_names",
            "avg_short_names",
            "benchmark_beta",
        ]
    ].copy()
    print(show_ls.round(6).to_string())

    print("\nSaved:")
    print(f"  {comparison_path}")
    print(f"  {long_short_path}")
    print(f"  {events_path}")


if __name__ == "__main__":
    main()
