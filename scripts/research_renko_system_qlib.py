"""Backtest the renko.md v1.0 system on Qlib daily data: full system + ablations.

The sibling ``research_signal_filters_qlib.py`` showed only **volume confirmation**
robustly improves the raw ``turn_up`` event, and nothing controls the ~-40% drawdown.
This script operationalises the *additional* rules in ``renko.md`` that were never
backtested and measures each rule's marginal contribution by ablation.

Rules operationalised (authoritative definitions cross-checked against the original
``zxt_brick`` implementation in the sibling ``PYPlugins`` project):

  - §4.1 strong-red candle quality  -> ``signal_filters.strong_red_body``
  - §3.2 bullish MA stack           -> ``signal_filters.dual_ma_bull``
        (standard short>mid stack ``白 > 黄``; renko.md's literal ``Yellow > White``
         is a transcription error vs the zxt_brick source)
  - §3.1 AMV macro regime gate       -> ``src.research.amv_regime`` (AMV index trend
        gate; the proprietary ``-2.37`` oscillator level is not in the daily data, so
        we reproduce the "sleep in a bear regime" intent with a causal close>=SMA gate)
  - §5.2 "数四块砖" profit-take       -> ``metrics.event_strategy(max_red_bricks=4)``
  - §5.1 trend-break stop            -> close < MA(yellow), OR'd into the exit
  - §5.1 layered policy stops        -> ``event_positions`` kwargs
        ``ma_yellow`` (zero-tolerance), ``ma_white`` + ``white_grace_bars`` (2 consecutive
        breaks default), and ``no_rise_check_bars`` (T+N close ≤ entry close ⇒ exit).
        Bundled together as the ``full_policy`` variant; ablated by ``policy_minus_*``.

Not modelled (documented fidelity limit): the §5.1 *next-day verification* (sell by
T+1 09:37 if price hasn't risen) and the 14:57/09:30 timing are intraday rules that
daily bars cannot represent; a naive daily analog conflates with turn_down / trend-break.

Entries are always a subset of ``turn_up`` so every variant stays comparable to the
``baseline_turn_up`` row (which reproduces the signal-filter study's baseline).

Example:
    /home/x1843/venvs/qlib/bin/python scripts/research_renko_system_qlib.py \
        --market csi300 --start 2018-01-01 --end 2020-09-25 \
        --output-prefix results/renko_system_csi300_2018
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indicators.brick_alpha import compute_brick_alpha
from src.indicators.factor_momentum import FactorMomentumResult, factor_momentum_filter
from src.indicators.signal_filters import (
    combine,
    combine_any,
    dual_ma_bull,
    moving_average,
    strong_red_body,
    volume_ratio,
)
from src.research.amv_regime import DEFAULT_AMV_CSV, amv_regime_mask, broadcast_to_panel, load_amv_close
from src.research.metrics import event_return_summary, event_strategy, forward_returns
from src.research.qlib_data import calendar_bounds, load_features, warmup_start

DEFAULT_PROVIDER_URI = Path(
    os.environ.get("QLIB_PROVIDER_URI", "/home/x1843/.qlib/qlib_data/cn_data")
)


@dataclass(frozen=True)
class Variant:
    """One backtest cell: a turn_up-subset entry mask plus an exit configuration."""

    name: str
    entries: pd.DataFrame
    exits: pd.DataFrame
    max_hold: int
    max_red_bricks: int | None
    # §5.1 layered stops (default OFF ⇒ legacy behaviour).
    no_rise_bars: int | None = None
    enforce_ma_stop: bool = False


def _build_variants(
    alpha,
    *,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    amv: pd.DataFrame,
    body_ratio_min: float,
    upper_shadow_max: float,
    white: int,
    yellow: int,
    vol_window: int,
    max_hold: int,
    max_red_bricks: int,
    no_rise_bars: int = 2,
    factor_momentum: pd.DataFrame | None = None,
) -> list[Variant]:
    """Full renko.md system + single-rule entry/exit ablations.

    Entry-effect group (fixed exit = turn_down | max_hold): isolates each entry rule.
    Exit-effect group (fixed entry = full): isolates each exit rule.  ``full_calendar5``
    is the hinge (full entry, baseline exit) shared by both groups.

    The ``full_policy`` variant operationalises the user-facing exit policy in one
    cell: red→green soft exit + §5.2 four-brick + §5.1 yellow zero-tolerance, white
    grace-1 break, and T+``no_rise_bars`` no-rise check.  ``policy_minus_*`` rows
    ablate one new rule at a time so the marginal effect is measurable.
    """
    turn_up, turn_down, rising = alpha.turn_up, alpha.turn_down, alpha.rising

    strong = strong_red_body(
        open_, high, low, close,
        body_ratio_min=body_ratio_min, upper_shadow_max=upper_shadow_max,
    )
    dual = dual_ma_bull(close, white=white, yellow=yellow)
    vol = volume_ratio(volume, vol_window) >= 1.5
    trend_break = (close < moving_average(close, yellow)).fillna(False)

    e_full = combine(turn_up, strong, dual, amv)
    exit_td = turn_down
    exit_tb = combine_any(turn_down, trend_break)
    off = len(close.index) + 1  # disables the calendar cap for spec-exit variants

    variants = [
        # --- entry-effect group (exit fixed = turn_down | max_hold) ---
        Variant("baseline_turn_up", turn_up, exit_td, max_hold, None),
        Variant("vol_ge_1.5x", combine(turn_up, vol), exit_td, max_hold, None),
        Variant("strong_red", combine(turn_up, strong), exit_td, max_hold, None),
        Variant("dual_ma_bull", combine(turn_up, dual), exit_td, max_hold, None),
        Variant("amv_regime", combine(turn_up, amv), exit_td, max_hold, None),
        Variant("full_minus_strong_red", combine(turn_up, dual, amv), exit_td, max_hold, None),
        Variant("full_minus_dual_ma", combine(turn_up, strong, amv), exit_td, max_hold, None),
        Variant("full_minus_amv", combine(turn_up, strong, dual), exit_td, max_hold, None),
        # --- hinge: full entry, baseline exit ---
        Variant("full_calendar5", e_full, exit_td, max_hold, None),
        # --- exit-effect group (entry fixed = full) ---
        Variant("full_brick4", e_full, exit_td, off, max_red_bricks),
        Variant("full_trendbreak", e_full, exit_tb, off, None),
        Variant("full_system", e_full, exit_tb, off, max_red_bricks),
        # --- policy-stack group (red→green + §5.2 brick + §5.1 layered stops) ---
        Variant(
            "full_policy", e_full, exit_td, off, max_red_bricks,
            no_rise_bars=no_rise_bars, enforce_ma_stop=True,
        ),
        Variant(
            "policy_minus_no_rise", e_full, exit_td, off, max_red_bricks,
            no_rise_bars=None, enforce_ma_stop=True,
        ),
        Variant(
            "policy_minus_ma_stop", e_full, exit_td, off, max_red_bricks,
            no_rise_bars=no_rise_bars, enforce_ma_stop=False,
        ),
        Variant(
            "policy_minus_brick", e_full, exit_td, off, None,
            no_rise_bars=no_rise_bars, enforce_ma_stop=True,
        ),
    ]
    if factor_momentum is not None:
        fm = factor_momentum.reindex(index=close.index, columns=close.columns).fillna(False)
        e_factor = combine(turn_up, fm)
        e_full_factor = combine(e_full, fm)
        variants.extend(
            [
                Variant("factor_momentum", e_factor, exit_td, max_hold, None),
                Variant("vol1.5_factor_momentum", combine(turn_up, vol, fm), exit_td, max_hold, None),
                Variant("strong_factor_momentum", combine(turn_up, strong, fm), exit_td, max_hold, None),
                Variant("full_factor_momentum", e_full_factor, exit_td, max_hold, None),
                Variant("full_system_factor_momentum", e_full_factor, exit_tb, off, max_red_bricks),
            ]
        )
    return variants


def parse_args() -> argparse.Namespace:
    cal_start, cal_end = calendar_bounds(DEFAULT_PROVIDER_URI)
    default_start = "2018-01-01"
    if cal_start and default_start < cal_start:
        default_start = cal_start
    default_end = cal_end or "2020-09-25"

    parser = argparse.ArgumentParser(description="Backtest the renko.md system + ablations.")
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--start", default=default_start)
    parser.add_argument("--end", default=default_end)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--max-hold", type=int, default=5)
    parser.add_argument("--max-red-bricks", type=int, default=4, help="renko.md §5.2 数四块砖.")
    parser.add_argument(
        "--no-rise-bars", type=int, default=2,
        help="renko.md §5.1 次日不涨即走 (T+N proxy on daily bars; 0 disables for full_policy).",
    )
    parser.add_argument(
        "--white-grace-bars", type=int, default=1,
        help="White-line break grace days (default 1 ⇒ need 2 consecutive breaks).",
    )
    parser.add_argument("--amv-csv", type=Path, default=DEFAULT_AMV_CSV)
    parser.add_argument("--amv-sma", type=int, default=60, help="AMV regime trend-gate SMA window.")
    parser.add_argument("--body-ratio-min", type=float, default=2.0 / 3.0)
    parser.add_argument("--upper-shadow-max", type=float, default=0.25)
    parser.add_argument("--white", type=int, default=20, help="Short (白) MA period.")
    parser.add_argument("--yellow", type=int, default=60, help="Mid (黄) MA period.")
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument(
        "--factor-momentum",
        action="store_true",
        help="Enable price-derived factor-momentum/regime filters for Renko entries.",
    )
    parser.add_argument("--factor-momentum-window", type=int, default=60)
    parser.add_argument("--factor-high-window", type=int, default=120)
    parser.add_argument("--factor-reversal-window", type=int, default=5)
    parser.add_argument("--factor-volatility-window", type=int, default=20)
    parser.add_argument("--factor-lookback", type=int, default=20)
    parser.add_argument("--factor-top-n", type=int, default=2)
    parser.add_argument("--factor-spread-quantile", type=float, default=0.2)
    parser.add_argument("--factor-stock-quantile", type=float, default=0.2)
    parser.add_argument("--factor-min-names", type=int, default=20)
    parser.add_argument(
        "--no-factor-uncertainty-regime",
        action="store_true",
        help="Disable low-uncertainty momentum / high-uncertainty reversal switching.",
    )
    parser.add_argument("--factor-uncertainty-window", type=int, default=20)
    parser.add_argument("--factor-uncertainty-quantile-window", type=int, default=252)
    parser.add_argument("--factor-uncertainty-threshold", type=float, default=0.6)
    parser.add_argument(
        "--warmup-bars", type=int, default=120,
        help="Extra trading bars fetched before --start to warm up recursive SMA / MAs.",
    )
    parser.add_argument(
        "--output-prefix", type=Path, default=PROJECT_ROOT / "results" / "renko_system",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    warmup_bars = args.warmup_bars
    if args.factor_momentum:
        warmup_bars = max(
            warmup_bars,
            args.factor_high_window + args.factor_lookback + 5,
            args.factor_momentum_window + args.factor_lookback + 5,
            args.factor_uncertainty_window + args.factor_uncertainty_quantile_window + 5,
        )
    load_start = warmup_start(args.provider_uri, args.start, warmup_bars)
    feats = load_features(
        args.provider_uri, args.market, load_start, args.end,
        fields=("$open", "$high", "$low", "$close", "$volume"),
    )
    open_f, high_f, low_f, close_f, vol_f = (
        feats["$open"], feats["$high"], feats["$low"], feats["$close"], feats["$volume"],
    )
    alpha_full = compute_brick_alpha(high=high_f, low=low_f, close=close_f)
    factor_result: FactorMomentumResult | None = None
    if args.factor_momentum:
        factor_result = factor_momentum_filter(
            close_f,
            high=high_f,
            momentum_window=args.factor_momentum_window,
            high_window=args.factor_high_window,
            reversal_window=args.factor_reversal_window,
            volatility_window=args.factor_volatility_window,
            factor_lookback=args.factor_lookback,
            factor_top_n=args.factor_top_n,
            factor_spread_quantile=args.factor_spread_quantile,
            stock_quantile=args.factor_stock_quantile,
            min_names=args.factor_min_names,
            use_uncertainty_regime=not args.no_factor_uncertainty_regime,
            uncertainty_window=args.factor_uncertainty_window,
            uncertainty_quantile_window=args.factor_uncertainty_quantile_window,
            uncertainty_threshold=args.factor_uncertainty_threshold,
        )

    amv_mask = amv_regime_mask(load_amv_close(args.amv_csv), sma_window=args.amv_sma)

    # Warm up recursive SMA / rolling MAs on pre-start bars, then trim to [start, end].
    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)
    keep = (close_f.index >= start_ts) & (close_f.index <= end_ts)

    def trim(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[keep]

    open_, high, low, close, volume = (trim(open_f), trim(high_f), trim(low_f), trim(close_f), trim(vol_f))
    alpha = replace(
        alpha_full,
        brick_value=trim(alpha_full.brick_value),
        rising=trim(alpha_full.rising),
        falling=trim(alpha_full.falling),
        turn_up=trim(alpha_full.turn_up),
        turn_down=trim(alpha_full.turn_down),
    )
    amv = broadcast_to_panel(amv_mask, close.index, close.columns)
    factor_mask = trim(factor_result.stock_filter) if factor_result is not None else None

    variants = _build_variants(
        alpha,
        open_=open_, high=high, low=low, close=close, volume=volume, amv=amv,
        body_ratio_min=args.body_ratio_min, upper_shadow_max=args.upper_shadow_max,
        white=args.white, yellow=args.yellow, vol_window=args.vol_window,
        max_hold=args.max_hold, max_red_bricks=args.max_red_bricks,
        no_rise_bars=args.no_rise_bars if args.no_rise_bars > 0 else None,
        factor_momentum=factor_mask,
    )

    fwd = {h: forward_returns(close, h) for h in args.horizons}

    # MA frames are reused across variants for the §5.1 policy stops.
    ma_white_frame = moving_average(close, args.white)
    ma_yellow_frame = moving_average(close, args.yellow)

    comparison_rows: list[dict[str, float | int | str]] = []
    event_rows: list[dict[str, float | int | str]] = []
    benchmark_row: dict[str, float] | None = None

    for v in variants:
        entries = v.entries.reindex(index=close.index, columns=close.columns).fillna(False)

        for h in args.horizons:
            summary = event_return_summary(entries, fwd[h], label=v.name)
            summary["horizon"] = h
            event_rows.append(summary)

        daily, strat = event_strategy(
            close, entries, v.exits,
            max_hold=v.max_hold, red_brick=alpha.rising, max_red_bricks=v.max_red_bricks,
            no_rise_check_bars=v.no_rise_bars,
            ma_white=ma_white_frame if v.enforce_ma_stop else None,
            ma_yellow=ma_yellow_frame if v.enforce_ma_stop else None,
            white_grace_bars=args.white_grace_bars,
        )

        if benchmark_row is None:
            bench_total = float(daily["benchmark_nav"].iloc[-1] - 1.0)
            bench_vol = float(daily["benchmark_return"].std(ddof=0))
            bench_sharpe = (
                float(daily["benchmark_return"].mean() / bench_vol * (252.0 ** 0.5))
                if bench_vol > 0 else float("nan")
            )
            benchmark_row = {
                "variant": "benchmark_equal_weight",
                "total_return": bench_total,
                "sharpe": bench_sharpe,
            }

        h3 = next((r for r in event_rows if r["event"] == v.name and r["horizon"] == 3), None)
        h5 = next((r for r in event_rows if r["event"] == v.name and r["horizon"] == 5), None)
        comparison_rows.append(
            {
                "variant": v.name,
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
    factor_path = None
    if factor_result is not None:
        factor_diag = factor_result.selected_factors.loc[keep].astype(int)
        if factor_result.low_uncertainty is not None:
            factor_diag["low_uncertainty"] = factor_result.low_uncertainty.loc[keep].astype(int)
        factor_diag["selected_factor_count"] = factor_result.selected_factors.loc[keep].sum(axis=1)
        factor_diag["factor_stock_pass_count"] = factor_mask.sum(axis=1)
        factor_path = args.output_prefix.with_name(
            args.output_prefix.name + "_factor_momentum_daily.csv"
        )
        factor_diag.to_csv(factor_path)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"Market: {args.market}  range: {close.index.min().date()} -> {close.index.max().date()}")
    print(f"Panel: {close.shape[0]} days x {close.shape[1]} names | max_hold={args.max_hold} "
          f"max_red_bricks={args.max_red_bricks} white={args.white} yellow={args.yellow} "
          f"amv_sma={args.amv_sma} no_rise_bars={args.no_rise_bars} "
          f"white_grace_bars={args.white_grace_bars}")
    if factor_result is not None:
        print(
            "Factor momentum: "
            f"top_factors={args.factor_top_n} stock_quantile={args.factor_stock_quantile:.2f} "
            f"uncertainty_regime={not args.no_factor_uncertainty_regime}"
        )
    if benchmark_row is not None:
        print(
            f"Benchmark (equal-weight): total_return={benchmark_row['total_return']:.4f} "
            f"sharpe={benchmark_row['sharpe']:.4f}"
        )
    print("\nVariant comparison (sorted by Sharpe):")
    show = comparison[
        ["total_entry_signals", "avg_active_names", "exposure", "total_return",
         "sharpe", "max_drawdown", "excess_3d", "winrate_3d", "excess_5d"]
    ].copy()
    print(show.round(6).to_string())
    print("\nSaved:")
    print(f"  {comparison_path}")
    print(f"  {events_path}")
    if factor_path is not None:
        print(f"  {factor_path}")


if __name__ == "__main__":
    main()
