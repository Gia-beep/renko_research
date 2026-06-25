"""Backtest long-turn confirmation variants on Qlib daily data.

This script focuses on the rule suggested by the visual diagnosis:

  - ``long.turn_up`` confirms the N-shaped wave has turned up.
  - ``shot.rising`` allows the short brick to have turned up earlier.
  - holdings are capped by ``--max-hold`` and exit on ``shot.falling``.

It reports two execution proxies:

  - ``daily_rebalanced``: active positions are equal-weighted every day.
  - ``hold_without_rebalance``: each entry day uses a fixed cash tranche equally,
    then weights drift until exit/max-hold; existing holdings are not rebalanced.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indicators.brick_combo import compute_brick_combo_signals, consecutive_true
from src.indicators.signal_filters import cross_sectional_topk, turn_strength
from src.research.metrics import (
    benchmark_beta,
    benchmark_corr,
    event_positions,
    event_return_summary,
    event_strategy,
    forward_returns,
    max_drawdown,
)
from src.research.qlib_data import calendar_bounds, load_features, warmup_start

DEFAULT_PROVIDER_URI = Path(
    os.environ.get("QLIB_PROVIDER_URI", "/home/x1843/.qlib/qlib_data/cn_data")
)


@dataclass(frozen=True)
class Variant:
    name: str
    entries: pd.DataFrame
    max_hold: int


def parse_args() -> argparse.Namespace:
    cal_start, cal_end = calendar_bounds(DEFAULT_PROVIDER_URI)
    default_start = "2018-01-01"
    if cal_start and default_start < cal_start:
        default_start = cal_start

    parser = argparse.ArgumentParser(description="Qlib backtest for brick turn confirmation.")
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--market", default="csi1000")
    parser.add_argument("--start", default=default_start)
    parser.add_argument("--end", default=cal_end or "2020-09-25")
    parser.add_argument("--shot-n", "--short-n", dest="shot_n", type=int, default=4)
    parser.add_argument("--shot-m", "--short-m", dest="shot_m", type=int, default=6)
    parser.add_argument("--long-n", type=int, default=21)
    parser.add_argument("--long-m", type=int, default=28)
    parser.add_argument("--long-green-bars", type=int, default=2)
    parser.add_argument("--trigger-level", type=float, default=4.0)
    parser.add_argument("--max-hold", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--entry-tranche",
        type=float,
        default=0.0,
        help="Fraction of portfolio allocated to each entry batch. Use 0 for 1/max_hold.",
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument("--cost-bps", nargs="+", type=float, default=[0, 5, 10, 20])
    parser.add_argument("--warmup-bars", type=int, default=250)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "results" / "brick_turn_confirm_qlib",
    )
    return parser.parse_args()


def _bool(frame: pd.DataFrame, index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    return frame.reindex(index=index, columns=columns).fillna(False).astype(bool)


def _return_stats(returns: pd.Series) -> dict[str, float]:
    returns = pd.Series(returns).fillna(0.0)
    nav = (1.0 + returns).cumprod()
    vol = float(returns.std(ddof=0))
    return {
        "total_return": float(nav.iloc[-1] - 1.0),
        "annualized_return": float((nav.iloc[-1]) ** (252.0 / max(1, len(nav))) - 1.0),
        "sharpe": float(returns.mean() / vol * math.sqrt(252.0)) if vol > 0 else np.nan,
        "max_drawdown": max_drawdown(nav),
    }


def _cost_proxy(
    close: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    *,
    max_hold: int,
    cost_bps: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = event_positions(entries, exits, max_hold=max_hold)
    next_ret = close.shift(-1) / close - 1.0
    valid = positions & next_ret.notna()
    active = valid.sum(axis=1).replace(0, np.nan)
    weights = valid.div(active, axis=0).fillna(0.0)
    gross_return = (weights * next_ret.fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = weights.iloc[0].abs().sum()

    daily = pd.DataFrame(
        {
            "gross_return": gross_return,
            "turnover": turnover,
            "active_names": valid.sum(axis=1),
        }
    )
    rows = []
    for bps in cost_bps:
        cost_rate = float(bps) / 10000.0
        net_return = gross_return - turnover * cost_rate
        stats = _return_stats(net_return)
        rows.append(
            {
                "cost_bps": bps,
                **stats,
                "avg_daily_turnover": float(turnover.mean()),
                "annualized_turnover": float(turnover.mean() * 252.0),
            }
        )
        daily[f"net_return_cost_{bps:g}bps"] = net_return
        daily[f"net_nav_cost_{bps:g}bps"] = (1.0 + net_return.fillna(0.0)).cumprod()
    return pd.DataFrame(rows), daily


def _hold_without_rebalance_proxy(
    close: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    *,
    max_hold: int,
    entry_tranche: float,
    cost_bps: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Capital proxy that does not rebalance holdings during their holding period.

    Trades occur at the signal-day close for the next day's return.  Existing
    holdings are sold on ``exits`` or ``max_hold`` before new entries are bought.
    New entries consume at most ``entry_tranche`` of current equity, split equally.
    With ``entry_tranche=1/max_hold``, a 3-day holding rule creates three
    overlapping entry batches without rebalancing older holdings.
    """
    entries = entries.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    exits = exits.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    next_ret = close.shift(-1) / close - 1.0

    weights = pd.Series(0.0, index=close.columns, dtype=float)
    ages = pd.Series(0, index=close.columns, dtype=int)
    cash = 1.0
    rows = []

    for dt in close.index:
        turnover = 0.0
        held = weights.gt(1e-12)
        exit_mask = exits.loc[dt]
        if max_hold > 0:
            exit_mask = exit_mask | ages.ge(max_hold)
        sells = held & exit_mask
        if bool(sells.any()):
            sell_value = float(weights.loc[sells].sum())
            cash += sell_value
            weights.loc[sells] = 0.0
            ages.loc[sells] = 0
            turnover += sell_value

        valid_entry = entries.loc[dt] & weights.le(1e-12) & close.loc[dt].notna() & next_ret.loc[dt].notna()
        if cash > 1e-12 and bool(valid_entry.any()):
            names = valid_entry.index[valid_entry]
            buy_value = min(cash, entry_tranche)
            weights.loc[names] = buy_value / len(names)
            cash -= buy_value
            turnover += buy_value
            ages.loc[names] = 0

        day_ret = next_ret.loc[dt].fillna(0.0)
        gross_return = float((weights * day_ret).sum())
        active = weights.gt(1e-12)
        rows.append(
            {
                "datetime": dt,
                "gross_return": gross_return,
                "turnover": turnover,
                "active_names": int(active.sum()),
                "cash_weight": float(cash),
                "gross_exposure": float(weights.sum()),
            }
        )

        denom = 1.0 + gross_return
        if denom <= 0:
            weights.loc[:] = 0.0
            ages.loc[:] = 0
            cash = 0.0
            continue

        weights = weights * (1.0 + day_ret) / denom
        cash = cash / denom
        active = weights.gt(1e-12)
        ages.loc[active] += 1
        ages.loc[~active] = 0

    daily = pd.DataFrame(rows).set_index("datetime")
    daily["gross_nav"] = (1.0 + daily["gross_return"].fillna(0.0)).cumprod()

    cost_rows = []
    for bps in cost_bps:
        cost_rate = float(bps) / 10000.0
        net_return = daily["gross_return"] - daily["turnover"] * cost_rate
        stats = _return_stats(net_return)
        cost_rows.append(
            {
                "cost_bps": bps,
                **stats,
                "avg_daily_turnover": float(daily["turnover"].mean()),
                "annualized_turnover": float(daily["turnover"].mean() * 252.0),
                "avg_cash_weight": float(daily["cash_weight"].mean()),
                "avg_gross_exposure": float(daily["gross_exposure"].mean()),
                "entry_tranche": entry_tranche,
            }
        )
        daily[f"net_return_cost_{bps:g}bps"] = net_return
        daily[f"net_nav_cost_{bps:g}bps"] = (1.0 + net_return.fillna(0.0)).cumprod()

    return pd.DataFrame(cost_rows), daily


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    warmup_bars = max(
        args.warmup_bars,
        args.long_m * 3,
        args.long_n + args.long_green_bars + 5,
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

    keep = (close_f.index >= pd.Timestamp(args.start)) & (close_f.index <= pd.Timestamp(args.end))
    close = close_f.loc[keep]
    index, columns = close.index, close.columns

    shot_turn_up = _bool(signals_full.shot.turn_up.loc[keep], index, columns)
    shot_rising = _bool(signals_full.shot.rising.loc[keep], index, columns)
    shot_falling = _bool(signals_full.shot.falling.loc[keep], index, columns)
    long_falling = _bool(signals_full.long.falling.loc[keep], index, columns)
    long_turn_up = _bool(signals_full.long.turn_up.loc[keep], index, columns)
    long_strength = turn_strength(signals_full.long.brick_value).loc[keep].reindex(
        index=index,
        columns=columns,
    )

    falling2_shot_turn = consecutive_true(long_falling, args.long_green_bars) & shot_turn_up
    turn_confirm_all = long_turn_up & shot_rising
    turn_confirm_topk = cross_sectional_topk(long_strength, turn_confirm_all, args.top_k)

    variants = [
        Variant("falling2_shot_turn_mh0", falling2_shot_turn, 0),
        Variant(f"falling2_shot_turn_mh{args.max_hold}", falling2_shot_turn, args.max_hold),
        Variant(f"turn_confirm_all_mh{args.max_hold}", turn_confirm_all, args.max_hold),
        Variant(f"turn_confirm_top{args.top_k}_long_strength_mh{args.max_hold}", turn_confirm_topk, args.max_hold),
    ]

    benchmark_return = close.shift(-1).div(close).sub(1.0).mean(axis=1, skipna=True).fillna(0.0)
    benchmark_total = float((1.0 + benchmark_return).cumprod().iloc[-1] - 1.0)
    fwd = {h: forward_returns(close, h) for h in args.horizons}
    summary_rows = []
    event_rows = []
    yearly_rows = []
    cost_rows = []
    hold_daily_parts = []
    daily_parts = [pd.DataFrame({"benchmark_return": benchmark_return}, index=index)]

    for variant in variants:
        variant_entry_tranche = (
            args.entry_tranche
            if args.entry_tranche > 0
            else (1.0 / variant.max_hold if variant.max_hold > 0 else 1.0)
        )
        entries = variant.entries.reindex(index=index, columns=columns).fillna(False).astype(bool)
        daily, stats = event_strategy(close, entries, shot_falling, max_hold=variant.max_hold)
        daily_parts.append(
            daily[["strategy_return", "strategy_nav", "active_names"]].rename(
                columns={
                    "strategy_return": f"{variant.name}_return",
                    "strategy_nav": f"{variant.name}_nav",
                    "active_names": f"{variant.name}_active_names",
                }
            )
        )

        for horizon, ret in fwd.items():
            event = event_return_summary(entries, ret, label=variant.name)
            event["horizon"] = horizon
            event_rows.append(event)

        summary_rows.append(
            {
                "variant": variant.name,
                "max_hold": variant.max_hold,
                "entry_signals": int(entries.to_numpy().sum()),
                "benchmark_total_return": benchmark_total,
                **stats,
                "benchmark_beta": benchmark_beta(daily["strategy_return"], daily["benchmark_return"]),
                "benchmark_corr": benchmark_corr(daily["strategy_return"], daily["benchmark_return"]),
            }
        )

        for year, frame in daily.groupby(daily.index.year):
            yearly_rows.append(
                {
                    "variant": variant.name,
                    "year": int(year),
                    "strategy_return": float((1.0 + frame["strategy_return"]).prod() - 1.0),
                    "benchmark_return": float((1.0 + frame["benchmark_return"]).prod() - 1.0),
                    "avg_active_names": float(frame["active_names"].mean()),
                    "zero_active_days": int((frame["active_names"] == 0).sum()),
                }
            )

        costs, _ = _cost_proxy(
            close,
            entries,
            shot_falling,
            max_hold=variant.max_hold,
            cost_bps=args.cost_bps,
        )
        costs.insert(0, "variant", variant.name)
        costs.insert(1, "execution", "daily_rebalanced")
        cost_rows.extend(costs.to_dict("records"))

        hold_costs, hold_daily = _hold_without_rebalance_proxy(
            close,
            entries,
            shot_falling,
            max_hold=variant.max_hold,
            entry_tranche=variant_entry_tranche,
            cost_bps=args.cost_bps,
        )
        hold_costs.insert(0, "variant", variant.name)
        hold_costs.insert(1, "execution", "hold_without_rebalance")
        cost_rows.extend(hold_costs.to_dict("records"))
        hold_daily_parts.append(
            hold_daily[
                [
                    "gross_return",
                    "gross_nav",
                    "turnover",
                    "active_names",
                    "cash_weight",
                    "gross_exposure",
                ]
            ].rename(
                columns={
                    "gross_return": f"{variant.name}_hold_return",
                    "gross_nav": f"{variant.name}_hold_nav",
                    "turnover": f"{variant.name}_hold_turnover",
                    "active_names": f"{variant.name}_hold_active_names",
                    "cash_weight": f"{variant.name}_hold_cash_weight",
                    "gross_exposure": f"{variant.name}_hold_gross_exposure",
                }
            )
        )

    summary = pd.DataFrame(summary_rows).set_index("variant")
    events = pd.DataFrame(event_rows)
    yearly = pd.DataFrame(yearly_rows)
    costs = pd.DataFrame(cost_rows)
    daily_returns = pd.concat(daily_parts + hold_daily_parts, axis=1)

    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    events_path = args.output_prefix.with_name(args.output_prefix.name + "_events.csv")
    yearly_path = args.output_prefix.with_name(args.output_prefix.name + "_yearly.csv")
    costs_path = args.output_prefix.with_name(args.output_prefix.name + "_costs.csv")
    daily_path = args.output_prefix.with_name(args.output_prefix.name + "_daily_returns.csv")

    summary.to_csv(summary_path)
    events.to_csv(events_path, index=False)
    yearly.to_csv(yearly_path, index=False)
    costs.to_csv(costs_path, index=False)
    daily_returns.to_csv(daily_path)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(f"Market: {args.market}  range: {close.index.min().date()} -> {close.index.max().date()}")
    print(
        "Params: "
        f"shot=({args.shot_n},{args.shot_m}) long=({args.long_n},{args.long_m}) "
        f"max_hold={args.max_hold} top_k={args.top_k}"
    )
    print(f"Panel: {close.shape[0]} days x {close.shape[1]} names")
    print("\nSummary:")
    print(summary.round(6).to_string())
    print("\nCost proxy:")
    print(costs.round(6).to_string(index=False))
    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {events_path}")
    print(f"  {yearly_path}")
    print(f"  {costs_path}")
    print(f"  {daily_path}")


if __name__ == "__main__":
    main()
