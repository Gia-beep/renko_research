"""Search for higher-quality long-only reversal signals with Qlib data.

This script tests interpretable reversal candidates rather than individual-stock
short legs.  Signals are long-only entry masks built from:

  - short-term losers / drawdown ranks,
  - RSI oversold recovery,
  - capitulation volume and lower-shadow candles,
  - brick ``turn_up`` after a pullback.

For each signal and forward horizon, it reports event-level return quality and
an equal-weight "one-period portfolio" quality metric.  Forward returns use a
one-bar entry delay by default, matching practical A-share next-day execution.

Example:
    /home/x1843/venvs/qlib/bin/python scripts/research_reversal_signals_qlib.py \
        --market csi500 --start 2019-01-01 --end 2020-09-25 \
        --output-prefix results/reversal_signals_csi500_2019
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indicators.brick_alpha import compute_brick_alpha
from src.indicators.signal_filters import (
    combine,
    moving_average,
    volume_ratio,
    bars_since_last_turn_down,
    preceding_downtrend_depth,
    preceding_brick_drop,
    atr_ratio,
    close_above_vwap,
    bullish_divergence,
    market_breadth_filter,
    strict_candle_filter,
)
from src.research.amv_regime import DEFAULT_AMV_CSV, amv_regime_mask, broadcast_to_panel, load_amv_close
from src.research.metrics import stack_non_null
from src.research.qlib_data import calendar_bounds, load_features, warmup_start

DEFAULT_PROVIDER_URI = Path(
    os.environ.get("QLIB_PROVIDER_URI", "/home/x1843/.qlib/qlib_data/cn_data")
)


def delayed_forward_returns(close: pd.DataFrame, *, horizon: int, entry_shift: int) -> pd.DataFrame:
    """Return from delayed entry close to ``horizon`` bars later."""
    return close.shift(-(entry_shift + horizon)) / close.shift(-entry_shift) - 1.0


def rsi(close: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Wilder RSI on wide date x instrument close frames."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.where(avg_loss > 0)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(avg_loss > 0, 100.0)


def lower_shadow_ratio(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    """Lower-shadow fraction of the day's range."""
    spread = (high - low).where(high > low)
    body_bottom = close.where(close <= open_, open_)
    return (body_bottom - low) / spread


def bottomk(score: pd.DataFrame, eligible: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep the per-date bottom-k eligible names by score."""
    if k <= 0:
        raise ValueError("k must be positive")
    eligible = eligible.fillna(False).astype(bool)
    masked = score.reindex(index=eligible.index, columns=eligible.columns).where(eligible)
    ranks = masked.rank(axis=1, ascending=True, method="first")
    return (ranks.le(k) & eligible).fillna(False).astype(bool)


def topk(score: pd.DataFrame, eligible: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep the per-date top-k eligible names by score."""
    if k <= 0:
        raise ValueError("k must be positive")
    eligible = eligible.fillna(False).astype(bool)
    masked = score.reindex(index=eligible.index, columns=eligible.columns).where(eligible)
    ranks = masked.rank(axis=1, ascending=False, method="first")
    return (ranks.le(k) & eligible).fillna(False).astype(bool)


def signal_quality(
    signal: pd.DataFrame,
    forward_ret: pd.DataFrame,
    *,
    label: str,
    horizon: int,
) -> dict[str, float | int | str]:
    """Event and daily equal-weight period-return quality for a signal."""
    signal, forward_ret = signal.align(forward_ret, join="inner", axis=None)
    signal = signal.fillna(False).astype(bool)
    counts = signal.sum(axis=1)
    active_dates = counts.index[counts > 0]
    event_rets = stack_non_null(forward_ret.where(signal))
    if len(active_dates):
        benchmark_rets = stack_non_null(forward_ret.loc[active_dates])
        portfolio_rets = (forward_ret.where(signal).sum(axis=1) / counts.replace(0, np.nan)).loc[active_dates]
        portfolio_rets = portfolio_rets.dropna()
    else:
        benchmark_rets = pd.Series(dtype=float)
        portfolio_rets = pd.Series(dtype=float)

    wins = event_rets[event_rets > 0]
    losses = event_rets[event_rets < 0]
    port_wins = portfolio_rets[portfolio_rets > 0]
    port_losses = portfolio_rets[portfolio_rets < 0]
    avg_win = float(wins.mean()) if not wins.empty else np.nan
    avg_loss = float(losses.mean()) if not losses.empty else np.nan
    port_avg_win = float(port_wins.mean()) if not port_wins.empty else np.nan
    port_avg_loss = float(port_losses.mean()) if not port_losses.empty else np.nan

    event_mean = float(event_rets.mean()) if not event_rets.empty else np.nan
    benchmark_mean = float(benchmark_rets.mean()) if not benchmark_rets.empty else np.nan
    return {
        "signal": label,
        "horizon": horizon,
        "event_count": int(event_rets.shape[0]),
        "active_days": int(len(active_dates)),
        "avg_names": float(counts.where(counts > 0).mean()),
        "event_mean": event_mean,
        "event_median": float(event_rets.median()) if not event_rets.empty else np.nan,
        "event_win_rate": float((event_rets > 0).mean()) if not event_rets.empty else np.nan,
        "event_avg_win": avg_win,
        "event_avg_loss": avg_loss,
        "event_win_loss_ratio": avg_win / abs(avg_loss)
        if np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss < 0
        else np.nan,
        "event_profit_factor": float(wins.sum() / abs(losses.sum()))
        if not wins.empty and not losses.empty
        else np.nan,
        "benchmark_mean_same_dates": benchmark_mean,
        "event_excess_mean": event_mean - benchmark_mean
        if np.isfinite(event_mean) and np.isfinite(benchmark_mean)
        else np.nan,
        "portfolio_period_mean": float(portfolio_rets.mean()) if not portfolio_rets.empty else np.nan,
        "portfolio_period_win_rate": float((portfolio_rets > 0).mean()) if not portfolio_rets.empty else np.nan,
        "portfolio_period_avg_win": port_avg_win,
        "portfolio_period_avg_loss": port_avg_loss,
        "portfolio_period_win_loss_ratio": port_avg_win / abs(port_avg_loss)
        if np.isfinite(port_avg_win) and np.isfinite(port_avg_loss) and port_avg_loss < 0
        else np.nan,
        "portfolio_period_profit_factor": float(port_wins.sum() / abs(port_losses.sum()))
        if not port_wins.empty and not port_losses.empty
        else np.nan,
    }


def build_reversal_signals(
    *,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    amv: pd.DataFrame,
    top_n: int,
    vwap: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Construct long-only reversal entry masks."""
    alpha = compute_brick_alpha(high=high, low=low, close=close)
    ret1 = close / close.shift(1) - 1.0
    ret3 = close / close.shift(3) - 1.0
    ret5 = close / close.shift(5) - 1.0
    ret10 = close / close.shift(10) - 1.0
    drawdown20 = close / close.rolling(20, min_periods=20).max() - 1.0
    rsi6 = rsi(close, 6)
    vr20 = volume_ratio(volume, 20)
    ma20 = moving_average(close, 20)
    ma60 = moving_average(close, 60)
    lower_shadow = lower_shadow_ratio(open_, high, low, close)
    body_up = close > open_
    rebound_day = ret1 > 0
    not_limit_down = ret1 > -0.095
    tradable = close.notna() & not_limit_down
    base = combine(amv, tradable)

    # Base groups
    losers3 = bottomk(ret3, base, top_n)
    losers5 = bottomk(ret5, base, top_n)
    losers10 = bottomk(ret10, base, top_n)
    deep_dd20 = bottomk(drawdown20, base, top_n)
    low_rsi6 = bottomk(rsi6, base, top_n)
    strongest_rebound_after_loss3 = topk(ret1, losers3, max(1, top_n // 2))
    strongest_rebound_after_loss = topk(ret1, losers5, max(1, top_n // 2))
    strongest_rebound_after_rsi = topk(ret1, low_rsi6, max(1, top_n // 2))

    # New upgraded components
    turn_down = alpha.turn_down
    bars_since_down = bars_since_last_turn_down(turn_down)
    preceding_drop = preceding_downtrend_depth(close, turn_down, high=high)
    preceding_brick_val_drop = preceding_brick_drop(alpha.brick_value, turn_down)

    atr_comp = atr_ratio(high, low, close, short_window=10, long_window=30)

    if vwap is not None and not vwap.isna().all().all():
        above_vwap = close_above_vwap(close, vwap)
    else:
        above_vwap = close > (high + low) / 2.0

    div_brick = bullish_divergence(close, alpha.brick_value, window=20)
    div_rsi = bullish_divergence(close, rsi6, window=20)

    strict_candle = strict_candle_filter(open_, high, low, close, upper_shadow_max=0.20)
    breadth_turn_up = market_breadth_filter(alpha.turn_up, min_ratio=0.10)

    return {
        # Baseline / legacy signals
        "ret3_bottom30_amv": losers3,
        "ret3_bottom30_rebound": combine(losers3, rebound_day),
        "ret3_bottom30_strong_rebound_top15": strongest_rebound_after_loss3,
        "ret3_bottom30_vol_rebound": combine(losers3, vr20 >= 1.5, rebound_day),
        "ret5_bottom30_amv": losers5,
        "ret10_bottom30_amv": losers10,
        "drawdown20_bottom30_amv": deep_dd20,
        "rsi6_bottom30_amv": low_rsi6,
        "rsi6_bottom30_rebound": combine(low_rsi6, rebound_day),
        "rsi6_bottom30_strong_rebound_top15": strongest_rebound_after_rsi,
        "rsi6_bottom30_vol_rebound": combine(low_rsi6, vr20 >= 1.5, rebound_day),
        "rsi6_bottom30_lower_shadow": combine(low_rsi6, lower_shadow >= 0.35),
        "rsi6_bottom30_bull_shadow": combine(low_rsi6, lower_shadow >= 0.35, body_up),
        "ret5_bottom30_rebound": combine(losers5, rebound_day),
        "ret5_bottom30_strong_rebound_top15": strongest_rebound_after_loss,
        "ret5_bottom30_lower_shadow": combine(losers5, lower_shadow >= 0.35),
        "ret5_bottom30_bull_shadow": combine(losers5, lower_shadow >= 0.35, body_up),
        "ret5_bottom30_vol_rebound": combine(losers5, vr20 >= 1.5, rebound_day),
        "ret5_bottom30_rsi_oversold": combine(losers5, rsi6 < 35),
        "ret5_bottom30_rsi_rebound": combine(losers5, rsi6 < 35, rebound_day),
        "ret5_bottom30_rsi_recover": combine(losers5, rsi6.shift(1) < 35, rsi6 > rsi6.shift(1)),
        "ret10_bottom30_rsi_recover": combine(losers10, rsi6.shift(1) < 35, rsi6 > rsi6.shift(1)),
        "panic_shadow_vol": combine(losers10, lower_shadow >= 0.4, vr20 >= 1.5),
        "uptrend_pullback_ret5": combine(losers5, close > ma60, ma20 > ma60),
        "brick_turn_pullback5": combine(alpha.turn_up, base, ret5 < 0),
        "brick_turn_ret5_bottom30": combine(alpha.turn_up, losers5),
        "brick_turn_rsi6_bottom30": combine(alpha.turn_up, low_rsi6),
        "brick_turn_deep_drawdown20": combine(alpha.turn_up, base, drawdown20 <= -0.08),
        "brick_turn_rsi_recover": combine(alpha.turn_up, base, rsi6.shift(1) < 45, rsi6 > rsi6.shift(1)),

        # Upgraded reversal momentum signals (Refine Turn Strength)
        "turn_up_dur5": combine(alpha.turn_up, base, bars_since_down >= 5),
        "turn_up_dur10": combine(alpha.turn_up, base, bars_since_down >= 10),
        "turn_up_depth8": combine(alpha.turn_up, base, preceding_drop >= 0.08),
        "turn_up_depth12": combine(alpha.turn_up, base, preceding_drop >= 0.12),
        "turn_up_brick_drop20": combine(alpha.turn_up, base, preceding_brick_val_drop >= 20.0),

        # Upgraded signals with Volatility Compression
        "turn_up_atr_compressed": combine(alpha.turn_up, base, atr_comp <= 0.85),
        "turn_up_depth_atr_comp": combine(alpha.turn_up, base, preceding_drop >= 0.08, atr_comp <= 0.90),

        # Upgraded signals with VWAP and Volume confirmation
        "turn_up_vol_vwap": combine(alpha.turn_up, base, vr20 >= 1.5, above_vwap),
        "turn_up_strong_vol_vwap": combine(alpha.turn_up, base, vr20 >= 2.0, above_vwap),

        # Upgraded signals with Divergence
        "turn_up_div_brick": combine(alpha.turn_up, base, div_brick),
        "turn_up_div_rsi": combine(alpha.turn_up, base, div_rsi),
        "turn_up_div_combined": combine(alpha.turn_up, base, div_brick | div_rsi),

        # Upgraded signals with Regime and Micro-Structure
        "turn_up_strict_candle": combine(alpha.turn_up, base, strict_candle),
        "turn_up_breadth10": combine(alpha.turn_up, base, breadth_turn_up),
        "turn_up_breadth_strict_candle": combine(alpha.turn_up, base, breadth_turn_up, strict_candle),
    }


def parse_args() -> argparse.Namespace:
    cal_start, cal_end = calendar_bounds(DEFAULT_PROVIDER_URI)
    default_start = "2019-01-01"
    if cal_start and default_start < cal_start:
        default_start = cal_start
    parser = argparse.ArgumentParser(description="Research long-only reversal signals with Qlib data.")
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--start", default=default_start)
    parser.add_argument("--end", default=cal_end or "2020-09-25")
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--entry-shift", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--amv-csv", type=Path, default=DEFAULT_AMV_CSV)
    parser.add_argument("--amv-sma", type=int, default=60)
    parser.add_argument("--warmup-bars", type=int, default=260)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "results" / "reversal_signals",
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
        fields=("$open", "$high", "$low", "$close", "$volume", "$vwap"),
    )
    open_f, high_f, low_f, close_f, volume_f, vwap_f = (
        feats["$open"],
        feats["$high"],
        feats["$low"],
        feats["$close"],
        feats["$volume"],
        feats["$vwap"],
    )
    amv_full = broadcast_to_panel(
        amv_regime_mask(load_amv_close(args.amv_csv), sma_window=args.amv_sma),
        close_f.index,
        close_f.columns,
    )
    signals_full = build_reversal_signals(
        open_=open_f,
        high=high_f,
        low=low_f,
        close=close_f,
        volume=volume_f,
        vwap=vwap_f,
        amv=amv_full,
        top_n=args.top_n,
    )

    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)
    keep = (close_f.index >= start_ts) & (close_f.index <= end_ts)
    close = close_f.loc[keep]
    signals = {
        name: sig.reindex(index=close.index, columns=close.columns, fill_value=False).fillna(False).astype(bool)
        for name, sig in signals_full.items()
    }

    rows: list[dict[str, float | int | str]] = []
    for horizon in args.horizons:
        fwd = delayed_forward_returns(close_f, horizon=horizon, entry_shift=args.entry_shift)
        fwd = fwd.reindex(index=close.index, columns=close.columns)
        for name, signal in signals.items():
            rows.append(signal_quality(signal, fwd, label=name, horizon=horizon))

    comparison_path = args.output_prefix.with_name(args.output_prefix.name + "_comparison.csv")
    summary_paths: list[Path] = []
    summaries: dict[int, pd.DataFrame] = {}
    comparison = pd.DataFrame(rows)
    comparison.to_csv(comparison_path, index=False)
    for horizon in args.horizons:
        summary = comparison[comparison["horizon"].eq(horizon)].sort_values(
            ["portfolio_period_profit_factor", "event_excess_mean"],
            ascending=[False, False],
        )
        summary_path = args.output_prefix.with_name(args.output_prefix.name + f"_h{horizon}_summary.csv")
        summary.to_csv(summary_path, index=False)
        summary_paths.append(summary_path)
        summaries[horizon] = summary

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 24)
    print(f"Market: {args.market}  range: {close.index.min().date()} -> {close.index.max().date()}")
    print(f"Panel: {close.shape[0]} days x {close.shape[1]} names | top_n={args.top_n} entry_shift={args.entry_shift}")
    summary_columns = [
        "signal",
        "event_count",
        "active_days",
        "avg_names",
        "event_mean",
        "event_win_rate",
        "event_win_loss_ratio",
        "event_profit_factor",
        "event_excess_mean",
        "portfolio_period_mean",
        "portfolio_period_win_rate",
        "portfolio_period_win_loss_ratio",
        "portfolio_period_profit_factor",
    ]
    for horizon in args.horizons:
        print(f"\nHorizon={horizon} summary (sorted by portfolio Profit Factor):")
        print(summaries[horizon][summary_columns].round(6).to_string(index=False))
    print("\nSaved:")
    print(f"  {comparison_path}")
    for summary_path in summary_paths:
        print(f"  {summary_path}")


if __name__ == "__main__":
    main()
