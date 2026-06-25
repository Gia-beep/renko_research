"""Analyze brick turn-up samples that burst on T+1 or T+2.

The event is the brick-alpha green/red turn proxy from
``src.indicators.brick_alpha``.  By default this script uses the Tongdaxin
``XG`` definition (``turn_up``); pass ``--turn-mode strict_green_to_red`` to
require the previous bar to be an actual falling/green brick.
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
from src.indicators.signal_filters import moving_average, strong_red_body, volume_ratio
from src.research.amv_regime import DEFAULT_AMV_CSV, amv_regime_mask, broadcast_to_panel, load_amv_close
from src.research.metrics import stack_non_null
from src.research.qlib_data import calendar_bounds, load_features, warmup_start

DEFAULT_PROVIDER_URI = Path(
    os.environ.get("QLIB_PROVIDER_URI", "/home/x1843/.qlib/qlib_data/cn_data")
)

INTERACTION_FEATURES = (
    "brick_value",
    "rsi6",
    "ret3_to_t",
    "ret5_to_t",
    "ret10_to_t",
    "volume_ratio20",
)


def rsi(close: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.where(avg_loss > 0)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(avg_loss > 0, 100.0)


def ratio_frame(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    return numerator / denominator.where(denominator != 0)


def stack_feature(frame: pd.DataFrame, events: pd.DataFrame, name: str) -> pd.Series:
    out = stack_non_null(frame.where(events)).rename(name)
    out.index = out.index.set_names(["datetime", "instrument"])
    return out


def event_sample_frame(
    *,
    market: str,
    window_label: str,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    events: pd.DataFrame,
    previous_green: pd.DataFrame,
    amv: pd.DataFrame,
    burst_threshold: float,
    burst_metric: str,
) -> pd.DataFrame:
    alpha = compute_brick_alpha(high=high, low=low, close=close)
    spread = (high - low).where(high > low)
    ret1 = close / close.shift(1) - 1.0
    t1_close_ret = close.shift(-1) / close - 1.0
    t2_close_day_ret = close.shift(-2) / close.shift(-1) - 1.0
    t2_cum_close_ret = close.shift(-2) / close - 1.0
    t1_high_from_signal = high.shift(-1) / close - 1.0
    t2_high_from_signal = high.shift(-2) / close - 1.0
    t1_open_gap = open_.shift(-1) / close - 1.0
    t2_open_from_signal = open_.shift(-2) / close - 1.0
    upper_shadow = (high - open_.where(open_ >= close, close)) / spread
    lower_shadow = (open_.where(open_ <= close, close) - low) / spread
    body_ratio = (close - open_) / spread
    close_pos = (close - low) / spread
    ma20 = moving_average(close, 20)
    ma60 = moving_average(close, 60)
    features = {
        "prev_green": previous_green.astype(float),
        "amv_regime": amv.astype(float),
        "brick_value": alpha.brick_value,
        "brick_delta": alpha.brick_value.fillna(0.0).diff(),
        "ret_t": ret1,
        "ret3_to_t": close / close.shift(3) - 1.0,
        "ret5_to_t": close / close.shift(5) - 1.0,
        "ret10_to_t": close / close.shift(10) - 1.0,
        "rsi6": rsi(close, 6),
        "volume_ratio20": volume_ratio(volume, 20),
        "body_ratio": body_ratio,
        "upper_shadow": upper_shadow,
        "lower_shadow": lower_shadow,
        "close_pos": close_pos,
        "strong_red_body": strong_red_body(open_, high, low, close).astype(float),
        "above_ma20": (close > ma20).astype(float),
        "ma20_above_ma60": (ma20 > ma60).astype(float),
        "t1_open_gap": t1_open_gap,
        "t1_close_ret": t1_close_ret,
        "t2_close_day_ret": t2_close_day_ret,
        "t2_cum_close_ret": t2_cum_close_ret,
        "t1_high_from_signal": t1_high_from_signal,
        "t2_open_from_signal": t2_open_from_signal,
        "t2_high_from_signal": t2_high_from_signal,
    }

    parts = [stack_feature(frame, events, name) for name, frame in features.items()]
    out = pd.concat(parts, axis=1).reset_index()
    out.insert(0, "market", market)
    out.insert(1, "window", window_label)
    out["signal_date"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d")
    out = out.drop(columns=["datetime"])

    if burst_metric == "close_day":
        t1_hit = out["t1_close_ret"] >= burst_threshold
        t2_hit = out["t2_close_day_ret"] >= burst_threshold
        out["max_burst_ret"] = out[["t1_close_ret", "t2_close_day_ret"]].max(axis=1)
    elif burst_metric == "high_from_signal":
        t1_hit = out["t1_high_from_signal"] >= burst_threshold
        t2_hit = out["t2_high_from_signal"] >= burst_threshold
        out["max_burst_ret"] = out[["t1_high_from_signal", "t2_high_from_signal"]].max(axis=1)
    elif burst_metric == "either":
        t1_hit = (out["t1_close_ret"] >= burst_threshold) | (out["t1_high_from_signal"] >= burst_threshold)
        t2_hit = (out["t2_close_day_ret"] >= burst_threshold) | (out["t2_high_from_signal"] >= burst_threshold)
        out["max_burst_ret"] = out[
            ["t1_close_ret", "t2_close_day_ret", "t1_high_from_signal", "t2_high_from_signal"]
        ].max(axis=1)
    else:
        raise ValueError(f"unsupported burst_metric={burst_metric}")

    out["burst"] = t1_hit | t2_hit
    out["burst_day"] = np.select(
        [t1_hit, ~t1_hit & t2_hit],
        ["T+1", "T+2"],
        default="",
    )
    out["burst_threshold"] = burst_threshold
    out["burst_metric"] = burst_metric
    return out.sort_values(["signal_date", "instrument"]).reset_index(drop=True)


def feature_contrast(events: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "prev_green",
        "amv_regime",
        "brick_value",
        "brick_delta",
        "ret_t",
        "ret3_to_t",
        "ret5_to_t",
        "ret10_to_t",
        "rsi6",
        "volume_ratio20",
        "body_ratio",
        "upper_shadow",
        "lower_shadow",
        "close_pos",
        "strong_red_body",
        "above_ma20",
        "ma20_above_ma60",
        "t1_open_gap",
        "t1_close_ret",
        "t2_close_day_ret",
        "t2_cum_close_ret",
        "t1_high_from_signal",
        "t2_open_from_signal",
        "t2_high_from_signal",
    ]
    rows = []
    for col in numeric_cols:
        non = events.loc[~events["burst"], col].dropna()
        hit = events.loc[events["burst"], col].dropna()
        rows.append(
            {
                "feature": col,
                "non_burst_mean": float(non.mean()) if not non.empty else np.nan,
                "burst_mean": float(hit.mean()) if not hit.empty else np.nan,
                "mean_diff": float(hit.mean() - non.mean()) if not non.empty and not hit.empty else np.nan,
                "non_burst_median": float(non.median()) if not non.empty else np.nan,
                "burst_median": float(hit.median()) if not hit.empty else np.nan,
                "median_diff": float(hit.median() - non.median()) if not non.empty and not hit.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summary_frame(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["market", "window"]
    for keys, group in events.groupby(group_cols, dropna=False):
        market, window = keys
        bursts = group[group["burst"]]
        rows.append(
            {
                "market": market,
                "window": window,
                "events": int(group.shape[0]),
                "burst_count": int(bursts.shape[0]),
                "burst_rate": float(group["burst"].mean()),
                "t1_burst_count": int((group["burst_day"] == "T+1").sum()),
                "t2_burst_count": int((group["burst_day"] == "T+2").sum()),
                "avg_t1_close_ret": float(group["t1_close_ret"].mean()),
                "avg_t2_close_day_ret": float(group["t2_close_day_ret"].mean()),
                "avg_t2_cum_close_ret": float(group["t2_cum_close_ret"].mean()),
                "burst_avg_max_ret": float(bursts["max_burst_ret"].mean()) if not bursts.empty else np.nan,
                "burst_median_max_ret": float(bursts["max_burst_ret"].median()) if not bursts.empty else np.nan,
            }
        )
    total = events
    bursts = total[total["burst"]]
    rows.append(
        {
            "market": "ALL",
            "window": "ALL",
            "events": int(total.shape[0]),
            "burst_count": int(bursts.shape[0]),
            "burst_rate": float(total["burst"].mean()),
            "t1_burst_count": int((total["burst_day"] == "T+1").sum()),
            "t2_burst_count": int((total["burst_day"] == "T+2").sum()),
            "avg_t1_close_ret": float(total["t1_close_ret"].mean()),
            "avg_t2_close_day_ret": float(total["t2_close_day_ret"].mean()),
            "avg_t2_cum_close_ret": float(total["t2_cum_close_ret"].mean()),
            "burst_avg_max_ret": float(bursts["max_burst_ret"].mean()) if not bursts.empty else np.nan,
            "burst_median_max_ret": float(bursts["max_burst_ret"].median()) if not bursts.empty else np.nan,
        }
    )
    return pd.DataFrame(rows)


def daily_burst_frame(events: pd.DataFrame) -> pd.DataFrame:
    daily = events.groupby(["market", "window", "signal_date"], dropna=False).agg(
        events=("burst", "size"),
        burst_count=("burst", "sum"),
        burst_rate=("burst", "mean"),
        avg_max_burst_ret=("max_burst_ret", "mean"),
    )
    return daily.reset_index().sort_values(["burst_count", "burst_rate"], ascending=[False, False])


def quantile_bucket(
    events: pd.DataFrame,
    feature: str,
    *,
    q: int = 5,
    within: tuple[str, ...] = ("market", "window"),
) -> pd.Series:
    """Assign causal-analysis buckets within each market/window sample.

    The local Qlib bundle does not include daily market-cap fields, so this
    script uses ``market`` (CSI300 vs CSI500) as a coarse size proxy.  Feature
    buckets are computed inside each market/window to avoid one period's level
    shift dominating another period's bucket assignment.
    """
    if q <= 1:
        raise ValueError("q must be greater than 1")

    out = pd.Series(pd.NA, index=events.index, dtype="object")
    labels = [f"Q{i}_{'low' if i == 1 else 'high' if i == q else 'mid'}" for i in range(1, q + 1)]

    for _, idx in events.groupby(list(within), dropna=False).groups.items():
        values = pd.to_numeric(events.loc[idx, feature], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue
        bins = min(q, valid.size)
        if bins == 1:
            out.loc[valid.index] = labels[0]
            continue
        ranks = valid.rank(method="first")
        bucket = pd.qcut(ranks, q=bins, labels=labels[:bins])
        out.loc[valid.index] = bucket.astype("object")
    return out


def burst_group_summary(
    events: pd.DataFrame,
    group_cols: list[str],
    *,
    feature: str | None = None,
) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, str]] = {
        "events": ("burst", "size"),
        "burst_count": ("burst", "sum"),
        "burst_rate": ("burst", "mean"),
        "avg_t1_close_ret": ("t1_close_ret", "mean"),
        "avg_t2_close_day_ret": ("t2_close_day_ret", "mean"),
        "avg_t2_cum_close_ret": ("t2_cum_close_ret", "mean"),
        "avg_max_burst_ret": ("max_burst_ret", "mean"),
        "median_max_burst_ret": ("max_burst_ret", "median"),
        "amv_bull_rate": ("amv_regime", "mean"),
    }
    if feature is not None:
        agg_spec["feature_mean"] = (feature, "mean")
        agg_spec["feature_median"] = (feature, "median")

    grouped = events.groupby(group_cols, dropna=False)
    out = grouped.agg(**agg_spec).reset_index()
    t1 = grouped["burst_day"].apply(lambda x: int((x == "T+1").sum())).rename("t1_burst_count")
    t2 = grouped["burst_day"].apply(lambda x: int((x == "T+2").sum())).rename("t2_burst_count")
    out = out.merge(t1.reset_index(), on=group_cols, how="left")
    out = out.merge(t2.reset_index(), on=group_cols, how="left")
    out["t1_share_of_bursts"] = out["t1_burst_count"] / out["burst_count"].where(out["burst_count"] > 0)
    out["t2_share_of_bursts"] = out["t2_burst_count"] / out["burst_count"].where(out["burst_count"] > 0)
    out["events"] = out["events"].astype(int)
    out["burst_count"] = out["burst_count"].astype(int)
    return out


def interaction_frames(
    events: pd.DataFrame,
    *,
    q: int = 5,
    features: tuple[str, ...] = INTERACTION_FEATURES,
    include_window: bool = True,
) -> dict[str, pd.DataFrame]:
    base = events.copy()
    base["market_regime"] = np.where(base["amv_regime"] >= 0.5, "amv_bull", "amv_bear")
    size_cols = ["market", "window"] if include_window else ["market"]

    regime = burst_group_summary(base, size_cols + ["market_regime"])
    momentum_parts = []
    regime_momentum_parts = []
    for feature in features:
        frame = base.copy()
        frame["feature"] = feature
        frame["feature_bucket"] = quantile_bucket(frame, feature, q=q)
        frame = frame.dropna(subset=["feature_bucket"])
        momentum_parts.append(
            burst_group_summary(
                frame,
                size_cols + ["feature", "feature_bucket"],
                feature=feature,
            )
        )
        regime_momentum_parts.append(
            burst_group_summary(
                frame,
                size_cols + ["market_regime", "feature", "feature_bucket"],
                feature=feature,
            )
        )

    momentum = pd.concat(momentum_parts, ignore_index=True)
    regime_momentum = pd.concat(regime_momentum_parts, ignore_index=True)
    return {
        "regime": regime.sort_values(size_cols + ["market_regime"]).reset_index(drop=True),
        "size_momentum": momentum.sort_values(size_cols + ["feature", "feature_bucket"]).reset_index(drop=True),
        "size_momentum_regime": regime_momentum.sort_values(
            size_cols + ["market_regime", "feature", "feature_bucket"]
        ).reset_index(drop=True),
        "shape": interaction_shape_summary(momentum, size_cols=size_cols),
    }


def interaction_shape_summary(momentum: pd.DataFrame, *, size_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in momentum.groupby(size_cols + ["feature"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(size_cols + ["feature"], keys))
        ordered = group.assign(
            bucket_num=group["feature_bucket"].astype(str).str.extract(r"Q(\d+)_")[0].astype(int)
        ).sort_values("bucket_num")
        if ordered.empty:
            continue
        low = ordered.iloc[0]
        high = ordered.iloc[-1]
        mid = ordered.iloc[len(ordered) // 2]
        best = ordered.sort_values(["burst_rate", "events"], ascending=[False, False]).iloc[0]
        rows.append(
            {
                **key_map,
                "low_bucket": low["feature_bucket"],
                "mid_bucket": mid["feature_bucket"],
                "high_bucket": high["feature_bucket"],
                "low_burst_rate": float(low["burst_rate"]),
                "mid_burst_rate": float(mid["burst_rate"]),
                "high_burst_rate": float(high["burst_rate"]),
                "high_minus_low": float(high["burst_rate"] - low["burst_rate"]),
                "high_minus_mid": float(high["burst_rate"] - mid["burst_rate"]),
                "low_minus_mid": float(low["burst_rate"] - mid["burst_rate"]),
                "best_bucket": best["feature_bucket"],
                "best_burst_rate": float(best["burst_rate"]),
                "events": int(ordered["events"].sum()),
                "burst_count": int(ordered["burst_count"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(size_cols + ["feature"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    cal_start, cal_end = calendar_bounds(DEFAULT_PROVIDER_URI)
    default_start = "2019-01-01"
    if cal_start and default_start < cal_start:
        default_start = cal_start
    parser = argparse.ArgumentParser(description="Analyze T+1/T+2 burst samples after brick turn-up.")
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--start", default=default_start)
    parser.add_argument("--end", default=cal_end or "2020-09-25")
    parser.add_argument("--window-label", default="")
    parser.add_argument(
        "--turn-mode",
        choices=["xg", "strict_green_to_red"],
        default="xg",
        help="xg uses Tongdaxin turn_up; strict_green_to_red also requires the previous bar to be falling/green.",
    )
    parser.add_argument("--burst-threshold", type=float, default=0.05)
    parser.add_argument(
        "--max-valid-day-ret",
        type=float,
        default=None,
        help="Drop events whose T+1 or T+2 single-day close return exceeds this absolute value.",
    )
    parser.add_argument(
        "--burst-metric",
        choices=["close_day", "high_from_signal", "either"],
        default="close_day",
    )
    parser.add_argument("--amv-csv", type=Path, default=DEFAULT_AMV_CSV)
    parser.add_argument("--amv-sma", type=int, default=60)
    parser.add_argument("--warmup-bars", type=int, default=260)
    parser.add_argument("--top-samples", type=int, default=200)
    parser.add_argument("--interaction-quantiles", type=int, default=5)
    parser.add_argument("--save-all-events", action="store_true")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "results" / "turn_burst_samples",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    window_label = args.window_label or f"{args.market}_{args.start[:4]}_{args.end[:4]}"

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

    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)
    keep = (close_f.index >= start_ts) & (close_f.index <= end_ts)
    open_ = open_f.loc[keep]
    high = high_f.loc[keep]
    low = low_f.loc[keep]
    close = close_f.loc[keep]
    volume = volume_f.loc[keep]
    alpha = replace(
        alpha_full,
        brick_value=alpha_full.brick_value.loc[keep],
        rising=alpha_full.rising.loc[keep],
        falling=alpha_full.falling.loc[keep],
        turn_up=alpha_full.turn_up.loc[keep],
        turn_down=alpha_full.turn_down.loc[keep],
    )
    previous_green = alpha_full.falling.shift(1, fill_value=False).loc[keep].reindex(
        index=close.index,
        columns=close.columns,
        fill_value=False,
    )
    if args.turn_mode == "strict_green_to_red":
        events = (alpha.rising & previous_green).fillna(False).astype(bool)
    else:
        events = alpha.turn_up.fillna(False).astype(bool)

    amv = broadcast_to_panel(
        amv_regime_mask(load_amv_close(args.amv_csv), sma_window=args.amv_sma),
        close.index,
        close.columns,
    )

    event_rows = event_sample_frame(
        market=args.market,
        window_label=window_label,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        events=events,
        previous_green=previous_green,
        amv=amv,
        burst_threshold=args.burst_threshold,
        burst_metric=args.burst_metric,
    )
    if args.max_valid_day_ret is not None:
        valid = (
            event_rows["t1_close_ret"].abs().le(args.max_valid_day_ret)
            & event_rows["t2_close_day_ret"].abs().le(args.max_valid_day_ret)
        )
        event_rows = event_rows[valid].copy()

    burst_samples = event_rows[event_rows["burst"]].sort_values(
        ["max_burst_ret", "t2_cum_close_ret"],
        ascending=[False, False],
    )
    top_samples = burst_samples.head(args.top_samples)
    summary = summary_frame(event_rows)
    contrast = feature_contrast(event_rows)
    daily = daily_burst_frame(event_rows)
    interactions = interaction_frames(event_rows, q=args.interaction_quantiles)

    sample_path = args.output_prefix.with_name(args.output_prefix.name + "_burst_samples.csv")
    top_path = args.output_prefix.with_name(args.output_prefix.name + "_top_samples.csv")
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    contrast_path = args.output_prefix.with_name(args.output_prefix.name + "_feature_contrast.csv")
    daily_path = args.output_prefix.with_name(args.output_prefix.name + "_daily.csv")
    events_path = args.output_prefix.with_name(args.output_prefix.name + "_events.csv")
    regime_path = args.output_prefix.with_name(args.output_prefix.name + "_interaction_regime.csv")
    size_momentum_path = args.output_prefix.with_name(args.output_prefix.name + "_interaction_size_momentum.csv")
    size_momentum_regime_path = args.output_prefix.with_name(
        args.output_prefix.name + "_interaction_size_momentum_regime.csv"
    )
    shape_path = args.output_prefix.with_name(args.output_prefix.name + "_interaction_shape.csv")

    burst_samples.to_csv(sample_path, index=False)
    top_samples.to_csv(top_path, index=False)
    summary.to_csv(summary_path, index=False)
    contrast.to_csv(contrast_path, index=False)
    daily.to_csv(daily_path, index=False)
    interactions["regime"].to_csv(regime_path, index=False)
    interactions["size_momentum"].to_csv(size_momentum_path, index=False)
    interactions["size_momentum_regime"].to_csv(size_momentum_regime_path, index=False)
    interactions["shape"].to_csv(shape_path, index=False)
    if args.save_all_events:
        event_rows.to_csv(events_path, index=False)

    pd.set_option("display.width", 180)
    print(
        f"Market: {args.market}  window: {window_label}  "
        f"range: {close.index.min().date()} -> {close.index.max().date()}"
    )
    print(f"Turn mode: {args.turn_mode}  burst: {args.burst_metric} >= {args.burst_threshold:.2%}")
    if args.max_valid_day_ret is not None:
        print(f"Valid-day filter: abs(T+1/T+2 close-day return) <= {args.max_valid_day_ret:.2%}")
    print("\nSummary:")
    print(summary.round(6).to_string(index=False))
    print("\nTop burst samples:")
    show_cols = [
        "signal_date",
        "instrument",
        "burst_day",
        "max_burst_ret",
        "t1_close_ret",
        "t2_close_day_ret",
        "t2_cum_close_ret",
        "ret3_to_t",
        "rsi6",
        "volume_ratio20",
        "brick_delta",
        "prev_green",
    ]
    print(top_samples[show_cols].head(20).round(6).to_string(index=False))
    print("\nSaved:")
    print(f"  {sample_path}")
    print(f"  {top_path}")
    print(f"  {summary_path}")
    print(f"  {contrast_path}")
    print(f"  {daily_path}")
    print(f"  {regime_path}")
    print(f"  {size_momentum_path}")
    print(f"  {size_momentum_regime_path}")
    print(f"  {shape_path}")
    if args.save_all_events:
        print(f"  {events_path}")


if __name__ == "__main__":
    main()
