"""Analyze T+1..T+20 forward paths after brick turn-up events."""
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

from scripts.analyze_turn_burst_samples_qlib import quantile_bucket, rsi, stack_feature
from src.indicators.brick_alpha import compute_brick_alpha
from src.indicators.signal_filters import volume_ratio
from src.research.amv_regime import DEFAULT_AMV_CSV, amv_regime_mask, broadcast_to_panel, load_amv_close
from src.research.qlib_data import calendar_bounds, load_features, warmup_start

DEFAULT_PROVIDER_URI = Path(
    os.environ.get("QLIB_PROVIDER_URI", "/home/x1843/.qlib/qlib_data/cn_data")
)


def forward_calendar_end(provider_uri: Path, end: str, horizon: int) -> str:
    path = provider_uri / "calendars" / "day.txt"
    if horizon <= 0 or not path.exists():
        return end
    dates = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not dates:
        return end
    end_ts = pd.Timestamp(end)
    idx = None
    for i, date in enumerate(dates):
        if pd.Timestamp(date) >= end_ts:
            idx = i
            break
    if idx is None:
        return end
    return dates[min(len(dates) - 1, idx + horizon)]


def nanmax_frame(left: pd.DataFrame | None, right: pd.DataFrame) -> pd.DataFrame:
    if left is None:
        return right.copy()
    values = np.fmax(left.to_numpy(dtype=float), right.to_numpy(dtype=float))
    return pd.DataFrame(values, index=left.index, columns=left.columns)


def abnormal_forward_day_mask(close: pd.DataFrame, max_horizon: int, threshold: float | None) -> pd.DataFrame:
    bad = pd.DataFrame(False, index=close.index, columns=close.columns)
    if threshold is None:
        return bad
    for h in range(1, max_horizon + 1):
        day_ret = close.shift(-h) / close.shift(-(h - 1)) - 1.0
        bad = bad | day_ret.abs().gt(threshold).fillna(False)
    return bad


def event_path_frame(
    *,
    market: str,
    window_label: str,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    events: pd.DataFrame,
    amv: pd.DataFrame,
    max_horizon: int,
    max_valid_day_ret: float | None,
    quantiles: int,
) -> pd.DataFrame:
    alpha = compute_brick_alpha(high=high, low=low, close=close)
    entry_open = open_.shift(-1)
    t1_high = high.shift(-1)
    t1_close = close.shift(-1)
    t1_open_gap = entry_open / close - 1.0
    features: dict[str, pd.DataFrame] = {
        "amv_regime": amv.astype(float),
        "brick_value": alpha.brick_value,
        "ret3_to_t": close / close.shift(3) - 1.0,
        "ret5_to_t": close / close.shift(5) - 1.0,
        "ret10_to_t": close / close.shift(10) - 1.0,
        "rsi6": rsi(close, 6),
        "volume_ratio20": volume_ratio(volume, 20),
        "t1_open_gap": t1_open_gap,
        "t1_open_to_close": t1_close / entry_open - 1.0,
        "t1_close_from_high": t1_close / t1_high - 1.0,
        "bad_forward_day_ret": abnormal_forward_day_mask(close, max_horizon, max_valid_day_ret).astype(float),
    }

    running_max_high: pd.DataFrame | None = None
    for h in range(1, max_horizon + 1):
        future_close = close.shift(-h)
        future_high = high.shift(-h)
        running_max_high = nanmax_frame(running_max_high, future_high)
        features[f"close_ret_t{h}"] = future_close / close - 1.0
        features[f"open_entry_ret_t{h}"] = future_close / entry_open - 1.0
        features[f"max_runup_t1_open_t{h}"] = running_max_high / entry_open - 1.0
        features[f"close_vs_peak_t{h}"] = future_close / running_max_high - 1.0

    parts = [stack_feature(frame, events, name) for name, frame in features.items()]
    out = pd.concat(parts, axis=1).reset_index()
    out.insert(0, "market", market)
    out.insert(1, "window", window_label)
    out["signal_date"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d")
    out = out.drop(columns=["datetime"])
    out = out[out["bad_forward_day_ret"].lt(0.5)].drop(columns=["bad_forward_day_ret"])

    if max_horizon >= 20:
        out["late_ret_5_20"] = (1.0 + out["close_ret_t20"]) / (1.0 + out["close_ret_t5"]) - 1.0
        out["late_ret_10_20"] = (1.0 + out["close_ret_t20"]) / (1.0 + out["close_ret_t10"]) - 1.0
        out["t20_minus_t5_close_ret"] = out["close_ret_t20"] - out["close_ret_t5"]
        out["early_peak5_to_t20_close"] = (
            (1.0 + out["open_entry_ret_t20"]) / (1.0 + out["max_runup_t1_open_t5"]) - 1.0
        )
    out["market_regime"] = np.where(out["amv_regime"] >= 0.5, "amv_bull", "amv_bear")
    out["ret3_bucket"] = quantile_bucket(out, "ret3_to_t", q=quantiles)
    out = out.dropna(subset=["ret3_bucket"]).copy()
    out["strength_group"] = np.select(
        [out["ret3_bucket"].eq("Q1_low"), out["ret3_bucket"].eq(f"Q{quantiles}_high")],
        ["oversold", "strong"],
        default="middle",
    )
    return out.sort_values(["signal_date", "instrument"]).reset_index(drop=True)


def _group_iterator(events: pd.DataFrame, group_cols: list[str]):
    for keys, group in events.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        yield dict(zip(group_cols, keys)), group


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def _median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _win_rate(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.gt(0).mean()) if not values.empty else np.nan


def horizon_summary(
    events: pd.DataFrame,
    *,
    group_cols: list[str],
    horizons: range,
) -> pd.DataFrame:
    rows = []
    for key_map, group in _group_iterator(events, group_cols):
        for h in horizons:
            close_col = f"close_ret_t{h}"
            entry_col = f"open_entry_ret_t{h}"
            runup_col = f"max_runup_t1_open_t{h}"
            peak_col = f"close_vs_peak_t{h}"
            valid = group[close_col].notna()
            g = group.loc[valid]
            rows.append(
                {
                    **key_map,
                    "horizon": h,
                    "events": int(g.shape[0]),
                    "avg_ret3_to_t": _mean(g["ret3_to_t"]),
                    "avg_t1_open_gap": _mean(g["t1_open_gap"]),
                    "gap_up_rate": float(g["t1_open_gap"].gt(0).mean()) if not g.empty else np.nan,
                    "avg_t1_open_to_close": _mean(g["t1_open_to_close"]),
                    "avg_t1_close_from_high": _mean(g["t1_close_from_high"]),
                    "avg_close_ret": _mean(g[close_col]),
                    "median_close_ret": _median(g[close_col]),
                    "close_win_rate": _win_rate(g[close_col]),
                    "avg_open_entry_ret": _mean(g[entry_col]),
                    "median_open_entry_ret": _median(g[entry_col]),
                    "open_entry_win_rate": _win_rate(g[entry_col]),
                    "avg_max_runup_t1_open": _mean(g[runup_col]),
                    "avg_close_vs_peak": _mean(g[peak_col]),
                }
            )
    return pd.DataFrame(rows)


def key_summary(events: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key_map, group in _group_iterator(events, group_cols):
        row = {
            **key_map,
            "events": int(group.shape[0]),
            "avg_ret3_to_t": _mean(group["ret3_to_t"]),
            "avg_rsi6": _mean(group["rsi6"]),
            "avg_volume_ratio20": _mean(group["volume_ratio20"]),
            "avg_t1_open_gap": _mean(group["t1_open_gap"]),
            "gap_up_rate": float(group["t1_open_gap"].gt(0).mean()) if not group.empty else np.nan,
            "avg_t1_open_to_close": _mean(group["t1_open_to_close"]),
            "avg_t1_close_from_high": _mean(group["t1_close_from_high"]),
            "avg_late_ret_5_20": _mean(group["late_ret_5_20"]) if "late_ret_5_20" in group else np.nan,
            "median_late_ret_5_20": _median(group["late_ret_5_20"]) if "late_ret_5_20" in group else np.nan,
            "late_5_20_win_rate": _win_rate(group["late_ret_5_20"]) if "late_ret_5_20" in group else np.nan,
            "valid_late_ret_5_20": (
                int(group["late_ret_5_20"].notna().sum()) if "late_ret_5_20" in group else 0
            ),
            "avg_early_peak5_to_t20_close": (
                _mean(group["early_peak5_to_t20_close"]) if "early_peak5_to_t20_close" in group else np.nan
            ),
        }
        for h in (5, 10, 20):
            for prefix in ("close_ret", "open_entry_ret", "max_runup_t1_open", "close_vs_peak"):
                col = f"{prefix}_t{h}"
                if col in group:
                    row[f"avg_{col}"] = _mean(group[col])
                    row[f"median_{col}"] = _median(group[col])
            close_col = f"close_ret_t{h}"
            entry_col = f"open_entry_ret_t{h}"
            if close_col in group:
                row[f"valid_close_ret_t{h}"] = int(group[close_col].notna().sum())
                row[f"close_t{h}_win_rate"] = _win_rate(group[close_col])
            if entry_col in group:
                row[f"valid_open_entry_ret_t{h}"] = int(group[entry_col].notna().sum())
                row[f"open_entry_t{h}_win_rate"] = _win_rate(group[entry_col])
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    cal_start, cal_end = calendar_bounds(DEFAULT_PROVIDER_URI)
    default_start = "2019-01-01"
    if cal_start and default_start < cal_start:
        default_start = cal_start
    parser = argparse.ArgumentParser(description="Analyze T+1..T+20 paths after brick turn-up.")
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--start", default=default_start)
    parser.add_argument("--end", default=cal_end or "2020-09-25")
    parser.add_argument("--window-label", default="")
    parser.add_argument("--turn-mode", choices=["xg", "strict_green_to_red"], default="xg")
    parser.add_argument("--amv-csv", type=Path, default=DEFAULT_AMV_CSV)
    parser.add_argument("--amv-sma", type=int, default=60)
    parser.add_argument("--warmup-bars", type=int, default=260)
    parser.add_argument("--max-horizon", type=int, default=20)
    parser.add_argument("--summary-start-horizon", type=int, default=5)
    parser.add_argument("--max-valid-day-ret", type=float, default=0.205)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "results" / "turn_forward_paths",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    window_label = args.window_label or f"{args.market}_{args.start[:4]}_{args.end[:4]}"
    load_start = warmup_start(args.provider_uri, args.start, args.warmup_bars)
    load_end = forward_calendar_end(args.provider_uri, args.end, args.max_horizon)

    feats = load_features(
        args.provider_uri,
        args.market,
        load_start,
        load_end,
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
    event_keep = (close_f.index >= start_ts) & (close_f.index <= end_ts)
    previous_green = alpha_full.falling.shift(1, fill_value=False).reindex(
        index=close_f.index,
        columns=close_f.columns,
        fill_value=False,
    )
    if args.turn_mode == "strict_green_to_red":
        events_full = (alpha_full.rising & previous_green).fillna(False).astype(bool)
    else:
        events_full = alpha_full.turn_up.fillna(False).astype(bool)
    events = events_full.copy()
    events.loc[~event_keep, :] = False
    events = events.fillna(False).astype(bool)

    amv = broadcast_to_panel(
        amv_regime_mask(load_amv_close(args.amv_csv), sma_window=args.amv_sma),
        close_f.index,
        close_f.columns,
    )

    paths = event_path_frame(
        market=args.market,
        window_label=window_label,
        open_=open_f,
        high=high_f,
        low=low_f,
        close=close_f,
        volume=volume_f,
        events=events,
        amv=amv,
        max_horizon=args.max_horizon,
        max_valid_day_ret=args.max_valid_day_ret,
        quantiles=args.quantiles,
    )
    paths = paths[paths["signal_date"].between(args.start, args.end)].copy()

    horizon_range = range(args.summary_start_horizon, args.max_horizon + 1)
    by_bucket = horizon_summary(
        paths,
        group_cols=["market", "window", "ret3_bucket", "strength_group"],
        horizons=horizon_range,
    )
    by_bucket_regime = horizon_summary(
        paths,
        group_cols=["market", "window", "market_regime", "ret3_bucket", "strength_group"],
        horizons=horizon_range,
    )
    key = key_summary(paths, group_cols=["market", "window", "ret3_bucket", "strength_group"])
    key_regime = key_summary(
        paths,
        group_cols=["market", "window", "market_regime", "ret3_bucket", "strength_group"],
    )

    events_path = args.output_prefix.with_name(args.output_prefix.name + "_events.csv")
    horizon_path = args.output_prefix.with_name(args.output_prefix.name + "_horizon_summary.csv")
    horizon_regime_path = args.output_prefix.with_name(
        args.output_prefix.name + "_horizon_summary_regime.csv"
    )
    key_path = args.output_prefix.with_name(args.output_prefix.name + "_key_summary.csv")
    key_regime_path = args.output_prefix.with_name(args.output_prefix.name + "_key_summary_regime.csv")

    paths.to_csv(events_path, index=False)
    by_bucket.to_csv(horizon_path, index=False)
    by_bucket_regime.to_csv(horizon_regime_path, index=False)
    key.to_csv(key_path, index=False)
    key_regime.to_csv(key_regime_path, index=False)

    pd.set_option("display.width", 220)
    print(
        f"Market: {args.market}  window: {window_label}  "
        f"events: {len(paths):,}  load: {load_start} -> {load_end}"
    )
    print(f"Turn mode: {args.turn_mode}  max_horizon={args.max_horizon}  q={args.quantiles}")
    if args.max_valid_day_ret is not None:
        print(f"Forward clean filter: abs(each T+1..T+{args.max_horizon} close-day return) <= {args.max_valid_day_ret:.2%}")
    print("\nKey summary:")
    show_cols = [
        "ret3_bucket",
        "strength_group",
        "events",
        "avg_t1_open_gap",
        "avg_t1_open_to_close",
        "avg_close_ret_t5",
        "avg_open_entry_ret_t5",
        "avg_close_ret_t20",
        "avg_open_entry_ret_t20",
        "avg_late_ret_5_20",
        "avg_early_peak5_to_t20_close",
    ]
    print(key[show_cols].round(6).to_string(index=False))
    print("\nSaved:")
    print(f"  {events_path}")
    print(f"  {horizon_path}")
    print(f"  {horizon_regime_path}")
    print(f"  {key_path}")
    print(f"  {key_regime_path}")


if __name__ == "__main__":
    main()
