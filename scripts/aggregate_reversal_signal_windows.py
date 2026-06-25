"""Aggregate reversal-signal comparison CSVs across robustness windows."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def infer_window(path: Path) -> str:
    """Infer a readable window label from a comparison CSV filename."""
    name = path.stem
    if name.endswith("_comparison"):
        name = name[: -len("_comparison")]
    for prefix in ("reversal_signals_short_", "reversal_signals_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def aggregate(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, dropna=False)
    out = grouped.agg(
        cases=("portfolio_period_profit_factor", "count"),
        windows=("window", "nunique"),
        min_pf=("portfolio_period_profit_factor", "min"),
        median_pf=("portfolio_period_profit_factor", "median"),
        mean_pf=("portfolio_period_profit_factor", "mean"),
        min_event_excess=("event_excess_mean", "min"),
        mean_event_excess=("event_excess_mean", "mean"),
        mean_port_ret=("portfolio_period_mean", "mean"),
        mean_win_rate=("portfolio_period_win_rate", "mean"),
        mean_avg_names=("avg_names", "mean"),
        total_events=("event_count", "sum"),
    ).reset_index()
    if "horizon" not in group_cols:
        out["horizons"] = grouped["horizon"].nunique().to_numpy()
    return out.sort_values(
        ["min_pf", "median_pf", "mean_port_ret"],
        ascending=[False, False, False],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate reversal-signal robustness windows.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for path in args.inputs:
        df = pd.read_csv(path)
        df["window"] = infer_window(path)
        frames.append(df)
    all_windows = pd.concat(frames, ignore_index=True)
    if args.horizons is not None:
        all_windows = all_windows[all_windows["horizon"].isin(args.horizons)].copy()
    if all_windows.empty:
        raise ValueError("No rows remain after horizon filtering")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    all_path = args.output_prefix.with_name(args.output_prefix.name + "_all_windows.csv")
    by_horizon_path = args.output_prefix.with_name(args.output_prefix.name + "_aggregate_by_horizon.csv")
    aggregate_path = args.output_prefix.with_name(args.output_prefix.name + "_aggregate.csv")

    all_windows.to_csv(all_path, index=False)
    by_horizon = aggregate(all_windows, ["signal", "horizon"])
    by_signal = aggregate(all_windows, ["signal"])
    by_horizon.to_csv(by_horizon_path, index=False)
    by_signal.to_csv(aggregate_path, index=False)

    pd.set_option("display.width", 180)
    show_cols = [
        "signal",
        "cases",
        "windows",
        "horizons",
        "min_pf",
        "median_pf",
        "mean_port_ret",
        "mean_win_rate",
        "mean_avg_names",
        "total_events",
    ]
    print("Short-horizon aggregate:")
    print(by_signal[show_cols].replace({np.nan: None}).round(6).to_string(index=False))
    print("\nSaved:")
    print(f"  {all_path}")
    print(f"  {by_horizon_path}")
    print(f"  {aggregate_path}")


if __name__ == "__main__":
    main()
