"""Aggregate T+1..T+20 turn-up forward path event CSVs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_turn_forward_paths_qlib import horizon_summary, key_summary


REQUIRED_COLUMNS = {
    "market",
    "window",
    "instrument",
    "signal_date",
    "market_regime",
    "ret3_bucket",
    "strength_group",
    "ret3_to_t",
    "rsi6",
    "volume_ratio20",
    "t1_open_gap",
    "t1_open_to_close",
    "t1_close_from_high",
    "close_ret_t5",
    "open_entry_ret_t5",
    "close_ret_t10",
    "open_entry_ret_t10",
    "close_ret_t20",
    "open_entry_ret_t20",
    "late_ret_5_20",
    "early_peak5_to_t20_close",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate turn forward path event files.")
    parser.add_argument("events_csv", type=Path, nargs="+")
    parser.add_argument("--summary-start-horizon", type=int, default=5)
    parser.add_argument("--max-horizon", type=int, default=20)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "results" / "turn_forward_paths_all",
    )
    parser.add_argument("--save-events", action="store_true")
    return parser.parse_args()


def read_events(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing required columns: {missing_cols}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    events = read_events(args.events_csv)
    horizons = range(args.summary_start_horizon, args.max_horizon + 1)

    key_market = key_summary(events, group_cols=["market", "ret3_bucket", "strength_group"])
    key_market_regime = key_summary(
        events,
        group_cols=["market", "market_regime", "ret3_bucket", "strength_group"],
    )
    key_window = key_summary(events, group_cols=["market", "window", "ret3_bucket", "strength_group"])
    horizon_market = horizon_summary(
        events,
        group_cols=["market", "ret3_bucket", "strength_group"],
        horizons=horizons,
    )
    horizon_market_regime = horizon_summary(
        events,
        group_cols=["market", "market_regime", "ret3_bucket", "strength_group"],
        horizons=horizons,
    )
    horizon_window = horizon_summary(
        events,
        group_cols=["market", "window", "ret3_bucket", "strength_group"],
        horizons=horizons,
    )

    events_path = args.output_prefix.with_name(args.output_prefix.name + "_events.csv")
    key_market_path = args.output_prefix.with_name(args.output_prefix.name + "_key_summary.csv")
    key_regime_path = args.output_prefix.with_name(args.output_prefix.name + "_key_summary_regime.csv")
    key_window_path = args.output_prefix.with_name(args.output_prefix.name + "_key_summary_by_window.csv")
    horizon_market_path = args.output_prefix.with_name(args.output_prefix.name + "_horizon_summary.csv")
    horizon_regime_path = args.output_prefix.with_name(
        args.output_prefix.name + "_horizon_summary_regime.csv"
    )
    horizon_window_path = args.output_prefix.with_name(
        args.output_prefix.name + "_horizon_summary_by_window.csv"
    )

    key_market.to_csv(key_market_path, index=False)
    key_market_regime.to_csv(key_regime_path, index=False)
    key_window.to_csv(key_window_path, index=False)
    horizon_market.to_csv(horizon_market_path, index=False)
    horizon_market_regime.to_csv(horizon_regime_path, index=False)
    horizon_window.to_csv(horizon_window_path, index=False)
    if args.save_events:
        events.to_csv(events_path, index=False)

    pd.set_option("display.width", 220)
    print(f"Events: {len(events):,}")
    print("\nMarket-level key summary:")
    show_cols = [
        "market",
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
    print(key_market[show_cols].round(6).to_string(index=False))
    print("\nSaved:")
    print(f"  {key_market_path}")
    print(f"  {key_regime_path}")
    print(f"  {key_window_path}")
    print(f"  {horizon_market_path}")
    print(f"  {horizon_regime_path}")
    print(f"  {horizon_window_path}")
    if args.save_events:
        print(f"  {events_path}")


if __name__ == "__main__":
    main()
