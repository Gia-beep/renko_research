"""Aggregate turn-burst event CSVs into size/momentum/regime diagnostics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_turn_burst_samples_qlib import (
    feature_contrast,
    interaction_frames,
    summary_frame,
)


REQUIRED_COLUMNS = {
    "market",
    "window",
    "instrument",
    "signal_date",
    "burst",
    "burst_day",
    "max_burst_ret",
    "t1_close_ret",
    "t2_close_day_ret",
    "t2_cum_close_ret",
    "amv_regime",
    "brick_value",
    "rsi6",
    "ret3_to_t",
    "ret5_to_t",
    "ret10_to_t",
    "volume_ratio20",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate saved turn-burst event rows into interaction summaries."
    )
    parser.add_argument("events_csv", type=Path, nargs="+")
    parser.add_argument("--interaction-quantiles", type=int, default=5)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "results" / "turn_burst_xg_clean_all",
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
    events = pd.concat(frames, ignore_index=True)
    events["burst"] = events["burst"].astype(bool)
    return events


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    events = read_events(args.events_csv)

    summary = summary_frame(events)
    contrast = feature_contrast(events)
    pooled = interaction_frames(events, q=args.interaction_quantiles, include_window=False)
    by_window = interaction_frames(events, q=args.interaction_quantiles, include_window=True)

    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    contrast_path = args.output_prefix.with_name(args.output_prefix.name + "_feature_contrast.csv")
    events_path = args.output_prefix.with_name(args.output_prefix.name + "_events.csv")

    pooled_paths = {
        "regime": args.output_prefix.with_name(args.output_prefix.name + "_interaction_regime.csv"),
        "size_momentum": args.output_prefix.with_name(
            args.output_prefix.name + "_interaction_size_momentum.csv"
        ),
        "size_momentum_regime": args.output_prefix.with_name(
            args.output_prefix.name + "_interaction_size_momentum_regime.csv"
        ),
        "shape": args.output_prefix.with_name(args.output_prefix.name + "_interaction_shape.csv"),
    }
    by_window_paths = {
        "regime": args.output_prefix.with_name(args.output_prefix.name + "_interaction_regime_by_window.csv"),
        "size_momentum": args.output_prefix.with_name(
            args.output_prefix.name + "_interaction_size_momentum_by_window.csv"
        ),
        "size_momentum_regime": args.output_prefix.with_name(
            args.output_prefix.name + "_interaction_size_momentum_regime_by_window.csv"
        ),
        "shape": args.output_prefix.with_name(args.output_prefix.name + "_interaction_shape_by_window.csv"),
    }

    summary.to_csv(summary_path, index=False)
    contrast.to_csv(contrast_path, index=False)
    for name, path in pooled_paths.items():
        pooled[name].to_csv(path, index=False)
    for name, path in by_window_paths.items():
        by_window[name].to_csv(path, index=False)
    if args.save_events:
        events.to_csv(events_path, index=False)

    pd.set_option("display.width", 180)
    print(f"Events: {len(events):,}")
    print("\nSummary:")
    print(summary.round(6).to_string(index=False))
    print("\nPooled interaction shape:")
    print(pooled["shape"].round(6).to_string(index=False))
    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {contrast_path}")
    for path in pooled_paths.values():
        print(f"  {path}")
    for path in by_window_paths.values():
        print(f"  {path}")
    if args.save_events:
        print(f"  {events_path}")


if __name__ == "__main__":
    main()
