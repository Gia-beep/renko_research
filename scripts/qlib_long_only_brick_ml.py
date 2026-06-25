"""Train a Qlib long-only ML model using the translated brick-turn indicator.

The brick formula is computed by ``src.indicators.brick_alpha`` so Tongdaxin's
recursive ``SMA`` semantics stay exact.  Qlib is used for market data and the
LightGBM model interface; the portfolio proxy is strictly long-only:

  - train a Qlib ``LGBModel`` on brick/volume/trend features;
  - predict cross-sectional forward-return scores;
  - buy only the top-N eligible names, otherwise hold cash;
  - never create an individual-stock short leg.

Default execution gate is conservative for A-share use but still lets the ML
model rank a broad cross-section:

    AMV regime is tradeable, then buy the model's top-N names

Use ``--entry-gate turn_up_vol_amv`` if you want the stricter event-only version:
``turn_up/XG AND volume_ratio_20 >= 1.5 AND AMV regime is tradeable``.

Example:
    /home/x1843/venvs/qlib/bin/python scripts/qlib_long_only_brick_ml.py \
        --market csi300 \
        --train-start 2014-01-01 --train-end 2017-12-31 \
        --valid-start 2018-01-01 --valid-end 2018-12-31 \
        --test-start 2019-01-01 --test-end 2020-09-25 \
        --output-prefix results/qlib_long_only_brick_ml_csi300
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indicators.brick_alpha import compute_brick_alpha
from src.indicators.signal_filters import combine, moving_average, roc, strong_red_body, volume_ratio
from src.research.amv_regime import DEFAULT_AMV_CSV, amv_regime_mask, broadcast_to_panel, load_amv_close
from src.research.metrics import benchmark_beta, max_drawdown
from src.research.qlib_data import calendar_bounds, load_features, warmup_start

DEFAULT_PROVIDER_URI = Path(
    os.environ.get("QLIB_PROVIDER_URI", "/home/x1843/.qlib/qlib_data/cn_data")
)


@dataclass(frozen=True)
class SegmentConfig:
    train: tuple[pd.Timestamp, pd.Timestamp]
    valid: tuple[pd.Timestamp, pd.Timestamp]
    test: tuple[pd.Timestamp, pd.Timestamp]


class PandasQlibDataset:
    """Minimal DatasetH-compatible wrapper around a Qlib-style pandas frame."""

    def __init__(self, data: pd.DataFrame, segments: dict[str, tuple[str, str]]):
        self.data = data.sort_index()
        self.segments = segments

    def prepare(
        self,
        segments,
        col_set="__all",
        data_key=None,
        **kwargs,
    ):
        if isinstance(segments, (list, tuple)) and all(isinstance(seg, str) for seg in segments):
            return [self.prepare(seg, col_set=col_set, data_key=data_key, **kwargs) for seg in segments]

        if isinstance(segments, str):
            if segments not in self.segments:
                raise KeyError(f"unknown segment: {segments}")
            start, end = self.segments[segments]
        elif isinstance(segments, slice):
            start, end = segments.start, segments.stop
        else:
            start, end = segments

        df = _slice_by_datetime(self.data, pd.Timestamp(start), pd.Timestamp(end))
        if col_set in ("__all", None):
            return df
        if isinstance(col_set, str):
            out = df[col_set]
        else:
            out = df.loc[:, pd.IndexSlice[list(col_set), :]]
        if isinstance(out.columns, pd.MultiIndex) and "label" in out.columns.get_level_values(0):
            out = out.dropna(subset=[("label", "LABEL0")])
        return out


def _slice_by_datetime(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = df.index.get_level_values("datetime")
    return df[(dates >= start) & (dates <= end)]


def _stack_wide(frame: pd.DataFrame) -> pd.Series:
    try:
        stacked = frame.stack(future_stack=True)
    except TypeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            stacked = frame.stack(dropna=False)
    stacked.index = stacked.index.set_names(["datetime", "instrument"])
    return stacked.sort_index()


def _cs_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1, skipna=True)
    std = frame.std(axis=1, skipna=True, ddof=0).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def _future_return(close: pd.DataFrame, *, horizon: int, entry_shift: int) -> pd.DataFrame:
    """Forward return from the delayed entry bar to ``horizon`` bars later."""
    return close.shift(-(entry_shift + horizon)) / close.shift(-entry_shift) - 1.0


def _build_feature_frames(
    *,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    amv: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    alpha = compute_brick_alpha(high=high, low=low, close=close)
    brick = alpha.brick_value.fillna(0.0)
    vr20 = volume_ratio(volume, 20).clip(lower=0.0, upper=10.0)
    ma20 = moving_average(close, 20)
    ma60 = moving_average(close, 60)
    spread = (high - low).where(high > low)

    bool_features = {
        "rising": alpha.rising,
        "falling": alpha.falling,
        "turn_up_xg": alpha.turn_up,
        "turn_down": alpha.turn_down,
        "vol_ge_1_5": vr20 >= 1.5,
        "close_above_ma20": close > ma20,
        "close_above_ma60": close > ma60,
        "strong_red": strong_red_body(open_, high, low, close),
        "amv_tradeable": amv,
    }
    features: dict[str, pd.DataFrame] = {
        "brick_value": brick,
        "brick_delta_1": brick.diff(),
        "brick_delta_3": brick - brick.shift(3),
        "brick_pct_rank_20": brick.rank(axis=1, pct=True),
        "volume_ratio_20": vr20,
        "ret_1": roc(close, 1),
        "ret_5": roc(close, 5),
        "ret_20": roc(close, 20),
        "ma20_gap": close / ma20 - 1.0,
        "ma60_gap": close / ma60 - 1.0,
        "ma20_slope_5": ma20 / ma20.shift(5) - 1.0,
        "range_pct": spread / close,
        "body_pct": (close - open_) / open_,
    }
    for name, mask in bool_features.items():
        features[name] = mask.astype(float)
    features["turn_up_x_vol"] = (alpha.turn_up & (vr20 >= 1.5)).astype(float)
    features["turn_up_x_vol_x_amv"] = combine(alpha.turn_up, vr20 >= 1.5, amv).astype(float)

    gates = {
        "none": pd.DataFrame(True, index=close.index, columns=close.columns),
        "amv": amv,
        "turn_up": alpha.turn_up,
        "turn_up_vol": combine(alpha.turn_up, vr20 >= 1.5),
        "turn_up_amv": combine(alpha.turn_up, amv),
        "turn_up_vol_amv": combine(alpha.turn_up, vr20 >= 1.5, amv),
        "vol_amv": combine(vr20 >= 1.5, amv),
    }
    return features, gates


def _make_dataset_frame(
    features: dict[str, pd.DataFrame],
    label: pd.DataFrame,
) -> pd.DataFrame:
    x = pd.concat({name: _stack_wide(frame) for name, frame in features.items()}, axis=1)
    y = pd.DataFrame({"LABEL0": _stack_wide(label)})
    df = pd.concat({"feature": x, "label": y}, axis=1)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.sort_index()


def _select_topn(
    score: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    top_n: int,
    min_score: float | None,
) -> pd.DataFrame:
    eligible = eligible.reindex(index=score.index, columns=score.columns, fill_value=False)
    masked = score.where(eligible.fillna(False).astype(bool))
    if min_score is not None:
        masked = masked.where(masked >= min_score)
    ranks = masked.rank(axis=1, ascending=False, method="first")
    return ranks.le(top_n).fillna(False).astype(bool)


def _weights_from_selection(selection: pd.DataFrame) -> pd.DataFrame:
    counts = selection.sum(axis=1).replace(0, np.nan)
    return selection.astype(float).div(counts, axis=0).fillna(0.0)


def _annualized_return(total_return: float, periods: int) -> float:
    if periods <= 0 or not np.isfinite(total_return) or total_return <= -1.0:
        return np.nan
    return float((1.0 + total_return) ** (252.0 / periods) - 1.0)


def _sharpe(returns: pd.Series) -> float:
    vol = float(returns.std(ddof=0))
    if vol <= 0:
        return np.nan
    return float(returns.mean() / vol * math.sqrt(252.0))


def _long_only_backtest(
    *,
    close: pd.DataFrame,
    score: pd.DataFrame,
    eligible: pd.DataFrame,
    top_n: int,
    min_score: float | None,
    entry_shift: int,
    hold_days: int,
    fee: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    tradable = close.reindex(index=score.index, columns=score.columns).notna()
    score = score.where(tradable)
    eligible = eligible.reindex(index=score.index, columns=score.columns, fill_value=False) & tradable
    selection = _select_topn(score, eligible, top_n=top_n, min_score=min_score)
    signal_weights = _weights_from_selection(selection)

    close = close.sort_index()
    score = score.sort_index()
    close_index = close.index
    columns = score.columns
    target_points: list[tuple[pd.Timestamp, pd.Series, pd.Timestamp]] = []
    period_returns: list[float] = []

    rebalance_signal_dates = list(score.index[::hold_days])
    for signal_dt in rebalance_signal_dates:
        loc = close_index.get_indexer([signal_dt])
        if len(loc) == 0 or loc[0] < 0:
            continue
        entry_pos = int(loc[0]) + entry_shift
        exit_pos = entry_pos + hold_days
        if entry_pos >= len(close_index):
            continue
        entry_dt = close_index[entry_pos]
        weight = signal_weights.loc[signal_dt].reindex(columns).fillna(0.0)
        target_points.append((entry_dt, weight, signal_dt))
        if exit_pos < len(close_index) and float(weight.sum()) > 0:
            holding_ret = close.iloc[exit_pos].reindex(columns) / close.iloc[entry_pos].reindex(columns) - 1.0
            period_returns.append(float((weight * holding_ret.fillna(0.0)).sum()))

    raw_targets = pd.DataFrame(np.nan, index=close_index, columns=columns, dtype=float)
    for entry_dt, weight, _signal_dt in target_points:
        raw_targets.loc[entry_dt, columns] = weight.to_numpy(dtype=float)
    weights = raw_targets.ffill().fillna(0.0)

    next_ret = close.shift(-1) / close - 1.0
    next_ret = next_ret.reindex(index=weights.index, columns=weights.columns)
    gross_ret = (weights * next_ret.fillna(0.0)).sum(axis=1)
    benchmark_ret = next_ret.mean(axis=1, skipna=True).fillna(0.0)

    turnover = weights.diff().abs().sum(axis=1)
    if not turnover.empty:
        turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost = turnover.fillna(0.0) * fee
    strategy_ret = gross_ret - cost
    active_names = weights.gt(0).sum(axis=1)

    eval_index = close_index[(close_index >= score.index.min()) & (close_index <= score.index.max())]

    daily = pd.DataFrame(
        {
            "strategy_return": strategy_ret,
            "gross_return": gross_ret,
            "benchmark_return": benchmark_ret,
            "turnover": turnover.fillna(0.0),
            "cost": cost,
            "active_names": active_names,
        }
    ).loc[eval_index]
    daily["strategy_nav"] = (1.0 + daily["strategy_return"]).cumprod()
    daily["benchmark_nav"] = (1.0 + daily["benchmark_return"]).cumprod()
    active = daily["active_names"] > 0
    daily["active_benchmark_return"] = daily["benchmark_return"].where(active, 0.0)
    daily["excess_return"] = daily["strategy_return"] - daily["active_benchmark_return"]
    daily["excess_nav"] = (1.0 + daily["excess_return"]).cumprod()

    total_return = float(daily["strategy_nav"].iloc[-1] - 1.0)
    benchmark_total = float(daily["benchmark_nav"].iloc[-1] - 1.0)
    excess_total = float(daily["excess_nav"].iloc[-1] - 1.0)
    period_series = pd.Series(period_returns, dtype=float)
    evaluated_periods = period_series[period_series != 0]
    wins = evaluated_periods[evaluated_periods > 0]
    losses = evaluated_periods[evaluated_periods < 0]
    avg_win = float(wins.mean()) if not wins.empty else np.nan
    avg_loss = float(losses.mean()) if not losses.empty else np.nan
    summary = {
        "total_return": total_return,
        "annualized_return": _annualized_return(total_return, len(daily)),
        "sharpe": _sharpe(daily["strategy_return"]),
        "max_drawdown": max_drawdown(daily["strategy_nav"]),
        "benchmark_total_return": benchmark_total,
        "benchmark_sharpe": _sharpe(daily["benchmark_return"]),
        "excess_total_return": excess_total,
        "excess_sharpe": _sharpe(daily["excess_return"]),
        "excess_max_drawdown": max_drawdown(daily["excess_nav"]),
        "benchmark_beta": benchmark_beta(daily["strategy_return"], daily["benchmark_return"]),
        "exposure": float(active.mean()),
        "avg_active_names": float(daily["active_names"].where(daily["active_names"] > 0).mean()),
        "avg_turnover": float(daily["turnover"].mean()),
        "active_days": float(active.sum()),
        "period_count": float(len(period_series)),
        "period_evaluated_count": float(len(evaluated_periods)),
        "period_win_rate": float((evaluated_periods > 0).mean()) if len(evaluated_periods) else np.nan,
        "period_avg_win": avg_win,
        "period_avg_loss": avg_loss,
        "period_win_loss_ratio": avg_win / abs(avg_loss) if np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss < 0 else np.nan,
        "period_profit_factor": float(wins.sum() / abs(losses.sum())) if not wins.empty and not losses.empty else np.nan,
    }
    return daily, weights, summary


def _feature_importance(model, feature_names: Iterable[str]) -> pd.DataFrame:
    booster = getattr(model, "model", None)
    if booster is None:
        return pd.DataFrame()
    names = list(feature_names)
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    return (
        pd.DataFrame({"feature": names, "gain": gain, "split": split})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    cal_start, cal_end = calendar_bounds(DEFAULT_PROVIDER_URI)
    train_start = "2014-01-01"
    if cal_start and train_start < cal_start:
        train_start = cal_start
    default_test_end = cal_end or "2020-09-25"

    parser = argparse.ArgumentParser(description="Qlib long-only ML model for brick alpha.")
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--train-start", default=train_start)
    parser.add_argument("--train-end", default="2017-12-31")
    parser.add_argument("--valid-start", default="2018-01-01")
    parser.add_argument("--valid-end", default="2018-12-31")
    parser.add_argument("--test-start", default="2019-01-01")
    parser.add_argument("--test-end", default=default_test_end)
    parser.add_argument("--label-horizon", type=int, default=5)
    parser.add_argument("--entry-shift", type=int, default=1)
    parser.add_argument(
        "--hold-days",
        type=int,
        default=None,
        help="Rebalance/holding cycle in trading days. Defaults to --label-horizon.",
    )
    parser.add_argument("--label-mode", choices=["raw", "zscore"], default="zscore")
    parser.add_argument(
        "--entry-gate",
        choices=["none", "amv", "turn_up", "turn_up_vol", "turn_up_amv", "turn_up_vol_amv", "vol_amv"],
        default="amv",
    )
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--fee", type=float, default=0.001, help="One-way turnover cost.")
    parser.add_argument("--amv-csv", type=Path, default=DEFAULT_AMV_CSV)
    parser.add_argument("--amv-sma", type=int, default=60)
    parser.add_argument("--warmup-bars", type=int, default=260)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--verbose-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--experiment-name", default="brick_alpha_long_only_ml")
    parser.add_argument(
        "--use-qlib-recorder",
        action="store_true",
        help="Enable Qlib workflow/mlflow recording. Disabled by default to avoid git noise in non-git research dirs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "results" / "qlib_long_only_brick_ml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hold_days is None:
        args.hold_days = args.label_horizon
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.label_horizon <= 0 or args.entry_shift < 0 or args.hold_days <= 0:
        raise ValueError("--label-horizon/--hold-days must be positive and --entry-shift non-negative")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    load_start = warmup_start(args.provider_uri, args.train_start, args.warmup_bars)
    feats = load_features(
        args.provider_uri,
        args.market,
        load_start,
        args.test_end,
        fields=("$open", "$high", "$low", "$close", "$volume"),
    )
    open_f, high_f, low_f, close_f, volume_f = (
        feats["$open"],
        feats["$high"],
        feats["$low"],
        feats["$close"],
        feats["$volume"],
    )
    amv = broadcast_to_panel(
        amv_regime_mask(load_amv_close(args.amv_csv), sma_window=args.amv_sma),
        close_f.index,
        close_f.columns,
    )

    features, gates = _build_feature_frames(
        open_=open_f,
        high=high_f,
        low=low_f,
        close=close_f,
        volume=volume_f,
        amv=amv,
    )
    raw_label = _future_return(close_f, horizon=args.label_horizon, entry_shift=args.entry_shift)
    label = _cs_zscore(raw_label) if args.label_mode == "zscore" else raw_label
    data = _make_dataset_frame(features, label)

    segments = {
        "train": (args.train_start, args.train_end),
        "valid": (args.valid_start, args.valid_end),
        "test": (args.test_start, args.test_end),
    }
    dataset = PandasQlibDataset(data, segments)
    feature_names = list(data["feature"].columns)

    from qlib.contrib.model import gbdt as qlib_gbdt
    from qlib.contrib.model.gbdt import LGBModel

    model = LGBModel(
        loss="mse",
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        lambda_l2=1.0,
        verbosity=-1,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    evals_result: dict = {}
    if args.use_qlib_recorder:
        from qlib.workflow import R

        with R.start(experiment_name=args.experiment_name):
            model.fit(dataset, evals_result=evals_result, verbose_eval=args.verbose_eval)
            pred = model.predict(dataset, segment="test").rename("score")
    else:
        # Qlib's LGBModel logs metrics through qlib.workflow.R.  In this research
        # repo (not a git repo), that recorder emits noisy git-diff failures, so
        # default to a no-op metric logger while still using Qlib's model class.
        original_log_metrics = qlib_gbdt.R.log_metrics
        qlib_gbdt.R.log_metrics = lambda *args, **kwargs: None
        try:
            model.fit(dataset, evals_result=evals_result, verbose_eval=args.verbose_eval)
            pred = model.predict(dataset, segment="test").rename("score")
        finally:
            qlib_gbdt.R.log_metrics = original_log_metrics

    score = pred.unstack("instrument").sort_index()
    score = score.loc[(score.index >= pd.Timestamp(args.test_start)) & (score.index <= pd.Timestamp(args.test_end))]
    tradable = close_f.reindex(index=score.index, columns=score.columns).notna()
    score = score.where(tradable)
    eligible = gates[args.entry_gate].reindex(index=score.index, columns=score.columns, fill_value=False) & tradable
    close_test = close_f.reindex(index=score.index.union(close_f.index)).sort_index()

    daily, weights, summary = _long_only_backtest(
        close=close_test,
        score=score,
        eligible=eligible,
        top_n=args.top_n,
        min_score=args.min_score,
        entry_shift=args.entry_shift,
        hold_days=args.hold_days,
        fee=args.fee,
    )
    daily = daily.reindex(score.index).fillna(0.0)
    if not daily.empty:
        daily["strategy_nav"] = (1.0 + daily["strategy_return"]).cumprod()
        daily["benchmark_nav"] = (1.0 + daily["benchmark_return"]).cumprod()
        daily["excess_nav"] = (1.0 + daily["excess_return"]).cumprod()

    metrics = pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in {
                **summary,
                "train_rows": float(dataset.prepare("train", col_set=["feature", "label"]).shape[0]),
                "valid_rows": float(dataset.prepare("valid", col_set=["feature", "label"]).shape[0]),
                "test_rows": float(dataset.prepare("test", col_set=["feature", "label"]).shape[0]),
                "feature_count": float(len(feature_names)),
            }.items()
        ]
    )
    importance = _feature_importance(model, feature_names)
    predictions = pred.reset_index()
    weights_aligned = weights.reindex(index=score.index, columns=score.columns).fillna(0.0)
    active_dates = weights_aligned.index[weights_aligned.sum(axis=1) > 0]
    latest_dt = active_dates[-1] if len(active_dates) else score.index.max()
    latest = pd.DataFrame(
        {
            "score": score.loc[latest_dt],
            "eligible": eligible.loc[latest_dt],
            "weight": weights_aligned.loc[latest_dt],
        }
    ).sort_values(["weight", "score"], ascending=[False, False])

    metrics_path = args.output_prefix.with_name(args.output_prefix.name + "_metrics.csv")
    daily_path = args.output_prefix.with_name(args.output_prefix.name + "_daily_returns.csv")
    pred_path = args.output_prefix.with_name(args.output_prefix.name + "_predictions.csv")
    weights_path = args.output_prefix.with_name(args.output_prefix.name + "_weights.csv")
    importance_path = args.output_prefix.with_name(args.output_prefix.name + "_feature_importance.csv")
    latest_path = args.output_prefix.with_name(args.output_prefix.name + "_latest_candidates.csv")
    metrics.to_csv(metrics_path, index=False)
    daily.to_csv(daily_path)
    predictions.to_csv(pred_path, index=False)
    weights.to_csv(weights_path)
    importance.to_csv(importance_path, index=False)
    latest.to_csv(latest_path)

    pd.set_option("display.width", 180)
    print(f"Market: {args.market}")
    print(
        f"Segments: train={args.train_start}->{args.train_end} "
        f"valid={args.valid_start}->{args.valid_end} test={args.test_start}->{args.test_end}"
    )
    print(
        f"Rows(with labels): train={dataset.prepare('train', col_set=['feature', 'label']).shape[0]} "
        f"valid={dataset.prepare('valid', col_set=['feature', 'label']).shape[0]} "
        f"test={dataset.prepare('test', col_set=['feature', 'label']).shape[0]} "
        f"features={len(feature_names)}"
    )
    print(
        f"Long-only gate={args.entry_gate} top_n={args.top_n} min_score={args.min_score} "
        f"label={args.label_mode}/{args.label_horizon}d hold_days={args.hold_days} "
        f"entry_shift={args.entry_shift}"
    )
    print("\nBacktest summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    print("\nTop feature importance:")
    if not importance.empty:
        print(importance.head(12).round(4).to_string(index=False))
    print(f"\nLatest selected/candidates ({pd.Timestamp(latest_dt).date()}):")
    print(latest.head(max(args.top_n, 10)).round(6).to_string())
    print("\nSaved:")
    print(f"  {metrics_path}")
    print(f"  {daily_path}")
    print(f"  {pred_path}")
    print(f"  {weights_path}")
    print(f"  {importance_path}")
    print(f"  {latest_path}")


if __name__ == "__main__":
    main()
