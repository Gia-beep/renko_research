"""Factor-evaluation primitives shared by the brick-alpha research scripts.

Every function operates on wide ``date x instrument`` frames and is free of any
Qlib dependency so it can be unit-tested with synthetic frames.  The logic was
extracted verbatim from ``scripts/research_brick_alpha_qlib.py`` and made public.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward ``horizon``-day simple return for every name (NaN at the tail)."""
    return close.shift(-horizon) / close - 1.0


def stack_non_null(frame: pd.DataFrame) -> pd.Series:
    """Stack a wide frame to a long Series, dropping NaNs (pandas-version safe)."""
    try:
        return frame.stack(future_stack=True).dropna()
    except TypeError:
        return frame.stack(dropna=True)


def rank_ic(
    factor: pd.DataFrame,
    forward_ret: pd.DataFrame,
    *,
    min_names: int = 20,
) -> pd.Series:
    """Daily cross-sectional Spearman rank IC between factor and forward return."""
    values: list[tuple[pd.Timestamp, float]] = []
    for dt in factor.index.intersection(forward_ret.index):
        x = factor.loc[dt]
        y = forward_ret.loc[dt]
        mask = x.notna() & y.notna()
        if int(mask.sum()) < min_names:
            continue
        corr = x[mask].rank(method="average").corr(y[mask].rank(method="average"))
        if pd.notna(corr):
            values.append((dt, float(corr)))
    if not values:
        return pd.Series(dtype=float, name="rank_ic")
    index, data = zip(*values)
    return pd.Series(data, index=pd.Index(index, name="datetime"), name="rank_ic")


def quantile_returns(
    factor: pd.DataFrame,
    forward_ret: pd.DataFrame,
    *,
    quantiles: int,
    min_names: int = 50,
) -> pd.DataFrame:
    """Mean forward return per factor quantile, plus a top-minus-bottom spread."""
    rows: list[dict[str, float | pd.Timestamp]] = []
    for dt in factor.index.intersection(forward_ret.index):
        x = factor.loc[dt]
        y = forward_ret.loc[dt]
        mask = x.notna() & y.notna()
        if int(mask.sum()) < max(min_names, quantiles):
            continue

        try:
            labels = pd.qcut(
                x[mask].rank(method="first"),
                quantiles,
                labels=range(1, quantiles + 1),
            )
        except ValueError:
            continue

        grouped = y[mask].groupby(labels, observed=False).mean()
        row: dict[str, float | pd.Timestamp] = {"datetime": dt}
        for q, value in grouped.items():
            row[f"q{int(q)}"] = float(value)
        if 1 in grouped.index and quantiles in grouped.index:
            row["top_minus_bottom"] = float(grouped.loc[quantiles] - grouped.loc[1])
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("datetime").sort_index()


def event_return_summary(
    signal: pd.DataFrame,
    forward_ret: pd.DataFrame,
    *,
    label: str,
) -> dict[str, float | int | str]:
    """Mean/median/win-rate of forward returns conditioned on a boolean event.

    The benchmark is the unconditional mean forward return over the same
    (aligned) panel, so ``excess_mean`` isolates the event's edge.
    """
    aligned_signal, aligned_ret = signal.align(forward_ret, join="inner", axis=None)
    event_rets = stack_non_null(aligned_ret.where(aligned_signal))
    all_rets = stack_non_null(aligned_ret)
    if event_rets.empty:
        return {
            "event": label,
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "win_rate": np.nan,
            "benchmark_mean": float(all_rets.mean()) if not all_rets.empty else np.nan,
            "excess_mean": np.nan,
        }
    benchmark_mean = float(all_rets.mean()) if not all_rets.empty else np.nan
    mean = float(event_rets.mean())
    return {
        "event": label,
        "count": int(event_rets.shape[0]),
        "mean": mean,
        "median": float(event_rets.median()),
        "win_rate": float((event_rets > 0).mean()),
        "benchmark_mean": benchmark_mean,
        "excess_mean": mean - benchmark_mean,
    }


def event_positions(
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    *,
    max_hold: int,
    red_brick: pd.DataFrame | None = None,
    max_red_bricks: int | None = None,
) -> pd.DataFrame:
    """Per-name holding mask: enter on ``entries``, leave on ``exits`` or ``max_hold``.

    ``max_hold <= 0`` disables the calendar holding cap.

    When both ``red_brick`` (a boolean red/rising-brick frame) and ``max_red_bricks``
    are given, the renko.md §5.2 "数四块砖" exit also applies: the entry bar is red
    brick #1 and each later ``red_brick`` day increments the count, so the position is
    closed *on* the ``max_red_bricks``-th red brick (it does not earn that day's
    forward return).  Leaving either ``None`` (the default) disables the brick exit and
    reproduces the original ``exits``/``max_hold`` behaviour exactly.
    """
    entries = entries.fillna(False).astype(bool)
    exits = exits.fillna(False).astype(bool)
    use_max_hold = max_hold > 0
    use_brick = red_brick is not None and max_red_bricks is not None
    if use_brick:
        red_brick = red_brick.reindex(index=entries.index, columns=entries.columns).fillna(False).astype(bool)
    positions = pd.DataFrame(False, index=entries.index, columns=entries.columns)

    for col in entries.columns:
        holding = False
        age = 0
        red_count = 0
        for dt in entries.index:
            if holding and (bool(exits.at[dt, col]) or (use_max_hold and age >= max_hold)):
                holding = False
                age = 0
                red_count = 0
            if bool(entries.at[dt, col]):
                holding = True
                age = 0
                red_count = 0
            # renko.md §5.2: entry bar is red brick #1; exit on the max_red_bricks-th.
            if use_brick and holding and bool(red_brick.at[dt, col]):
                red_count += 1
                if red_count >= max_red_bricks:
                    holding = False
                    age = 0
                    red_count = 0
            positions.at[dt, col] = holding
            if holding:
                age += 1

    return positions


def max_drawdown(nav: pd.Series) -> float:
    """Most negative peak-to-trough drawdown of a NAV series."""
    drawdown = nav / nav.cummax() - 1.0
    return float(drawdown.min()) if not drawdown.empty else np.nan


def _return_stats(returns: pd.Series, nav: pd.Series) -> dict[str, float]:
    """Common return statistics for a daily return/NAV pair."""
    returns = pd.Series(returns).fillna(0.0)
    nav = pd.Series(nav)
    if returns.empty or nav.empty:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }
    total_return = float(nav.iloc[-1] - 1.0)
    ann_return = float((1.0 + total_return) ** (252.0 / max(1, len(returns))) - 1.0)
    vol = float(returns.std(ddof=0))
    sharpe = (
        float(returns.mean() / vol * math.sqrt(252.0))
        if vol > 0
        else np.nan
    )
    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(nav),
    }


def benchmark_beta(strategy_return: pd.Series, benchmark_return: pd.Series) -> float:
    """Daily beta of ``strategy_return`` versus ``benchmark_return``."""
    strategy_return, benchmark_return = strategy_return.align(benchmark_return, join="inner")
    mask = strategy_return.notna() & benchmark_return.notna()
    if int(mask.sum()) < 2:
        return np.nan
    bench = benchmark_return[mask]
    var = float(bench.var(ddof=0))
    if var <= 0:
        return np.nan
    cov = float(((strategy_return[mask] - strategy_return[mask].mean()) * (bench - bench.mean())).mean())
    return cov / var


def benchmark_corr(strategy_return: pd.Series, benchmark_return: pd.Series) -> float:
    """Daily correlation of ``strategy_return`` versus ``benchmark_return``."""
    strategy_return, benchmark_return = strategy_return.align(benchmark_return, join="inner")
    mask = strategy_return.notna() & benchmark_return.notna()
    if int(mask.sum()) < 2:
        return np.nan
    strat = strategy_return[mask]
    bench = benchmark_return[mask]
    if float(strat.std(ddof=0)) <= 0 or float(bench.std(ddof=0)) <= 0:
        return np.nan
    return float(strat.corr(bench))


def event_strategy(
    close: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    *,
    max_hold: int,
    red_brick: pd.DataFrame | None = None,
    max_red_bricks: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Equal-weight next-day-return proxy for holding event positions.

    Returns the daily frame (strategy/benchmark returns + NAVs) and a summary of
    total/annualized return, Sharpe, max drawdown, exposure and average breadth.
    ``red_brick`` / ``max_red_bricks`` are forwarded to :func:`event_positions` to
    enable the renko.md §5.2 "数四块砖" exit (default ``None`` ⇒ disabled).
    """
    positions = event_positions(
        entries,
        exits,
        max_hold=max_hold,
        red_brick=red_brick,
        max_red_bricks=max_red_bricks,
    )
    next_ret = close.shift(-1) / close - 1.0
    valid_positions = positions & next_ret.notna()
    active_names = valid_positions.sum(axis=1)
    strategy_ret = next_ret.where(valid_positions).sum(axis=1) / active_names.replace(0, np.nan)
    strategy_ret = strategy_ret.fillna(0.0)
    benchmark_ret = next_ret.mean(axis=1, skipna=True).fillna(0.0)

    daily = pd.DataFrame(
        {
            "strategy_return": strategy_ret,
            "benchmark_return": benchmark_ret,
            "active_names": active_names,
        }
    )
    daily["strategy_nav"] = (1.0 + daily["strategy_return"]).cumprod()
    daily["benchmark_nav"] = (1.0 + daily["benchmark_return"]).cumprod()

    active_days = int((daily["active_names"] > 0).sum())
    total_return = float(daily["strategy_nav"].iloc[-1] - 1.0)
    ann_return = float((1.0 + total_return) ** (252.0 / max(1, len(daily))) - 1.0)
    vol = float(daily["strategy_return"].std(ddof=0))
    sharpe = (
        float(daily["strategy_return"].mean() / vol * math.sqrt(252.0))
        if vol > 0
        else np.nan
    )

    summary = {
        "total_return": total_return,
        "annualized_return": ann_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(daily["strategy_nav"]),
        "exposure": float((daily["active_names"] > 0).mean()),
        "avg_active_names": float(daily["active_names"].where(daily["active_names"] > 0).mean()),
        "active_days": float(active_days),
    }
    return daily, summary


def benchmark_hedged_event_strategy(
    close: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    *,
    max_hold: int,
    red_brick: pd.DataFrame | None = None,
    max_red_bricks: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Long event proxy plus an active-day equal-weight benchmark hedge.

    The hedge is applied only on dates with at least one active event position.
    This keeps inactive dates flat instead of implicitly running a naked short
    benchmark book when the signal has no holdings.
    """
    daily, long_summary = event_strategy(
        close,
        entries,
        exits,
        max_hold=max_hold,
        red_brick=red_brick,
        max_red_bricks=max_red_bricks,
    )
    active = daily["active_names"] > 0
    daily["active_benchmark_return"] = daily["benchmark_return"].where(active, 0.0)
    daily["excess_return"] = daily["strategy_return"] - daily["active_benchmark_return"]
    daily["excess_nav"] = (1.0 + daily["excess_return"]).cumprod()

    excess = _return_stats(daily["excess_return"], daily["excess_nav"])
    summary = {
        **long_summary,
        "excess_total_return": excess["total_return"],
        "excess_annualized_return": excess["annualized_return"],
        "excess_sharpe": excess["sharpe"],
        "excess_max_drawdown": excess["max_drawdown"],
        "benchmark_beta": benchmark_beta(daily["strategy_return"], daily["benchmark_return"]),
        "benchmark_corr": benchmark_corr(daily["strategy_return"], daily["benchmark_return"]),
    }
    return daily, summary


def long_short_event_strategy(
    close: pd.DataFrame,
    long_entries: pd.DataFrame,
    long_exits: pd.DataFrame,
    short_entries: pd.DataFrame,
    short_exits: pd.DataFrame,
    *,
    max_hold: int,
    require_both_sides: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Dollar-neutral event proxy: 50% long events and 50% short events.

    Short-leg return is the negative of the selected names' next-day return.
    With ``require_both_sides=True`` (default), the strategy is flat unless both
    long and short books are active on the same date.
    """
    long_positions = event_positions(long_entries, long_exits, max_hold=max_hold)
    short_positions = event_positions(short_entries, short_exits, max_hold=max_hold)
    next_ret = close.shift(-1) / close - 1.0

    valid_long = long_positions & next_ret.notna()
    valid_short = short_positions & next_ret.notna()
    long_names = valid_long.sum(axis=1)
    short_names = valid_short.sum(axis=1)

    long_ret = next_ret.where(valid_long).sum(axis=1) / long_names.replace(0, np.nan)
    short_underlying_ret = next_ret.where(valid_short).sum(axis=1) / short_names.replace(0, np.nan)
    if require_both_sides:
        active = (long_names > 0) & (short_names > 0)
        strategy_ret = (0.5 * long_ret - 0.5 * short_underlying_ret).where(active, 0.0)
    else:
        strategy_ret = 0.5 * long_ret.fillna(0.0) - 0.5 * short_underlying_ret.fillna(0.0)
        active = (long_names > 0) | (short_names > 0)
    strategy_ret = strategy_ret.fillna(0.0)
    benchmark_ret = next_ret.mean(axis=1, skipna=True).fillna(0.0)

    daily = pd.DataFrame(
        {
            "strategy_return": strategy_ret,
            "benchmark_return": benchmark_ret,
            "long_return": long_ret.fillna(0.0),
            "short_underlying_return": short_underlying_ret.fillna(0.0),
            "long_names": long_names,
            "short_names": short_names,
        }
    )
    daily["strategy_nav"] = (1.0 + daily["strategy_return"]).cumprod()
    daily["benchmark_nav"] = (1.0 + daily["benchmark_return"]).cumprod()

    stats = _return_stats(daily["strategy_return"], daily["strategy_nav"])
    active_days = int(active.sum())
    summary = {
        **stats,
        "exposure": float(active.mean()),
        "long_exposure": float((long_names > 0).mean()),
        "short_exposure": float((short_names > 0).mean()),
        "avg_long_names": float(long_names.where(long_names > 0).mean()),
        "avg_short_names": float(short_names.where(short_names > 0).mean()),
        "active_days": float(active_days),
        "benchmark_beta": benchmark_beta(daily["strategy_return"], daily["benchmark_return"]),
        "benchmark_corr": benchmark_corr(daily["strategy_return"], daily["benchmark_return"]),
    }
    return daily, summary
