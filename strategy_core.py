"""
融合策略核心模块 — 常量 / 宏观过滤器 / 选股条件
共 fused_strategy_picker.py（通达信实盘）和 backtest（Qlib回测）使用
"""
import numpy as np
import pandas as pd

# ============================================================
# 策略参数常量
# ============================================================
# 均线周期：白线（短期趋势线）与黄线（中期支撑线）
MA_WHITE = 20
MA_YELLOW = 60
# 强红砖：实体占波幅比例 >= 66.67%
STRONG_RED_THRESHOLD = 2.0 / 3.0
# 短上影：上影线占波幅比例 <= 15%
SHORT_SHADOW_THRESHOLD = 0.15
# 放量倍数：成交量 >= 1.5 × 20日均量
HIGH_VOL_THRESHOLD = 1.5
# 回调节奏：前 N 天至少有一天阴线
PULLBACK_LOOKBACK = 4

# §5.1 出场参数
NO_RISE_CHECK_BARS = 2          # T+N 不涨即走（T+2 = 持仓第二根 K 线）
WHITE_GRACE_BARS = 1            # 白线击穿宽限：>=2 连破才出（grace=1 ⇒ 1 容忍 + 1 触发）

# 宏观状态机
MACRO_BEAR_THRESHOLD = -0.023
MACRO_BULL_THRESHOLD = 0.04
MACRO_BULL_TWO_DAY = 0.04

# 通达信 Amount(万元) → 元, Volume(手) → 股
AMOUNT_SCALE = 10000
VOLUME_HAND = 100

# Qlib 表达式（供 backtest 导入）
EXPRESSIONS = {
    "is_red": "$close > $open",
    "is_strong_red": "($close - $open) / ($high - $low + 1e-8) >= 0.666667",
    "is_short_shadow": "($high - $close) / ($high - $low + 1e-8) <= 0.15",
    # renko.md §3.2 双均线 stack：收盘价 > 黄线 AND 白线 > 黄线（不操作黄线之下）
    "is_dual_ma_bull": "($close > Mean($close, 60)) & (Mean($close, 20) > Mean($close, 60))",
    "has_pullback": "Ref(Sum($close < $open, 4), 1) > 0",
    "is_high_vol": "$volume > 1.5 * Mean($volume, 20)",
    "is_above_vwap": "$close > $amount * 10000 / $volume",
}
CONDITION_NAMES = list(EXPRESSIONS.keys())

# 活跃市值 CSV 路径
MACRO_CSV_WIN = r"C:\new_tdx_test\PYPlugins\file\0AMV_daily_official.csv"
MACRO_CSV_WSL = "/mnt/c/new_tdx_test/PYPlugins/file/0AMV_daily_official.csv"


# ============================================================
# 宏观状态机
# ============================================================

def compute_macro_states(ret_series):
    """
    基于活跃市值收益率推演多空状态。
    
    参数
    ---
    ret_series : array-like
        每日收益率序列，时间从早到晚（最新在最后）
        
    返回
    ---
    numpy.ndarray : bool 数组，True=多头安全，False=空头锁定
    """
    ret_values = np.asarray(ret_series, dtype=float)
    n = len(ret_values)
    states = np.ones(n, dtype=bool)
    state = True

    for i in range(1, n):
        cur = ret_values[i]
        two_day = cur + ret_values[i - 1]
        if cur <= MACRO_BEAR_THRESHOLD:
            state = False
        elif cur >= MACRO_BULL_THRESHOLD or two_day > MACRO_BULL_TWO_DAY:
            state = True
        states[i] = state

    return states


def read_macro_csv(csv_path=None):
    """
    读取活跃市值 CSV，返回 (ret_series, dates)。
    
    参数
    ---
    csv_path : str or None
        CSV 路径，None 则自动在 Win / WSL 路径中查找
        
    返回
    ---
    (ret_series: pd.Series, dates: pd.DatetimeIndex) or (None, None)
    """
    import os
    if csv_path is None:
        csv_path = MACRO_CSV_WIN if os.path.exists(MACRO_CSV_WIN) else MACRO_CSV_WSL
    if not os.path.exists(csv_path):
        return None, None

    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "ret_1d" in raw.columns:
        ret = pd.to_numeric(raw["ret_1d"], errors="coerce").fillna(0)
    else:
        close = pd.to_numeric(raw["close"], errors="coerce")
        ret = close.pct_change().fillna(0)
    dates = (pd.to_datetime(raw["date"])
             if "date" in raw.columns
             else pd.to_datetime(raw.iloc[:, 0]))
    ret.index = dates
    return ret, dates


def get_macro_signal(csv_path=None, verbose=True):
    """
    获取当前宏观状态（最近一个交易日的信号）。
    
    返回 bool : True=安全可开仓, False=空头锁定
    """
    ret, dates = read_macro_csv(csv_path)
    if ret is None:
        if verbose:
            print("警告：未找到活跃市值文件，默认全天候安全。")
        return True

    states = compute_macro_states(ret.values)
    is_safe = states[-1]

    if verbose:
        tag = "多头区间" if is_safe else "空头区间"
        prev_ret = ret.values[-2] if len(ret) > 1 else 0
        print(f"宏观环境【{tag}】：安全={is_safe}  "
              f"(近两日 {prev_ret*100:.2f}%, {ret.values[-1]*100:.2f}%)")
    return is_safe


def get_macro_states_df(csv_path=None):
    """
    获取完整历史宏观状态 DataFrame，每日期 signa。
    
    返回 pd.DataFrame index=dates, columns=['is_safe']
    """
    ret, dates = read_macro_csv(csv_path)
    if ret is None:
        return None
    states = compute_macro_states(ret.values)
    return pd.DataFrame({"is_safe": states}, index=dates)


# ============================================================
# 选股条件计算（通达信数据格式）
# ============================================================

def compute_conditions(data_dict):
    """
    对通达信格式的 KLine dict 计算 7 个选股条件。

    参数
    ---
    data_dict : dict of pd.DataFrame
        形如 {'Open': df, 'High': df, ...}，每个 df 的
        index=交易日(≥MA_YELLOW 行), columns=股票代码

    返回
    ---
    pd.DataFrame : 每行一只股票，7列 bool 条件 + 详细指标
    """
    close = data_dict["Close"].astype(float)
    open_ = data_dict["Open"].astype(float)
    high = data_dict["High"].astype(float)
    low = data_dict["Low"].astype(float)
    vol = data_dict["Volume"].astype(float)
    amt = data_dict["Amount"].astype(float)

    ma_white_close = close.rolling(window=MA_WHITE).mean()
    ma_yellow_close = close.rolling(window=MA_YELLOW).mean()
    ma_vol = vol.rolling(window=MA_WHITE).mean()
    # VWAP: Amount(万元) → 元 / Volume(手=100股) → 股
    vwap = (amt * AMOUNT_SCALE) / (vol * VOLUME_HAND).replace(0, np.nan)

    # 取最新一期
    lc = close.iloc[-1]
    lo = open_.iloc[-1]
    lh = high.iloc[-1]
    ll = low.iloc[-1]
    lv = vol.iloc[-1]

    body = lc - lo
    total_range = lh - ll
    total_range = total_range.replace(0, np.nan)
    upper = lh - lc

    # Pullback: 前 N 天至少有一天收阴
    pullback = (close.iloc[-(PULLBACK_LOOKBACK + 1):-1]
                < open_.iloc[-(PULLBACK_LOOKBACK + 1):-1]).any()

    # §3.2 双均线 stack：收盘价 > 黄线 AND 白线 > 黄线（renko.md「黄线之下不操作」）
    latest_ma_white = ma_white_close.iloc[-1]
    latest_ma_yellow = ma_yellow_close.iloc[-1]
    is_dual_ma_bull = (lc > latest_ma_yellow) & (latest_ma_white > latest_ma_yellow)

    cond = pd.DataFrame({
        "is_red": lc > lo,
        "is_strong_red": (body / total_range) >= STRONG_RED_THRESHOLD,
        "is_short_shadow": (upper / total_range) <= SHORT_SHADOW_THRESHOLD,
        "is_dual_ma_bull": is_dual_ma_bull,
        "has_pullback": pullback,
        "is_high_vol": lv > (HIGH_VOL_THRESHOLD * ma_vol.iloc[-1]),
        "is_above_vwap": lc > vwap.iloc[-1],
    })

    # 附加诊断列
    cond["close"] = lc
    cond["vwap"] = vwap.iloc[-1]
    cond["ma_white"] = latest_ma_white
    cond["ma_yellow"] = latest_ma_yellow
    cond["vol_ratio"] = lv / ma_vol.iloc[-1]
    cond["body_ratio"] = body / total_range
    return cond


def pick(data_dict):
    """
    完整选股流程：计算条件 → AND 过滤 → 返回股票代码列表。

    参数
    ---
    data_dict : dict of pd.DataFrame
        通达信格式的K线数据

    返回
    ---
    selected : list[str]  符合条件的股票代码
    details : pd.DataFrame  详细条件表
    """
    details = compute_conditions(data_dict)
    signal = details[CONDITION_NAMES].all(axis=1)
    selected = signal[signal].index.tolist()
    return selected, details


# ============================================================
# 出场判定（持仓状态下的红线 / 黄线 / 白线 / 次日不涨 / 数四块砖）
# ============================================================

def _consecutive_below(close_df, ma_df, lookback):
    """Count trailing consecutive bars where close < ma, per column.

    Returns a Series indexed by columns with the streak length (0 if today is
    not below the MA, otherwise the count of consecutive trailing days below).
    """
    below = (close_df < ma_df).iloc[-lookback:]  # latest `lookback` rows
    # Walk from the most recent bar backward; stop at the first non-break day.
    streaks = {}
    for code in below.columns:
        col = below[code].values
        n = 0
        for v in col[::-1]:
            if bool(v):
                n += 1
            else:
                break
        streaks[code] = n
    return pd.Series(streaks)


def compute_exit_conditions(data_dict, entry_records, red_brick=None):
    """Per-holding sell-decision table implementing renko.md §5.1/§5.2.

    Parameters
    ----------
    data_dict : dict of pd.DataFrame
        Tongdaxin OHLCV+Amount frames covering at least ``MA_YELLOW + WHITE_GRACE_BARS + 1``
        bars; columns are the held stock codes (subset of the universe is fine, the
        function will reindex to the entries supplied below).
    entry_records : dict[str, dict]
        ``{code: {"entry_date": pd.Timestamp, "entry_close": float,
                  "red_brick_count": int}}``.
        ``red_brick_count`` is the cumulative number of red bricks seen since
        entry (entry bar itself counts as brick #1 per §5.2).
    red_brick : pd.DataFrame, optional
        Boolean rising-brick mask aligned to ``data_dict``; if supplied the
        latest bar's red-brick flag is added to the running ``red_brick_count``
        and the four-brick exit fires accordingly. Pass ``None`` to skip the
        brick check (e.g. when running the picker without the砖型图 alpha).

    Returns
    -------
    pd.DataFrame indexed by held stock code, with columns:
        ``should_exit`` (bool), ``exit_reason`` (str; '' when not exiting),
        ``yellow_break``, ``white_break_streak``, ``no_rise``, ``red_brick_count``,
        ``close``, ``entry_close``.
    The first matching rule (yellow → white → no-rise → brick) wins, mirroring
    the priority used by :func:`src.research.metrics.event_positions`.
    """
    close = data_dict["Close"].astype(float)
    held = list(entry_records.keys())
    close = close.reindex(columns=held)
    ma_white = close.rolling(window=MA_WHITE).mean()
    ma_yellow = close.rolling(window=MA_YELLOW).mean()

    latest_dt = close.index[-1]
    lc = close.iloc[-1]
    lw = ma_white.iloc[-1]
    ly = ma_yellow.iloc[-1]

    yellow_break = (lc < ly).fillna(False)
    white_streak = _consecutive_below(close, ma_white, WHITE_GRACE_BARS + 1)
    white_break = white_streak >= (WHITE_GRACE_BARS + 1)

    # 次日不涨即走：仅在持仓 age == NO_RISE_CHECK_BARS 的当日触发。
    no_rise_flags = {}
    for code in held:
        rec = entry_records[code]
        entry_dt = pd.Timestamp(rec["entry_date"])
        age = close.index.get_indexer([latest_dt])[0] - close.index.get_indexer([entry_dt])[0]
        if age != NO_RISE_CHECK_BARS:
            no_rise_flags[code] = False
            continue
        c_val = lc.get(code)
        entry_close = float(rec.get("entry_close", float("nan")))
        no_rise_flags[code] = (
            pd.isna(c_val) or pd.isna(entry_close) or float(c_val) <= entry_close
        )
    no_rise = pd.Series(no_rise_flags)

    brick_count = pd.Series(
        {code: int(entry_records[code].get("red_brick_count", 0)) for code in held}
    )
    if red_brick is not None:
        rb_today = red_brick.reindex(columns=held).iloc[-1].fillna(False).astype(bool)
        brick_count = brick_count + rb_today.astype(int)
    brick_exit = brick_count >= 4

    reasons = []
    should = []
    for code in held:
        if bool(yellow_break.get(code, False)):
            reasons.append("yellow_break")
            should.append(True)
        elif bool(white_break.get(code, False)):
            reasons.append("white_grace_break")
            should.append(True)
        elif bool(no_rise.get(code, False)):
            reasons.append("no_rise_t" + str(NO_RISE_CHECK_BARS))
            should.append(True)
        elif bool(brick_exit.get(code, False)):
            reasons.append("four_red_bricks")
            should.append(True)
        else:
            reasons.append("")
            should.append(False)

    return pd.DataFrame({
        "should_exit": should,
        "exit_reason": reasons,
        "yellow_break": yellow_break.reindex(held).fillna(False).astype(bool).values,
        "white_break_streak": white_streak.reindex(held).fillna(0).astype(int).values,
        "no_rise": no_rise.reindex(held).fillna(False).astype(bool).values,
        "red_brick_count": brick_count.reindex(held).fillna(0).astype(int).values,
        "close": lc.reindex(held).values,
        "entry_close": [
            float(entry_records[code].get("entry_close", float("nan"))) for code in held
        ],
    }, index=pd.Index(held, name="code"))
