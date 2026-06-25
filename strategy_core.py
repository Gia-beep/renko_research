"""
融合策略核心模块 — 常量 / 宏观过滤器 / 选股条件
共 fused_strategy_picker.py（通达信实盘）和 backtest（Qlib回测）使用
"""
import numpy as np
import pandas as pd

# ============================================================
# 策略参数常量
# ============================================================
# 均线周期
MA_PERIOD = 20
# 强红砖：实体占波幅比例 >= 66.67%
STRONG_RED_THRESHOLD = 2.0 / 3.0
# 短上影：上影线占波幅比例 <= 15%
SHORT_SHADOW_THRESHOLD = 0.15
# 放量倍数：成交量 >= 1.5 × 20日均量
HIGH_VOL_THRESHOLD = 1.5
# 回调节奏：前 N 天至少有一天阴线
PULLBACK_LOOKBACK = 4

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
    "is_above_ma20": "$close > Mean($close, 20)",
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
        index=交易日(≥20行), columns=股票代码
        
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

    ma20_close = close.rolling(window=MA_PERIOD).mean()
    ma20_vol = vol.rolling(window=MA_PERIOD).mean()
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

    cond = pd.DataFrame({
        "is_red": lc > lo,
        "is_strong_red": (body / total_range) >= STRONG_RED_THRESHOLD,
        "is_short_shadow": (upper / total_range) <= SHORT_SHADOW_THRESHOLD,
        "is_above_ma20": lc > ma20_close.iloc[-1],
        "has_pullback": pullback,
        "is_high_vol": lv > (HIGH_VOL_THRESHOLD * ma20_vol.iloc[-1]),
        "is_above_vwap": lc > vwap.iloc[-1],
    })

    # 附加诊断列
    cond["close"] = lc
    cond["vwap"] = vwap.iloc[-1]
    cond["ma20"] = ma20_close.iloc[-1]
    cond["vol_ratio"] = lv / ma20_vol.iloc[-1]
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
