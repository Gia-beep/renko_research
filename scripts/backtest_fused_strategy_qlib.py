import pandas as pd
import numpy as np
import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH
# Removed unused imports
import os

# 1. Initialize Qlib
provider_uri = "~/.qlib/qlib_data/cn_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)

# 2. Define Macro Environment Filter
def get_macro_state():
    csv_path = "/mnt/c/new_tdx_test/PYPlugins/file/0AMV_daily_official.csv"
    if not os.path.exists(csv_path):
        print("警告：未找到活跃市值文件 0AMV_daily_official.csv，默认全天候安全。")
        return None
    
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "ret_1d" in raw.columns:
        ret_series = pd.to_numeric(raw["ret_1d"], errors="coerce").fillna(0)
    else:
        close_series = pd.to_numeric(raw["close"], errors="coerce")
        ret_series = close_series.pct_change().fillna(0)
        
    ret_values = ret_series.values
    dates = pd.to_datetime(raw['date']) if 'date' in raw.columns else pd.to_datetime(raw.iloc[:, 0])
    
    n = len(ret_values)
    states = np.ones(n, dtype=bool)
    state = True
    for i in range(1, n):
        current_ret = ret_values[i]
        prev_ret = ret_values[i-1]
        two_day_ret = current_ret + prev_ret
        if current_ret <= -0.023:
            state = False
        elif current_ret >= 0.04 or two_day_ret > 0.04:
            state = True
        states[i] = state
        
    macro_df = pd.DataFrame({'is_safe': states}, index=dates)
    return macro_df

# 3. Define the Qlib Expression for Fused Strategy
# fused_strategy_picker conditions:
# is_red: $close > $open
# is_strong_red: ($close - $open) / ($high - $low + 1e-8) >= 0.666667
# is_short_shadow: ($high - $close) / ($high - $low + 1e-8) <= 0.15
# is_above_ma20: $close > Mean($close, 20)
# has_pullback: Ref(Sum($close < $open, 4), 1) > 0
# is_high_vol: $volume > 1.5 * Mean($volume, 20)
# is_above_vwap: $close > $vwap

expressions = {
    "is_red": "$close > $open",
    "is_strong_red": "($close - $open) / ($high - $low + 1e-8) >= 0.666667",
    "is_short_shadow": "($high - $close) / ($high - $low + 1e-8) <= 0.15",
    "is_above_ma20": "$close > Mean($close, 20)",
    "has_pullback": "Ref(Sum($close < $open, 4), 1) > 0",
    "is_high_vol": "$volume > 1.5 * Mean($volume, 20)",
    "is_above_vwap": "$close > ($open + $high + $low + $close) / 4"
}

# The final signal is the logical AND of all these conditions
# In Qlib, True is 1.0, False is 0.0
# We can sum them up and check if the sum equals 7, or multiply them
fields = [f"({expr})" for expr in expressions.values()] + ["$close"]
names = list(expressions.keys()) + ["close"]

start_time = "2015-01-01"
end_time = "2020-12-31"
market = "csi1000" # We can use csi300 as a default pool, or 'all'

print(f"Loading data from {start_time} to {end_time} for market {market}...")
# Create a DataHandler using expressions
data_handler_args = {
    "instruments": market,
    "start_time": start_time,
    "end_time": end_time,
    "infer_processors": [],
    "learn_processors": [],
    "fit_start_time": start_time,
    "fit_end_time": end_time,
}

# D.features
features_df = D.features(D.instruments(market), fields, start_time, end_time)
features_df.columns = names

print("Individual Condition Hit Rates:")
for cond in expressions.keys():
    print(f"  {cond}: {features_df[cond].mean():.4%}")

# vwap debug removed

features_df['signal'] = features_df[list(expressions.keys())].product(axis=1)

# 4. Integrate Macro Filter
macro_df = get_macro_state()
if macro_df is not None:
    # merge by date
    features_df = features_df.reset_index()
    features_df['datetime'] = pd.to_datetime(features_df['datetime'])
    features_df = features_df.merge(macro_df, left_on='datetime', right_index=True, how='left')
    # Forward fill safe state if missing, default to True
    features_df['is_safe'] = features_df['is_safe'].fillna(True).astype(bool)
    
    # Apply macro filter to signal
    features_df['signal'] = features_df['signal'] * features_df['is_safe']
    features_df = features_df.set_index(['instrument', 'datetime'])
else:
    features_df['signal'] = features_df['signal']

# 5. Generate Backtest trades
# If signal is 1, we want to buy. Qlib TopkDropoutStrategy uses a score to rank.
# Since our signal is binary (1 or 0), we can just use the signal as a score. 
# We'll use a simple logic: hold the stocks with signal == 1 for N days, or equal weight them.
# A simple way is to use Qlib's TopkDropoutStrategy, but since we have a binary signal, it might just randomly pick among 1s if we limit k.
# Or we can just build a custom target position:
# Target position: Equal weight across all stocks with signal == 1 on that day.
# If no stock has signal == 1, target position is 0.
# We hold the position until next day. If we want to hold for N days, we can do a rolling sum.
# According to fused_strategy_picker, it's a picker. It just picks. We'll simulate a 1-day holding period for simplicity, or 5-day.

# Let's do a simple 1-day holding to evaluate the picker's immediate edge.
dates = features_df.index.get_level_values('datetime').unique().sort_values()
positions = {}

print("Generating daily target positions...")
for d in dates:
    day_data = features_df.xs(d, level='datetime')
    selected = day_data[day_data['signal'] > 0.5]
    if len(selected) > 0:
        weight = 1.0 / len(selected)
        positions[d] = {stock: weight for stock in selected.index}
    else:
        positions[d] = {}

# 6. Evaluate Returns
print("Calculating returns...")
# To evaluate returns simply without full backtest engine (which requires TopkDropout),
# we can just calculate the daily returns from close to close.
# Next day return:
features_df['next_ret'] = features_df.groupby('instrument')['close'].pct_change().shift(-1)

portfolio_returns = []
for d in dates[:-1]:
    pos = positions.get(d, {})
    if not pos:
        portfolio_returns.append(0.0)
        continue
    
    day_data = features_df.xs(d, level='datetime')
    # Get next_ret for selected stocks
    rets = day_data.loc[list(pos.keys()), 'next_ret']
    # Mean return
    port_ret = rets.mean()
    # Fill nan with 0 (in case of suspension)
    port_ret = 0.0 if pd.isna(port_ret) else port_ret
    portfolio_returns.append(port_ret)

portfolio_returns.append(0.0) # last day

ret_series = pd.Series(portfolio_returns, index=dates)

print("\nYearly Annualized Returns:")
yearly_returns = ret_series.groupby(ret_series.index.year).apply(lambda x: x.mean() * 252 * 100)
for year, ret in yearly_returns.items():
    print(f"{year}: {ret:.2f}%")

cum_ret = (1 + ret_series).cumprod()

active_days = sum(1 for pos in positions.values() if pos)
print(f"\nActive Trading Days (Days with >0 stocks picked): {active_days} / {len(dates)}")
print(f"Total Cumulative Return (1-day hold): {cum_ret.iloc[-1]:.4f}")
print(f"Overall Annualized Return: {ret_series.mean() * 252 * 100:.2f}%")
print(f"Overall Annualized Volatility: {ret_series.std() * np.sqrt(252) * 100:.2f}%")
if ret_series.std() > 0:
    print(f"Overall Sharpe Ratio: {(ret_series.mean() / ret_series.std()) * np.sqrt(252):.2f}")
else:
    print("Overall Sharpe Ratio: 0.0")

# Save results
out_dir = "/mnt/c/new_tdx_test/renko_research/results"
os.makedirs(out_dir, exist_ok=True)
ret_series.to_csv(os.path.join(out_dir, "fused_strategy_qlib_returns_2015_2020.csv"))
print(f"Results saved to {out_dir}/fused_strategy_qlib_returns_2015_2020.csv")
