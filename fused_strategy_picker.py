import sys
import os
import winreg
import pandas as pd
import numpy as np

def init_tdx():
    """初始化通达信接口"""
    paths = [
        r'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\通达信金融终端64',
        r'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\通达信专业版',
        r'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\通达信金融终端(量化模拟)',
        r'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\通达信金融终端(测试)',
    ]
    
    tdx_root = None
    for p in paths:
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, p) as key:
                tdx_root, _ = winreg.QueryValueEx(key, "InstallLocation")
                break
        except FileNotFoundError:
            continue
            
    if not tdx_root:
        # Fallback to check if the current workspace parent directory is the root
        fallback_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if os.path.exists(os.path.join(fallback_root, 'PYPlugins', 'user', 'tqcenter.py')):
            tdx_root = fallback_root
        else:
            print("未找到通达信安装路径，请确认是否已安装64位通达信，或者将脚本放置于通达信目录下运行。")
            sys.exit(1)

    user_path = os.path.join(tdx_root, 'PYPlugins', 'user')
    if user_path not in sys.path:
        sys.path.insert(0, user_path)
    
    try:
        from tqcenter import tq
        tq.initialize(__file__)
        return tq
    except Exception as e:
        print(f"初始化 tqcenter 失败: {e}")
        sys.exit(1)

def check_macro_environment(tq):
    """
    第一层：大盘环境过滤 (The Gatekeeper)
    读取活跃市值CSV：0AMV_daily_official.csv
    根据状态机逻辑判断多空区间：
    1. 单日跌幅 <= -2.3% (-0.023)，进入空头区间 (False)
    2. 单日涨幅 >= 4% (0.04) 或 连续两日涨幅相加 > 4%，进入多头区间 (True)
    3. 其他情况维持上一日的状态
    """
    import os
    import pandas as pd
    
    # 兼容 Windows 盘符路径和 WSL 挂载路径
    csv_path = r"C:\new_tdx_test\PYPlugins\file\0AMV_daily_official.csv"
    if not os.path.exists(csv_path):
        csv_path = "/mnt/c/new_tdx_test/PYPlugins/file/0AMV_daily_official.csv"
        
    if not os.path.exists(csv_path):
        print(">>> 警告：未找到活跃市值文件 0AMV_daily_official.csv，默认环境安全。")
        return True
        
    try:
        print(">>> 正在读取宏观环境活跃市值数据并计算多空状态...")
        # 尝试读取包含 ret_1d 的列
        raw = pd.read_csv(csv_path, encoding="utf-8-sig")
        
        # 兼容如果CSV里没有直接叫ret_1d的情况，通过close计算
        if "ret_1d" in raw.columns:
            ret_series = pd.to_numeric(raw["ret_1d"], errors="coerce").fillna(0)
        else:
            close_series = pd.to_numeric(raw["close"], errors="coerce")
            ret_series = close_series.pct_change().fillna(0)
            
        ret_values = ret_series.values
        n = len(ret_values)
        
        # 状态机：初始状态设为 True (多头)
        state = True 
        
        # 遍历历史数据推演当前状态
        for i in range(1, n):
            current_ret = ret_values[i]
            prev_ret = ret_values[i-1]
            two_day_ret = current_ret + prev_ret
            
            # 切换状态
            if current_ret <= -0.023:
                state = False
            elif current_ret >= 0.04 or two_day_ret > 0.04:
                state = True
            # 其他情况 state 保持不变
            
        is_safe = state
        latest_ret = ret_values[-1]
        prev_ret_val = ret_values[-2] if n > 1 else 0
        
        if is_safe:
            print(f">>> 宏观环境【多头区间】：系统安全允许开仓。(近两日涨跌幅: {prev_ret_val*100:.2f}%, {latest_ret*100:.2f}%)")
        else:
            print(f">>> 宏观环境【空头区间】：系统已锁定，今日建议空仓休息。(近两日涨跌幅: {prev_ret_val*100:.2f}%, {latest_ret*100:.2f}%)")
            
        return is_safe
        
    except Exception as e:
        print(f">>> 警告：读取活跃市值数据或计算状态失败 ({e})，默认环境安全。")
        return True

def pick_stocks(tq, stock_list):
    """
    第二层 & 第三层：选股与买点触发器
    基于：量价VWAP共振 + 砖形图强红结构 + 均线趋势
    """
    print(f">>> 开始获取 {len(stock_list)} 只股票的K线数据...")
    
    # 获取过去30天的日线数据（用于计算20日均线）
    data = tq.get_market_data(
        field_list=['Open', 'High', 'Low', 'Close', 'Volume', 'Amount'],
        stock_list=stock_list,
        period='1d',
        count=30,
        dividend_type='front' # 前复权
    )
    
    if not data or 'Close' not in data:
        print("未能获取到数据。")
        return []

    close_df = data['Close'].astype(float)
    open_df = data['Open'].astype(float)
    high_df = data['High'].astype(float)
    low_df = data['Low'].astype(float)
    vol_df = data['Volume'].astype(float)
    amt_df = data['Amount'].astype(float)
    
    # --- 指标计算 ---
    # 1. 均线系统 (代表趋势的黄线，通常取20日或30日)
    ma20_close = close_df.rolling(window=20).mean()
    # 2. 成交量均线
    ma20_vol = vol_df.rolling(window=20).mean()
    
    # 3. 计算日内 VWAP (均价)
    # 通达信中 Amount单位是万元，Volume单位是手(100股)
    # VWAP = (Amount * 10000) / (Volume * 100) = Amount * 100 / Volume
    vwap_df = (amt_df * 100) / vol_df.replace(0, np.nan)
    
    # 取最近一个交易日的数据
    latest_close = close_df.iloc[-1]
    latest_open = open_df.iloc[-1]
    latest_high = high_df.iloc[-1]
    latest_low = low_df.iloc[-1]
    latest_vol = vol_df.iloc[-1]
    
    latest_ma20_close = ma20_close.iloc[-1]
    latest_ma20_vol = ma20_vol.iloc[-1]
    latest_vwap = vwap_df.iloc[-1]
    
    # --- 条件判断 ---
    
    # 1. 红砖判断：收盘 > 开盘
    is_red = latest_close > latest_open
    
    # 2. 强红砖判断：实体比例 > 2/3
    body = latest_close - latest_open
    total_range = latest_high - latest_low
    total_range = total_range.replace(0, np.nan) # 防止除以0
    is_strong_red = (body / total_range) >= (2.0 / 3.0)
    
    # 3. 形态过滤：上影线极短 (上影线占总波幅 < 15%)
    upper_shadow = latest_high - latest_close
    is_short_shadow = (upper_shadow / total_range) <= 0.15
    
    # 4. 趋势过滤：股价位于黄线之上 (Close > MA20)
    is_above_ma20 = latest_close > latest_ma20_close
    
    # 5. N型/横盘 起跳前置条件：前5天内存在调整（至少有一天是阴线或下跌）
    # 这里用一个简单的条件：过去5天内有收阴的日子，证明不是连续逼空，存在起跳支点
    has_pullback = (close_df.iloc[-5:-1] < open_df.iloc[-5:-1]).any()
    
    # 6. 量价灵魂注入：放量 > 1.5倍的20日均量
    is_high_vol = latest_vol > (1.5 * latest_ma20_vol)
    
    # 7. VWAP确认：收盘价必须严格站上日内均价
    is_above_vwap = latest_close > latest_vwap
    
    # 组合所有条件
    final_condition = (
        is_red & 
        is_strong_red & 
        is_short_shadow & 
        is_above_ma20 & 
        has_pullback & 
        is_high_vol & 
        is_above_vwap
    )
    
    # 提取符合条件的股票代码
    selected_stocks = final_condition[final_condition == True].index.tolist()
    
    # 打印部分选股的详细参数，方便复盘
    if selected_stocks:
        print(f"\n>>> 恭喜！共扫描出 {len(selected_stocks)} 只符合【量价结构共振】的标的：")
        print(f"{'代码':<12} {'收盘价':<8} {'VWAP':<8} {'MA20':<8} {'放量倍数':<10} {'实体比例'}")
        print("-" * 65)
        for code in selected_stocks:
            vol_ratio = latest_vol[code] / latest_ma20_vol[code]
            body_ratio = body[code] / total_range[code]
            print(f"{code:<12} {latest_close[code]:<8.2f} {latest_vwap[code]:<8.2f} {latest_ma20_close[code]:<8.2f} {vol_ratio:<10.2f} {body_ratio:.1%}")
    else:
        print("\n>>> 今日无符合条件的股票。坚持纪律，宁可错过不可做错。")
        
    return selected_stocks

if __name__ == "__main__":
    print("==================================================")
    print("      量价结构共振选股系统 (Renko + Vol/VWAP)     ")
    print("==================================================")
    
    # 1. 初始化 TQ API
    tq = init_tdx()
    
    try:
        # 2. 环境检查
        is_safe = check_macro_environment(tq)
        if not is_safe:
            print("警告：宏观环境恶劣，系统已锁定，今日空仓休息。")
            sys.exit(0)
            
        # 3. 准备股票池（这里以几个测试股票或某个板块成分股为例）
        # 实际使用中可以调用 tq 获取全部A股列表，此处为演示，选取一部分标的
        test_pool = [
            '600000.SH', '600519.SH', '000001.SZ', '000858.SZ', 
            '300750.SZ', '688318.SH', '002594.SZ', '601318.SH'
        ]
        
        # 获取真实环境下的某个板块（例如沪深300）成分股，这里代码供参考：
        # block_data = tq.get_market_data(..., stock_list=['000300.CSI'])
        
        # 4. 执行选股
        selected = pick_stocks(tq, test_pool)
        
    finally:
        # 5. 关闭连接
        tq.close()
        print("\n>>> 扫描结束，接口已断开。")
