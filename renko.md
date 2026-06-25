量化工程技术开发文档：砖形图策略系统 (Brick Chart Strategy System v1.0)
文档版本: 1.0.0
更新日期: 2026-06-19
文档状态: 生产就绪
适用框架: 事件驱动型量化交易系统

1. 系统概述
1.1 策略定位
本系统是一个基于“砖形图”形态捕捉与严格风控纪律的短线波段交易引擎。其核心 Alpha 来源于对市场“四根K线（砖）周期韵律”的捕捉，以及在特定高胜率结构定式下的动量爆发点入场。
1.2 核心执行逻辑
系统采用规则驱动、机械执行的无脑化框架，浓缩为三大底层指令：

RED_BUY (红买): 满足多级过滤条件的强红砖信号触发买入。
GREEN_SELL (绿卖): 持仓态出现红翻绿信号触发无条件平仓（最高优先级）。
COUNT_4_SELL (四砖减仓): 持仓态达到4根红砖周期触发主动止盈。

2. 系统架构与数据流
2.1 数据输入层

行情数据: 日频/分钟级 OHLCV 数据。
指标数据: 活跃市值指数 (Active Market Value, AMV)，砖形图指标。
均线数据: 白线 (White_MA, 短期趋势线)，黄线 (Yellow_MA, 中期支撑线)。

2.2 核心模块

Environment_Filter (环境过滤模块): 大盘多空状态与个股趋势校验。
Signal_Engine (信号引擎): 砖块质量评估与定式形态识别。
Execution_Engine (执行引擎): 开仓与平仓订单路由。
Risk_Manager (风控引擎): 止损、止盈、次日验证等生命周期管理。

3. 环境过滤模块
本模块为系统的前置网关，任何不满足以下条件的标的将被直接剔除，禁止进入信号引擎。
3.1 宏观市场环境校验

参数定义: AMV_Threshold = -2.37
逻辑:
if Active_Market_Value <= AMV_Threshold:
    return BLOCK_TRADE  # 空头区间，策略休眠

3.2 个股趋势状态校验

条件 1: Current_Price > Yellow_MA (黄线之上为绝对铁律)
条件 2: Yellow_MA > White_MA (均线多头排列，确保主升浪环境)

4. 信号引擎：入场逻辑
入场信号必须在交易日收盘前3分钟完成确认，并在次日早盘执行。
4.1 砖块质量评估
系统仅接受“强红”砖块作为有效信号。

强红比例: (Close - Open) / (High - Low) >= 2/3
上影线约束: (High - Max(Close, Open)) / (High - Low) <= Upper_Shadow_Limit (要求短小)

4.2 形态定式识别
信号引擎内置三大经典定式模式匹配器：
定式类型形态特征触发条件N型起跳 上涨 -> 回调 -> 再启动。整体呈"N"字。 回调结束（不破白线/黄线）后出现的第一根强红砖。 横盘起跳 长时间窄幅箱体震荡，波动率收敛。横盘平台末期，伴随成交量放大的突破性强红砖。 上升波段延续 处于清晰上升通道，沿白线稳步上行。 上升趋势中，短暂回调（如小绿砖）后立即出现的强红砖。 
4.3 订单生成

触发时间: T日 14:57 (收盘前3分钟) 确认信号。
执行时间: T+1日 09:30 - 09:37 (早盘集合竞价及开盘初段)。
动作: 买入开仓。

5. 风控与出场引擎
出场逻辑由高度量化的状态机控制，优先级从高到低依次为：
5.1 状态机：被动止损与验证

最高优先级：红翻绿离场

触发条件: 持仓任何时点（收盘前3分钟确认），砖形图颜色由红转绿。
执行动作: T日收盘前或T+1日开盘无条件市价清仓。

次高优先级：次日验证失败

触发条件: T+1日 09:33 至 09:37，股价未如期大幅拉升或处于下跌状态。
执行动作: T+1日 09:37 前无条件市价清仓。判定公式: Price(T+1, 09:37) <= Price(T, Close)

第三优先级：趋势破位止损

触发条件: 盘中价格跌破 White_MA 或 Yellow_MA。
执行动作: 服从高级别趋势信号，立即清仓。

5.2 状态机：主动止盈机制

“数四块砖”法则

触发条件: 从入场的那根红砖记为 Brick_Count = 1，后续每出现一根红砖 Brick_Count += 1。当 Brick_Count == 4 时触发。
执行动作: 于第4根红砖当日减仓50%或全部清仓，主动规避“四砖循环”变盘点。

6. 参数字典
参数名称类型默认值描述AMV_ThresholdFloat-2.37活跃市值多空分界线Yellow_MA_PeriodInt-中期趋势支撑线周期White_MA_PeriodInt-短期趋势线周期Strong_Red_RatioFloat0.667(2/3) 强红实体占比下限Brick_Cycle_LimitInt4触发主动减仓的红砖计数阈值 Next_Day_Check_EndTime09:37次日验证失效的最后卖出时点 Signal_Confirm_TimeTime14:57尾盘信号确认时间点

7. 生命周期状态机
系统针对单一标的的生命周期流转如下：
<code>stateDiagram-v2
    [*] --> Idle: 初始化
    Idle --> Monitoring: AMV > -2.37 & Price > Yellow_MA
    Monitoring --> Pending_Signal: 持续监听砖形图
    Pending_Signal --> Order_Placed: T日14:57 满足强红+定式
    Order_Placed --> Holding: T+1日09:30 买入成交
    Holding --> Exited_NextDay_Fail: T+1日09:37 股价未涨
    Holding --> Exited_Green: 砖形图红翻绿
    Holding --> Exited_Trend_Break: 跌破白线/黄线
    Holding --> Exited_Profit_Take: 数至第4根红砖
    Exited_NextDay_Fail --> Idle
    Exited_Green --> Idle
    Exited_Trend_Break --> Idle
    Exited_Profit_Take --> Idle
</code>
8. 系统失效场景警告
开发与运维人员需在日志监控系统中针对以下场景设置异常报警，因在这些场景下策略胜率将大幅下降：

逆势操作警报: 系统在 AMV < -2.37 时产生信号（表明环境过滤模块可能失效）。
接飞刀警报: 标的处于 A杀（连续暴跌）或阶梯缩量阴跌时触发信号。
低质量信号警报: 买入的红砖实体比例 < 0.5 或带有极长上影线。
执行延迟警报: 触发卖出条件（如红翻绿）后，订单延迟超过 $T+1$ 日开盘未成交。
