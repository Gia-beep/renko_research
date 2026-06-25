# renko_research — Renko + 动量指标策略研究

基于 **TdxQuant（通达信量化 / tqcenter）** 的动量指标策略研究项目。

核心假设：先用 **Renko（砖形图）** 对价格序列降噪，再在降噪后的序列上计算
**动量类指标**（ROC / MTM / RSI / Williams %R 等），相比直接在时间序列上计算，
能减少震荡假信号、提升动量策略的稳健性。

---

## 目录结构

```
renko_research/
├── README.md               # 本文件：研究计划与使用说明
├── requirements.txt        # 依赖
├── .gitignore
│
├── config/                 # 配置（路径、参数）
│   ├── settings.py         #   tqcenter 路径解析 + 项目路径常量
│   └── params.yaml         #   所有可调参数（砖块大小、回看窗口、费率、股票池）
│
├── src/                    # 可复用代码（库）
│   ├── connection.py       #   tq.initialize 封装（上下文管理器）
│   ├── data/
│   │   ├── loader.py       #   get_market_data 封装 + 本地 parquet 缓存
│   │   └── renko.py        #   OHLC → Renko 砖块转换（fixed / ATR）
│   ├── indicators/
│   │   └── momentum.py     #   ROC / MTM / RSI / Williams %R + 横截面强度排名
│   ├── strategies/
│   │   ├── base.py         #   策略接口：generate_signals → (entries, exits)
│   │   └── renko_momentum.py  # 示例策略
│   ├── backtest/
│   │   └── engine.py       #   vectorbt 回测封装 + 绩效指标
│   └── viz/
│       └── plots.py        #   收益曲线 / 回撤 / Renko 砖块图
│
├── scripts/                # 入口脚本（可执行）
│   ├── fetch_data.py       #   批量下载股票池数据 → 缓存
│   ├── run_backtest.py     #   配置 → 策略 → 回测 → 保存结果
│   ├── select_stocks.py    #   动量排名选股 → 写入自定义板块
│   └── monitor.py          #   实时订阅 + 动量突破预警
│
├── notebooks/              # 探索性分析（Jupyter）
├── data/                   # 数据缓存（raw 原始K线 / renko 砖块），不入库
├── results/                # 回测产出（绩效、交易日志、图表、报告），不入库
└── tests/                  # 单元测试（重点：renko 转换、指标计算）
```

设计原则：`src/` 是纯函数库，`scripts/` 负责把库按研究流程串起来，`config/`
集中所有可调参数，`data/` 与 `results/` 只存产物、不入库。

---

## 研究流程（对应 TdxQuant 工作流）

| 步骤 | 模块 | 说明 |
|------|------|------|
| 1. 初始化连接 | `src/connection.py` | `tdx_session()` 上下文管理器封装 `tq.initialize/close` |
| 2. 确认股票池 | `config/params.yaml` → `universe` | 默认沪深300 / 中证500，可改板块名 |
| 3. 获取数据 | `src/data/loader.py` | `get_market_data` + 本地缓存，分钟线自动分批 |
| 4. Renko 降噪 | `src/data/renko.py` | OHLC → 砖块序列（fixed / ATR 两种砖块大小） |
| 5. 信号计算 | `src/indicators/momentum.py` + `src/strategies/` | 在砖块序列上算动量指标 → entries/exits |
| 6. 回测 / 选股 / 预警 | `src/backtest/engine.py` / `scripts/select_stocks.py` / `scripts/monitor.py` | vectorbt 回测；或选股入板块；或实时预警 |
| 7. 结果输出 | `src/viz/plots.py` → `results/` | 收益曲线、回撤、砖块图、绩效报告 |

---

## 动量指标研究清单

**待研究指标**（均在 `src/indicators/momentum.py`，纯函数，对时间序列与砖块序列通用）：

- `roc`  变动率 Rate of Change
- `mtm`  动量 Momentum
- `rsi`  相对强弱指标（Wilder 平滑）
- `williams_r`  威廉指标 %R
- `cross_sectional_rank`  横截面相对强度排名（组合选股用）
- *(待补)* MACD、KDJ、DMI/ADX —— 可直接调通达信公式 `tq.formula_zb`

**实验设计**：

1. **砖块大小敏感性** — fixed vs ATR(window, multiplier)，扫描 multiplier ∈ [1, 3]。
2. **回看窗口敏感性** — 各指标 window 网格扫描，看绩效稳定区间。
3. **时序动量 vs 横截面动量** — 单标的择时 vs 组合截面排名持有 top-N。
4. **Renko vs 时间序列对照** — 同一策略在砖块序列与原始日线上分别回测，验证降噪收益。
5. **股票池稳健性** — 沪深300 / 中证500 / 行业板块分别验证。

---

## 快速开始

```bash
pip install -r requirements.txt
```

1. 确认通达信终端已启动登录；若注册表自动探测失败，在
   `config/settings.py` 设置 `TDX_ROOT_OVERRIDE`。
2. 调整 `config/params.yaml`（股票池、砖块、指标、回测参数）。
3. 拉数据：`python scripts/fetch_data.py`
4. 跑回测：`python scripts/run_backtest.py` → 产物落在 `results/`。

> 入口脚本须先 `from config.settings import ensure_tqcenter_on_path` 并调用，
> 再 `from tqcenter import tq`（路径用 `sys.path.insert(0, ...)`，见 settings.py）。

## Qlib 砖型转折 Alpha 研究

`src/indicators/brick_alpha.py` 将通达信公式转成两个可研究对象：

- `brick_value`：公式里的 `砖型图` 连续强度因子。
- `turn_up`：公式里的 `XG`，即 `AA := REF(砖型图,1)<砖型图` 从 false 转 true 的红砖转折事件。

使用本机 Qlib 数据运行因子研究：

```bash
/home/x1843/venvs/qlib/bin/python scripts/research_brick_alpha_qlib.py \
  --provider-uri /home/x1843/.qlib/qlib_data/cn_data \
  --market csi300 \
  --start 2018-01-01 \
  --end 2020-09-25
```

脚本输出到 `results/`：

- `brick_alpha_qlib_metrics.csv`：连续因子 RankIC、分位数组合收益、事件策略代理绩效。
- `brick_alpha_qlib_events.csv`：`turn_up/XG`、红砖持有、绿砖下行等事件后的 1/3/5/10 日收益。
- `brick_alpha_qlib_daily_returns.csv`：按 `turn_up` 买入、`turn_down` 或 `max_hold` 退出的等权代理日收益。
- `brick_alpha_qlib_recent_candidates.csv`：最近触发或仍处于红砖上升状态的股票列表。

### 双周期砖型图组合（shot / long）

`scripts/research_brick_combo_qlib.py` 回测 `shot(7,3)` + `long(21,28)` 组合规则：

- `long` 连续 2 根绿色序列中，`shot` 出现红色转折时入场。
- 入场后 `long` 只作为入场过滤，不触发退出。
- 可选信号日 yellow line 过滤：默认 `MA60`，价格低于 yellow line 超过 5% 的信号剔除。
- `shot` 出现绿色立即退出；`--max-hold 0` 表示不启用日历持仓上限。

```bash
/home/x1843/venvs/qlib/bin/python scripts/research_brick_combo_qlib.py \
  --market csi300 \
  --start 2018-01-01 \
  --end 2020-09-25 \
  --output-prefix results/brick_combo_csi300_2018
```

输出 `*_comparison.csv`、`*_events.csv`、`*_daily_returns.csv`、`*_latest_candidates.csv`。
`comparison` 会同时给出原始 `shot`、`shot+yellow`、`long绿色+shot`、`long绿色+shot+yellow`
四组结果。

### 信号增强滤波研究（volume / trend / 横截面 top-K）

基线结论：`turn_up/XG` 事件有正的事件后超额收益，但每天对全市场约 1/3 的标的触发
（沪深300 样本日均约 130 只、敞口 99%），代理策略几乎等同满仓做多。`src/indicators/
signal_filters.py` 提供三类可组合（按位 AND）的滤波器来筛出更高质量的入场：

- **量能确认** `volume_ratio`：当日量能 / N 日均量 ≥ 阈值。
- **趋势环境** `above_ma` / `ma_uptrend`：仅在均线之上 / 均线上行时入场。
- **横截面 top-K** `cross_sectional_topk`：每日只保留打分最高的 K 只转折标的，直接压缩敞口。

`src/research/metrics.py`、`src/research/qlib_data.py` 抽出两脚本共用的评估与取数原语
（重构后基线脚本产出逐字节不变）。运行对比：

```bash
/home/x1843/venvs/qlib/bin/python scripts/research_signal_filters_qlib.py \
  --market csi300 --start 2018-01-01 --end 2020-09-25
```

输出 `results/signal_filter_comparison.csv`（每个变体一行：信号数、日均持仓、敞口、
总/年化收益、Sharpe、最大回撤、3/5 日事件超额与胜率）与 `signal_filter_events.csv`
（各变体 1/3/5/10 日事件后收益明细）。

### 绿红转折后 T+1/T+2 爆发样本分析

`scripts/analyze_turn_burst_samples_qlib.py` 专门抽取砖型图 `turn_up/XG` 后 T+1 或 T+2
兑现爆发上涨的历史样本。默认爆发定义为 T+1 或 T+2 任一单日收盘涨幅 ≥ 5%；本轮汇总用
`--max-valid-day-ret 0.205` 剔除复权、停复牌或特殊交易造成的极端日收益异常。

```bash
/home/x1843/venvs/qlib/bin/python scripts/analyze_turn_burst_samples_qlib.py \
  --market csi500 --start 2019-01-01 --end 2020-09-25 \
  --window-label csi500_2019 \
  --burst-threshold 0.05 --max-valid-day-ret 0.205 \
  --output-prefix results/turn_burst_xg_clean_csi500_2019
```

输出：

- `turn_burst_xg_clean_*_burst_samples.csv`：所有 T+1/T+2 爆发样本明细。
- `turn_burst_xg_clean_*_top_samples.csv`：按爆发幅度排序的前 200 个样本。
- `turn_burst_xg_clean_*_feature_contrast.csv`：爆发 vs 未爆发样本的特征均值/中位数差异。
- `turn_burst_xg_clean_*_daily.csv`：按信号日聚合的爆发簇。
- `turn_burst_xg_clean_all_*.csv`：四窗口合并结果。

四窗口合并结果（CSI300/CSI500 × 2014-2017/2019-2020）：

| 窗口 | 转折事件数 | 爆发数 | 爆发率 | T+1 爆发 | T+2 爆发 | 爆发样本中位最大涨幅 |
|---|---:|---:|---:|---:|---:|---:|
| CSI300 2014-2017 | 42,600 | 3,224 | 7.57% | 1,950 | 1,274 | 7.26% |
| CSI500 2014-2017 | 68,454 | 6,185 | 9.04% | 3,712 | 2,473 | 7.18% |
| CSI300 2019-2020 | 19,637 | 1,321 | 6.73% | 744 | 577 | 6.70% |
| CSI500 2019-2020 | 33,028 | 2,794 | 8.46% | 1,597 | 1,197 | 7.01% |
| **合计** | **163,719** | **13,524** | **8.26%** | **8,003** | **5,521** | **7.12%** |

样本共性：

1. **爆发更多是市场级反弹簇，而不是孤立个股事件**。高发日期集中在 2015-07-09、2015-08-27、
   2015-10-08、2019-02-22 等普涨修复日。
2. **T+1 比 T+2 更常见**。合计 59.2% 的爆发在 T+1 兑现，40.8% 在 T+2 兑现。
3. **爆发样本不是深度超跌形态，而是“转红时已经偏强”**。相对未爆发样本，爆发样本的 `brick_value`
   均值高约 4.74，RSI6 高约 3.52，量比高约 0.14，强红实体占比高约 5.9pct，站上 MA20 的比例高约
   7.3pct。
4. **AMV 多头环境有帮助但不是决定项**。爆发样本 AMV 多头占比约 53.9%，非爆发样本约 49.4%，边际提升
   约 4.5pct。

### 绿红转折爆发的动量 × 市值代理 × 市场状态交互

2026-06-20 继续补充交互诊断。注意：本机 Qlib 日频数据只有 OHLCV、factor、change，缺少逐日总市值/
流通市值字段；这里先用 **CSI300 = 大盘代理、CSI500 = 中小盘代理**，不是严格的市值分位。若后续从
TdxQuant 财务/行情字段补齐市值，可把同一套分桶逻辑替换为真实市值分层。

新增输出：

- `turn_burst_xg_clean_*_events.csv`：完整转折事件明细（含未爆发样本）。
- `turn_burst_xg_clean_*_interaction_regime.csv`：市场池 × AMV 状态。
- `turn_burst_xg_clean_*_interaction_size_momentum.csv`：市场池 × 动量/量能分位。
- `turn_burst_xg_clean_*_interaction_size_momentum_regime.csv`：市场池 × AMV 状态 × 动量/量能分位。
- `turn_burst_xg_clean_all_interaction_*_by_window.csv`：四窗口逐窗口交互稳健性表。

合并四窗口后的关键结果：

| 分组 | CSI300 爆发率 | CSI500 爆发率 | 结论 |
|---|---:|---:|---|
| AMV 空头 | 6.46% | 8.27% | 中小盘代理池弹性更强，但也更依赖市场级反弹 |
| AMV 多头 | 8.16% | 9.42% | 市场状态改善会同步抬升两类股票池 |
| `ret3_to_t` Q1 超跌 | 8.39% | 10.01% | 存在恐慌修复爆发，不是传统动量延续 |
| `ret3_to_t` Q3 中位 | 4.43% | 5.54% | 中间状态最弱，确认非线性 |
| `ret3_to_t` Q5 强势 | 12.99% | 15.41% | 转红时已经偏强的样本最容易 T+1/T+2 爆发 |
| `volume_ratio20` Q1 | 4.58% | 5.77% | 缩量转红质量差 |
| `volume_ratio20` Q5 | 10.25% | 11.34% | 放量确认仍是最稳健的增强项 |

动量与市场状态的交互更细：

- `ret3_to_t` Q5 强势组在 AMV 多头下继续增强：CSI300 14.04%，CSI500 15.97%。
- `ret3_to_t` Q1 超跌组在 AMV 空头下反而更高：CSI300 9.32%，CSI500 11.20%，说明这类爆发更多是
  空头环境里的急跌修复，不应和趋势动量混用。
- 四个窗口里 `ret3_to_t`、`rsi6`、`volume_ratio20` 的最高分桶均为最佳分桶，方向稳定；但
  `ret3_to_t` 的 Q1 也明显高于 Q2/Q3，呈“两端抬升、强势端最高”的形态。

结论：绿红转折后的 T+1/T+2 爆发不是单一动量因子。更准确的解释是 **市值代理池决定弹性上限，
AMV 决定风险状态，动量分桶呈非线性**：强势转红是主线，超跌转红是另一类短促修复，两者应在模型中
分开编码，而不是只用一个线性动量权重。

### 绿红转折后的 T+5~T+20 路径回测

`scripts/analyze_turn_forward_paths_qlib.py` 将同一批 `turn_up/XG` 事件扩展到 T+1~T+20 路径。分组仍用
`ret3_to_t` 五分位：Q1 为超跌端，Q5 为强势端。收益同时输出两种口径：

- `close_ret_tN`：相对信号日收盘的 T+N 收盘收益，用于观察事件后价格路径。
- `open_entry_ret_tN`：按 A 股可执行性更接近的 T+1 开盘追入，到 T+N 收盘的收益。

```bash
/home/x1843/venvs/qlib/bin/python scripts/analyze_turn_forward_paths_qlib.py \
  --market csi500 --start 2019-01-01 --end 2020-09-25 \
  --window-label csi500_2019 \
  --max-valid-day-ret 0.205 \
  --output-prefix results/turn_forward_paths_csi500_2019

/home/x1843/venvs/qlib/bin/python scripts/aggregate_turn_forward_paths.py \
  results/turn_forward_paths_csi300_2014_2017_events.csv \
  results/turn_forward_paths_csi500_2014_2017_events.csv \
  results/turn_forward_paths_csi300_2019_events.csv \
  results/turn_forward_paths_csi500_2019_events.csv \
  --output-prefix results/turn_forward_paths_all --save-events
```

主要输出：

- `turn_forward_paths_*_events.csv`：事件级 T+1~T+20 路径。
- `turn_forward_paths_*_horizon_summary.csv`：T+5 到 T+20 逐 horizon 汇总。
- `turn_forward_paths_*_key_summary.csv`：T+5/T+10/T+20 和后半程关键指标。
- `turn_forward_paths_all_*`：四窗口合并结果。

四窗口合并的关键路径：

| 市场池 | 分组 | T+1 开盘缺口 | T+5 收盘 | T+5 开盘追入 | T+20 收盘 | T+20 开盘追入 | T+5→T+20 | T+20 相对前5日高点 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CSI300 | Q1 超跌 | −0.09% | −0.08% | −0.00% | +0.96% | +1.03% | +1.08% | −3.59% |
| CSI300 | Q5 强势 | +0.05% | +0.53% | +0.47% | +1.98% | +1.92% | +1.40% | −3.86% |
| CSI500 | Q1 超跌 | −0.05% | +0.10% | +0.14% | +1.73% | +1.77% | +1.71% | −3.41% |
| CSI500 | Q5 强势 | +0.06% | +0.69% | +0.62% | +2.40% | +2.36% | +1.72% | −3.97% |

结论：

1. **强势端不是平均意义上的“高开后亏损”**。Q5 到 T+20 仍是收益最高的一端；但它的 T+1 开盘缺口最高，
   T+1 开盘追入收益略低于信号收盘口径，且 T+20 相对前 5 日高点回吐约 −3.9%，追高主要风险是
   “早期冲高后的持仓回撤”，不是立即失效。
2. **超跌端后半程补涨成立**。Q1 的 T+5 最弱，CSI300 甚至仍为负；但 T+5→T+20 转正，CSI500 超跌端
   后半程收益 +1.71%，几乎追平强势端的 +1.72%。这说明超跌转红更适合等待修复，而不是只看
   T+1/T+2 爆发。
3. **市场状态改变路径形态**。AMV 多头下强势端 T+5 表现最好；AMV 空头下 Q1/Q5 的 T+20 和 T+5→T+20
   反而更强，更多来自急跌后的反弹修复。模型中应把“强势延续”和“空头修复”分开建特征。

**三窗口稳健性结论**（沪深300 / 中证500 两个池 + 2014-2017 / 2018-2020 两段独立区间）：

| 变体（均为 `turn_up` 子集） | 事件后超额（3 日） | 跨窗口一致性 |
|---|---|---|
| `vol_ge_1.5x`（量能确认） | +0.36% ~ +0.48% | **稳健为正**，且把日均持仓从 ~130 压到 ~20 |
| `vol1.5_trend_ma20`（量能+趋势） | +0.38% ~ +0.54% | 单笔超额最高，但交易更少、参与度更低 |
| `topk20_by_brick` / `topk20_by_roc`（按强度排名） | −0.02% ~ −0.18% | **稳健为负**：高 brick / 高动量的转折标的反而跑输 |

要点：

1. **量能确认是唯一稳健的信号增强器** —— 三个窗口里都抬升事件后超额，并大幅降低持仓宽度。
2. **按 `brick_value` / 动量做横截面排名会破坏信号**（事件后超额转负），与连续因子的
   倒 U 形 / 弱反转特征一致；“挑最强的转折”这条路走不通。
3. **代理策略的绝对收益由市场 β 主导**：2014-2017 牛市里基线满仓做多 Sharpe 反而最高
   （+218%），2018-2020 中证500 阴跌时所有变体都亏 —— 滤波器改善的是“相对/单笔”质量，
   并不去除市场敞口，也未压住 ~−40% 的回撤。

**已验证的下一步**：见下一节。量能确认的单笔超额在市场中性诊断下仍能保留，但 A 股普通股票组合
不能直接做空个股；因此对冲研究只用于识别更好的多头过滤条件，落地仍回到多头/空仓框架。

### 市场中性 / 对冲研究（alpha 诊断，不作为 A 股直接交易方案）

`scripts/research_hedged_brick_alpha_qlib.py` 继续验证上一节遗留的问题：滤波器改善的是单笔相对收益，
还是只是降低/切换市场 β。A 股普通股票组合不能直接做空个股，所以下面的 long/short 只用于判断
信号是否有相对 alpha，不作为可落地交易方案。脚本输出两套诊断口径：

- **active-day benchmark hedge**：有持仓日才扣等权股票池基准，空仓日保持 0，不隐含裸空指数。
- **event long/short**：50% 做多 `turn_up` 变体、50% 做空 `turn_down` 变体；默认要求多空两侧同时有仓。

```bash
/home/x1843/venvs/qlib/bin/python scripts/research_hedged_brick_alpha_qlib.py \
  --market csi300 --start 2018-01-01 --end 2020-09-25 \
  --output-prefix results/hedged_brick_alpha_csi300_2018
# 另两窗口：--market csi500 同期；--market csi300 --start 2014-01-01 --end 2017-12-31
```

输出：

- `hedged_brick_alpha_*_comparison.csv`：多头变体的原始收益、active-day 基准对冲收益、beta、事件超额。
- `hedged_brick_alpha_*_long_short.csv`：事件多空组合收益、Sharpe、回撤、beta、两侧持仓宽度。
- `hedged_brick_alpha_*_events.csv`：多头/空头事件各 horizon 的事件后收益。

三窗口关键结果：

| 口径 / 变体 | 300 '18 Sh / maxDD | 500 '18 Sh / maxDD | 300 '14 Sh / maxDD | 结论 |
|---|---|---|---|---|
| active hedge `vol1.5_amv` | 1.18 / -13.1% | -0.75 / -33.0% | 0.59 / -11.4% | 长多对冲后仍受股票池影响 |
| active hedge `renko_full_plus_vol` | 0.57 / -18.3% | 0.17 / -28.1% | 1.06 / -15.3% | 三窗口为正，但中证500 边际较弱 |
| L/S `vol1.5` vs `vol1.5 turn_down` | 1.96 / -7.6% | 0.58 / -15.4% | 1.05 / -18.6% | 量能多空有效，但窗口差异大 |
| **L/S `renko_full_plus_vol` vs `vol1.5 turn_down_amv`** | **1.12 / -10.1%** | **1.07 / -10.5%** | **1.25 / -9.0%** | **诊断上最稳健，beta 近 0；A 股不可直接照搬** |

研究结论：

1. **`turn_down` 空腿证明了相对 alpha 存在，但不能作为 A 股普通股票组合的执行腿**。它的价值是帮我们
   识别哪些多头过滤条件真的改善了信号，而不是被市场 β 掩盖。
2. **可落地版本应退回多头/空仓框架**：用 `amv_regime` 控制是否开仓，用 `vol_ge_1.5x` 做量能确认；
   若需要进一步压缩候选池，再叠加 `strong_red` / `dual_ma_bull`，但不能期待它们单独解决回撤。
3. **多头版本不能只靠个股过滤解决市场状态问题**。`vol1.5_amv` 在 CSI300 很好，但中证500 2018-2020
   对冲后为负；这说明股票池和市场阶段仍是主要风险源。
4. 横截面 top-K / 强度排名已被前一节证伪，本轮没有继续投入。

### Long-Only 反转信号研究（短跌 / RSI / 下影线 / 放量）

`scripts/research_reversal_signals_qlib.py` 用可解释规则寻找更高质量的反转候选，不使用个股空腿。
所有 forward return 默认延后一根 K 线入场（`entry_shift=1`），更接近 A 股次日可执行约束。

```bash
/home/x1843/venvs/qlib/bin/python scripts/research_reversal_signals_qlib.py \
  --market csi500 --start 2019-01-01 --end 2020-09-25 \
  --output-prefix results/reversal_signals_csi500_2019
# 稳健性窗口：csi300/csi500 × 2014-2017 / 2019-2020
```

输出：

- `reversal_signals_*_comparison.csv`：各信号在指定 horizon 下的事件收益、胜率、盈亏比、Profit Factor。
- `reversal_signals_*_h{N}_summary.csv`：第 N 日 horizon 摘要，按组合级 Profit Factor 排序。
- `reversal_signals_h10_aggregate.csv`：四窗口聚合稳健性排序。
- `reversal_signals_h4_h6_aggregate.csv`：4-6 日短周期四窗口聚合稳健性排序。

关键候选定义：

- `ret3_bottom30_amv`：AMV 多头期，每日选 3 日跌幅最差的 30 只。
- `ret3_bottom30_strong_rebound_top15`：3 日跌幅最差 30 只中，当日反弹强度最高的 15 只。
- `rsi6_bottom30_amv`：AMV 多头期，每日选 RSI6 最低的 30 只。
- `panic_shadow_vol`：10 日跌幅最差 30 只中，长下影（下影/全日振幅 ≥ 40%）且量比 ≥ 1.5。
- `ret5_bottom30_vol_rebound`：5 日跌幅最差 30 只中，量比 ≥ 1.5 且当日收涨。
- `brick_turn_rsi_recover`：砖型图 `turn_up/XG` 同时 RSI6 从弱势区回升。

四窗口 10 日 horizon 聚合结果（CSI300/CSI500 × 2014-2017/2019-2020）：

| 信号 | 最低组合 PF | 中位组合 PF | 平均10日组合收益 | 平均胜率 | 平均持仓宽度 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| **`rsi6_bottom30_amv`** | **1.33** | **1.35** | 0.76% | 57.5% | 30.0 | 最稳健的反转候选池，适合做 ML/规则基座 |
| **`panic_shadow_vol`** | **1.24** | **1.43** | **1.80%** | 58.5% | 1.8 | 最高质量但很稀疏，适合做强反转确认 |
| `ret5_bottom30_vol_rebound` | 1.21 | 1.60 | 1.31% | 54.3% | 2.3 | 弹性好，2019-2020 很强，但牛市窗口稳定性弱一点 |
| `brick_turn_rsi_recover` | 1.16 | 1.41 | 0.93% | 56.3% | 29.9 | 可作为砖型图转红后的辅助确认 |

结论：

1. **更优质的反转不是单纯“跌得多”**。`ret5/ret10_bottom30` 本身可用，但加入 RSI、下影线、放量或转红确认后，
   盈亏比明显改善。
2. **首选稳健基座**：`rsi6_bottom30_amv`。它不是最暴利，但四个窗口最低 Profit Factor 仍有 1.33，
   覆盖足够宽，适合放入 long-only ML 候选池。
3. **首选强确认信号**：`panic_shadow_vol`。它很稀疏，但 10 日平均组合收益最高，尤其适合做人工观察池或
   给 ML 模型增加“恐慌反转确认”特征。
4. **砖型图自身更适合做确认，而不是单独反转入口**：`brick_turn_rsi_recover` 稳定性尚可，但不如 RSI 超跌篮子。

四窗口 4-6 日短周期聚合结果（CSI300/CSI500 × 2014-2017/2019-2020；每个信号共 12 个
窗口-horizon 样本点）：

| 信号 | 最低组合 PF | 中位组合 PF | 平均4-6日组合收益 | 平均胜率 | 平均持仓宽度 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| **`ret3_bottom30_strong_rebound_top15`** | **1.29** | 1.36 | **0.55%** | 55.4% | 15.0 | 短周期最佳：3 日超跌后选当日反弹最强的一半 |
| **`ret3_bottom30_amv`** | **1.27** | 1.39 | 0.53% | 55.7% | 30.0 | 更宽的短周期候选池，适合做 ML 基座 |
| **`rsi6_bottom30_amv`** | **1.23** | **1.42** | 0.52% | **56.1%** | 30.0 | 仍然稳健，但短周期不如 3 日超跌反弹 |
| `ret5_bottom30_amv` | 1.21 | 1.32 | 0.43% | 54.9% | 30.0 | 可用，但短周期弱于 3 日口径 |
| `uptrend_pullback_ret5` | 1.19 | 1.28 | 0.45% | 53.9% | 15.0 | 趋势回撤型备选，超额稳定性较弱 |

短周期结论：

1. **4-6 日窗口首选 `ret3_bottom30_strong_rebound_top15`**。它把 3 日超跌和当日反弹动量结合起来，
   四窗口 × 三 horizon 的最低组合 PF 仍有 1.29，优于 10 日研究里的 RSI 基座。
2. **宽候选池用 `ret3_bottom30_amv` 或 `rsi6_bottom30_amv`**。前者更贴近短周期反转动量，后者胜率最高、
   与 10 日结论连续性最好。
3. **`panic_shadow_vol` 不适合 4-6 日主信号**。它平均收益高，但 CSI500 2014-2017 的 4/5/6 日 PF 分别只有
   0.73/0.77/0.91，更像 10 日级别的强反转确认。
4. **放量反弹类信号弹性大但窗口不稳**。`rsi6_bottom30_vol_rebound` 和 `ret3_bottom30_vol_rebound`
   在 2019-2020 很强，但早期窗口最差 PF 明显回落，只适合作为特征或辅助确认。

### Qlib Long-Only 机器学习模型（砖型图特征 + LGBModel）

`scripts/qlib_long_only_brick_ml.py` 是可落地到 A 股普通股票约束的纯做多机器学习入口：

- Qlib 负责取数、训练 `qlib.contrib.model.gbdt.LGBModel`、预测打分。
- `src/indicators/brick_alpha.py` 负责精确复现通达信公式里的 `SMA`、`砖型图`、`XG`。
- 回测代理只允许非负权重：每天在可交易候选池里买入模型分数最高的 top-N，候选不足则持现金。
- 默认不启用 Qlib workflow/mlflow recorder，避免当前非 git 研究目录下刷 `git diff` 噪声；需要时加
  `--use-qlib-recorder`。

默认推荐从 AMV 控仓 + ML top30 开始，而不是把当天 `XG` 作为硬门槛；`XG`、`brick_value`、红绿砖状态、
量能、均线和 K 线实体都作为特征进入模型：

```bash
/home/x1843/venvs/qlib/bin/python scripts/qlib_long_only_brick_ml.py \
  --market csi300 \
  --train-start 2014-01-01 --train-end 2017-12-31 \
  --valid-start 2018-01-01 --valid-end 2018-12-31 \
  --test-start 2019-01-01 --test-end 2020-09-25 \
  --num-boost-round 200 --early-stopping-rounds 30 \
  --output-prefix results/qlib_long_only_brick_ml_csi300
```

输出：

- `qlib_long_only_brick_ml_*_metrics.csv`：收益、Sharpe、回撤、beta、暴露率、样本行数、周期盈亏比。
- `qlib_long_only_brick_ml_*_daily_returns.csv`：策略/基准/超额日收益和 NAV。
- `qlib_long_only_brick_ml_*_predictions.csv`：测试集逐股模型分数。
- `qlib_long_only_brick_ml_*_weights.csv`：每日 long-only 权重，所有权重均非负且总仓位 ≤ 1。
- `qlib_long_only_brick_ml_*_feature_importance.csv`：LightGBM gain/split 重要性。
- `qlib_long_only_brick_ml_*_latest_candidates.csv`：最近实际建仓日的 top-N 候选。

CSI300 标签周期工程样例（train 2014-2017，valid 2018，test 2019-2020，`entry_gate=amv`，
top30，单边费率 0.1%；`hold_days` 默认与 `label_horizon` 对齐）：

| 标签 / 持仓 | 总收益 | Sharpe | 最大回撤 | 周期胜率 | 盈亏比 | Profit Factor | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| 5 日 / 5 日 | +22.63% | 0.73 | -19.0% | 57.1% | 1.07 | 1.43 | 收益最高，盈亏比较低但胜率较好 |
| **10 日 / 10 日** | **+20.57%** | **0.72** | **-13.9%** | 55.0% | **1.38** | **1.69** | **盈亏比和回撤最好，当前更匹配砖型趋势属性** |
| 20 日 / 20 日 | -10.11% | -0.29 | -23.8% | 40.0% | 1.17 | 0.78 | 周期过长，趋势延续不足且回撤扩大 |

对应汇总文件：`results/qlib_long_only_brick_ml_label_horizon_summary.csv`。

10 日参数市场迁移（同一套 `label_horizon=10` / `hold_days=10` / `entry_gate=amv` / top30 参数）：

| 股票池 | 总收益 | Sharpe | 最大回撤 | 周期胜率 | 盈亏比 | Profit Factor | beta | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CSI300 | +20.57% | 0.72 | -13.9% | 55.0% | 1.38 | 1.69 | 0.62 | 基准窗口，盈亏比较好 |
| CSI500 | +20.42% | 0.65 | -14.0% | 55.0% | 1.08 | 1.32 | 0.67 | 收益接近，但高波动池里亏损幅度放大、交易质量下降 |

对应汇总文件：`results/qlib_long_only_brick_ml_10d_market_transfer_summary.csv`。
本机 Qlib 数据目前只有 `all/csi100/csi300/csi500` 股票池，没有 `csi1000.txt`；补齐中证1000 成份文件后，
同一脚本可直接用 `--market csi1000` 复跑。

初步判断：这个脚本已经完成“Qlib + 砖型图指标 + long-only ML”的工程闭环，但当前特征集还不是稳定
跑赢基准的生产 alpha。标签周期上，10 日比 5 日有更好的周期盈亏比和回撤，20 日明显劣化；下一步
应围绕 10 日标签做滚动训练和特征扩展。市场迁移上，CSI500 未崩但盈亏比明显收缩，说明需要继续加入
可在 A 股落地的特征（行业中性排名、
流动性/波动率约束、基本面或指数状态特征），并用滚动训练验证，而不是回到不可交易的个股空腿。

### renko.md 完整系统回测（AMV 宏观闸门 / 强红 / 双均线 / 数四块砖 / 破位出场）

`renko.md`（v1.0 生产规格）比上面的滤波研究多出几条**从未被回测**的规则。
`scripts/research_renko_system_qlib.py` 把它们落进同一套流水线，并对每条规则做**单规则消融**
（入场组固定出场=红翻绿|`max_hold`，隔离入场效应；出场组固定入场=full，隔离出场效应）。
规则的权威定义对照了同机 `PYPlugins` 项目里更成熟的 `zxt_brick` 实现。

```bash
/home/x1843/venvs/qlib/bin/python scripts/research_renko_system_qlib.py \
  --market csi300 --start 2018-01-01 --end 2020-09-25 \
  --output-prefix results/renko_system_csi300_2018
# 另两窗口：--market csi500 同期；--market csi300 --start 2014-01-01 --end 2017-12-31
```

若要把“因子动量 + 不确定性 regime 切换”用于砖型图回测，开启 `--factor-momentum`：

```bash
/home/x1843/venvs/qlib/bin/python scripts/research_renko_system_qlib.py \
  --market csi300 --start 2018-01-01 --end 2020-09-25 \
  --factor-momentum \
  --output-prefix results/renko_system_factor_mom_csi300_2018
```

新增因子层位于 `src/indicators/factor_momentum.py`，从日线价格面板构造五类 high-is-better
因子：传统价格动量、距滚动高点回撤、残差动量、短期反转、低波动。脚本先用各因子 top-minus-bottom
组合的历史收益选择当前强势因子，再把 `turn_up` 入场限制到因子评分前 `--factor-stock-quantile`
的股票；默认还会在低不确定性环境使用因子动量，在高不确定性环境切到短期反转。输出会额外生成
`*_factor_momentum_daily.csv`，记录每日被选中的因子、低不确定性状态和通过因子过滤的股票数。

三窗口 Sharpe / 最大回撤（基准等权：300'18 +13.7%/0.33，500'18 +5.6%/0.21，300'14 +70.9%/0.64）。
2026-06-20 修正了 §5.1 破位出场实现：应为 `turn_down OR trend_break`，旧脚本误用 AND；下表为修正后结果。

| 变体（入场均为 `turn_up` 子集） | 300 '18 Sh / maxDD | 500 '18 Sh / maxDD | 300 '14 Sh / maxDD |
|---|---|---|---|
| `baseline_turn_up`（满仓代理，敞口~99%） | 0.35 / −39.7% | −0.36 / −49.5% | 1.21 / −25.6% |
| **`amv_regime`**（turn_up & AMV 多头，敞口~0.5） | **0.51 / −15.6%** | **0.36 / −20.4%** | 1.11 / −23.0% |
| `strong_red`（§4.1 强红实体） | 0.17 / −42.6% | −0.23 / −49.5% | 1.45 / −31.6% |
| `dual_ma_bull`（§3.2 白>黄多头） | −0.20 / −51.1% | −0.25 / −52.4% | 0.81 / −49.7% |
| `vol_ge_1.5x`（量能，旧稳健项） | 0.67 / −38.2% | −0.65 / −53.7% | 1.06 / −40.6% |
| `full_calendar5`（full 入场 + 日历出场） | 0.38 / −25.4% | 0.13 / −22.1% | **1.60 / −18.0%** |
| `full_brick4`（+ §5.2 数四块砖出场） | 0.39 / −29.8% | 0.47 / −22.9% | 1.45 / −27.9% |
| `full_trendbreak`（+ §5.1 破位出场，无日历帽） | 0.32 / −29.3% | −0.14 / −27.3% | 0.30 / **−72.8%** |
| `full_system`（破位 + 数四块砖全开） | 0.38 / −29.7% | **0.55 / −22.9%** | 1.45 / −28.0% |
| `full_minus_amv`（full_system 去掉 AMV） | −0.17 / −47.7% | −0.23 / −50.3% | 1.20 / −45.3% |

**三窗口稳健结论**：

1. **AMV 宏观闸门是唯一稳健的回撤控制杠杆**——这正是上一节遗留的未解问题。它把两个
   2018-2020 窗口的回撤从 ~−40%/−50% 压到 ~−16%/−20%，并把中证500 最差窗口从
   −27%/Sharpe −0.36 翻成 +14%/+0.36；牛市窗口仅微降。敞口减半（~0.45–0.56）。
   **去掉 AMV（`full_minus_amv`）每个窗口都把深回撤打回 −45%~−50%**，确证回撤改善由它而非其他滤波贡献。
2. **破位出场不能单独使用**：OR 口径修正后，`full_trendbreak` 在三个窗口都不稳，2014-2017 最大回撤
   扩到 −72.8%。**日历快出（`max_hold=5`）最稳健**（牛市最佳 1.60，其余窗口回撤 −22%~−25%）。
   **§5.2「数四块砖」止盈**比破位更有保护价值；`full_system` 的主要稳定性来自四块砖止盈，而不是破位本身。
3. **逐股入场质量滤波（强红 / 双均线）不具跨窗口稳健性**：`dual_ma_bull` 多数窗口为负、整体减分；
   `strong_red` 牛市强（+334%）但中证500 阴跌为负。与上一节一致——**稳健的是市场/量能级闸门，
   不是逐股「强度 / 质量」排名**。

**采纳建议**：多头版本默认用 **AMV 多头闸门 + 日历快出** 控回撤；破位出场只允许配合四块砖止盈，
不建议单独放开日历帽。若目标是提高 alpha 质量，只把上一节的对冲结果当作诊断工具，再回到
`turn_up & AMV & volume` 的多头候选池落地。

**三处保真度说明**（实现判定，见 `amv_regime.py` / `signal_filters.dual_ma_bull` docstring）：

- **AMV `−2.37` 不可复原**：`0AMV_daily_official.csv` 只有 OHLC+量（close ~15 万量级），
  专有振荡器的 −2.37 阈值不在数据里。这里用与 `PYPlugins/oos_regime_filter.py` 同款的
  **因果趋势闸门**（`AMV_close ≥ SMA(AMV_close, 60)`）复现「空头区间休眠」之意图（`--amv-sma` 可调）。
- **白/黄方向**：renko.md §3.2 字面写 `Yellow > White`，但权威 `zxt_brick` 用 `白 > 黄`（短>中标准多头），
  本实现采用后者并视前者为笔误；周期默认 white=20 / yellow=60（`--white/--yellow` 可调）。
- **日内规则未建模**：14:57 确认、09:30–09:37 执行、§5.1 次日 09:37 验证均为分钟级，日线无法表示，
  故省略次日验证出场（其朴素日线近似会与红翻绿 / 破位混淆）。

---

## 关键约束（来自 TdxQuant）

- `get_market_data` 单次最多 24000 条，分钟线需分批；`subscribe_hq` 最多 100 只。
- 含 SMA 的指标（RSI/KDJ/WR）批量计算时 count 要够（日线 ≥ 250）以避免初始收敛误差。
- 复权：行情 API 用字符串 `front`/`back`/`none`；公式 API 用整数 `1`/`2`/`0`。
- A股 T+1、整手（科创/北交 ≥ 200 股）；实盘下单 `order_stock` 返回 1=待确认、2=模拟成功、0=失败。
