import pandas as pd
import numpy as np
import json
import os

def generate_html_report(csv_path, out_html_path):
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path, names=['date', 'daily_return'], header=0)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # Calculate cumulative return
    df['cum_return'] = (1 + df['daily_return']).cumprod()
    
    # Extract data for JS
    dates = df.index.strftime('%Y-%m-%d').tolist()
    cum_returns = df['cum_return'].tolist()
    
    # Calculate yearly metrics
    yearly_returns = df['daily_return'].groupby(df.index.year).apply(lambda x: x.mean() * 252 * 100)
    
    # Calculate overall metrics
    ann_ret = df['daily_return'].mean() * 252 * 100
    ann_vol = df['daily_return'].std() * np.sqrt(252) * 100
    sharpe = (df['daily_return'].mean() / df['daily_return'].std()) * np.sqrt(252) if df['daily_return'].std() > 0 else 0
    total_cum = df['cum_return'].iloc[-1]
    
    yearly_trs = ""
    for year, ret in yearly_returns.items():
        color = "#10b981" if ret > 0 else "#ef4444"
        yearly_trs += f"<tr><td>{year}</td><td style='color: {color};'>{ret:.2f}%</td></tr>\n"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fused Strategy Backtest Report (2015-2020)</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; margin: 0; padding: 2rem; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ font-size: 2.5rem; font-weight: 700; margin-bottom: 2rem; text-align: center; background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .metrics {{ display: flex; gap: 1.5rem; margin-bottom: 2rem; flex-wrap: wrap; justify-content: center; }}
        .card {{ background: #1e293b; padding: 1.5rem; border-radius: 1rem; flex: 1; min-width: 200px; text-align: center; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1); border: 1px solid #334155; transition: transform 0.2s; }}
        .card:hover {{ transform: translateY(-5px); }}
        .card h3 {{ margin: 0; font-size: 1rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }}
        .card p {{ margin: 0.5rem 0 0; font-size: 2rem; font-weight: 700; color: #f1f5f9; }}
        .chart-container {{ background: #1e293b; border-radius: 1rem; padding: 1.5rem; height: 500px; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.3); }}
        .layout {{ display: grid; grid-template-columns: 3fr 1fr; gap: 1.5rem; }}
        .yearly-table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 1rem; overflow: hidden; border: 1px solid #334155; }}
        .yearly-table th, .yearly-table td {{ padding: 1rem; text-align: center; border-bottom: 1px solid #334155; font-size: 1.1rem; }}
        .yearly-table th {{ background: #0f172a; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
        .yearly-table tr:last-child td {{ border-bottom: none; }}
        .yearly-table tr:hover {{ background-color: #2dd4bf11; }}
        
        @media (max-width: 900px) {{
            .layout {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Quant Strategy Performance (2015 - 2020)</h1>
        
        <div class="metrics">
            <div class="card">
                <h3>Total Cum. Return</h3>
                <p style="color: #34d399;">{total_cum:.2f}x</p>
            </div>
            <div class="card">
                <h3>Annualized Return</h3>
                <p style="color: #38bdf8;">{ann_ret:.2f}%</p>
            </div>
            <div class="card">
                <h3>Annualized Volatility</h3>
                <p style="color: #fbbf24;">{ann_vol:.2f}%</p>
            </div>
            <div class="card">
                <h3>Sharpe Ratio</h3>
                <p style="color: #a78bfa;">{sharpe:.2f}</p>
            </div>
        </div>

        <div class="layout">
            <div id="chart" class="chart-container"></div>
            <div>
                <table class="yearly-table">
                    <thead>
                        <tr>
                            <th>Year</th>
                            <th>Return</th>
                        </tr>
                    </thead>
                    <tbody>
                        {yearly_trs}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        var chartDom = document.getElementById('chart');
        var myChart = echarts.init(chartDom, 'dark');
        var option;

        option = {{
            backgroundColor: 'transparent',
            tooltip: {{
                trigger: 'axis',
                axisPointer: {{
                    type: 'cross',
                    label: {{ backgroundColor: '#6a7985' }}
                }}
            }},
            grid: {{
                top: '5%',
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            }},
            xAxis: [
                {{
                    type: 'category',
                    boundaryGap: false,
                    data: {json.dumps(dates)},
                    axisLine: {{ lineStyle: {{ color: '#475569' }} }},
                    axisLabel: {{ color: '#94a3b8' }}
                }}
            ],
            yAxis: [
                {{
                    type: 'value',
                    axisLine: {{ lineStyle: {{ color: '#475569' }} }},
                    axisLabel: {{ color: '#94a3b8' }},
                    splitLine: {{ lineStyle: {{ color: '#334155', type: 'dashed' }} }}
                }}
            ],
            dataZoom: [
                {{
                    type: 'inside',
                    start: 0,
                    end: 100
                }},
                {{
                    start: 0,
                    end: 100,
                    borderColor: 'transparent',
                    backgroundColor: '#0f172a',
                    fillerColor: 'rgba(56, 189, 248, 0.2)',
                    handleStyle: {{ color: '#38bdf8' }}
                }}
            ],
            series: [
                {{
                    name: 'Cumulative Return',
                    type: 'line',
                    smooth: true,
                    lineStyle: {{
                        width: 3,
                        color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                            {{ offset: 0, color: '#38bdf8' }},
                            {{ offset: 1, color: '#818cf8' }}
                        ])
                    }},
                    showSymbol: false,
                    areaStyle: {{
                        opacity: 0.2,
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            {{ offset: 0, color: '#38bdf8' }},
                            {{ offset: 1, color: 'rgba(56, 189, 248, 0)' }}
                        ])
                    }},
                    data: {json.dumps(cum_returns)}
                }}
            ]
        }};

        option && myChart.setOption(option);
        
        window.addEventListener('resize', function() {{
            myChart.resize();
        }});
    </script>
</body>
</html>"""
    
    with open(out_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Successfully generated HTML report at {out_html_path}")

if __name__ == "__main__":
    csv_in = "/mnt/c/new_tdx_test/renko_research/results/fused_strategy_qlib_returns_2015_2020.csv"
    html_out = "/mnt/c/new_tdx_test/renko_research/results/backtest_report_2015_2020.html"
    generate_html_report(csv_in, html_out)
