"""Run a momentum strategy backtest end-to-end.

    python scripts/run_backtest.py

Maps directly onto the TdxQuant research workflow:
  1. init connection      -> src.connection.tdx_session
  2. load universe data   -> src.data.loader.load_prices (from cache)
  3. build renko series   -> src.data.renko.build_renko
  4. compute indicators   -> src.indicators.momentum
  5. generate signals     -> src.strategies.renko_momentum.RenkoMomentum
  6. backtest             -> src.backtest.engine.run_backtest / summarize
  7. save & plot results  -> src.viz.plots + results/
"""
from __future__ import annotations

from config.settings import RESULTS_DIR, load_params

# from src.data.loader import load_prices
# from src.strategies.renko_momentum import RenkoMomentum
# from src.backtest.engine import run_backtest, summarize
# from src.viz.plots import plot_equity, plot_drawdown


def main() -> None:
    params = load_params()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # close = load_prices(...); high/low/open likewise (from cache)
    # strat = RenkoMomentum(params["strategy"] | params["renko"] | params["indicators"])
    # entries, exits = strat.generate_signals(close, high=high, low=low)
    # pf = run_backtest(close, entries, exits, price=open_, bt_params=params["backtest"])
    # print(summarize(pf)); plot_equity(pf, RESULTS_DIR / "equity.png")
    raise NotImplementedError("wire steps 2–7 using params from params.yaml")


if __name__ == "__main__":
    main()
