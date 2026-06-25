"""Cross-sectional momentum selection → write picks into a 通达信 block.

    python scripts/select_stocks.py

Ranks the universe by a momentum score and writes the top-N into a custom
user block so the picks show up in the 通达信 client.
"""
from __future__ import annotations

from config.settings import load_params
from src.connection import tdx_session

# from src.data.loader import load_prices
# from src.indicators.momentum import roc, cross_sectional_rank


def main() -> None:
    params = load_params()
    uni, strat = params["universe"], params["strategy"]

    with tdx_session(__file__) as tq:
        codes = tq.get_stock_list_in_sector(uni["source"])
        # close = load_prices(tq, codes, "Close", start=uni["start"], end=uni["end"])
        # score = roc(close, window=params["indicators"]["roc_window"])  # per-column
        # ranks = cross_sectional_rank(score)
        # top = ranks.iloc[-1].sort_values(ascending=False).head(strat["top_n"]).index.tolist()
        # tq.send_user_block(block_code="MOMENTUM", stocks=top, show=True)
        raise NotImplementedError("rank by momentum → send_user_block(top_n)")


if __name__ == "__main__":
    main()
