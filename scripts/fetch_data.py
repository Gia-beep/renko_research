"""Batch-download the configured universe into the local cache.

    python scripts/fetch_data.py

Resolves the stock pool from params.yaml's ``universe`` block, then pulls and
caches Close/Open/High/Low so backtests run offline afterwards.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import load_params
from src.connection import tdx_session
from src.data.loader import load_prices

FIELDS = ["Close", "Open", "High", "Low", "Volume", "Amount"]


def main() -> None:
    params = load_params()
    uni = params["universe"]
    period = uni.get("period", "1d")

    with tdx_session(__file__) as tq:
        if uni["source"] == "中证1000":
            codes = []
            code_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'csi1000_codes.txt')
            with open(code_file, 'r') as f:
                for line in f:
                    sym = line.strip()
                    if sym:
                        market, code = sym[:2].upper(), sym[2:]
                        # TDX usually uses code.SH or code.SZ
                        codes.append(f"{code}.{market}")
        else:
            codes = tq.get_stock_list_in_sector(uni["source"])
            
        print(f"universe '{uni['source']}': {len(codes)} codes")
        if not codes:
            print("no codes resolved — check universe.source / block_type")
            return

        for field in FIELDS:
            df = load_prices(
                tq,
                codes,
                field,
                start=uni["start"],
                end=uni["end"],
                period=period,
                use_cache=True,
            )
            print(f"  {field}: {df.shape[0]} rows x {df.shape[1]} cols cached")


if __name__ == "__main__":
    main()
