"""Real-time momentum-breakout alerts via subscribe_hq.

    python scripts/monitor.py

Subscribes to the universe (<=100 names — API cap) and fires a 通达信 warning
when intraday momentum breaks out. One-shot per code (unsubscribe on trigger).
"""
from __future__ import annotations

import json
import time

from config.settings import load_params
from src.connection import tdx_session


def main() -> None:
    params = load_params()
    triggered: set[str] = set()

    with tdx_session(__file__) as tq:
        codes = tq.get_stock_list_in_sector(params["universe"]["source"])[:100]

        def on_update(data_str: str) -> None:
            try:
                code = json.loads(data_str).get("Code")
                if not code or code in triggered:
                    return
                # TODO: pull snapshot, compute intraday momentum (e.g. ROC vs
                # open / vs prev close), and if it breaks threshold:
                #   triggered.add(code); tq.unsubscribe_hq(stock_list=[code])
                #   tq.send_warn(...)  # reason_list element <= 25 汉字
            except Exception as exc:  # keep the callback alive
                print(f"callback error: {exc}")

        tq.subscribe_hq(stock_list=codes, callback=on_update)
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
