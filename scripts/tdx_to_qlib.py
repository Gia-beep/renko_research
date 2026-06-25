import os
import pandas as pd
import numpy as np

# This script converts the parquet files in data/raw/ into Qlib CSV format
# and then invokes Qlib's DumpDataUpdate to update the binary dataset.

import glob

RAW_DATA_DIR = "/mnt/c/new_tdx_test/renko_research/data/raw"
CSV_OUT_DIR = "/mnt/c/new_tdx_test/renko_research/data/qlib_csvs"

def main():
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Directory {RAW_DATA_DIR} does not exist. Please run scripts/fetch_data.py first.")
        return

    os.makedirs(CSV_OUT_DIR, exist_ok=True)
    
    # Find files
    close_file = glob.glob(os.path.join(RAW_DATA_DIR, "Close_*.parquet"))[0]
    open_file = glob.glob(os.path.join(RAW_DATA_DIR, "Open_*.parquet"))[0]
    high_file = glob.glob(os.path.join(RAW_DATA_DIR, "High_*.parquet"))[0]
    low_file = glob.glob(os.path.join(RAW_DATA_DIR, "Low_*.parquet"))[0]
    vol_file = glob.glob(os.path.join(RAW_DATA_DIR, "Volume_*.parquet"))[0]
    amt_file = glob.glob(os.path.join(RAW_DATA_DIR, "Amount_*.parquet"))[0]

    # Read all parquet files (Close, Open, High, Low, Volume, Amount)
    print("Loading parquet files...")
    close_df = pd.read_parquet(close_file)
    open_df = pd.read_parquet(open_file)
    high_df = pd.read_parquet(high_file)
    low_df = pd.read_parquet(low_file)
    vol_df = pd.read_parquet(vol_file)
    amt_df = pd.read_parquet(amt_file)
    
    symbols = close_df.columns
    print(f"Processing {len(symbols)} symbols...")
    
    for sym in symbols:
        df = pd.DataFrame({
            "open": open_df[sym],
            "high": high_df[sym],
            "low": low_df[sym],
            "close": close_df[sym],
            "volume": vol_df[sym],
            "amount": amt_df[sym],
            "factor": 1.0,
            "change": close_df[sym].pct_change()
        })
        # drop nan rows
        df = df.dropna(subset=["close"])
        if df.empty:
            continue
            
        # Qlib expects the index to be 'date' and symbols to be like sh600000
        df.index.name = "date"
        
        # Convert TDX symbol (e.g., 600006.SH) to Qlib symbol (sh600006)
        parts = sym.split('.')
        if len(parts) == 2:
            qlib_sym = parts[1].lower() + parts[0]
        else:
            qlib_sym = sym.lower()
            
        csv_path = os.path.join(CSV_OUT_DIR, f"{qlib_sym}.csv")
        df.to_csv(csv_path)

    print(f"Generated {len(symbols)} CSV files in {CSV_OUT_DIR}.")
    print("Now run Qlib dump_bin command to ingest the CSVs.")

if __name__ == "__main__":
    main()
