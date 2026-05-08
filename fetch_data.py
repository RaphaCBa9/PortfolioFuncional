#!/usr/bin/env python3
"""Download historical prices for all 30 Dow Jones stocks via yfinance."""

import os
import sys
import yfinance as yf

TICKERS = [
    "AAPL", "AMGN", "AXP",  "BA",   "CAT",
    "CRM",  "CSCO", "CVX",  "DIS",  "DOW",
    "GS",   "HD",   "HON",  "IBM",  "JNJ",
    "JPM",  "KO",   "MCD",  "MMM",  "MRK",
    "MSFT", "NKE",  "NVDA", "PG",   "SHW",
    "TRV",  "UNH",  "V",    "VZ",   "WMT",
]

def download_period(start, end, subdir):
    out_dir = os.path.join("data", "raw", subdir)
    os.makedirs(out_dir, exist_ok=True)
    failed = []
    for ticker in TICKERS:
        path = os.path.join(out_dir, f"{ticker}.csv")
        try:
            df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if df.empty:
                print(f"  WARN: {ticker} — no data", file=sys.stderr)
                failed.append(ticker)
                continue
            # Keep only Date and Close columns; reset index to get Date as column
            df = df[["Close"]].copy()
            df.index = df.index.strftime("%Y-%m-%d")
            df.index.name = "Date"
            df.to_csv(path)
            print(f"  OK: {ticker} ({len(df)} rows) -> {path}")
        except Exception as ex:
            print(f"  ERROR: {ticker}: {ex}", file=sys.stderr)
            failed.append(ticker)
    if failed:
        print(f"\nFailed tickers: {failed}", file=sys.stderr)
    else:
        print(f"\nAll {len(TICKERS)} tickers downloaded to {out_dir}/")

if __name__ == "__main__":
    print("=== Downloading H2 2025 (Jul–Dec) ===")
    download_period("2025-07-01", "2025-12-31", "h2_2025")

    print("\n=== Downloading Q1 2025 (Jan–Mar) ===")
    download_period("2025-01-01", "2025-03-31", "q1_2025")
