#!/usr/bin/env python
"""Build repo-compatible wide time-series CSV from ticker metadata.

Input metadata CSV example columns:
    col_name,group,ticker,name,nation

Output CSV columns are wide-format numeric series with a date column named '날짜'.
The script uses Yahoo Finance symbols already present in the metadata file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Exact repo-friendly renames for key columns used by the project
NAME_MAP = {
    "S&P 500": "SPX",
    "VIX": "VIX",
    "KOSPI": "KOSPI",
    "KOSDAQ": "KOSDAQ",
    "Gold": "Gold",
    "Crude Oil": "WTI",
    "Brent Crude": "Brent Crude",
    "Natural Gas": "Natural Gas",
    "USD/KRW": "USD/KRW",
    "USD/CNY": "USD/CNY",
    "USD/JPY": "USD/JPY",
    "USD/EUR": "USD/EUR",
    "USD/GBP": "USD/GBP",
    "USD/CAD": "USD/CAD",
    "Dow Jones": "Dow Jones",
    "Nasdaq": "Nasdaq",
    "FTSE 100": "FTSE 100",
    "Nikkei 225": "Nikkei 225",
    "DAX": "DAX",
    "CAC 40": "CAC 40",
}

# Optional extras if you want closer parity with the synthetic example.
# Uncomment / edit if needed.
EXTRA_TICKERS = {
    # "DXY": "DX-Y.NYB",
    # "Silver": "SI=F",
    # "Copper": "HG=F",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--meta-csv", required=True, help="Ticker metadata CSV (your uploaded file)")
    p.add_argument("--out-csv", default="data/raw/timeseries_data_real.csv")
    p.add_argument("--start", default="2004-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--price-field", default="Close", choices=["Close", "Adj Close", "Open", "High", "Low"])
    return p.parse_args()


def load_meta(path: str) -> pd.DataFrame:
    meta = pd.read_csv(path)
    req = {"ticker", "name"}
    miss = req - set(meta.columns)
    if miss:
        raise ValueError(f"metadata CSV must contain columns {sorted(req)}; missing={sorted(miss)}")
    meta = meta[[c for c in ["col_name", "group", "ticker", "name", "nation"] if c in meta.columns]].copy()
    meta = meta.dropna(subset=["ticker", "name"]).drop_duplicates(subset=["ticker"])
    return meta


def build_series_map(meta: pd.DataFrame) -> dict[str, str]:
    series_map: dict[str, str] = {}
    for _, row in meta.iterrows():
        output_name = NAME_MAP.get(str(row["name"]), str(row["name"]))
        series_map[output_name] = str(row["ticker"])
    for k, v in EXTRA_TICKERS.items():
        series_map.setdefault(k, v)
    return series_map


def download_history(series_map: dict[str, str], start: str, end: str | None, price_field: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:
        raise SystemExit(
            "yfinance가 필요합니다. 먼저 `pip install yfinance` 를 실행하세요."
        ) from e

    tickers = list(series_map.values())
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=True,
        group_by="ticker",
        threads=True,
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError("다운로드된 데이터가 없습니다. ticker / 날짜 범위를 확인하세요.")

    wide = pd.DataFrame(index=raw.index)
    reverse_map = {ticker: out_name for out_name, ticker in series_map.items()}

    if len(tickers) == 1:
        ticker = tickers[0]
        field = price_field if price_field in raw.columns else ("Adj Close" if "Adj Close" in raw.columns else "Close")
        wide[reverse_map[ticker]] = raw[field]
    else:
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(0):
                continue
            sub = raw[ticker]
            field = price_field if price_field in sub.columns else ("Adj Close" if "Adj Close" in sub.columns else "Close")
            wide[reverse_map[ticker]] = sub[field]

    wide = wide.sort_index().ffill().dropna(how="all")
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "날짜"
    wide = wide.reset_index()
    return wide


def main():
    args = parse_args()
    meta = load_meta(args.meta_csv)
    series_map = build_series_map(meta)
    df = download_history(series_map, start=args.start, end=args.end, price_field=args.price_field)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"saved: {out_path}")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    missing_core = [c for c in ["SPX", "VIX", "KOSPI", "Gold", "WTI", "DXY"] if c not in df.columns]
    if missing_core:
        print(f"warning: core columns missing from output: {missing_core}")


if __name__ == "__main__":
    main()
