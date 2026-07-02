from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from fixed8_backtest.data import read_universe
from fixed8_backtest.finmind_history import fetch_history_for_symbols


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical Taiwan stock fundamentals from FinMind.")
    parser.add_argument("--universe", default="data/universe_twse_all.csv")
    parser.add_argument("--benchmark-symbol", default="0050.TW")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output", default="data/fundamentals_finmind_history.csv")
    parser.add_argument("--cache-dir", default="work/finmind_cache")
    parser.add_argument("--token", default=os.environ.get("FINMIND_TOKEN"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    args = parser.parse_args()

    symbols = read_universe(args.universe, args.benchmark_symbol)
    if args.offset:
        symbols = symbols[args.offset :]
    if args.limit:
        symbols = symbols[: args.limit]

    history = fetch_history_for_symbols(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        cache_dir=args.cache_dir,
        token=args.token,
        sleep_seconds=args.sleep_seconds,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(history)} historical fundamental rows for {history['symbol'].nunique() if not history.empty else 0} symbols to {output}")
    if not history.empty:
        print(history[["symbol", "period_date", "as_of_date", "roe", "roic_proxy", "revenue_growth", "eva_like_score"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
