from __future__ import annotations

import argparse

from fixed8_backtest.universe import (
    fetch_twse_listed_companies,
    normalize_twse_universe,
    write_universe_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch TWSE listed companies and create a yfinance universe CSV.")
    parser.add_argument("--output", default="data/universe_twse_all.csv")
    args = parser.parse_args()

    rows = normalize_twse_universe(fetch_twse_listed_companies())
    write_universe_csv(rows, args.output)
    print(f"Saved {len(rows)} TWSE listed stock symbols to {args.output}")


if __name__ == "__main__":
    main()
