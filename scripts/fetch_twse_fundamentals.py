from __future__ import annotations

import argparse

from fixed8_backtest.twse_fundamentals import fetch_twse_fundamentals, write_fundamentals_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch latest TWSE financial statements and create EVA-like fundamentals.")
    parser.add_argument("--output", default="data/fundamentals_twse_latest.csv")
    args = parser.parse_args()

    fundamentals = fetch_twse_fundamentals()
    write_fundamentals_csv(fundamentals, args.output)
    print(f"Saved {len(fundamentals)} TWSE latest fundamental rows to {args.output}")
    print(fundamentals[["symbol", "name", "as_of_date", "roe", "roic_proxy", "debt_to_equity", "eva_like_score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
