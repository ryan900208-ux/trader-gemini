from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .data import download_ohlcv, read_universe
from .fundamentals import fetch_yfinance_snapshot, load_fundamentals
from .indicators import add_indicators, market_regime
from .reports import summarize
from .strategy import add_entry_diagnostics, add_strategy_scores, build_feature_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    benchmark_symbol = config["benchmark_symbol"]
    symbols = read_universe(config["universe_csv"], benchmark_symbol)
    all_symbols = sorted(set(symbols + [benchmark_symbol]))

    print(f"Downloading OHLCV for {len(all_symbols)} symbols...")
    raw_data = download_ohlcv(
        all_symbols,
        config["start"],
        config["end"],
        batch_size=config.get("yfinance_batch_size", 80),
        cache_dir=config.get("price_cache_dir"),
    )
    benchmark = raw_data.pop(benchmark_symbol)
    data = {symbol: add_indicators(frame) for symbol, frame in raw_data.items() if not frame.empty}
    benchmark = add_indicators(benchmark)
    regime = market_regime(benchmark, config)

    fundamentals = load_fundamentals(config.get("fundamentals_csv"))
    if config.get("use_yfinance_fundamental_snapshot"):
        snapshot = fetch_yfinance_snapshot(symbols)
        fundamentals = pd.concat([fundamentals, snapshot], ignore_index=True)

    panel = build_feature_panel(data, benchmark, regime)
    panel = add_strategy_scores(panel, fundamentals, config)
    panel = add_entry_diagnostics(panel, config)
    equity, trades = run_backtest(panel, data, config)
    summary = summarize(equity, trades)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_dir / "daily_features.csv", index=False)
    equity.to_csv(output_dir / "equity_curve.csv", index=False)
    trades.to_csv(output_dir / "trades.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    diagnostic_cols = [column for column in panel.columns if column.startswith("pass_")]
    diagnostics = (
        panel[diagnostic_cols]
        .mean()
        .rename_axis("condition")
        .reset_index(name="pass_rate")
        .sort_values("pass_rate")
    )
    diagnostics.to_csv(output_dir / "entry_diagnostics.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
