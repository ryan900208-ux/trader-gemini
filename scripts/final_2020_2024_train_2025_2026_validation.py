from __future__ import annotations

from pathlib import Path

import pandas as pd

from fixed8_backtest.reports import summarize

from two_stage_fundamental_ml_filter import main as run_two_stage


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "outputs" / "two_stage_fundamental_ml_filter"
OUTPUT_DIR = ROOT / "outputs" / "final_2020_2024_train_2025_2026_validation"

STRATEGIES = [
    "fundamental_rank_score",
    "final_score",
    "stable_ensemble_score",
    "fund_ml_85_15",
    "fund_final_ml",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_two_stage()

    summary = pd.read_csv(SOURCE_DIR / "two_stage_backtest_summary.csv")
    summary = summary[summary["score_col"].isin(STRATEGIES)].copy()
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    rows = []
    exit_rows = []
    top_trade_rows = []
    for strategy in STRATEGIES:
        equity_path = SOURCE_DIR / f"{strategy}_equity_curve.csv"
        trades_path = SOURCE_DIR / f"{strategy}_trades.csv"
        if not equity_path.exists() or not trades_path.exists():
            continue
        equity = pd.read_csv(equity_path, parse_dates=["date"])
        trades = pd.read_csv(trades_path, parse_dates=["entry_date", "exit_date", "exit_signal_date"])
        equity.to_csv(OUTPUT_DIR / f"{strategy}_equity_curve.csv", index=False)
        trades.to_csv(OUTPUT_DIR / f"{strategy}_trades.csv", index=False)
        rows.extend(_yearly_rows(strategy, equity, trades))
        exit_rows.extend(_exit_rows(strategy, trades))
        top_trade_rows.extend(_top_trade_rows(strategy, trades))

    yearly = pd.DataFrame(rows)
    exits = pd.DataFrame(exit_rows)
    top_trades = pd.DataFrame(top_trade_rows)
    selection = pd.read_csv(SOURCE_DIR / "two_stage_selection_power.csv")
    selection = selection[selection["score_col"].isin(STRATEGIES)].copy()

    yearly.to_csv(OUTPUT_DIR / "yearly_summary.csv", index=False)
    exits.to_csv(OUTPUT_DIR / "exit_reason_summary.csv", index=False)
    top_trades.to_csv(OUTPUT_DIR / "top_trades.csv", index=False)
    selection.to_csv(OUTPUT_DIR / "selection_power.csv", index=False)

    print("Final frozen validation: train 2020-07-01 to 2024-10-03, validate 2025-01-01 onward")
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nYearly summary:")
    print(yearly.to_string(index=False))
    print("\nExit reasons:")
    print(exits.to_string(index=False))
    print("\nTop trades:")
    print(top_trades.to_string(index=False))
    print("\nSelection power:")
    print(selection.to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


def _yearly_rows(strategy: str, equity: pd.DataFrame, trades: pd.DataFrame) -> list[dict]:
    rows = []
    equity = equity.sort_values("date")
    equity["year"] = equity["date"].dt.year
    trades["exit_year"] = trades["exit_date"].dt.year if not trades.empty else pd.Series(dtype=int)
    for year, year_equity in equity.groupby("year"):
        if year < 2025:
            continue
        start_equity = float(year_equity.iloc[0]["equity"])
        end_equity = float(year_equity.iloc[-1]["equity"])
        year_trades = trades[trades["exit_year"] == year] if not trades.empty else trades
        rows.append(
            {
                "strategy": strategy,
                "year": int(year),
                "start_equity": start_equity,
                "end_equity": end_equity,
                "year_return": end_equity / start_equity - 1 if start_equity else None,
                "max_drawdown": _max_drawdown(year_equity["equity"]),
                "trades": len(year_trades),
                "win_rate": (year_trades["pnl"] > 0).mean() if len(year_trades) else None,
                "realized_pnl": year_trades["pnl"].sum() if len(year_trades) else 0.0,
            }
        )
    return rows


def _exit_rows(strategy: str, trades: pd.DataFrame) -> list[dict]:
    if trades.empty:
        return []
    counts = trades.groupby("exit_reason").agg(trades=("symbol", "count"), pnl=("pnl", "sum")).reset_index()
    counts.insert(0, "strategy", strategy)
    return counts.to_dict("records")


def _top_trade_rows(strategy: str, trades: pd.DataFrame) -> list[dict]:
    if trades.empty:
        return []
    top = trades.sort_values("pnl", ascending=False).head(10).copy()
    top.insert(0, "strategy", strategy)
    return top[
        [
            "strategy",
            "symbol",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "pnl",
            "return_pct",
            "holding_days",
            "exit_reason",
        ]
    ].to_dict("records")


def _max_drawdown(equity: pd.Series) -> float:
    curve = equity.astype(float)
    return float((curve / curve.cummax() - 1).min())


if __name__ == "__main__":
    main()
