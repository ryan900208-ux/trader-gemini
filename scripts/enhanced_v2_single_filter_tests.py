from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fixed8_backtest.data import download_ohlcv, read_universe
from fixed8_backtest.reports import summarize


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "fixed8_control_eva_top15_pool_best_sweep.json"
PANEL_PATH = ROOT / "outputs" / "fixed8_control_eva_top15_pool_best_sweep" / "daily_features.csv"
PRED_PATH = ROOT / "outputs" / "enhanced_lgbm_v21" / "v2_enhanced_features" / "candidate_predictions.csv"
OUTPUT_DIR = ROOT / "outputs" / "enhanced_v2_single_filter_tests"


@dataclass
class Position:
    symbol: str
    shares: int
    entry_date: pd.Timestamp
    entry_price: float
    last_rs20_rank_pct: float
    rs20_weak_count: int = 0
    holding_bars: int = 0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"], low_memory=False)
    panel = _add_enhanced_features(panel)
    pred = pd.read_csv(PRED_PATH, parse_dates=["date"])
    panel = panel.merge(pred[["date", "symbol", "model_score"]], on=["date", "symbol"], how="left")
    data = _load_price_data(config)

    runs = _runs()
    summaries = []
    for idx, run in enumerate(runs, start=1):
        print(f"[{idx}/{len(runs)}] {run['name']}", flush=True)
        equity, trades = _run_backtest(panel, data, config, run)
        summary = summarize(equity, trades)
        summary.insert(0, "category", run["category"])
        summary.insert(1, "scenario", run["name"])
        summaries.append(summary)

    out = pd.concat(summaries, ignore_index=True)
    pivot = out.pivot_table(index=["category", "scenario"], columns="metric", values="value", aggfunc="first")
    cols = ["total_return", "annual_return", "max_drawdown", "sharpe", "trades", "win_rate", "avg_trade_return", "final_equity"]
    pivot = pivot[[c for c in cols if c in pivot.columns]].reset_index()
    pivot = pivot.sort_values(["category", "annual_return"], ascending=[True, False])
    out.to_csv(OUTPUT_DIR / "single_filter_summary.csv", index=False)
    pivot.to_csv(OUTPUT_DIR / "single_filter_pivot.csv", index=False)
    print(pivot.to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


def _runs() -> list[dict]:
    base = {"name": "baseline_enhanced", "category": "baseline", "stop_loss": 0.12, "filter": None}
    runs = [base]
    for value in [0.08, 0.10, 0.12, 0.15]:
        runs.append({"name": f"stop_{value:.2f}", "category": "stop_loss", "stop_loss": value, "filter": None})
    for value in [20, 30, 50, 100]:
        runs.append({
            "name": f"price_min_{value}",
            "category": "price_filter",
            "stop_loss": 0.12,
            "filter": lambda df, value=value: df["Close"] >= value,
        })
    for value in [0.2, 0.4, 0.6, 0.8]:
        runs.append({
            "name": f"liquidity_top_{int(value*100)}",
            "category": "liquidity_filter",
            "stop_loss": 0.12,
            "filter": lambda df, value=value: df["liquidity_rank"] <= value,
        })
    for value in [15, 30, 50, 100, 200]:
        runs.append({
            "name": f"eva_rank_lte_{value}",
            "category": "eva_filter",
            "stop_loss": 0.12,
            "filter": lambda df, value=value: df["fundamental_rank"] <= value,
        })
    for trend in ["ma20", "ma60", "ma120"]:
        runs.append({
            "name": f"close_gt_{trend}",
            "category": "trend_filter",
            "stop_loss": 0.12,
            "filter": lambda df, trend=trend: df["Close"] > df[trend],
        })
    for name, lo, hi in [("vol_0.5_4.5", 0.5, 4.5), ("vol_0.8_3.0", 0.8, 3.0), ("vol_0.3_6.0", 0.3, 6.0)]:
        runs.append({
            "name": name,
            "category": "volume_filter",
            "stop_loss": 0.12,
            "filter": lambda df, lo=lo, hi=hi: df["volume_ratio"].between(lo, hi),
        })
    # A few promising pairs, not the full hard filter bundle.
    for value in [0.08, 0.10]:
        runs.append({
            "name": f"stop_{value:.2f}_liq_top80",
            "category": "combo_small",
            "stop_loss": value,
            "filter": lambda df: df["liquidity_rank"] <= 0.8,
        })
        runs.append({
            "name": f"stop_{value:.2f}_price_ge30",
            "category": "combo_small",
            "stop_loss": value,
            "filter": lambda df: df["Close"] >= 30,
        })
    return runs


def _base_candidate_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["market_regime"].isin(["bull", "neutral"])
        & (df["score"] >= 55)
        & (df["rs20_rank_pct"] <= 0.50)
        & (df["rs60_rank_pct"] <= 0.60)
        & (df["Close"] > df["ma60"])
        & (df["ma20_slope"] > -0.005)
        & df["rsi14"].between(45, 78)
        & (df["ret20"] > df["benchmark_ret20"] * 0.4)
        & (df["ret60"] > df["benchmark_ret60"] * 0.4)
        & (df["ret5"] <= 0.15)
        & (df["ret20"] <= 0.65)
        & df["volume_ratio"].between(0.5, 4.5)
        & (df["ma20_deviation"].abs() <= 0.25)
        & df["fundamental_score"].notna()
    )


def _run_backtest(panel: pd.DataFrame, data: dict[str, pd.DataFrame], config: dict, run: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cash = float(config["initial_cash"])
    positions: dict[str, Position] = {}
    cooldown_until: dict[str, pd.Timestamp] = {}
    trades = []
    equity_rows = []
    day_groups = {date: group.copy() for date, group in panel.groupby("date", sort=True)}
    dates = [date for date in sorted(day_groups) if date >= pd.Timestamp("2023-01-01")]

    for idx, date in enumerate(dates[:-1]):
        next_date = dates[idx + 1]
        day = day_groups[date].set_index("symbol", drop=False)
        for symbol in list(positions):
            if symbol not in day.index:
                continue
            pos = positions[symbol]
            row = day.loc[symbol]
            pos.holding_bars += 1
            pos.rs20_weak_count = pos.rs20_weak_count + 1 if row["rs20_rank_pct"] > pos.last_rs20_rank_pct else 0
            pos.last_rs20_rank_pct = float(row["rs20_rank_pct"])
            reason = _exit_reason(pos, row, run["stop_loss"])
            if not reason:
                continue
            open_price = _next_open(data[symbol], date)
            if pd.isna(open_price):
                continue
            proceeds = pos.shares * open_price * (1 - config["slippage_rate"])
            fee = proceeds * config["commission_rate"]
            tax = proceeds * config["tax_rate"]
            cash += proceeds - fee - tax
            pnl = proceeds - fee - tax - pos.shares * pos.entry_price
            trades.append({
                "symbol": symbol,
                "entry_date": pos.entry_date,
                "exit_signal_date": date,
                "exit_date": next_date,
                "entry_price": pos.entry_price,
                "exit_price": open_price,
                "shares": pos.shares,
                "pnl": pnl,
                "return_pct": open_price / pos.entry_price - 1,
                "holding_days": pos.holding_bars,
                "exit_reason": reason,
            })
            if reason == "stop_loss":
                cooldown_until[symbol] = date + pd.Timedelta(days=config["cooldown_days_after_stop"])
            del positions[symbol]

        free_slots = config["max_positions"] - len(positions)
        if free_slots > 0:
            day_frame = day_groups[date]
            mask = _base_candidate_mask(day_frame)
            if run["filter"] is not None:
                mask = mask & run["filter"](day_frame)
            candidates = day_frame[mask].copy()
            if not candidates.empty and "model_score" in candidates.columns:
                candidates = candidates[~candidates["symbol"].isin(positions)]
                candidates = candidates[candidates["symbol"].map(lambda symbol: cooldown_until.get(symbol, pd.Timestamp.min) <= date)]
                if candidates.empty or "model_score" not in candidates.columns:
                    continue
                candidates = candidates.dropna(subset=["model_score"])
                if candidates.empty:
                    continue
                candidates = candidates.sort_values(["model_score", "final_score"], ascending=False)
                for candidate in candidates.head(free_slots).itertuples(index=False):
                    open_price = _next_open(data[candidate.symbol], date)
                    if pd.isna(open_price):
                        continue
                    budget = min(cash, _portfolio_value(cash, positions, day) * config["position_weight"])
                    buy_price = open_price * (1 + config["slippage_rate"])
                    fee_adjusted = buy_price * (1 + config["commission_rate"])
                    shares = int(budget // fee_adjusted)
                    if shares <= 0:
                        continue
                    cash -= shares * fee_adjusted
                    positions[candidate.symbol] = Position(
                        symbol=candidate.symbol,
                        shares=shares,
                        entry_date=next_date,
                        entry_price=buy_price,
                        last_rs20_rank_pct=float(candidate.rs20_rank_pct),
                    )

        market_value = 0.0
        for symbol, pos in positions.items():
            if symbol in day.index and not pd.isna(day.at[symbol, "Close"]):
                market_value += pos.shares * float(day.at[symbol, "Close"])
        equity_rows.append({"date": date, "cash": cash, "market_value": market_value, "equity": cash + market_value})
    return pd.DataFrame(equity_rows), pd.DataFrame(trades)


def _exit_reason(pos: Position, row: pd.Series, stop_loss: float) -> str | None:
    if row["market_regime"] == "bear":
        return "market_bear"
    if row["Close"] <= pos.entry_price * (1 - stop_loss):
        return "stop_loss"
    if pos.holding_bars >= 252:
        return "max_holding_days"
    if row["Close"] < row["ma120"]:
        return "below_ma120"
    return None


def _add_enhanced_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["close_ma60_ratio"] = df["Close"] / df["ma60"] - 1
    df["close_ma120_ratio"] = df["Close"] / df["ma120"] - 1
    df["above_ma120"] = (df["Close"] > df["ma120"]).astype(float)
    df["turnover20"] = df["Close"] * df["volume_ma20"]
    df["liquidity_rank"] = df.groupby("date")["turnover20"].rank(ascending=False, pct=True)
    df["price_bucket"] = pd.cut(df["Close"], bins=[-np.inf, 30, 100, 500, np.inf], labels=[0, 1, 2, 3]).astype(float)
    df["eva_rank_bucket"] = pd.cut(df["fundamental_rank"], bins=[-np.inf, 15, 30, 100, np.inf], labels=[0, 1, 2, 3]).astype(float)
    return df


def _portfolio_value(cash: float, positions: dict[str, Position], day: pd.DataFrame) -> float:
    value = cash
    for symbol, pos in positions.items():
        if symbol in day.index and not pd.isna(day.at[symbol, "Close"]):
            value += pos.shares * float(day.at[symbol, "Close"])
    return value


def _next_open(frame: pd.DataFrame, signal_date: pd.Timestamp) -> float:
    location = frame.index.searchsorted(signal_date, side="right")
    if location >= len(frame):
        return float("nan")
    return float(frame.iloc[location]["Open"])


def _load_price_data(config: dict) -> dict[str, pd.DataFrame]:
    symbols = read_universe(ROOT / config["universe_csv"], config["benchmark_symbol"])
    all_symbols = sorted(set(symbols + [config["benchmark_symbol"]]))
    raw_data = download_ohlcv(
        all_symbols,
        config["start"],
        config["end"],
        batch_size=config.get("yfinance_batch_size", 80),
        cache_dir=ROOT / config["price_cache_dir"],
    )
    raw_data.pop(config["benchmark_symbol"], None)
    return {symbol: frame for symbol, frame in raw_data.items() if not frame.empty}


if __name__ == "__main__":
    main()
