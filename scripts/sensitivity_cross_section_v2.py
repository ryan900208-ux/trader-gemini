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
PRED_PATH = (
    ROOT
    / "outputs"
    / "no_pool_lgbm_anti_overfit"
    / "rolling3y_purged_forward_60d_return"
    / "candidate_predictions.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "v2_sensitivity_cross_section"


@dataclass
class Position:
    symbol: str
    shares: int
    entry_date: pd.Timestamp
    entry_price: float
    last_rs20_rank_pct: float
    rs20_weak_count: int = 0
    holding_bars: int = 0


BASE_RULES = {
    "rs20_max": 0.50,
    "rs60_max": 0.60,
    "score_min": 55,
    "trend": "ma60",
    "rsi_min": 45,
    "rsi_max": 78,
    "ret20_max": 0.65,
    "volume_min": 0.5,
    "volume_max": 4.5,
    "ma20_dev_max": 0.25,
}


BASE_EXEC = {
    "stop_loss": 0.12,
    "trend_ma": 120,
    "max_holding_days": 252,
    "bear_exit": True,
    "cooldown_days": 30,
    "max_positions": 5,
    "position_weight": 0.2,
    "slippage_rate": 0.001,
    "entry_price": "open",
    "max_entry_gap": None,
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"], low_memory=False)
    pred = pd.read_csv(PRED_PATH, parse_dates=["date"])
    panel = panel.merge(pred[["date", "symbol", "model_score"]], on=["date", "symbol"], how="left")
    universe = pd.read_csv(ROOT / config["universe_csv"])
    industry_map = dict(zip(universe["symbol"].astype(str), universe["industry"].astype(str)))
    data = _load_price_data(config)

    runs = []
    runs.extend(_candidate_sensitivity_runs())
    runs.extend(_exit_sensitivity_runs())
    runs.extend(_execution_sensitivity_runs())
    runs.extend(_cross_section_runs(panel, industry_map))

    summaries = []
    for idx, run in enumerate(runs, start=1):
        print(f"[{idx}/{len(runs)}] {run['name']}", flush=True)
        equity, trades = _run_backtest(panel, data, config, run)
        summary = summarize(equity, trades)
        summary.insert(0, "category", run["category"])
        summary.insert(1, "scenario", run["name"])
        summaries.append(summary)
        if run.get("save_detail"):
            detail_dir = OUTPUT_DIR / run["name"]
            detail_dir.mkdir(parents=True, exist_ok=True)
            equity.to_csv(detail_dir / "equity_curve.csv", index=False)
            trades.to_csv(detail_dir / "trades.csv", index=False)

    out = pd.concat(summaries, ignore_index=True)
    out.to_csv(OUTPUT_DIR / "sensitivity_cross_section_summary.csv", index=False)
    _pivot_summary(out).to_csv(OUTPUT_DIR / "sensitivity_cross_section_pivot.csv", index=False)
    print(_pivot_summary(out).to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


def _base_run(name: str, category: str) -> dict:
    return {
        "name": name,
        "category": category,
        "rules": dict(BASE_RULES),
        "exec": dict(BASE_EXEC),
        "filter": None,
    }


def _candidate_sensitivity_runs() -> list[dict]:
    runs = [_base_run("baseline", "baseline")]
    for value in [0.40, 0.50, 0.60]:
        run = _base_run(f"rs20_{value:.2f}", "candidate_rs20")
        run["rules"]["rs20_max"] = value
        runs.append(run)
    for value in [0.50, 0.60, 0.70]:
        run = _base_run(f"rs60_{value:.2f}", "candidate_rs60")
        run["rules"]["rs60_max"] = value
        runs.append(run)
    for value in [50, 55, 60]:
        run = _base_run(f"score_{value}", "candidate_score")
        run["rules"]["score_min"] = value
        runs.append(run)
    for trend in ["ma20", "ma60", "ma120"]:
        run = _base_run(f"trend_close_gt_{trend}", "candidate_trend")
        run["rules"]["trend"] = trend
        runs.append(run)
    for name, lo, hi in [("rsi_45_78", 45, 78), ("rsi_50_75", 50, 75), ("rsi_45_85", 45, 85)]:
        run = _base_run(name, "candidate_rsi")
        run["rules"]["rsi_min"] = lo
        run["rules"]["rsi_max"] = hi
        runs.append(run)
    for value in [0.50, 0.65, 0.80]:
        run = _base_run(f"ret20cap_{value:.2f}", "candidate_ret20cap")
        run["rules"]["ret20_max"] = value
        runs.append(run)
    for name, lo, hi in [("vol_0.5_4.5", 0.5, 4.5), ("vol_0.8_3.0", 0.8, 3.0), ("vol_0.3_6.0", 0.3, 6.0)]:
        run = _base_run(name, "candidate_volume")
        run["rules"]["volume_min"] = lo
        run["rules"]["volume_max"] = hi
        runs.append(run)
    return runs


def _exit_sensitivity_runs() -> list[dict]:
    runs = []
    for value in [0.08, 0.10, 0.12, 0.15]:
        run = _base_run(f"stop_{value:.2f}", "exit_stop")
        run["exec"]["stop_loss"] = value
        runs.append(run)
    for value in [60, 90, 120, 150]:
        run = _base_run(f"trend_exit_ma{value}", "exit_trend_ma")
        run["exec"]["trend_ma"] = value
        runs.append(run)
    for value in [120, 180, 252, 360]:
        run = _base_run(f"hold_{value}", "exit_holding")
        run["exec"]["max_holding_days"] = value
        runs.append(run)
    for value in [False, True]:
        run = _base_run(f"bear_exit_{value}", "exit_bear")
        run["exec"]["bear_exit"] = value
        runs.append(run)
    for value in [15, 30, 60]:
        run = _base_run(f"cooldown_{value}", "exit_cooldown")
        run["exec"]["cooldown_days"] = value
        runs.append(run)
    return runs


def _execution_sensitivity_runs() -> list[dict]:
    runs = []
    for value in [0.001, 0.003, 0.005, 0.010]:
        run = _base_run(f"slippage_{value:.3f}", "execution_slippage")
        run["exec"]["slippage_rate"] = value
        runs.append(run)
    for value in ["open", "close"]:
        run = _base_run(f"entry_{value}", "execution_entry_price")
        run["exec"]["entry_price"] = value
        runs.append(run)
    for value in [None, 0.03, 0.05, 0.08]:
        run = _base_run("entry_gap_none" if value is None else f"entry_gap_{value:.2f}", "execution_entry_gap")
        run["exec"]["max_entry_gap"] = value
        runs.append(run)
    for value in [3, 5, 8, 10]:
        run = _base_run(f"max_positions_{value}", "portfolio_positions")
        run["exec"]["max_positions"] = value
        run["exec"]["position_weight"] = min(0.2, 1 / value)
        runs.append(run)
    for value in [0.15, 0.20, 0.25]:
        run = _base_run(f"position_weight_{value:.2f}", "portfolio_weight")
        run["exec"]["position_weight"] = value
        runs.append(run)
    return runs


def _cross_section_runs(panel: pd.DataFrame, industry_map: dict[str, str]) -> list[dict]:
    runs = []
    for name, lo, hi in [
        ("eva_rank_1_15", 1, 15),
        ("eva_rank_16_30", 16, 30),
        ("eva_rank_31_100", 31, 100),
        ("eva_rank_gt100", 101, np.inf),
    ]:
        run = _base_run(name, "cross_eva_rank")
        run["filter"] = lambda df, lo=lo, hi=hi: (df["fundamental_rank"] >= lo) & (df["fundamental_rank"] <= hi)
        runs.append(run)
    for name, lo, hi in [
        ("price_lt30", -np.inf, 30),
        ("price_30_100", 30, 100),
        ("price_100_500", 100, 500),
        ("price_gt500", 500, np.inf),
    ]:
        run = _base_run(name, "cross_price")
        run["filter"] = lambda df, lo=lo, hi=hi: (df["Close"] >= lo) & (df["Close"] < hi)
        runs.append(run)

    liquidity = panel.copy()
    liquidity["turnover20"] = liquidity["Close"] * liquidity["volume_ma20"]
    liquidity["liq_rank"] = liquidity.groupby("date")["turnover20"].rank(ascending=False, pct=True)
    liq_lookup = liquidity[["date", "symbol", "liq_rank"]]
    for name, lo, hi in [("liquidity_top20", 0, 0.2), ("liquidity_mid60", 0.2, 0.8), ("liquidity_bottom20", 0.8, 1.0)]:
        run = _base_run(name, "cross_liquidity")
        run["filter"] = lambda df, lo=lo, hi=hi: (df["liq_rank"] > lo) & (df["liq_rank"] <= hi)
        run["extra_panel"] = liq_lookup
        runs.append(run)

    industry_series = pd.Series(industry_map)
    top_industries = industry_series.value_counts().head(8).index.tolist()
    for industry in top_industries:
        run = _base_run(f"industry_{industry}", "cross_industry")
        run["filter"] = lambda df, industry=industry: df["industry"] == industry
        run["industry_map"] = industry_map
        runs.append(run)
    for regime in ["bull", "neutral"]:
        run = _base_run(f"regime_{regime}", "cross_market_regime")
        run["filter"] = lambda df, regime=regime: df["market_regime"] == regime
        runs.append(run)
    for name, condition in [
        ("benchmark_ret20_positive", lambda df: df["benchmark_ret20"] > 0),
        ("benchmark_ret20_negative", lambda df: df["benchmark_ret20"] <= 0),
        ("benchmark_ret60_positive", lambda df: df["benchmark_ret60"] > 0),
        ("benchmark_ret60_negative", lambda df: df["benchmark_ret60"] <= 0),
    ]:
        run = _base_run(name, "cross_market_return")
        run["filter"] = condition
        runs.append(run)
    return runs


def _run_backtest(
    base_panel: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    config: dict,
    run: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = base_panel
    if run.get("extra_panel") is not None:
        panel = panel.merge(run["extra_panel"], on=["date", "symbol"], how="left")
    if run.get("industry_map") is not None:
        panel = panel.copy()
        panel["industry"] = panel["symbol"].map(run["industry_map"])

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
            reason = _exit_reason(pos, row, run)
            if not reason:
                continue
            exit_price = _next_price(data[symbol], date, run["exec"]["entry_price"])
            if pd.isna(exit_price):
                continue
            proceeds = pos.shares * exit_price * (1 - run["exec"]["slippage_rate"])
            fee = proceeds * config["commission_rate"]
            tax = proceeds * config["tax_rate"]
            cash += proceeds - fee - tax
            pnl = proceeds - fee - tax - pos.shares * pos.entry_price
            trades.append(
                {
                    "symbol": symbol,
                    "entry_date": pos.entry_date,
                    "exit_signal_date": date,
                    "exit_date": next_date,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "shares": pos.shares,
                    "pnl": pnl,
                    "return_pct": exit_price / pos.entry_price - 1,
                    "holding_days": pos.holding_bars,
                    "exit_reason": reason,
                }
            )
            if reason == "stop_loss":
                cooldown_until[symbol] = date + pd.Timedelta(days=run["exec"]["cooldown_days"])
            del positions[symbol]

        free_slots = run["exec"]["max_positions"] - len(positions)
        if free_slots > 0:
            candidates = _candidate_frame(day_groups[date], run)
            if candidates.empty or "model_score" not in candidates.columns:
                candidates = candidates.iloc[0:0]
            else:
                candidates = candidates[~candidates["symbol"].isin(positions)]
                candidates = candidates[
                    candidates["symbol"].map(lambda symbol: cooldown_until.get(symbol, pd.Timestamp.min) <= date)
                ]
                if candidates.empty or "model_score" not in candidates.columns:
                    candidates = candidates.iloc[0:0]
                else:
                    candidates = candidates.dropna(subset=["model_score"])
                    candidates = candidates.sort_values(["model_score", "final_score"], ascending=False)
            for candidate in candidates.head(free_slots).itertuples(index=False):
                price = _next_price(data[candidate.symbol], date, run["exec"]["entry_price"])
                if pd.isna(price):
                    continue
                max_gap = run["exec"]["max_entry_gap"]
                if max_gap is not None and price / candidate.Close - 1 > max_gap:
                    continue
                budget = min(cash, _portfolio_value(cash, positions, day) * run["exec"]["position_weight"])
                buy_price = price * (1 + run["exec"]["slippage_rate"])
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


def _candidate_frame(day: pd.DataFrame, run: dict) -> pd.DataFrame:
    rules = run["rules"]
    trend_col = rules["trend"]
    if trend_col not in day:
        trend_col = "ma60"
    mask = (
        day["market_regime"].isin(["bull", "neutral"])
        & (day["score"] >= rules["score_min"])
        & (day["rs20_rank_pct"] <= rules["rs20_max"])
        & (day["rs60_rank_pct"] <= rules["rs60_max"])
        & (day["Close"] > day[trend_col])
        & (day["ma20_slope"] > -0.005)
        & day["rsi14"].between(rules["rsi_min"], rules["rsi_max"])
        & (day["ret20"] > day["benchmark_ret20"] * 0.4)
        & (day["ret60"] > day["benchmark_ret60"] * 0.4)
        & (day["ret5"] <= 0.15)
        & (day["ret20"] <= rules["ret20_max"])
        & day["volume_ratio"].between(rules["volume_min"], rules["volume_max"])
        & (day["ma20_deviation"].abs() <= rules["ma20_dev_max"])
        & (day["fundamental_score"].notna())
    )
    if run.get("filter") is not None:
        mask = mask & run["filter"](day)
    return day[mask].copy()


def _exit_reason(pos: Position, row: pd.Series, run: dict) -> str | None:
    if run["exec"]["bear_exit"] and row["market_regime"] == "bear":
        return "market_bear"
    if row["Close"] <= pos.entry_price * (1 - run["exec"]["stop_loss"]):
        return "stop_loss"
    if pos.holding_bars >= run["exec"]["max_holding_days"]:
        return "max_holding_days"
    trend_col = f"ma{run['exec']['trend_ma']}"
    if trend_col in row and row["Close"] < row[trend_col]:
        return f"below_{trend_col}"
    return None


def _portfolio_value(cash: float, positions: dict[str, Position], day: pd.DataFrame) -> float:
    value = cash
    for symbol, pos in positions.items():
        if symbol in day.index and not pd.isna(day.at[symbol, "Close"]):
            value += pos.shares * float(day.at[symbol, "Close"])
    return value


def _next_price(frame: pd.DataFrame, signal_date: pd.Timestamp, price_type: str) -> float:
    location = frame.index.searchsorted(signal_date, side="right")
    if location >= len(frame):
        return float("nan")
    column = "Close" if price_type == "close" else "Open"
    return float(frame.iloc[location][column])


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


def _pivot_summary(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot_table(index=["category", "scenario"], columns="metric", values="value", aggfunc="first")
    wanted = [
        "total_return",
        "annual_return",
        "max_drawdown",
        "sharpe",
        "trades",
        "win_rate",
        "avg_trade_return",
        "final_equity",
    ]
    pivot = pivot[[col for col in wanted if col in pivot.columns]].reset_index()
    return pivot.sort_values(["category", "annual_return"], ascending=[True, False])


if __name__ == "__main__":
    main()
