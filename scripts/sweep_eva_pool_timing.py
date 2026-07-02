from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fixed8_backtest.data import download_ohlcv, read_universe
from fixed8_backtest.reports import summarize


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = ROOT / "config" / "fixed8_control_eva_top20_pool_trend_equity.json"
PANEL_PATH = ROOT / "outputs" / "fixed8_control_eva_top20_pool_trend_equity" / "daily_features.csv"
OUTPUT_DIR = ROOT / "outputs" / "eva_pool_timing_sweep"
RESULT_PATH = OUTPUT_DIR / "sweep_results.csv"


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
    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        base_config = json.load(handle)

    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"], low_memory=False)
    day_groups = {date: group.copy() for date, group in panel.groupby("date", sort=True)}
    dates = sorted(day_groups)
    symbols = read_universe(ROOT / base_config["universe_csv"], base_config["benchmark_symbol"])
    all_symbols = sorted(set(symbols + [base_config["benchmark_symbol"]]))
    raw_data = download_ohlcv(
        all_symbols,
        base_config["start"],
        base_config["end"],
        batch_size=base_config.get("yfinance_batch_size", 80),
        cache_dir=ROOT / base_config["price_cache_dir"],
    )
    raw_data.pop(base_config["benchmark_symbol"], None)
    data = {symbol: frame for symbol, frame in raw_data.items() if not frame.empty}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_existing_rows()
    completed = {
        (
            int(row["top_n"]),
            float(row["stop_loss"]),
            int(row["trend_ma"]),
            _gap_key(row["max_entry_gap"]),
            float(row["rs20_top_pct"]),
            float(row["rs60_top_pct"]),
        )
        for row in rows
    }
    combos = list(
        itertools.product(
            [10, 15, 20, 25, 30],
            [0.10, 0.12, 0.15],
            [90, 120],
            [None, 0.03, 0.05],
            [(0.20, 0.30), (0.30, 0.40), (0.40, 0.50)],
        )
    )
    for index, (top_n, stop_loss, trend_ma, max_entry_gap, rs_pair) in enumerate(combos, start=1):
        rs20, rs60 = rs_pair
        config = copy.deepcopy(base_config)
        config["strategy_name"] = "eva_pool_timing_sweep"
        config["output_dir"] = str(OUTPUT_DIR)
        config["fundamental_universe_filter"]["top_n"] = top_n
        config["exit"]["stop_loss"] = stop_loss
        config["exit"]["trend_ma"] = trend_ma
        config["entry"]["rs20_top_pct"] = rs20
        config["entry"]["rs60_top_pct"] = rs60
        if max_entry_gap is None:
            config["entry"].pop("max_entry_gap", None)
        else:
            config["entry"]["max_entry_gap"] = max_entry_gap

        key = (top_n, stop_loss, trend_ma, _gap_key(max_entry_gap), rs20, rs60)
        if key in completed:
            continue
        print(
            f"[{index}/{len(combos)}] top={top_n} stop={stop_loss:.2f} "
            f"ma={trend_ma} gap={max_entry_gap} rs={rs20:.2f}/{rs60:.2f}",
            flush=True,
        )
        equity, trades = _run_fast_backtest(day_groups, dates, data, config)
        summary = summarize(equity, trades).set_index("metric")["value"].to_dict()
        avg_exposure = (equity["market_value"] / equity["equity"]).replace([pd.NA, pd.NaT], pd.NA).mean()
        rows.append(
            {
                "top_n": top_n,
                "stop_loss": stop_loss,
                "trend_ma": trend_ma,
                "max_entry_gap": "" if max_entry_gap is None else max_entry_gap,
                "rs20_top_pct": rs20,
                "rs60_top_pct": rs60,
                "total_return": summary.get("total_return"),
                "annual_return": summary.get("annual_return"),
                "max_drawdown": summary.get("max_drawdown"),
                "sharpe": summary.get("sharpe"),
                "trades": summary.get("trades"),
                "win_rate": summary.get("win_rate"),
                "avg_trade_return": summary.get("avg_trade_return"),
                "final_equity": summary.get("final_equity"),
                "avg_exposure": avg_exposure,
            }
        )
        _write_results(rows)

    result = pd.DataFrame(rows)
    result["score"] = (
        result["annual_return"].astype(float)
        + result["sharpe"].astype(float) * 0.08
        + result["max_drawdown"].astype(float) * 0.4
    )
    result = result.sort_values(["score", "annual_return"], ascending=False)
    result.to_csv(OUTPUT_DIR / "sweep_results.csv", index=False)
    result.head(30).to_csv(OUTPUT_DIR / "top30_results.csv", index=False)
    print(result.head(30).to_string(index=False))
    print(f"\nSaved sweep results to {OUTPUT_DIR}")


def _run_fast_backtest(
    day_groups: dict[pd.Timestamp, pd.DataFrame],
    dates: list[pd.Timestamp],
    data: dict[str, pd.DataFrame],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cash = float(config["initial_cash"])
    positions: dict[str, Position] = {}
    cooldown_until: dict[str, pd.Timestamp] = {}
    trades = []
    equity_rows = []

    for idx, date in enumerate(dates[:-1]):
        next_date = dates[idx + 1]
        day = day_groups[date].set_index("symbol", drop=False)

        for symbol in list(positions):
            pos = positions[symbol]
            if symbol not in day.index:
                continue
            row = day.loc[symbol]
            pos.holding_bars += 1
            if row["rs20_rank_pct"] > pos.last_rs20_rank_pct:
                pos.rs20_weak_count += 1
            else:
                pos.rs20_weak_count = 0
            pos.last_rs20_rank_pct = float(row["rs20_rank_pct"])
            exit_reason = _exit_reason(pos, row, config)
            if not exit_reason:
                continue
            open_price = _next_open(data[symbol], date)
            if pd.isna(open_price):
                continue
            proceeds = pos.shares * open_price * (1 - config["slippage_rate"])
            fee = proceeds * config["commission_rate"]
            tax = proceeds * config["tax_rate"]
            cash += proceeds - fee - tax
            pnl = proceeds - fee - tax - (pos.shares * pos.entry_price)
            trades.append(
                {
                    "symbol": symbol,
                    "entry_date": pos.entry_date,
                    "exit_signal_date": date,
                    "exit_date": next_date,
                    "entry_price": pos.entry_price,
                    "exit_price": open_price,
                    "shares": pos.shares,
                    "pnl": pnl,
                    "return_pct": (open_price / pos.entry_price) - 1,
                    "holding_days": pos.holding_bars,
                    "exit_reason": exit_reason,
                }
            )
            if exit_reason == "stop_loss":
                cooldown_until[symbol] = date + pd.Timedelta(days=config["cooldown_days_after_stop"])
            del positions[symbol]

        free_slots = config["max_positions"] - len(positions)
        if free_slots > 0:
            candidates = _entry_candidates(day_groups[date], config)
            candidates = candidates[~candidates["symbol"].isin(positions)]
            candidates = candidates[
                candidates["symbol"].map(lambda symbol: cooldown_until.get(symbol, pd.Timestamp.min) <= date)
            ]
            for candidate in candidates.head(free_slots).itertuples(index=False):
                open_price = _next_open(data[candidate.symbol], date)
                if pd.isna(open_price):
                    continue
                max_entry_gap = config["entry"].get("max_entry_gap")
                if max_entry_gap is not None and open_price / candidate.Close - 1 > max_entry_gap:
                    continue
                base_value = _portfolio_value(cash, positions, day)
                budget = min(cash, base_value * config["position_weight"])
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


def _entry_candidates(day: pd.DataFrame, config: dict) -> pd.DataFrame:
    entry = config["entry"]
    universe_filter = config["fundamental_universe_filter"]
    allowed_market_regimes = set(entry.get("allowed_market_regimes", ["bull", "neutral"]))
    universe_mask = _fundamental_universe_mask(day, universe_filter)
    pool_count = int(universe_mask.sum())
    if pool_count <= 0:
        return day.iloc[0:0]
    rs20_rank = day["ret20"].where(universe_mask).rank(ascending=False, pct=True)
    rs60_rank = day["ret60"].where(universe_mask).rank(ascending=False, pct=True)
    rs20_cutoff = max(entry["rs20_top_pct"], 1 / pool_count)
    rs60_cutoff = max(entry["rs60_top_pct"], 1 / pool_count)
    mask = (
        universe_mask
        & day["market_regime"].isin(allowed_market_regimes)
        & (day["score"] >= entry["min_score"])
        & (rs20_rank <= rs20_cutoff)
        & (rs60_rank <= rs60_cutoff)
        & (day["Close"] > day["ma20"])
        & (day["ma20"] > day["ma60"])
        & (day["ma20_slope"] > 0)
        & day["rsi14"].between(entry["rsi_min"], entry["rsi_max"])
        & (day["ret20"] > day["benchmark_ret20"])
        & (day["ret60"] > day["benchmark_ret60"])
        & (day["ret5"] <= entry["ret5_max"])
        & (day["ret20"] <= entry["ret20_max"])
        & day["volume_ratio"].between(entry["volume_ratio_min"], entry["volume_ratio_max"])
        & (day["ma20_deviation"].abs() <= entry["ma20_deviation_max"])
    )
    return day[mask].sort_values(["final_score", "score"], ascending=False)


def _fundamental_universe_mask(day: pd.DataFrame, universe_filter: dict) -> pd.Series:
    mask = day["fundamental_score"] >= universe_filter.get("min_score", 0)
    top_n = universe_filter.get("top_n")
    if top_n is not None:
        mask = mask & (day["fundamental_rank"] <= top_n)
    return mask


def _exit_reason(pos: Position, row: pd.Series, config: dict) -> str | None:
    if row["market_regime"] == "bear":
        return "market_bear"
    if row["Close"] <= pos.entry_price * (1 - config["exit"]["stop_loss"]):
        return "stop_loss"
    if pos.holding_bars >= config["exit"]["max_holding_days"]:
        return "max_holding_days"
    trend_ma = config["exit"].get("trend_ma", 120)
    trend_column = f"ma{trend_ma}"
    if trend_column in row and row["Close"] < row[trend_column]:
        return f"below_ma{trend_ma}"
    if pos.rs20_weak_count >= config["exit"]["rs20_weak_days"]:
        return "rs20_weak"
    return None


def _portfolio_value(cash: float, positions: dict[str, Position], day: pd.DataFrame) -> float:
    market_value = 0.0
    by_symbol = day.set_index("symbol", drop=False)
    for symbol, pos in positions.items():
        if symbol in by_symbol.index and not pd.isna(by_symbol.at[symbol, "Close"]):
            market_value += pos.shares * float(by_symbol.at[symbol, "Close"])
    return cash + market_value


def _next_open(frame: pd.DataFrame, signal_date: pd.Timestamp) -> float:
    location = frame.index.searchsorted(signal_date, side="right")
    if location >= len(frame):
        return float("nan")
    return float(frame.iloc[location]["Open"])


def _load_existing_rows() -> list[dict]:
    if not RESULT_PATH.exists():
        return []
    return pd.read_csv(RESULT_PATH).to_dict("records")


def _write_results(rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(RESULT_PATH, index=False)


def _gap_key(value: object) -> str:
    if value in ("", None) or pd.isna(value):
        return ""
    return f"{float(value):.2f}"


if __name__ == "__main__":
    main()
