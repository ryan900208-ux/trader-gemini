from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fixed8_backtest.data import download_ohlcv, read_universe
from fixed8_backtest.reports import summarize


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "fixed8_control_eva_top15_pool_best_sweep.json"
PANEL_PATH = ROOT / "outputs" / "fixed8_control_eva_top15_pool_best_sweep" / "daily_features.csv"
OUTPUT_DIR = ROOT / "outputs" / "holdout_2025_2026_enhanced_lgbm"


FEATURE_COLUMNS = [
    "technical_score",
    "fundamental_score",
    "eva_like_score",
    "fundamental_rank",
    "rs20_rank_pct",
    "rs60_rank_pct",
    "rsi14",
    "ret5",
    "ret20",
    "ret60",
    "ret120",
    "benchmark_ret20",
    "benchmark_ret60",
    "volume_ratio",
    "ma20_deviation",
    "ma20_slope",
    "roe",
    "roa",
    "roic_proxy",
    "eps",
    "debt_to_equity",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "revenue_growth",
    "Close",
    "close_ma60_ratio",
    "close_ma120_ratio",
    "above_ma120",
    "turnover20",
    "liquidity_rank",
    "price_bucket",
    "eva_rank_bucket",
]


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
    panel = panel.sort_values(["symbol", "date"], ignore_index=True)
    panel = _add_enhanced_features(panel)
    panel = _add_forward_target(panel)

    # Purge 60 trading-day target overlap approximately with a 90 calendar-day gap.
    train = panel[
        (panel["date"] >= pd.Timestamp("2020-07-01"))
        & (panel["date"] <= pd.Timestamp("2024-10-03"))
    ].dropna(subset=["forward_60d_return", *FEATURE_COLUMNS])
    holdout_candidates = panel[
        (panel["date"] >= pd.Timestamp("2025-01-01"))
        & (panel["date"] <= pd.Timestamp("2026-12-31"))
        & _candidate_mask(panel)
    ].copy()

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=550,
        learning_rate=0.035,
        num_leaves=23,
        max_depth=5,
        min_child_samples=350,
        subsample=0.72,
        subsample_freq=1,
        colsample_bytree=0.72,
        reg_alpha=0.5,
        reg_lambda=2.5,
        random_state=202024,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(_clean_features(train[FEATURE_COLUMNS]), _clip_target(train["forward_60d_return"]))
    holdout_candidates["model_score"] = model.predict(_clean_features(holdout_candidates[FEATURE_COLUMNS]))
    predictions = holdout_candidates[["date", "symbol", "model_score"]]
    predictions.to_csv(OUTPUT_DIR / "holdout_predictions.csv", index=False)

    scored_panel = panel.merge(predictions, on=["date", "symbol"], how="left")
    data = _load_price_data(config)
    runs = [
        ("holdout_2025_2026_stop12", 0.12, None),
        ("holdout_2025_2026_stop15", 0.15, None),
        ("holdout_2025_2026_stop15_vol0830", 0.15, lambda df: df["volume_ratio"].between(0.8, 3.0)),
    ]
    summaries = []
    for name, stop_loss, extra_filter in runs:
        equity, trades = _run_backtest(scored_panel, data, config, pd.Timestamp("2025-01-01"), stop_loss, extra_filter)
        summary = summarize(equity, trades)
        summary.insert(0, "strategy", name)
        summaries.append(summary)
        equity.to_csv(OUTPUT_DIR / f"{name}_equity_curve.csv", index=False)
        trades.to_csv(OUTPUT_DIR / f"{name}_trades.csv", index=False)

    metrics = _holdout_metrics(panel, predictions)
    metrics.to_csv(OUTPUT_DIR / "holdout_metrics.csv", index=False)
    summary_df = pd.concat(summaries, ignore_index=True)
    summary_df.to_csv(OUTPUT_DIR / "holdout_summary.csv", index=False)
    pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    print(summary_df.to_string(index=False))
    print("\nHoldout metrics:")
    print(metrics.to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


def _holdout_metrics(panel: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    merged = panel.merge(predictions, on=["date", "symbol"], how="inner")
    rows = []
    for year, group in merged.groupby(merged["date"].dt.year):
        rows.append(
            {
                "year": int(year),
                "candidate_rows": len(group),
                "rank_corr": _rank_corr(group["forward_60d_return"].to_numpy(), group["model_score"].to_numpy()),
                "top_half_forward60": group.loc[
                    group["model_score"] >= group["model_score"].median(), "forward_60d_return"
                ].mean(),
                "bottom_half_forward60": group.loc[
                    group["model_score"] < group["model_score"].median(), "forward_60d_return"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def _add_enhanced_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["close_ma60_ratio"] = df["Close"] / df["ma60"] - 1
    df["close_ma120_ratio"] = df["Close"] / df["ma120"] - 1
    df["above_ma120"] = (df["Close"] > df["ma120"]).astype(float)
    df["turnover20"] = df["Close"] * df["volume_ma20"]
    df["liquidity_rank"] = df.groupby("date")["turnover20"].rank(ascending=False, pct=True)
    df["price_bucket"] = pd.cut(df["Close"], [-np.inf, 30, 100, 500, np.inf], labels=[0, 1, 2, 3]).astype(float)
    df["eva_rank_bucket"] = pd.cut(df["fundamental_rank"], [-np.inf, 15, 30, 100, np.inf], labels=[0, 1, 2, 3]).astype(float)
    return df


def _add_forward_target(panel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in panel.groupby("symbol", sort=False):
        g = group.sort_values("date").copy()
        entry_open = g["Open"].shift(-1)
        close_60 = g["Close"].shift(-61)
        g["forward_60d_return"] = close_60 / entry_open - 1
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def _candidate_mask(df: pd.DataFrame) -> pd.Series:
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


def _run_backtest(
    panel: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    config: dict,
    start_date: pd.Timestamp,
    stop_loss: float,
    extra_filter,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cash = float(config["initial_cash"])
    positions: dict[str, Position] = {}
    cooldown_until: dict[str, pd.Timestamp] = {}
    trades = []
    equity_rows = []
    day_groups = {date: group.copy() for date, group in panel.groupby("date", sort=True)}
    dates = [date for date in sorted(day_groups) if date >= start_date]
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
            reason = _exit_reason(pos, row, stop_loss)
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
                    "return_pct": open_price / pos.entry_price - 1,
                    "holding_days": pos.holding_bars,
                    "exit_reason": reason,
                }
            )
            if reason == "stop_loss":
                cooldown_until[symbol] = date + pd.Timedelta(days=config["cooldown_days_after_stop"])
            del positions[symbol]

        free_slots = config["max_positions"] - len(positions)
        if free_slots > 0:
            day_frame = day_groups[date]
            mask = _candidate_mask(day_frame)
            if extra_filter is not None:
                mask = mask & extra_filter(day_frame)
            candidates = day_frame[mask].copy()
            if not candidates.empty and "model_score" in candidates:
                candidates = candidates[~candidates["symbol"].isin(positions)]
                candidates = candidates[candidates["symbol"].map(lambda symbol: cooldown_until.get(symbol, pd.Timestamp.min) <= date)]
                candidates = candidates.dropna(subset=["model_score"])
                candidates = candidates.sort_values(["model_score", "final_score"], ascending=False)
                for candidate in candidates.head(config["max_positions"] - len(positions)).itertuples(index=False):
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


def _clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _clip_target(target: pd.Series) -> pd.Series:
    low, high = target.quantile([0.01, 0.99])
    return target.clip(low, high)


def _rank_corr(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    return float(spearmanr(y, pred, nan_policy="omit").correlation)


if __name__ == "__main__":
    main()
