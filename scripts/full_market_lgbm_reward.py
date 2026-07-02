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
OUTPUT_DIR = ROOT / "outputs" / "full_market_lgbm_reward"


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
    panel = _add_forward_reward(panel)
    trainable = panel.dropna(subset=["reward_60", *FEATURE_COLUMNS]).copy()
    trainable = trainable[trainable["date"] >= pd.Timestamp("2020-07-01")]
    trainable["year"] = trainable["date"].dt.year

    candidates = panel[_candidate_mask(panel, config)].copy()
    candidates["year"] = candidates["date"].dt.year

    predictions = []
    metrics = []
    importances = []
    for year in [2023, 2024, 2025, 2026]:
        train = trainable[trainable["year"] < year].copy()
        valid = trainable[trainable["year"] == year - 1].copy()
        if not valid.empty and year > 2023:
            train = trainable[trainable["year"] < year - 1].copy()
        test_candidates = candidates[candidates["year"] == year].copy()
        if train.empty or test_candidates.empty:
            continue

        x_train = _clean_features(train[FEATURE_COLUMNS])
        y_train = _clip_target(train["reward_60"])
        eval_set = None
        callbacks = []
        if not valid.empty and len(valid) > 1000:
            x_valid = _clean_features(valid[FEATURE_COLUMNS])
            y_valid = _clip_target(valid["reward_60"])
            eval_set = [(x_valid, y_valid)]
            callbacks = [lgb.early_stopping(60, verbose=False)]

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=900,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=6,
            min_child_samples=250,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=1.5,
            random_state=year,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x_train, y_train, eval_set=eval_set, callbacks=callbacks)

        test_candidates["lgbm_reward_score"] = model.predict(_clean_features(test_candidates[FEATURE_COLUMNS]))
        predictions.append(test_candidates[["date", "symbol", "lgbm_reward_score"]])

        sampled = trainable[trainable["year"] == year].dropna(subset=["reward_60", *FEATURE_COLUMNS]).copy()
        sample_pred = model.predict(_clean_features(sampled[FEATURE_COLUMNS])) if not sampled.empty else np.array([])
        metrics.append(
            {
                "year": year,
                "train_rows": len(train),
                "candidate_rows": len(test_candidates),
                "test_daily_rows": len(sampled),
                "test_rank_corr": _rank_corr(sampled["reward_60"].to_numpy(), sample_pred),
                "candidate_score_mean": test_candidates["lgbm_reward_score"].mean(),
                "candidate_score_median": test_candidates["lgbm_reward_score"].median(),
                "best_iteration": getattr(model, "best_iteration_", None) or model.n_estimators,
            }
        )
        importances.append(
            pd.DataFrame(
                {
                    "year": year,
                    "feature": FEATURE_COLUMNS,
                    "importance": model.feature_importances_,
                }
            )
        )

    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(OUTPUT_DIR / "candidate_reward_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUTPUT_DIR / "walkforward_metrics.csv", index=False)
    pd.concat(importances, ignore_index=True).to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    scored_panel = panel.merge(pred, on=["date", "symbol"], how="left")
    data = _load_price_data(config)

    start_date = pd.Timestamp("2023-01-01")
    runs = [
        ("base_2023_oos", "base"),
        ("lgbm_reward_rank_2023_oos", "reward_rank"),
        ("hybrid_lgbm_rank_2023_oos", "hybrid_rank"),
        ("lgbm_filter_rank_2023_oos", "filter_rank"),
    ]
    summaries = []
    for name, mode in runs:
        equity, trades = _run_backtest(scored_panel, data, config, start_date, mode)
        summary = summarize(equity, trades)
        summary.insert(0, "strategy", name)
        summaries.append(summary)
        equity.to_csv(OUTPUT_DIR / f"{name}_equity_curve.csv", index=False)
        trades.to_csv(OUTPUT_DIR / f"{name}_trades.csv", index=False)

    output = pd.concat(summaries, ignore_index=True)
    output.to_csv(OUTPUT_DIR / "backtest_summaries.csv", index=False)
    print(output.to_string(index=False))
    print("\nWalk-forward metrics:")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


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


def _add_forward_reward(panel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in panel.groupby("symbol", sort=False):
        g = group.sort_values("date").copy()
        entry_open = g["Open"].shift(-1)
        close_60 = g["Close"].shift(-61)
        future_low = g["Low"].shift(-1).rolling(60, min_periods=20).min().shift(-59)
        g["forward_60d_return"] = close_60 / entry_open - 1
        g["forward_60d_max_drawdown"] = future_low / entry_open - 1
        g["reward_60"] = g["forward_60d_return"] - 0.5 * g["forward_60d_max_drawdown"].clip(upper=0).abs()
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def _candidate_mask(df: pd.DataFrame, config: dict) -> pd.Series:
    entry = config["entry"]
    universe = config["fundamental_universe_filter"]
    universe_mask = (df["fundamental_score"] >= universe.get("min_score", 0)) & (
        df["fundamental_rank"] <= universe["top_n"]
    )
    pool_counts = universe_mask.groupby(df["date"]).transform("sum").clip(lower=1)
    rs20_pool = df["ret20"].where(universe_mask).groupby(df["date"]).rank(ascending=False, pct=True)
    rs60_pool = df["ret60"].where(universe_mask).groupby(df["date"]).rank(ascending=False, pct=True)
    rs20_cutoff = np.maximum(entry["rs20_top_pct"], 1 / pool_counts)
    rs60_cutoff = np.maximum(entry["rs60_top_pct"], 1 / pool_counts)
    return (
        universe_mask
        & df["market_regime"].isin(entry.get("allowed_market_regimes", ["bull", "neutral"]))
        & (df["score"] >= entry["min_score"])
        & (rs20_pool <= rs20_cutoff)
        & (rs60_pool <= rs60_cutoff)
        & (df["Close"] > df["ma20"])
        & (df["ma20"] > df["ma60"])
        & (df["ma20_slope"] > 0)
        & df["rsi14"].between(entry["rsi_min"], entry["rsi_max"])
        & (df["ret20"] > df["benchmark_ret20"])
        & (df["ret60"] > df["benchmark_ret60"])
        & (df["ret5"] <= entry["ret5_max"])
        & (df["ret20"] <= entry["ret20_max"])
        & df["volume_ratio"].between(entry["volume_ratio_min"], entry["volume_ratio_max"])
        & (df["ma20_deviation"].abs() <= entry["ma20_deviation_max"])
    )


def _clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _clip_target(target: pd.Series) -> pd.Series:
    low, high = target.quantile([0.01, 0.99])
    return target.clip(low, high)


def _rank_corr(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    return float(spearmanr(y, pred, nan_policy="omit").correlation)


def _run_backtest(
    panel: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    config: dict,
    start_date: pd.Timestamp,
    mode: str,
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
            if row["rs20_rank_pct"] > pos.last_rs20_rank_pct:
                pos.rs20_weak_count += 1
            else:
                pos.rs20_weak_count = 0
            pos.last_rs20_rank_pct = float(row["rs20_rank_pct"])
            reason = _exit_reason(pos, row, config)
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
            candidates = day_groups[date][_candidate_mask(day_groups[date], config)].copy()
            if not candidates.empty:
                candidates = candidates[~candidates["symbol"].isin(positions)]
                candidates = candidates[
                    candidates["symbol"].map(lambda symbol: cooldown_until.get(symbol, pd.Timestamp.min) <= date)
                ]
                if not candidates.empty:
                    if mode == "base":
                        candidates = candidates.sort_values(["final_score", "score"], ascending=False)
                    else:
                        candidates = candidates.dropna(subset=["lgbm_reward_score"])
                        if mode == "filter_rank":
                            threshold = candidates["lgbm_reward_score"].quantile(0.55)
                            candidates = candidates[candidates["lgbm_reward_score"] >= threshold]
                        if mode == "hybrid_rank":
                            rank_ml = candidates["lgbm_reward_score"].rank(pct=True)
                            rank_base = candidates["final_score"].rank(pct=True)
                            candidates["hybrid_score"] = 0.55 * rank_base + 0.45 * rank_ml
                            candidates = candidates.sort_values(["hybrid_score", "final_score"], ascending=False)
                        else:
                            candidates = candidates.sort_values(["lgbm_reward_score", "final_score"], ascending=False)
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


def _exit_reason(pos: Position, row: pd.Series, config: dict) -> str | None:
    if row["market_regime"] == "bear":
        return "market_bear"
    if row["Close"] <= pos.entry_price * (1 - config["exit"]["stop_loss"]):
        return "stop_loss"
    if pos.holding_bars >= config["exit"]["max_holding_days"]:
        return "max_holding_days"
    trend_ma = config["exit"].get("trend_ma", 120)
    if row["Close"] < row[f"ma{trend_ma}"]:
        return f"below_ma{trend_ma}"
    if pos.rs20_weak_count >= config["exit"]["rs20_weak_days"]:
        return "rs20_weak"
    return None


def _portfolio_value(cash: float, positions: dict[str, Position], day: pd.DataFrame) -> float:
    value = cash
    by_symbol = day.set_index("symbol", drop=False)
    for symbol, pos in positions.items():
        if symbol in by_symbol.index and not pd.isna(by_symbol.at[symbol, "Close"]):
            value += pos.shares * float(by_symbol.at[symbol, "Close"])
    return value


def _next_open(frame: pd.DataFrame, signal_date: pd.Timestamp) -> float:
    location = frame.index.searchsorted(signal_date, side="right")
    if location >= len(frame):
        return float("nan")
    return float(frame.iloc[location]["Open"])


if __name__ == "__main__":
    main()
