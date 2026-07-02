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
OUTPUT_DIR = ROOT / "outputs" / "no_pool_lgbm_anti_overfit"

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
    panel = _add_forward_targets(panel)
    panel["year"] = panel["date"].dt.year
    trainable = panel.dropna(subset=[*FEATURE_COLUMNS, "reward_60"]).copy()
    trainable = trainable[trainable["date"] >= pd.Timestamp("2020-07-01")]
    candidates = panel[_no_eva_pool_candidate_mask(panel)].copy()
    candidates["year"] = candidates["date"].dt.year

    data = _load_price_data(config)
    protocols = ["expanding_purged", "rolling3y_purged"]
    targets = ["reward_60", "forward_60d_return", "reward_120", "top_quantile_60"]

    all_summaries = []
    all_metrics = []
    for protocol in protocols:
        for target in targets:
            print(f"Running {protocol} / {target}", flush=True)
            predictions, metrics = _walk_forward_predict(trainable, candidates, protocol, target)
            metrics["protocol"] = protocol
            metrics["target"] = target
            all_metrics.append(metrics)

            scored_panel = panel.merge(predictions, on=["date", "symbol"], how="left")
            equity, trades = _run_backtest(scored_panel, data, config, pd.Timestamp("2023-01-01"))
            summary = summarize(equity, trades)
            summary.insert(0, "strategy", f"{protocol}_{target}")
            all_summaries.append(summary)
            run_dir = OUTPUT_DIR / f"{protocol}_{target}"
            run_dir.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(run_dir / "candidate_predictions.csv", index=False)
            equity.to_csv(run_dir / "equity_curve.csv", index=False)
            trades.to_csv(run_dir / "trades.csv", index=False)

    summary_df = pd.concat(all_summaries, ignore_index=True)
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summary_df.to_csv(OUTPUT_DIR / "anti_overfit_summaries.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "walkforward_metrics.csv", index=False)
    print(summary_df.to_string(index=False))
    print("\nWalk-forward metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


def _walk_forward_predict(
    trainable: pd.DataFrame,
    candidates: pd.DataFrame,
    protocol: str,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    metrics = []
    for year in [2023, 2024, 2025, 2026]:
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year + 1}-01-01")
        train_end = test_start - pd.Timedelta(days=90)
        if protocol == "rolling3y_purged":
            train_start = test_start - pd.DateOffset(years=3)
        else:
            train_start = pd.Timestamp("2020-07-01")

        train = trainable[(trainable["date"] >= train_start) & (trainable["date"] <= train_end)].copy()
        test_candidates = candidates[(candidates["date"] >= test_start) & (candidates["date"] < test_end)].copy()
        train = train.dropna(subset=[target, *FEATURE_COLUMNS])
        if train.empty or test_candidates.empty:
            continue
        model = _fit_lgbm(train, target, seed=year + (1000 if protocol.startswith("rolling") else 0))
        test_candidates["model_score"] = model.predict(_clean_features(test_candidates[FEATURE_COLUMNS]))
        predictions.append(test_candidates[["date", "symbol", "model_score"]])

        test_daily = trainable[(trainable["date"] >= test_start) & (trainable["date"] < test_end)].dropna(
            subset=[target, *FEATURE_COLUMNS]
        )
        pred_daily = model.predict(_clean_features(test_daily[FEATURE_COLUMNS])) if not test_daily.empty else np.array([])
        metrics.append(
            {
                "year": year,
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "train_rows": len(train),
                "candidate_rows": len(test_candidates),
                "test_daily_rows": len(test_daily),
                "test_rank_corr": _rank_corr(test_daily[target].to_numpy(), pred_daily),
                "candidate_score_mean": test_candidates["model_score"].mean(),
                "candidate_score_median": test_candidates["model_score"].median(),
                "best_iteration": getattr(model, "best_iteration_", None) or model.n_estimators,
            }
        )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)


def _fit_lgbm(train: pd.DataFrame, target: str, seed: int) -> lgb.LGBMRegressor:
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
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(_clean_features(train[FEATURE_COLUMNS]), _clip_target(train[target]))
    return model


def _add_forward_targets(panel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in panel.groupby("symbol", sort=False):
        g = group.sort_values("date").copy()
        entry_open = g["Open"].shift(-1)
        close_20 = g["Close"].shift(-21)
        close_60 = g["Close"].shift(-61)
        close_120 = g["Close"].shift(-121)
        low_60 = g["Low"].shift(-1).rolling(60, min_periods=20).min().shift(-59)
        low_120 = g["Low"].shift(-1).rolling(120, min_periods=40).min().shift(-119)
        g["forward_20d_return"] = close_20 / entry_open - 1
        g["forward_60d_return"] = close_60 / entry_open - 1
        g["forward_120d_return"] = close_120 / entry_open - 1
        g["forward_60d_max_drawdown"] = low_60 / entry_open - 1
        g["forward_120d_max_drawdown"] = low_120 / entry_open - 1
        g["reward_60"] = g["forward_60d_return"] - 0.5 * g["forward_60d_max_drawdown"].clip(upper=0).abs()
        g["reward_120"] = g["forward_120d_return"] - 0.5 * g["forward_120d_max_drawdown"].clip(upper=0).abs()
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out["top_quantile_60"] = (
        out.groupby("date")["forward_60d_return"].rank(ascending=False, pct=True) <= 0.2
    ).astype(float)
    out.loc[out["forward_60d_return"].isna(), "top_quantile_60"] = np.nan
    return out


def _no_eva_pool_candidate_mask(df: pd.DataFrame) -> pd.Series:
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
        & (df["fundamental_score"].notna())
    )


def _run_backtest(
    panel: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    config: dict,
    start_date: pd.Timestamp,
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
            day_frame = day_groups[date]
            candidates = day_frame[_no_eva_pool_candidate_mask(day_frame)].copy()
            if "model_score" not in candidates.columns:
                continue
            candidates = candidates[~candidates["symbol"].isin(positions)]
            candidates = candidates[candidates["symbol"].map(lambda symbol: cooldown_until.get(symbol, pd.Timestamp.min) <= date)]
            if candidates.empty:
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
