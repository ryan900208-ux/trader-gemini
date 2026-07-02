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
OUTPUT_DIR = ROOT / "outputs" / "ml_top15_walkforward"


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
    candidates = _build_dataset(panel, config)
    candidates.to_csv(OUTPUT_DIR / "ml_candidates.csv", index=False)

    predictions, metrics = _walk_forward_predictions(candidates)
    predictions.to_csv(OUTPUT_DIR / "walkforward_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "walkforward_metrics.csv", index=False)

    scored_panel = panel.merge(
        predictions[["date", "symbol", "ml_win_prob", "tree_win_prob"]],
        on=["date", "symbol"],
        how="left",
    )

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
    data = {symbol: frame for symbol, frame in raw_data.items() if not frame.empty}

    start_date = pd.Timestamp("2023-01-01")
    base_equity, base_trades = _run_backtest(scored_panel, data, config, start_date, "base")
    rank_equity, rank_trades = _run_backtest(scored_panel, data, config, start_date, "ml_rank")
    filter_equity, filter_trades = _run_backtest(scored_panel, data, config, start_date, "ml_filter_rank")
    tree_rank_equity, tree_rank_trades = _run_backtest(scored_panel, data, config, start_date, "tree_rank")
    tree_filter_equity, tree_filter_trades = _run_backtest(
        scored_panel, data, config, start_date, "tree_filter_rank"
    )

    summaries = []
    for name, equity, trades in [
        ("base_2023_oos", base_equity, base_trades),
        ("ml_rank_2023_oos", rank_equity, rank_trades),
        ("ml_filter_rank_2023_oos", filter_equity, filter_trades),
        ("tree_rank_2023_oos", tree_rank_equity, tree_rank_trades),
        ("tree_filter_rank_2023_oos", tree_filter_equity, tree_filter_trades),
    ]:
        summary = summarize(equity, trades)
        summary.insert(0, "strategy", name)
        summaries.append(summary)
        equity.to_csv(OUTPUT_DIR / f"{name}_equity_curve.csv", index=False)
        trades.to_csv(OUTPUT_DIR / f"{name}_trades.csv", index=False)

    pd.concat(summaries, ignore_index=True).to_csv(OUTPUT_DIR / "backtest_summaries.csv", index=False)
    print(pd.concat(summaries, ignore_index=True).to_string(index=False))
    print("\nWalk-forward metrics:")
    print(metrics.to_string(index=False))
    print(f"\nSaved ML outputs to {OUTPUT_DIR}")


def _build_dataset(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    candidate_mask = _candidate_mask(panel, config)
    candidates = panel[candidate_mask].copy()
    candidates["forward_60d_return"] = np.nan
    candidates["forward_20d_return"] = np.nan
    candidates["hit_12_stop_20d"] = False
    by_symbol = {symbol: group.sort_values("date") for symbol, group in panel.groupby("symbol", sort=False)}
    for symbol, index_values in candidates.groupby("symbol").groups.items():
        frame = by_symbol[symbol].reset_index(drop=True)
        frame_index = pd.Series(frame.index.to_numpy(), index=frame["date"])
        for row_index in index_values:
            date = candidates.at[row_index, "date"]
            if date not in frame_index.index:
                continue
            loc = int(frame_index.at[date])
            entry_loc = loc + 1
            if entry_loc >= len(frame):
                continue
            entry_open = frame.at[entry_loc, "Open"]
            if pd.isna(entry_open) or entry_open <= 0:
                continue
            loc20 = min(entry_loc + 20, len(frame) - 1)
            loc60 = min(entry_loc + 60, len(frame) - 1)
            candidates.at[row_index, "forward_20d_return"] = frame.at[loc20, "Close"] / entry_open - 1
            candidates.at[row_index, "forward_60d_return"] = frame.at[loc60, "Close"] / entry_open - 1
            low20 = frame.loc[entry_loc:loc20, "Low"].min()
            candidates.at[row_index, "hit_12_stop_20d"] = bool(low20 <= entry_open * 0.88)
    candidates = candidates.dropna(subset=["forward_60d_return"]).copy()
    candidates["win60_label"] = (candidates["forward_60d_return"] > 0).astype(int)
    candidates["year"] = candidates["date"].dt.year
    return candidates[
        [
            "date",
            "symbol",
            "year",
            "win60_label",
            "forward_20d_return",
            "forward_60d_return",
            "hit_12_stop_20d",
            *FEATURE_COLUMNS,
        ]
    ]


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


def _walk_forward_predictions(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    metrics = []
    for year in [2023, 2024, 2025, 2026]:
        train = candidates[candidates["year"] < year].copy()
        test = candidates[candidates["year"] == year].copy()
        if train.empty or test.empty:
            continue
        model = _fit_logistic(train[FEATURE_COLUMNS], train["win60_label"])
        test_probs = _predict_logistic(model, test[FEATURE_COLUMNS])
        train_probs = _predict_logistic(model, train[FEATURE_COLUMNS])
        forest = _fit_forest(train[FEATURE_COLUMNS], train["win60_label"], seed=year)
        tree_test_probs = _predict_forest(forest, test[FEATURE_COLUMNS])
        tree_train_probs = _predict_forest(forest, train[FEATURE_COLUMNS])
        test = test.copy()
        test["ml_win_prob"] = test_probs
        test["tree_win_prob"] = tree_test_probs
        rows.append(test)
        metrics.append(
            {
                "year": year,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_win_rate": train["win60_label"].mean(),
                "test_win_rate": test["win60_label"].mean(),
                "test_auc": _auc(test["win60_label"].to_numpy(), test_probs),
                "train_auc": _auc(train["win60_label"].to_numpy(), train_probs),
                "top_half_win_rate": test.loc[test["ml_win_prob"] >= test["ml_win_prob"].median(), "win60_label"].mean(),
                "bottom_half_win_rate": test.loc[test["ml_win_prob"] < test["ml_win_prob"].median(), "win60_label"].mean(),
                "tree_test_auc": _auc(test["win60_label"].to_numpy(), tree_test_probs),
                "tree_train_auc": _auc(train["win60_label"].to_numpy(), tree_train_probs),
                "tree_top_half_win_rate": test.loc[
                    test["tree_win_prob"] >= test["tree_win_prob"].median(), "win60_label"
                ].mean(),
                "tree_bottom_half_win_rate": test.loc[
                    test["tree_win_prob"] < test["tree_win_prob"].median(), "win60_label"
                ].mean(),
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(metrics)


def _fit_logistic(x: pd.DataFrame, y: pd.Series) -> dict:
    x_arr = x.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = x_arr.median()
    x_arr = x_arr.fillna(medians).to_numpy(dtype=float)
    means = x_arr.mean(axis=0)
    stds = x_arr.std(axis=0)
    stds[stds == 0] = 1
    x_std = (x_arr - means) / stds
    x_design = np.column_stack([np.ones(len(x_std)), x_std])
    weights = np.zeros(x_design.shape[1])
    target = y.to_numpy(dtype=float)
    lr = 0.03
    l2 = 0.02
    for _ in range(1200):
        pred = 1 / (1 + np.exp(-(x_design @ weights).clip(-30, 30)))
        grad = (x_design.T @ (pred - target)) / len(target)
        grad[1:] += l2 * weights[1:]
        weights -= lr * grad
    return {"weights": weights, "means": means, "stds": stds, "medians": medians}


def _predict_logistic(model: dict, x: pd.DataFrame) -> np.ndarray:
    x_arr = x.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_arr = x_arr.fillna(model["medians"]).to_numpy(dtype=float)
    x_std = (x_arr - model["means"]) / model["stds"]
    x_design = np.column_stack([np.ones(len(x_std)), x_std])
    return 1 / (1 + np.exp(-(x_design @ model["weights"]).clip(-30, 30)))


def _fit_forest(x: pd.DataFrame, y: pd.Series, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x_arr, medians = _numeric_matrix(x)
    target = y.to_numpy(dtype=float)
    trees = []
    n_rows, n_features = x_arr.shape
    feature_count = max(3, int(np.sqrt(n_features)))
    for _ in range(120):
        sample_idx = rng.integers(0, n_rows, size=n_rows)
        features = rng.choice(n_features, size=feature_count, replace=False)
        tree = _build_tree(
            x_arr[sample_idx],
            target[sample_idx],
            depth=0,
            max_depth=4,
            min_leaf=12,
            feature_pool=features,
            rng=rng,
        )
        trees.append(tree)
    return {"trees": trees, "medians": medians}


def _predict_forest(model: dict, x: pd.DataFrame) -> np.ndarray:
    x_arr, _ = _numeric_matrix(x, medians=model["medians"])
    preds = np.zeros(len(x_arr), dtype=float)
    for tree in model["trees"]:
        preds += np.array([_predict_tree(tree, row) for row in x_arr])
    return preds / len(model["trees"])


def _numeric_matrix(x: pd.DataFrame, medians: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    frame = x.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = frame.median()
    return frame.fillna(medians).to_numpy(dtype=float), medians


def _build_tree(
    x: np.ndarray,
    y: np.ndarray,
    depth: int,
    max_depth: int,
    min_leaf: int,
    feature_pool: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    probability = float(y.mean()) if len(y) else 0.5
    if depth >= max_depth or len(y) < min_leaf * 2 or probability in {0.0, 1.0}:
        return {"probability": probability}

    best_feature = None
    best_threshold = None
    best_gain = 0.0
    parent_impurity = _gini(y)
    candidate_features = rng.choice(feature_pool, size=min(len(feature_pool), max(2, len(feature_pool) // 2)), replace=False)
    for feature in candidate_features:
        values = x[:, feature]
        thresholds = np.unique(np.nanpercentile(values, [20, 35, 50, 65, 80]))
        for threshold in thresholds:
            left = values <= threshold
            right = ~left
            left_count = int(left.sum())
            right_count = int(right.sum())
            if left_count < min_leaf or right_count < min_leaf:
                continue
            gain = parent_impurity - (
                left_count / len(y) * _gini(y[left]) + right_count / len(y) * _gini(y[right])
            )
            if gain > best_gain:
                best_gain = gain
                best_feature = int(feature)
                best_threshold = float(threshold)

    if best_feature is None:
        return {"probability": probability}

    left_mask = x[:, best_feature] <= best_threshold
    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "probability": probability,
        "left": _build_tree(x[left_mask], y[left_mask], depth + 1, max_depth, min_leaf, feature_pool, rng),
        "right": _build_tree(x[~left_mask], y[~left_mask], depth + 1, max_depth, min_leaf, feature_pool, rng),
    }


def _predict_tree(tree: dict, row: np.ndarray) -> float:
    while "feature" in tree:
        if row[tree["feature"]] <= tree["threshold"]:
            tree = tree["left"]
        else:
            tree = tree["right"]
    return float(tree["probability"])


def _gini(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    probability = float(y.mean())
    return 2 * probability * (1 - probability)


def _auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    positives = y_true == 1
    negatives = y_true == 0
    pos_count = positives.sum()
    neg_count = negatives.sum()
    if pos_count == 0 or neg_count == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[positives].sum() - pos_count * (pos_count + 1) / 2) / (pos_count * neg_count))


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
            if candidates.empty:
                market_value = 0.0
                for symbol, pos in positions.items():
                    if symbol in day.index and not pd.isna(day.at[symbol, "Close"]):
                        market_value += pos.shares * float(day.at[symbol, "Close"])
                equity_rows.append({"date": date, "cash": cash, "market_value": market_value, "equity": cash + market_value})
                continue
            candidates = candidates[~candidates["symbol"].isin(positions)]
            candidates = candidates[
                candidates["symbol"].map(lambda symbol: cooldown_until.get(symbol, pd.Timestamp.min) <= date)
            ]
            if candidates.empty:
                market_value = 0.0
                for symbol, pos in positions.items():
                    if symbol in day.index and not pd.isna(day.at[symbol, "Close"]):
                        market_value += pos.shares * float(day.at[symbol, "Close"])
                equity_rows.append({"date": date, "cash": cash, "market_value": market_value, "equity": cash + market_value})
                continue
            if mode != "base":
                score_column = "tree_win_prob" if mode.startswith("tree") else "ml_win_prob"
                candidates = candidates.dropna(subset=[score_column])
                if mode in {"ml_filter_rank", "tree_filter_rank"}:
                    candidates = candidates[candidates[score_column] >= 0.52]
                candidates = candidates.sort_values([score_column, "final_score"], ascending=False)
            else:
                if "final_score" not in candidates.columns:
                    candidates = candidates.copy()
                    candidates["final_score"] = (
                        config["score_weights"]["technical"] * candidates["technical_score"]
                        + config["score_weights"]["fundamental"] * candidates["fundamental_score"]
                    )
                candidates = candidates.sort_values(["final_score", "score"], ascending=False)
            for candidate in candidates.head(free_slots).itertuples(index=False):
                open_price = _next_open(data[candidate.symbol], date)
                if pd.isna(open_price):
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
