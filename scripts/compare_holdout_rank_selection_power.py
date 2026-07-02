from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fixed8_backtest.data import download_ohlcv, read_universe
from fixed8_backtest.reports import summarize

from holdout_ranker_ensemble import (
    _add_enhanced_features,
    _add_forward_targets,
    _candidate_mask,
    _load_price_data,
    _run_backtest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "fixed8_control_eva_top15_pool_best_sweep.json"
PANEL_PATH = ROOT / "outputs" / "fixed8_control_eva_top15_pool_best_sweep" / "daily_features.csv"
PREDICTIONS_PATH = ROOT / "outputs" / "holdout_ranker_ensemble" / "holdout_predictions.csv"
OUTPUT_DIR = ROOT / "outputs" / "holdout_rank_selection_power"


SCORE_COLUMNS = [
    "ensemble_score",
    "stable_ensemble_score",
    "ranker_score",
    "reg_forward60",
    "final_score",
    "score",
    "technical_score",
    "fundamental_score",
    "eva_like_score",
    "inverse_fundamental_rank",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"], low_memory=False)
    panel = panel.sort_values(["symbol", "date"], ignore_index=True)
    panel = _add_forward_targets(_add_enhanced_features(panel))
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    panel = panel.merge(predictions, on=["date", "symbol"], how="left")
    panel["inverse_fundamental_rank"] = -panel["fundamental_rank"]

    rng = np.random.default_rng(20260606)
    holdout_candidate_index = panel.index[(panel["date"] >= "2025-01-01") & _candidate_mask(panel)]
    for seed in range(10):
        panel[f"random_{seed}"] = np.nan
        panel.loc[holdout_candidate_index, f"random_{seed}"] = rng.normal(size=len(holdout_candidate_index))

    data = _load_price_data(config)
    summaries = []
    for score_col in [*SCORE_COLUMNS, *[f"random_{seed}" for seed in range(10)]]:
        if score_col not in panel:
            continue
        equity, trades = _run_backtest(panel, data, config, pd.Timestamp("2025-01-01"), score_col, 0.15)
        summary = summarize(equity, trades)
        summary.insert(0, "score_col", score_col)
        summaries.append(summary)
        equity.to_csv(OUTPUT_DIR / f"{score_col}_equity_curve.csv", index=False)
        trades.to_csv(OUTPUT_DIR / f"{score_col}_trades.csv", index=False)

    summary_df = pd.concat(summaries, ignore_index=True)
    random_summary = _aggregate_random(summary_df)
    summary_df.to_csv(OUTPUT_DIR / "backtest_by_score.csv", index=False)
    random_summary.to_csv(OUTPUT_DIR / "random_backtest_summary.csv", index=False)

    selection_power = _selection_power(panel)
    selection_power.to_csv(OUTPUT_DIR / "selection_power.csv", index=False)
    yearly_selection = _yearly_selection_power(panel)
    yearly_selection.to_csv(OUTPUT_DIR / "yearly_selection_power.csv", index=False)

    print("Backtest by score:")
    print(summary_df.to_string(index=False))
    print("\nRandom aggregate:")
    print(random_summary.to_string(index=False))
    print("\nSelection power:")
    print(selection_power.to_string(index=False))
    print("\nYearly selection power:")
    print(yearly_selection.to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


def _aggregate_random(summary_df: pd.DataFrame) -> pd.DataFrame:
    random_rows = summary_df[summary_df["score_col"].str.startswith("random_")]
    if random_rows.empty:
        return pd.DataFrame()
    metrics = ["total_return", "annual_return", "max_drawdown", "sharpe", "trades", "win_rate", "avg_trade_return"]
    rows = []
    for metric in metrics:
        values = random_rows.loc[random_rows["metric"] == metric, "value"].astype(float)
        rows.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def _selection_power(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidates = panel[(panel["date"] >= "2025-01-01") & _candidate_mask(panel)].copy()
    for score_col in [*SCORE_COLUMNS, "random_0"]:
        if score_col not in candidates:
            continue
        scored = candidates.dropna(subset=[score_col, "forward_60d_return"])
        for top_n in [1, 3, 5, 10]:
            selected = (
                scored.sort_values(["date", score_col], ascending=[True, False])
                .groupby("date", as_index=False)
                .head(top_n)
            )
            rows.append(_selection_row("all", score_col, top_n, selected))
    return pd.DataFrame(rows)


def _yearly_selection_power(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidates = panel[(panel["date"] >= "2025-01-01") & _candidate_mask(panel)].copy()
    candidates["year"] = candidates["date"].dt.year
    for year, year_group in candidates.groupby("year"):
        for score_col in [*SCORE_COLUMNS, "random_0"]:
            if score_col not in year_group:
                continue
            scored = year_group.dropna(subset=[score_col, "forward_60d_return"])
            for top_n in [1, 3, 5]:
                selected = (
                    scored.sort_values(["date", score_col], ascending=[True, False])
                    .groupby("date", as_index=False)
                    .head(top_n)
                )
                rows.append(_selection_row(int(year), score_col, top_n, selected))
    return pd.DataFrame(rows)


def _selection_row(period: str | int, score_col: str, top_n: int, selected: pd.DataFrame) -> dict:
    returns = selected["forward_60d_return"].astype(float)
    return {
        "period": period,
        "score_col": score_col,
        "top_n": top_n,
        "rows": len(selected),
        "avg_forward60": returns.mean(),
        "median_forward60": returns.median(),
        "hit_rate_forward60_positive": (returns > 0).mean(),
        "p75_forward60": returns.quantile(0.75),
        "p25_forward60": returns.quantile(0.25),
    }


if __name__ == "__main__":
    main()
