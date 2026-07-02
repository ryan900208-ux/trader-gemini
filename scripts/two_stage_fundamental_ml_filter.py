from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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
OUTPUT_DIR = ROOT / "outputs" / "two_stage_fundamental_ml_filter"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"], low_memory=False)
    panel = panel.sort_values(["symbol", "date"], ignore_index=True)
    panel = _add_forward_targets(_add_enhanced_features(panel))
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    panel = panel.merge(predictions, on=["date", "symbol"], how="left")

    holdout_mask = (panel["date"] >= "2025-01-01") & _candidate_mask(panel)
    panel.loc[holdout_mask, "fundamental_rank_score"] = -panel.loc[holdout_mask, "fundamental_rank"]
    panel.loc[holdout_mask, "fundamental_score_rank"] = panel.loc[holdout_mask].groupby("date")[
        "fundamental_rank_score"
    ].rank(pct=True)
    panel.loc[holdout_mask, "final_score_rank"] = panel.loc[holdout_mask].groupby("date")["final_score"].rank(pct=True)
    panel.loc[holdout_mask, "stable_ml_rank"] = panel.loc[holdout_mask].groupby("date")[
        "stable_ensemble_score"
    ].rank(pct=True)
    panel.loc[holdout_mask, "ensemble_ml_rank"] = panel.loc[holdout_mask].groupby("date")["ensemble_score"].rank(pct=True)

    # Fundamental remains the anchor. ML is used only as a quality gate or secondary tie-breaker.
    panel["fund_ml_70_30"] = 0.70 * panel["fundamental_score_rank"] + 0.30 * panel["stable_ml_rank"]
    panel["fund_ml_85_15"] = 0.85 * panel["fundamental_score_rank"] + 0.15 * panel["stable_ml_rank"]
    panel["final_ml_70_30"] = 0.70 * panel["final_score_rank"] + 0.30 * panel["stable_ml_rank"]
    panel["fund_final_ml"] = (
        0.50 * panel["fundamental_score_rank"] + 0.30 * panel["final_score_rank"] + 0.20 * panel["stable_ml_rank"]
    )

    gate_specs = {
        "fund_gate_ml_top70": ("fundamental_rank_score", panel["stable_ml_rank"] >= 0.30),
        "fund_gate_ml_top60": ("fundamental_rank_score", panel["stable_ml_rank"] >= 0.40),
        "fund_gate_ml_top50": ("fundamental_rank_score", panel["stable_ml_rank"] >= 0.50),
        "fund_gate_ensemble_top60": ("fundamental_rank_score", panel["ensemble_ml_rank"] >= 0.40),
        "final_gate_ml_top60": ("final_score", panel["stable_ml_rank"] >= 0.40),
    }
    for score_name, (_, gate) in gate_specs.items():
        panel[score_name] = np.nan
        panel.loc[holdout_mask & gate, score_name] = panel.loc[holdout_mask & gate, gate_specs[score_name][0]]

    score_cols = [
        "fundamental_rank_score",
        "final_score",
        "stable_ensemble_score",
        "fund_ml_70_30",
        "fund_ml_85_15",
        "final_ml_70_30",
        "fund_final_ml",
        *gate_specs.keys(),
    ]

    data = _load_price_data(config)
    summaries = []
    selection_rows = []
    for score_col in score_cols:
        equity, trades = _run_backtest(panel, data, config, pd.Timestamp("2025-01-01"), score_col, 0.15)
        summary = summarize(equity, trades)
        summary.insert(0, "score_col", score_col)
        summaries.append(summary)
        equity.to_csv(OUTPUT_DIR / f"{score_col}_equity_curve.csv", index=False)
        trades.to_csv(OUTPUT_DIR / f"{score_col}_trades.csv", index=False)
        selection_rows.extend(_selection_power(panel, score_col))

    summary_df = pd.concat(summaries, ignore_index=True)
    selection_df = pd.DataFrame(selection_rows)
    summary_df.to_csv(OUTPUT_DIR / "two_stage_backtest_summary.csv", index=False)
    selection_df.to_csv(OUTPUT_DIR / "two_stage_selection_power.csv", index=False)

    print("Two-stage backtest summary:")
    print(summary_df.to_string(index=False))
    print("\nTwo-stage selection power:")
    print(selection_df.to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


def _selection_power(panel: pd.DataFrame, score_col: str) -> list[dict]:
    candidates = panel[(panel["date"] >= "2025-01-01") & _candidate_mask(panel)].copy()
    scored = candidates.dropna(subset=[score_col, "forward_60d_return"])
    rows = []
    for top_n in [1, 3, 5]:
        selected = (
            scored.sort_values(["date", score_col], ascending=[True, False])
            .groupby("date", as_index=False)
            .head(top_n)
        )
        returns = selected["forward_60d_return"].astype(float)
        rows.append(
            {
                "score_col": score_col,
                "top_n": top_n,
                "rows": len(selected),
                "avg_forward60": returns.mean(),
                "median_forward60": returns.median(),
                "hit_rate_forward60_positive": (returns > 0).mean(),
                "p25_forward60": returns.quantile(0.25),
                "p75_forward60": returns.quantile(0.75),
            }
        )
    return rows


if __name__ == "__main__":
    main()
