from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

import full_market_lgbm_no_eva_pool as no_pool
from fixed8_backtest.reports import summarize


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "fixed8_control_eva_top15_pool_best_sweep.json"
PANEL_PATH = ROOT / "outputs" / "fixed8_control_eva_top15_pool_best_sweep" / "daily_features.csv"
OUTPUT_DIR = ROOT / "outputs" / "no_pool_lgbm_robustness"
OOS_OUTPUT_DIR = ROOT / "outputs" / "full_market_lgbm_no_eva_pool"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"], low_memory=False)
    panel = panel.sort_values(["symbol", "date"], ignore_index=True)
    panel = no_pool._add_forward_reward(panel)
    oos_pred = pd.read_csv(OOS_OUTPUT_DIR / "no_pool_candidate_predictions.csv", parse_dates=["date"])
    oos_panel = panel.merge(oos_pred, on=["date", "symbol"], how="left")
    data = no_pool._load_price_data(config)

    yearly_rows = []
    for year in [2023, 2024, 2025, 2026]:
        equity, trades = no_pool._run_backtest(oos_panel, data, config, pd.Timestamp(f"{year}-01-01"), "lgbm_rank")
        if not equity.empty:
            equity = equity[equity["date"] < pd.Timestamp(f"{year + 1}-01-01")]
        if not trades.empty:
            trades = trades[pd.to_datetime(trades["entry_date"]) < pd.Timestamp(f"{year + 1}-01-01")]
        yearly_rows.extend(_summary_rows(f"oos_{year}", equity, trades))
    pd.DataFrame(yearly_rows).to_csv(OUTPUT_DIR / "yearly_oos_summary.csv", index=False)

    base_equity, base_trades = no_pool._run_backtest(oos_panel, data, config, pd.Timestamp("2023-01-01"), "lgbm_rank")
    base_trades.to_csv(OUTPUT_DIR / "oos_lgbm_rank_trades.csv", index=False)
    robustness = []
    robustness.extend(_summary_rows("base_oos_2023_2026", base_equity, base_trades))
    for n in [1, 3, 5]:
        excluded = _top_winner_symbols(base_trades, n)
        filtered_panel = oos_panel[~oos_panel["symbol"].isin(excluded)].copy()
        equity, trades = no_pool._run_backtest(filtered_panel, data, config, pd.Timestamp("2023-01-01"), "lgbm_rank")
        robustness.extend(_summary_rows(f"exclude_top{n}_winners_{','.join(excluded)}", equity, trades))
    filtered_panel = oos_panel[oos_panel["date"].dt.year != 2023].copy()
    equity, trades = no_pool._run_backtest(filtered_panel, data, config, pd.Timestamp("2024-01-01"), "lgbm_rank")
    robustness.extend(_summary_rows("exclude_2023_start_2024", equity, trades))
    pd.DataFrame(robustness).to_csv(OUTPUT_DIR / "winner_exclusion_summary.csv", index=False)

    cost_rows = []
    for slippage in [0.001, 0.003, 0.005, 0.01]:
        cfg = json.loads(json.dumps(config))
        cfg["slippage_rate"] = slippage
        equity, trades = no_pool._run_backtest(oos_panel, data, cfg, pd.Timestamp("2023-01-01"), "lgbm_rank")
        cost_rows.extend(_summary_rows(f"slippage_{slippage:.3f}", equity, trades))
    pd.DataFrame(cost_rows).to_csv(OUTPUT_DIR / "cost_stress_summary.csv", index=False)

    full_panel = _full_sample_scored_panel(panel)
    full_equity, full_trades = no_pool._run_backtest(full_panel, data, config, pd.Timestamp("2020-01-02"), "lgbm_rank")
    full_summary = summarize(full_equity, full_trades)
    full_summary.insert(0, "strategy", "full_sample_insample_2020_2026")
    full_summary.to_csv(OUTPUT_DIR / "full_sample_insample_summary.csv", index=False)
    full_equity.to_csv(OUTPUT_DIR / "full_sample_insample_equity_curve.csv", index=False)
    full_trades.to_csv(OUTPUT_DIR / "full_sample_insample_trades.csv", index=False)

    print("Winner exclusion:")
    print(pd.DataFrame(robustness).to_string(index=False))
    print("\nCost stress:")
    print(pd.DataFrame(cost_rows).to_string(index=False))
    print("\nYearly OOS:")
    print(pd.DataFrame(yearly_rows).to_string(index=False))
    print("\nFull sample in-sample:")
    print(full_summary.to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


def _full_sample_scored_panel(panel: pd.DataFrame) -> pd.DataFrame:
    trainable = panel.dropna(subset=["reward_60", *no_pool.FEATURE_COLUMNS]).copy()
    trainable = trainable[trainable["date"] >= pd.Timestamp("2020-07-01")]
    candidates = panel[no_pool._no_eva_pool_candidate_mask(panel)].copy()
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
        random_state=20260606,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        no_pool._clean_features(trainable[no_pool.FEATURE_COLUMNS]),
        no_pool._clip_target(trainable["reward_60"]),
    )
    candidates["lgbm_reward_score"] = model.predict(no_pool._clean_features(candidates[no_pool.FEATURE_COLUMNS]))
    predictions = candidates[["date", "symbol", "lgbm_reward_score"]]
    predictions.to_csv(OUTPUT_DIR / "full_sample_insample_predictions.csv", index=False)
    return panel.merge(predictions, on=["date", "symbol"], how="left")


def _summary_rows(name: str, equity: pd.DataFrame, trades: pd.DataFrame) -> list[dict]:
    summary = summarize(equity, trades)
    rows = []
    for row in summary.itertuples(index=False):
        rows.append({"scenario": name, "metric": row.metric, "value": row.value})
    return rows


def _top_winner_symbols(trades: pd.DataFrame, n: int) -> list[str]:
    if trades.empty:
        return []
    ranked = trades.sort_values("pnl", ascending=False)
    return ranked["symbol"].head(n).astype(str).tolist()


if __name__ == "__main__":
    main()
