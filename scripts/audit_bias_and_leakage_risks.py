from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fixed8_backtest.fundamentals import load_fundamentals
from fixed8_backtest.reports import summarize
from fixed8_backtest.strategy import add_strategy_scores

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
OUTPUT_DIR = ROOT / "outputs" / "bias_leakage_audit"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    base_panel = pd.read_csv(PANEL_PATH, parse_dates=["date"], low_memory=False)
    base_panel = base_panel.sort_values(["symbol", "date"], ignore_index=True)
    data = _load_price_data(config)

    delay_summary = []
    delay_selection = []
    for delay_days in [0, 30, 60, 90, 120, 180]:
        panel = _panel_with_delayed_fundamentals(base_panel, config, delay_days)
        panel["fundamental_rank_score"] = -panel["fundamental_rank"]
        for score_col in ["fundamental_rank_score", "final_score"]:
            equity, trades = _run_backtest(panel, data, config, pd.Timestamp("2025-01-01"), score_col, 0.15)
            summary = summarize(equity, trades)
            summary.insert(0, "score_col", score_col)
            summary.insert(0, "delay_days", delay_days)
            delay_summary.append(summary)
            delay_selection.extend(_selection_power(panel, delay_days, score_col))
            equity.to_csv(OUTPUT_DIR / f"delay{delay_days}_{score_col}_equity_curve.csv", index=False)
            trades.to_csv(OUTPUT_DIR / f"delay{delay_days}_{score_col}_trades.csv", index=False)

    delay_summary_df = pd.concat(delay_summary, ignore_index=True)
    delay_selection_df = pd.DataFrame(delay_selection)
    delay_summary_df.to_csv(OUTPUT_DIR / "fundamental_delay_backtest_summary.csv", index=False)
    delay_selection_df.to_csv(OUTPUT_DIR / "fundamental_delay_selection_power.csv", index=False)

    concentration = _trade_concentration()
    concentration.to_csv(OUTPUT_DIR / "trade_concentration.csv", index=False)
    listing_age = _listing_age_at_entries(data)
    listing_age.to_csv(OUTPUT_DIR / "listing_age_at_entries.csv", index=False)
    universe_audit = _universe_audit(base_panel)
    universe_audit.to_csv(OUTPUT_DIR / "universe_audit.csv", index=False)

    print("Delay stress summary:")
    print(delay_summary_df.to_string(index=False))
    print("\nDelay selection power:")
    print(delay_selection_df.to_string(index=False))
    print("\nTrade concentration:")
    print(concentration.to_string(index=False))
    print("\nListing age at entries:")
    print(listing_age.to_string(index=False))
    print("\nUniverse audit:")
    print(universe_audit.to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


def _panel_with_delayed_fundamentals(base_panel: pd.DataFrame, config: dict, delay_days: int) -> pd.DataFrame:
    fund = load_fundamentals(ROOT / config["fundamentals_csv"])
    fund["as_of_date"] = fund["as_of_date"] + pd.Timedelta(days=delay_days)
    feature_drop = [
        "roe",
        "roa",
        "roic_proxy",
        "revenue_growth",
        "eps",
        "debt_to_equity",
        "pe",
        "pb",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "eva_like_score",
        "fundamental_score",
        "fundamental_pass",
        "technical_score",
        "score",
        "final_score",
        "fundamental_rank",
    ]
    panel = base_panel.drop(columns=[col for col in feature_drop if col in base_panel], errors="ignore")
    panel = add_strategy_scores(panel, fund, config)
    panel = _add_forward_targets(_add_enhanced_features(panel))
    return panel


def _selection_power(panel: pd.DataFrame, delay_days: int, score_col: str) -> list[dict]:
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
                "delay_days": delay_days,
                "score_col": score_col,
                "top_n": top_n,
                "rows": len(selected),
                "avg_forward60": returns.mean(),
                "median_forward60": returns.median(),
                "hit_rate_forward60_positive": (returns > 0).mean(),
            }
        )
    return rows


def _trade_concentration() -> pd.DataFrame:
    rows = []
    for output_name, label in [
        ("two_stage_fundamental_ml_filter", "fund_final_ml"),
        ("two_stage_fundamental_ml_filter", "fund_ml_85_15"),
        ("two_stage_fundamental_ml_filter", "fundamental_rank_score"),
        ("holdout_ranker_ensemble", "stable_ensemble_stop15"),
    ]:
        path = ROOT / "outputs" / output_name / f"{label}_trades.csv"
        if not path.exists():
            continue
        trades = pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
        total_pnl = trades["pnl"].sum()
        by_symbol = trades.groupby("symbol", as_index=False)["pnl"].sum().sort_values("pnl", ascending=False)
        rows.append(
            {
                "strategy": label,
                "trades": len(trades),
                "symbols": trades["symbol"].nunique(),
                "realized_pnl": total_pnl,
                "top1_symbol": by_symbol.iloc[0]["symbol"] if not by_symbol.empty else None,
                "top1_pnl_share": by_symbol.iloc[0]["pnl"] / total_pnl if total_pnl else None,
                "top3_pnl_share": by_symbol.head(3)["pnl"].sum() / total_pnl if total_pnl else None,
                "losing_trade_rate": (trades["pnl"] < 0).mean() if len(trades) else None,
            }
        )
    return pd.DataFrame(rows)


def _listing_age_at_entries(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for output_name, label in [
        ("two_stage_fundamental_ml_filter", "fund_final_ml"),
        ("two_stage_fundamental_ml_filter", "fund_ml_85_15"),
        ("two_stage_fundamental_ml_filter", "fundamental_rank_score"),
    ]:
        path = ROOT / "outputs" / output_name / f"{label}_trades.csv"
        if not path.exists():
            continue
        trades = pd.read_csv(path, parse_dates=["entry_date"])
        ages = []
        for row in trades.itertuples(index=False):
            frame = data.get(row.symbol)
            if frame is None or frame.empty:
                continue
            ages.append((row.entry_date - frame.index.min()).days)
        series = pd.Series(ages, dtype=float)
        rows.append(
            {
                "strategy": label,
                "entries_with_age": len(series),
                "median_listing_age_days": series.median(),
                "min_listing_age_days": series.min(),
                "entries_under_180d": (series < 180).sum(),
                "entries_under_365d": (series < 365).sum(),
            }
        )
    return pd.DataFrame(rows)


def _universe_audit(base_panel: pd.DataFrame) -> pd.DataFrame:
    first_dates = base_panel.groupby("symbol")["date"].min()
    last_dates = base_panel.groupby("symbol")["date"].max()
    return pd.DataFrame(
        [
            {
                "universe_symbols": int(base_panel["symbol"].nunique()),
                "symbols_with_data_on_2020_01_02_or_before": int((first_dates <= pd.Timestamp("2020-01-02")).sum()),
                "symbols_starting_after_2020": int((first_dates > pd.Timestamp("2020-01-02")).sum()),
                "symbols_with_last_date_before_2026": int((last_dates < pd.Timestamp("2026-01-01")).sum()),
                "min_first_date": first_dates.min(),
                "max_first_date": first_dates.max(),
                "min_last_date": last_dates.min(),
                "max_last_date": last_dates.max(),
            }
        ]
    )


if __name__ == "__main__":
    main()
