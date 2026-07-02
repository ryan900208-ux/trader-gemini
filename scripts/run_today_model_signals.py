from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from fixed8_backtest.data import download_ohlcv, read_universe
from fixed8_backtest.fundamentals import load_fundamentals
from fixed8_backtest.indicators import add_indicators, market_regime
from fixed8_backtest.strategy import add_entry_diagnostics, add_strategy_scores, build_feature_panel

from holdout_ranker_ensemble import (
    FEATURE_COLUMNS,
    _ClassifierProbWrapper,
    _add_enhanced_features,
    _add_forward_targets,
    _add_rank_labels,
    _candidate_mask,
    _clean_features,
    _fit_classifier,
    _fit_ranker,
    _fit_regressor,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "fixed8_control_eva_top15_pool_best_sweep.json"
DATA_ROOT = Path(os.environ.get("FIXED8_DATA_DIR", ROOT / "outputs"))
OUTPUT_DIR = DATA_ROOT / "today_model_signals"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    raw_data = _refresh_and_load_prices(config)
    benchmark = raw_data.pop(config["benchmark_symbol"])
    data = {symbol: add_indicators(frame) for symbol, frame in raw_data.items() if not frame.empty}
    benchmark = add_indicators(benchmark)
    regime = market_regime(benchmark, config)
    fundamentals = load_fundamentals(ROOT / config["fundamentals_csv"])

    panel = build_feature_panel(data, benchmark, regime)
    panel = add_strategy_scores(panel, fundamentals, config)
    panel = add_entry_diagnostics(panel, config)
    panel = _add_rank_labels(_add_forward_targets(_add_enhanced_features(panel)))
    panel = _add_current_model_scores(panel)
    panel = _add_two_stage_scores(panel)
    panel = _add_symbol_names(panel, config)
    latest_date = panel["date"].max()
    latest = panel[panel["date"] == latest_date].copy()
    candidates = latest[_candidate_mask(latest)].copy()

    rankings = {}
    for name, score_col in [
        ("aggressive_fund_final_ml", "fund_final_ml"),
        ("defensive_fund_ml_85_15", "fund_ml_85_15"),
        ("fundamental_only", "fundamental_rank_score"),
        ("final_score_only", "final_score"),
        ("stable_ml_only", "stable_ensemble_score"),
    ]:
        ranking = _ranking(candidates, score_col)
        rankings[name] = ranking
        ranking.to_csv(OUTPUT_DIR / f"{name}_{latest_date.date()}.csv", index=False)

    latest.to_csv(OUTPUT_DIR / f"latest_features_{latest_date.date()}.csv", index=False)
    candidates.to_csv(OUTPUT_DIR / f"latest_candidates_{latest_date.date()}.csv", index=False)
    
    strict_candidates = latest[latest["entry_all_pass"] == True].copy().sort_values(["final_score", "score"], ascending=False)
    strict_candidates.to_csv(OUTPUT_DIR / f"strict_candidates_{latest_date.date()}.csv", index=False)
    
    summary = _summary(panel, latest, candidates, rankings, config)
    summary.to_csv(OUTPUT_DIR / f"summary_{latest_date.date()}.csv", index=False)
    _write_named_outputs(latest_date.date())

    print(f"latest_date={latest_date.date()}")
    market_state = _latest_market_regime(panel, latest)
    print(f"market_regime={market_state}")
    print(f"universe_rows={len(latest)}")
    print(f"candidate_rows={len(candidates)}")
    print("\nSummary:")
    print(summary.to_string(index=False))
    for name, ranking in rankings.items():
        print(f"\n{name}:")
        if ranking.empty:
            print("No candidates")
        else:
            print(ranking.head(10).to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


def _refresh_and_load_prices(config: dict) -> dict[str, pd.DataFrame]:
    symbols = read_universe(ROOT / config["universe_csv"], config["benchmark_symbol"])
    all_symbols = sorted(set(symbols + [config["benchmark_symbol"]]))
    cache_dir = Path(os.environ.get("FIXED8_PRICE_CACHE_DIR", ROOT / config["price_cache_dir"]))
    cache_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("FIXED8_SKIP_PRICE_REFRESH") == "1":
        return _load_cached_prices(all_symbols, config["start"], cache_dir)
    latest_cached = _latest_cached_date(cache_dir)
    start = max(pd.Timestamp(config["start"]), latest_cached - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    fresh = download_ohlcv(
        all_symbols,
        start,
        None,
        batch_size=config.get("yfinance_batch_size", 80),
        cache_dir=None,
    )
    for symbol, frame in fresh.items():
        if frame.empty:
            continue
        path = cache_dir / f"{symbol.replace('.', '_')}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        old = pd.read_csv(path, parse_dates=["Date"], index_col="Date") if path.exists() else pd.DataFrame()
        merged = pd.concat([old, frame])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.index.name = "Date"
        merged.to_csv(path)
    cached = _load_cached_prices(all_symbols, config["start"], cache_dir)
    if config["benchmark_symbol"] not in cached:
        cached.update(
            download_ohlcv(
                [config["benchmark_symbol"]],
                config["start"],
                None,
                batch_size=1,
                cache_dir=cache_dir,
            )
        )
    return cached


def _latest_cached_date(cache_dir: Path) -> pd.Timestamp:
    dates = []
    for path in cache_dir.glob("*.csv"):
        try:
            frame = pd.read_csv(path, usecols=["Date"], parse_dates=["Date"])
        except Exception:
            continue
        if not frame.empty:
            dates.append(frame["Date"].max())
    return max(dates) if dates else pd.Timestamp("2020-01-01")


def _load_cached_prices(symbols: list[str], start: str, cache_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    start_ts = pd.Timestamp(start)
    for symbol in symbols:
        path = cache_dir / f"{symbol.replace('.', '_')}.csv"
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        except Exception:
            continue
        if frame.empty:
            continue
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        out[symbol] = frame[frame.index >= start_ts]
    return out


def _add_current_model_scores(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[
        (panel["date"] >= pd.Timestamp("2020-07-01"))
        & (panel["date"] <= pd.Timestamp("2024-10-03"))
    ].dropna(subset=[*FEATURE_COLUMNS, "forward_60d_return", "reward_60", "top_quantile_60"])
    score_mask = (panel["date"] >= pd.Timestamp("2025-01-01")) & _candidate_mask(panel)
    scored = panel[score_mask].copy()
    if train.empty or scored.empty:
        for col in [
            "reg_forward60",
            "reg_reward60",
            "cls_topq60",
            "rank_forward60",
            "ensemble_score",
            "ranker_score",
            "stable_ensemble_score",
        ]:
            panel[col] = np.nan
        return panel

    models = {
        "reg_forward60": _fit_regressor(train, "forward_60d_return", seed=101),
        "reg_reward60": _fit_regressor(train, "reward_60", seed=102),
        "cls_topq60": _fit_classifier(train, "top_quantile_60", seed=103),
        "rank_forward60": _fit_ranker(train, "fwd60_rank_label", seed=104),
    }
    for name, model in models.items():
        scored[name] = model.predict(_clean_features(scored[FEATURE_COLUMNS]))
        scored[f"{name}_pct"] = scored.groupby("date")[name].rank(pct=True)
    scored["ensemble_score"] = (
        0.30 * scored["reg_forward60_pct"]
        + 0.25 * scored["reg_reward60_pct"]
        + 0.25 * scored["cls_topq60_pct"]
        + 0.20 * scored["rank_forward60_pct"]
    )
    scored["ranker_score"] = scored["rank_forward60"]
    scored["stable_ensemble_score"] = scored.groupby("symbol")["ensemble_score"].transform(
        lambda series: series.rolling(5, min_periods=1).mean()
    )
    score_cols = [
        "reg_forward60",
        "reg_reward60",
        "cls_topq60",
        "rank_forward60",
        "ensemble_score",
        "ranker_score",
        "stable_ensemble_score",
    ]
    return panel.merge(scored[["date", "symbol", *score_cols]], on=["date", "symbol"], how="left")


def _add_two_stage_scores(panel: pd.DataFrame) -> pd.DataFrame:
    holdout_mask = (panel["date"] >= "2025-01-01") & _candidate_mask(panel)
    panel["fundamental_rank_score"] = np.nan
    panel.loc[holdout_mask, "fundamental_rank_score"] = -panel.loc[holdout_mask, "fundamental_rank"]
    panel.loc[holdout_mask, "fundamental_score_rank"] = panel.loc[holdout_mask].groupby("date")[
        "fundamental_rank_score"
    ].rank(pct=True)
    panel.loc[holdout_mask, "final_score_rank"] = panel.loc[holdout_mask].groupby("date")["final_score"].rank(pct=True)
    panel.loc[holdout_mask, "stable_ml_rank"] = panel.loc[holdout_mask].groupby("date")[
        "stable_ensemble_score"
    ].rank(pct=True)
    panel["fund_ml_85_15"] = 0.85 * panel["fundamental_score_rank"] + 0.15 * panel["stable_ml_rank"]
    panel["fund_final_ml"] = (
        0.50 * panel["fundamental_score_rank"] + 0.30 * panel["final_score_rank"] + 0.20 * panel["stable_ml_rank"]
    )
    return panel


def _ranking(candidates: pd.DataFrame, score_col: str) -> pd.DataFrame:
    if candidates.empty or score_col not in candidates:
        return pd.DataFrame()
    columns = [
        "date",
        "symbol",
        "name",
        "Close",
        "market_regime",
        score_col,
        "fundamental_rank",
        "fundamental_score",
        "final_score",
        "technical_score",
        "stable_ensemble_score",
        "rs20_rank_pct",
        "rs60_rank_pct",
        "ret5",
        "ret20",
        "ret60",
        "rsi14",
        "volume_ratio",
        "ma20_deviation",
    ]
    existing = list(dict.fromkeys(col for col in columns if col in candidates))
    return candidates.dropna(subset=[score_col]).sort_values(score_col, ascending=False)[existing].head(30)


def _add_symbol_names(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    universe = pd.read_csv(ROOT / config["universe_csv"], dtype={"symbol": str, "name": str})
    return panel.merge(universe[["symbol", "name"]], on="symbol", how="left")


def _summary(
    panel: pd.DataFrame,
    latest: pd.DataFrame,
    candidates: pd.DataFrame,
    rankings: dict[str, pd.DataFrame],
    config: dict,
) -> pd.DataFrame:
    rows = [
        {"item": "latest_date", "value": str(panel["date"].max().date())},
        {
            "item": "market_regime",
            "value": _latest_market_regime(panel, latest),
        },
        {"item": "universe_rows", "value": len(latest)},
        {"item": "candidate_rows", "value": len(candidates)},
        {"item": "max_positions", "value": config["max_positions"]},
        {"item": "position_weight", "value": config["position_weight"]},
    ]
    for name, ranking in rankings.items():
        top_symbols = (
            ",".join(
                f"{row.symbol} {row.name}" if pd.notna(row.name) else row.symbol
                for row in ranking.head(config["max_positions"]).itertuples(index=False)
            )
            if not ranking.empty
            else ""
        )
        rows.append({"item": f"{name}_top{config['max_positions']}", "value": top_symbols})
    return pd.DataFrame(rows)


def _latest_market_regime(panel: pd.DataFrame, latest: pd.DataFrame) -> str:
    if not latest.empty and "market_regime" in latest:
        latest_values = latest["market_regime"].dropna()
        if not latest_values.empty:
            return str(latest_values.iloc[0])

    if "market_regime" not in panel:
        return "NA"

    values = panel.sort_values("date")["market_regime"].dropna()
    return str(values.iloc[-1]) if not values.empty else "NA"


def _write_named_outputs(latest_date: object) -> None:
    suffix = f"_{latest_date}.csv"
    for path in OUTPUT_DIR.glob(f"*{suffix}"):
        if path.name.endswith("_named.csv"):
            continue
        frame = pd.read_csv(path)
        if path.name.startswith("summary_") and "value" in frame.columns:
            frame["value"] = frame["value"].map(_normalize_summary_symbols)
        frame.to_csv(OUTPUT_DIR / f"{path.stem}_named.csv", index=False, encoding="utf-8-sig")


def _normalize_summary_symbols(value: object) -> object:
    if not isinstance(value, str) or ".TW" not in value:
        return value
    parts = []
    for token in value.split(","):
        token = token.strip()
        pieces = token.split()
        if len(pieces) >= 2:
            parts.append(f"{pieces[0]} {pieces[1]}")
        else:
            parts.append(token)
    return ",".join(parts)


if __name__ == "__main__":
    main()
