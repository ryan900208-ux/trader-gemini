from __future__ import annotations

import pandas as pd

from .fundamentals import (
    fundamental_score,
    passes_fundamental_filters,
)


def build_feature_panel(
    data: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    market_regime: pd.Series,
) -> pd.DataFrame:
    frames = []
    for symbol, frame in data.items():
        if frame.empty:
            continue
        symbol_frame = frame.copy()
        symbol_frame["symbol"] = symbol
        symbol_frame["date"] = symbol_frame.index
        frames.append(symbol_frame)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    benchmark_features = pd.DataFrame(
        {
            "date": benchmark.index,
            "benchmark_ret20": benchmark["Close"].pct_change(20).to_numpy(),
            "benchmark_ret60": benchmark["Close"].pct_change(60).to_numpy(),
            "market_regime": market_regime.reindex(benchmark.index).to_numpy(),
        }
    )
    panel = panel.merge(benchmark_features, on="date", how="left")
    panel["rs20_rank_pct"] = panel.groupby("date")["ret20"].rank(ascending=False, pct=True)
    panel["rs60_rank_pct"] = panel.groupby("date")["ret60"].rank(ascending=False, pct=True)
    return panel.sort_values(["date", "symbol"], ignore_index=True)


def add_strategy_scores(panel: pd.DataFrame, fundamentals: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = panel.copy()
    use_fundamentals = config.get("use_fundamentals", True)
    if use_fundamentals:
        df = _merge_fundamentals_asof(df, fundamentals, config)
    else:
        df["fundamental_score"] = 0.0
        df["fundamental_pass"] = True

    tech = _technical_score(df)
    weights = config["score_weights"] if use_fundamentals else {"technical": 1.0, "fundamental": 0.0}
    df["technical_score"] = tech
    df["score"] = df["technical_score"]
    df["final_score"] = weights["technical"] * df["technical_score"] + weights["fundamental"] * df["fundamental_score"]
    df["fundamental_rank"] = df.groupby("date")["fundamental_score"].rank(ascending=False, method="first")
    return df


def _merge_fundamentals_asof(panel: pd.DataFrame, fundamentals: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = panel.sort_values(["symbol", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.as_unit("ns")
    if fundamentals.empty:
        df["fundamental_score"] = 50.0
        df["fundamental_pass"] = False
        return df

    fund = fundamentals.copy().sort_values(["symbol", "as_of_date"])
    fund["as_of_date"] = pd.to_datetime(fund["as_of_date"]).dt.tz_localize(None).dt.as_unit("ns")
    fund["fundamental_score"] = fund.apply(fundamental_score, axis=1)
    fund["fundamental_pass"] = fund.apply(
        lambda row: passes_fundamental_filters(row, config["fundamental_filters"]),
        axis=1,
    )

    merged = []
    fund_symbols = set(fund["symbol"].dropna().astype(str))
    for symbol, group in df.groupby("symbol", sort=False):
        if symbol not in fund_symbols:
            out = group.copy()
            out["fundamental_score"] = 50.0
            out["fundamental_pass"] = False
            merged.append(out)
            continue
        fund_group = fund[fund["symbol"] == symbol]
        out = pd.merge_asof(
            group.sort_values("date"),
            fund_group.sort_values("as_of_date"),
            left_on="date",
            right_on="as_of_date",
            by="symbol",
            direction="backward",
            suffixes=("", "_fundamental"),
        )
        out["fundamental_score"] = out["fundamental_score"].fillna(50.0)
        out["fundamental_pass"] = out["fundamental_pass"].fillna(False).astype(bool)
        merged.append(out)

    return pd.concat(merged, ignore_index=True).sort_values(["date", "symbol"], ignore_index=True)


def entry_candidates(panel: pd.DataFrame, date: pd.Timestamp, config: dict) -> pd.DataFrame:
    entry = config["entry"]
    filters = config["fundamental_filters"]
    use_fundamentals = config.get("use_fundamentals", True)
    use_fundamental_filter = config.get("use_fundamental_filter", True)
    universe_filter = config.get("fundamental_universe_filter", {})
    allowed_market_regimes = set(entry.get("allowed_market_regimes", ["bull", "neutral"]))
    day = panel[panel["date"] == date].copy()
    if day.empty:
        return day
    rs20_rank = day["rs20_rank_pct"]
    rs60_rank = day["rs60_rank_pct"]
    rank_count = len(day)
    if (
        use_fundamentals
        and universe_filter.get("enabled", False)
        and entry.get("rs_rank_scope") == "fundamental_universe"
    ):
        universe_mask = _fundamental_universe_mask(day, universe_filter)
        pool_count = int(universe_mask.sum())
        if pool_count > 0:
            rs20_rank = day["ret20"].where(universe_mask).rank(ascending=False, pct=True)
            rs60_rank = day["ret60"].where(universe_mask).rank(ascending=False, pct=True)
            rank_count = pool_count
    rs20_cutoff = max(entry["rs20_top_pct"], 1 / rank_count)
    rs60_cutoff = max(entry["rs60_top_pct"], 1 / rank_count)

    mask = (
        (day["market_regime"].isin(allowed_market_regimes))
        & (day["score"] >= entry["min_score"])
        & (rs20_rank <= rs20_cutoff)
        & (rs60_rank <= rs60_cutoff)
        & (day["Close"] > day["ma20"])
        & (day["ma20"] > day["ma60"])
        & (day["ma20_slope"] > 0)
        & (day["rsi14"].between(entry["rsi_min"], entry["rsi_max"]))
        & (day["ret20"] > day["benchmark_ret20"])
        & (day["ret60"] > day["benchmark_ret60"])
        & (day["ret5"] <= entry["ret5_max"])
        & (day["ret20"] <= entry["ret20_max"])
        & (day["volume_ratio"].between(entry["volume_ratio_min"], entry["volume_ratio_max"]))
        & (day["ma20_deviation"].abs() <= entry["ma20_deviation_max"])
    )
    if use_fundamentals and use_fundamental_filter:
        mask = mask & (day["fundamental_score"] >= filters["min_fundamental_score"]) & day["fundamental_pass"]
    if use_fundamentals and universe_filter.get("enabled", False):
        mask = mask & _fundamental_universe_mask(day, universe_filter)
    return day[mask].sort_values(["final_score", "score"], ascending=False)


def _fundamental_universe_mask(day: pd.DataFrame, universe_filter: dict) -> pd.Series:
    mask = day["fundamental_score"] >= universe_filter.get("min_score", 0)
    top_n = universe_filter.get("top_n")
    if top_n is not None:
        mask = mask & (day["fundamental_rank"] <= top_n)
    return mask


def add_entry_diagnostics(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    entry = config["entry"]
    filters = config["fundamental_filters"]
    use_fundamentals = config.get("use_fundamentals", True)
    use_fundamental_filter = config.get("use_fundamental_filter", True)
    universe_filter = config.get("fundamental_universe_filter", {})
    allowed_market_regimes = set(entry.get("allowed_market_regimes", ["bull", "neutral"]))
    df = panel.copy()
    day_counts = df.groupby("date")["symbol"].transform("count")
    rs20_cutoff = pd.Series(entry["rs20_top_pct"], index=df.index).mask(
        entry["rs20_top_pct"] < 1 / day_counts, 1 / day_counts
    )
    rs60_cutoff = pd.Series(entry["rs60_top_pct"], index=df.index).mask(
        entry["rs60_top_pct"] < 1 / day_counts, 1 / day_counts
    )
    checks = {
        "pass_market": df["market_regime"].isin(allowed_market_regimes),
        "pass_score": df["score"] >= entry["min_score"],
        "pass_rs20": df["rs20_rank_pct"] <= rs20_cutoff,
        "pass_rs60": df["rs60_rank_pct"] <= rs60_cutoff,
        "pass_close_ma20": df["Close"] > df["ma20"],
        "pass_ma20_ma60": df["ma20"] > df["ma60"],
        "pass_ma20_slope": df["ma20_slope"] > 0,
        "pass_rsi": df["rsi14"].between(entry["rsi_min"], entry["rsi_max"]),
        "pass_ret20_benchmark": df["ret20"] > df["benchmark_ret20"],
        "pass_ret60_benchmark": df["ret60"] > df["benchmark_ret60"],
        "pass_ret5_cap": df["ret5"] <= entry["ret5_max"],
        "pass_ret20_cap": df["ret20"] <= entry["ret20_max"],
        "pass_volume_ratio": df["volume_ratio"].between(entry["volume_ratio_min"], entry["volume_ratio_max"]),
        "pass_ma20_deviation": df["ma20_deviation"].abs() <= entry["ma20_deviation_max"],
    }
    if use_fundamentals:
        checks["pass_fundamental_score"] = df["fundamental_score"] >= filters["min_fundamental_score"]
        checks["pass_fundamental_filter"] = df["fundamental_pass"]
        if universe_filter.get("enabled", False):
            checks["pass_fundamental_universe"] = df["fundamental_score"] >= universe_filter.get("min_score", 0)
            top_n = universe_filter.get("top_n")
            if top_n is not None:
                checks["pass_fundamental_universe"] = (
                    checks["pass_fundamental_universe"] & (df["fundamental_rank"] <= top_n)
                )
    for name, values in checks.items():
        df[name] = values.fillna(False)
    pass_columns = [
        name
        for name in checks
        if (
            (use_fundamental_filter or name not in {"pass_fundamental_score", "pass_fundamental_filter"})
            and (universe_filter.get("enabled", False) or name != "pass_fundamental_universe")
        )
    ]
    df["entry_pass_count"] = df[pass_columns].sum(axis=1)
    df["entry_all_pass"] = df[pass_columns].all(axis=1)
    return df


def _technical_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    score += (1 - df["rs20_rank_pct"]).clip(0, 1) * 30
    score += (1 - df["rs60_rank_pct"]).clip(0, 1) * 25
    score += ((df["Close"] / df["ma20"]) - 1).clip(0, 0.12) / 0.12 * 10
    score += (df["ma20"] > df["ma60"]).astype(float) * 10
    score += (df["ma20_slope"] > 0).astype(float) * 10
    score += (1 - ((df["rsi14"] - 61).abs() / 9)).clip(0, 1) * 10
    score += (1 - ((df["volume_ratio"] - 1.6).abs() / 0.6)).clip(0, 1) * 5
    return score.clip(0, 100)
