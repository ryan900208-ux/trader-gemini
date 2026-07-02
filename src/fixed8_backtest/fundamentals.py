from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FUNDAMENTAL_COLUMNS = [
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
]


def load_fundamentals(path: str | Path | None) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["symbol", "as_of_date", *FUNDAMENTAL_COLUMNS])
    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].astype(str)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.tz_localize(None).dt.as_unit("ns")
    for column in FUNDAMENTAL_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values(["symbol", "as_of_date"])


def fetch_yfinance_snapshot(symbols: list[str]) -> pd.DataFrame:
    import yfinance as yf

    rows = []
    today = pd.Timestamp.today().normalize()
    for symbol in symbols:
        info = yf.Ticker(symbol).info
        rows.append(
            {
                "symbol": symbol,
                "as_of_date": today,
                "roe": info.get("returnOnEquity"),
                "revenue_growth": info.get("revenueGrowth"),
                "eps": info.get("trailingEps"),
                "debt_to_equity": _scale_percent(info.get("debtToEquity")),
                "pe": info.get("trailingPE"),
                "pb": info.get("priceToBook"),
                "gross_margin": info.get("grossMargins"),
                "operating_margin": info.get("operatingMargins"),
            }
        )
    return pd.DataFrame(rows)


def latest_asof(fundamentals: pd.DataFrame, symbol: str, date: pd.Timestamp) -> pd.Series | None:
    rows = fundamentals[(fundamentals["symbol"] == symbol) & (fundamentals["as_of_date"] <= date)]
    if rows.empty:
        return None
    return rows.iloc[-1]


def fundamental_score(row: pd.Series | None) -> float:
    if row is None:
        return 50.0
    if "eva_like_score" in row and not pd.isna(row.get("eva_like_score")):
        return float(row.get("eva_like_score"))

    points = {
        "roe": _higher_better(row.get("roe"), 0.05, 0.25),
        "revenue_growth": _higher_better(row.get("revenue_growth"), -0.1, 0.3),
        "eps": _higher_better(row.get("eps"), 0.0, 10.0),
        "debt_to_equity": _lower_better(row.get("debt_to_equity"), 2.0, 0.2),
        "pe": _middle_better(row.get("pe"), 5.0, 18.0, 45.0),
        "pb": _lower_better(row.get("pb"), 8.0, 1.0),
        "gross_margin": _higher_better(row.get("gross_margin"), 0.1, 0.5),
        "operating_margin": _higher_better(row.get("operating_margin"), 0.02, 0.25),
    }
    available = [value for value in points.values() if not np.isnan(value)]
    if not available:
        return 50.0
    return float(np.mean(available) * 100)


def passes_fundamental_filters(row: pd.Series | None, filters: dict) -> bool:
    if row is None:
        return False
    if not pd.isna(row.get("eva_like_score", np.nan)):
        checks = [
            row.get("eva_like_score") >= filters.get("min_eva_like_score", filters["min_fundamental_score"]),
            row.get("roe") >= filters["min_roe"],
            row.get("eps") > filters["min_eps"],
            row.get("debt_to_equity") <= filters["max_debt_to_equity"],
        ]
        return all(False if pd.isna(value) else bool(value) for value in checks)

    checks = [
        row.get("roe") >= filters["min_roe"],
        row.get("revenue_growth") >= filters["min_revenue_growth"],
        row.get("eps") > filters["min_eps"],
        row.get("debt_to_equity") <= filters["max_debt_to_equity"],
        filters["min_pe"] < row.get("pe") <= filters["max_pe"],
        filters["min_pb"] < row.get("pb") <= filters["max_pb"],
    ]
    return all(False if pd.isna(value) else bool(value) for value in checks)


def _scale_percent(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return value / 100 if value > 10 else value


def _higher_better(value: float | None, low: float, high: float) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(np.clip((value - low) / (high - low), 0, 1))


def _lower_better(value: float | None, high: float, low: float) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(np.clip((high - value) / (high - low), 0, 1))


def _middle_better(value: float | None, low: float, target: float, high: float) -> float:
    if value is None or pd.isna(value) or value <= low or value >= high:
        return np.nan if value is None or pd.isna(value) else 0.0
    if value <= target:
        return float(np.clip((value - low) / (target - low), 0, 1))
    return float(np.clip((high - value) / (high - target), 0, 1))
