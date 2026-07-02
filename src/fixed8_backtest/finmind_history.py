from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


API_URL = "https://api.finmindtrade.com/api/v4/data"

DATASETS = {
    "income": "TaiwanStockFinancialStatements",
    "balance": "TaiwanStockBalanceSheet",
    "cashflow": "TaiwanStockCashFlowsStatement",
}


def fetch_history_for_symbols(
    symbols: list[str],
    start_date: str,
    end_date: str | None,
    cache_dir: str | Path,
    token: str | None = None,
    sleep_seconds: float = 0.25,
) -> pd.DataFrame:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, symbol in enumerate(symbols, start=1):
        stock_id = symbol.split(".")[0]
        print(f"[{index}/{len(symbols)}] fetching {stock_id}", flush=True)
        income = fetch_dataset_cached(stock_id, DATASETS["income"], start_date, end_date, cache, token)
        time.sleep(sleep_seconds)
        balance = fetch_dataset_cached(stock_id, DATASETS["balance"], start_date, end_date, cache, token)
        time.sleep(sleep_seconds)
        if income.empty and balance.empty:
            continue
        rows.append(normalize_fundamental_history(symbol, income, balance))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["symbol", "as_of_date"])


def fetch_dataset_cached(
    stock_id: str,
    dataset: str,
    start_date: str,
    end_date: str | None,
    cache_dir: Path,
    token: str | None,
) -> pd.DataFrame:
    path = cache_dir / f"{dataset}_{stock_id}_{start_date}_{end_date or 'latest'}.csv"
    if path.exists():
        return pd.read_csv(path)
    frame = fetch_dataset(stock_id, dataset, start_date, end_date, token)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def fetch_dataset(
    stock_id: str,
    dataset: str,
    start_date: str,
    end_date: str | None,
    token: str | None,
) -> pd.DataFrame:
    params = {"dataset": dataset, "data_id": stock_id, "start_date": start_date}
    if end_date:
        params["end_date"] = end_date
    headers = {"User-Agent": "fixed8-control-backtest/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(f"{API_URL}?{urlencode(params)}", headers=headers)
    with urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != 200:
        raise RuntimeError(f"FinMind error for {dataset} {stock_id}: {payload.get('msg')}")
    return pd.DataFrame(payload.get("data", []))


def normalize_fundamental_history(symbol: str, income: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    income_wide = _pivot(income)
    balance_wide = _pivot(balance)
    df = income_wide.merge(balance_wide, on=["date", "stock_id"], how="outer", suffixes=("", "_balance"))
    df["symbol"] = symbol
    df["period_date"] = pd.to_datetime(df["date"])
    df["as_of_date"] = df["period_date"].map(_financial_statement_available_date)

    out = pd.DataFrame(
        {
            "symbol": df["symbol"],
            "period_date": df["period_date"],
            "as_of_date": df["as_of_date"],
            "revenue": _col(df, "Revenue"),
            "gross_profit": _col(df, "GrossProfit"),
            "operating_income": _col(df, "OperatingIncome"),
            "pretax_income": _col(df, "PreTaxIncome"),
            "tax_expense": _col(df, "TAX"),
            "net_income_parent": _col(df, "EquityAttributableToOwnersOfParent"),
            "eps": _col(df, "EPS"),
            "current_assets": _col(df, "CurrentAssets"),
            "current_liabilities": _col(df, "CurrentLiabilities"),
            "assets": _col(df, "TotalAssets"),
            "liabilities": _col(df, "Liabilities"),
            "equity": _col(df, "Equity"),
            "cash": _col(df, "CashAndCashEquivalents"),
            "short_borrowings": _col(df, "ShorttermBorrowings"),
            "long_borrowings": _col(df, "LongtermBorrowings"),
            "bonds_payable": _col(df, "BondsPayable"),
        }
    )
    out = add_history_metrics(out)
    return out


def add_history_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("period_date").copy()
    out["tax_rate"] = (out["tax_expense"] / out["pretax_income"]).replace([np.inf, -np.inf], np.nan).clip(0, 0.4)
    out["nopat"] = out["operating_income"] * (1 - out["tax_rate"].fillna(0.2))
    out["invested_capital_proxy"] = (out["assets"] - out["current_liabilities"].fillna(0)).where(out["assets"] > 0)
    out["roic_proxy"] = out["nopat"] / out["invested_capital_proxy"].replace(0, np.nan)
    out["roe"] = out["net_income_parent"] / out["equity"].replace(0, np.nan)
    out["roa"] = out["net_income_parent"] / out["assets"].replace(0, np.nan)
    out["debt_to_equity"] = out["liabilities"] / out["equity"].replace(0, np.nan)
    positive_revenue = out["revenue"].where(out["revenue"] > 0)
    out["gross_margin"] = out["gross_profit"] / positive_revenue
    out["operating_margin"] = out["operating_income"] / positive_revenue
    out["net_margin"] = out["net_income_parent"] / positive_revenue
    for column in ("gross_margin", "operating_margin", "net_margin"):
        out[column] = out[column].replace([np.inf, -np.inf], np.nan).where(out[column].between(-1, 3))
    out["revenue_growth"] = out["revenue"].pct_change(4)
    out["pe"] = np.nan
    out["pb"] = np.nan
    out["eva_like_score"] = _score_eva_like(out)
    return out


def _pivot(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "stock_id"])
    frame = frame.copy()
    frame["stock_id"] = frame["stock_id"].astype(str)
    return (
        frame.pivot_table(index=["date", "stock_id"], columns="type", values="value", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _financial_statement_available_date(period_date: pd.Timestamp) -> pd.Timestamp:
    year = period_date.year
    quarter = period_date.quarter
    if quarter == 1:
        return pd.Timestamp(year=year, month=5, day=15)
    if quarter == 2:
        return pd.Timestamp(year=year, month=8, day=14)
    if quarter == 3:
        return pd.Timestamp(year=year, month=11, day=14)
    return pd.Timestamp(year=year + 1, month=3, day=31)


def _score_eva_like(df: pd.DataFrame) -> pd.Series:
    score = pd.concat(
        [
            _higher(df["roic_proxy"], 0.0, 0.04),
            _higher(df["roe"], 0.0, 0.08),
            _higher(df["operating_margin"], 0.02, 0.2),
            _higher(df["gross_margin"], 0.1, 0.45),
            _lower(df["debt_to_equity"], 3.0, 0.3),
            _higher(df["roa"], 0.0, 0.03),
            _higher(df["revenue_growth"], -0.1, 0.3),
        ],
        axis=1,
    ).mean(axis=1)
    return (score * 100).fillna(50).clip(0, 100)


def _higher(series: pd.Series, low: float, high: float) -> pd.Series:
    return ((series - low) / (high - low)).clip(0, 1)


def _lower(series: pd.Series, high: float, low: float) -> pd.Series:
    return ((high - series) / (high - low)).clip(0, 1)
