from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


BASE_URL = "https://openapi.twse.com.tw/v1/opendata"

INCOME_ENDPOINTS = [
    "t187ap06_L_ci",
    "t187ap06_L_basi",
    "t187ap06_L_bd",
    "t187ap06_L_fh",
    "t187ap06_L_ins",
    "t187ap06_L_mim",
]

BALANCE_ENDPOINTS = [
    "t187ap07_L_ci",
    "t187ap07_L_basi",
    "t187ap07_L_bd",
    "t187ap07_L_fh",
    "t187ap07_L_ins",
    "t187ap07_L_mim",
]


def fetch_twse_fundamentals() -> pd.DataFrame:
    income = _fetch_group(INCOME_ENDPOINTS, normalize_income)
    balance = _fetch_group(BALANCE_ENDPOINTS, normalize_balance)
    if income.empty and balance.empty:
        return pd.DataFrame()

    keys = ["symbol", "code", "year", "quarter", "as_of_date"]
    merged = income.merge(balance, on=keys, how="outer", suffixes=("", "_balance"))
    if "name_balance" in merged:
        merged["name"] = merged["name"].fillna(merged["name_balance"])
        merged = merged.drop(columns=["name_balance"])

    merged = add_eva_like_metrics(merged)
    return merged.sort_values(["symbol", "as_of_date"]).reset_index(drop=True)


def normalize_income(row: dict) -> dict:
    revenue = _num(_pick(row, ["營業收入", "收益", "利息淨收益"]))
    gross_profit = _num(_pick(row, ["營業毛利（毛損）淨額", "營業毛利（毛損）"]))
    operating_income = _num(_pick(row, ["營業利益（損失）", "繼續營業單位稅前淨利（淨損）"]))
    pretax_income = _num(_pick(row, ["稅前淨利（淨損）", "繼續營業單位稅前淨利（淨損）"]))
    tax_expense = _num(_pick(row, ["所得稅費用（利益）"]))
    net_income = _num(_pick(row, ["淨利（淨損）歸屬於母公司業主", "淨利（損）歸屬於母公司業主", "本期淨利（淨損）"]))
    return {
        **_base_fields(row),
        "name": _pick(row, ["公司名稱"]),
        "revenue": revenue,
        "gross_profit": gross_profit,
        "operating_income": operating_income,
        "pretax_income": pretax_income,
        "tax_expense": tax_expense,
        "net_income_parent": net_income,
        "eps": _num(_pick(row, ["基本每股盈餘（元）"])),
    }


def normalize_balance(row: dict) -> dict:
    assets = _num(_pick(row, ["資產總額", "資產總計"]))
    liabilities = _num(_pick(row, ["負債總額", "負債總計"]))
    equity = _num(_pick(row, ["權益總額", "權益總計"]))
    return {
        **_base_fields(row),
        "name": _pick(row, ["公司名稱"]),
        "current_assets": _num(_pick(row, ["流動資產"])),
        "current_liabilities": _num(_pick(row, ["流動負債"])),
        "assets": assets,
        "liabilities": liabilities,
        "equity": equity,
        "book_value_per_share": _num(_pick(row, ["每股參考淨值"])),
    }


def add_eva_like_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tax_rate"] = (out["tax_expense"] / out["pretax_income"]).replace([np.inf, -np.inf], np.nan).clip(0, 0.4)
    out["nopat"] = out["operating_income"] * (1 - out["tax_rate"].fillna(0.2))
    invested_capital = out["assets"] - out["current_liabilities"].fillna(0)
    out["invested_capital_proxy"] = invested_capital.where(invested_capital > 0)
    out["roic_proxy"] = out["nopat"] / out["invested_capital_proxy"]
    out["roe"] = out["net_income_parent"] / out["equity"].replace(0, np.nan)
    out["roa"] = out["net_income_parent"] / out["assets"].replace(0, np.nan)
    out["debt_to_equity"] = out["liabilities"] / out["equity"].replace(0, np.nan)
    positive_revenue = out["revenue"].where(out["revenue"] > 0)
    out["gross_margin"] = out["gross_profit"] / positive_revenue
    out["operating_margin"] = out["operating_income"] / positive_revenue
    out["net_margin"] = out["net_income_parent"] / positive_revenue
    for column in ("gross_margin", "operating_margin", "net_margin"):
        out[column] = out[column].replace([np.inf, -np.inf], np.nan).where(out[column].between(-1, 3))
    out["revenue_growth"] = np.nan
    out["pe"] = np.nan
    out["pb"] = np.nan
    out["eva_like_score"] = _score_eva_like(out)
    return out


def write_fundamentals_csv(df: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")


def _fetch_group(endpoints: list[str], normalizer) -> pd.DataFrame:
    rows = []
    for endpoint in endpoints:
        payload = _fetch_json(f"{BASE_URL}/{endpoint}")
        for row in payload:
            normalized = normalizer(row)
            normalized["source_endpoint"] = endpoint
            rows.append(normalized)
    return pd.DataFrame(rows)


def _fetch_json(url: str) -> list[dict]:
    request = Request(url, headers={"User-Agent": "fixed8-control-backtest/0.1"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8-sig"))


def _base_fields(row: dict) -> dict:
    code = str(_pick(row, ["公司代號"]) or "").strip()
    return {
        "symbol": f"{code}.TW" if code else "",
        "code": code,
        "year": _num(_pick(row, ["年度"])),
        "quarter": _num(_pick(row, ["季別"])),
        "as_of_date": _roc_date(_pick(row, ["出表日期"])),
    }


def _pick(row: dict, names: list[str]) -> object | None:
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
    return None


def _num(value: object | None) -> float:
    if value in ("", None):
        return np.nan
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return np.nan


def _roc_date(value: object | None) -> pd.Timestamp:
    text = str(value or "").strip()
    if len(text) != 7 or not text.isdigit():
        return pd.Timestamp.today().normalize()
    year = int(text[:3]) + 1911
    month = int(text[3:5])
    day = int(text[5:7])
    return pd.Timestamp(year=year, month=month, day=day)


def _score_eva_like(df: pd.DataFrame) -> pd.Series:
    roic = _higher(df["roic_proxy"], 0.0, 0.04)
    roe = _higher(df["roe"], 0.0, 0.08)
    op_margin = _higher(df["operating_margin"], 0.02, 0.2)
    gross_margin = _higher(df["gross_margin"], 0.1, 0.45)
    leverage = _lower(df["debt_to_equity"], 3.0, 0.3)
    roa = _higher(df["roa"], 0.0, 0.03)
    score = pd.concat([roic, roe, op_margin, gross_margin, leverage, roa], axis=1).mean(axis=1)
    return (score * 100).fillna(50).clip(0, 100)


def _higher(series: pd.Series, low: float, high: float) -> pd.Series:
    return ((series - low) / (high - low)).clip(0, 1)


def _lower(series: pd.Series, high: float, low: float) -> pd.Series:
    return ((high - series) / (high - low)).clip(0, 1)
