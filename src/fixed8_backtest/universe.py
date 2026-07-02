from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen


TWSE_LISTED_COMPANIES_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"


def fetch_twse_listed_companies(url: str = TWSE_LISTED_COMPANIES_URL) -> list[dict]:
    request = Request(url, headers={"User-Agent": "fixed8-control-backtest/0.1"})
    with urlopen(request, timeout=60) as response:
        payload = response.read().decode("utf-8-sig")
    return json.loads(payload)


def normalize_twse_universe(rows: list[dict]) -> list[dict]:
    normalized = []
    seen = set()
    for row in rows:
        code = _pick(row, ["公司代號", "有價證券代號", "證券代號", "stockNo", "Code"])
        name = _pick(row, ["公司簡稱", "公司名稱", "有價證券名稱", "Name"])
        industry = _pick(row, ["產業別", "Industry"])
        if not code:
            continue
        code = str(code).strip()
        if not re.fullmatch(r"\d{4}", code):
            continue
        symbol = f"{code}.TW"
        if symbol in seen:
            continue
        seen.add(symbol)
        normalized.append({"symbol": symbol, "code": code, "name": str(name or "").strip(), "industry": str(industry or "").strip()})
    return sorted(normalized, key=lambda item: item["code"])


def write_universe_csv(rows: list[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "code", "name", "industry"])
        writer.writeheader()
        writer.writerows(rows)


def _pick(row: dict, names: list[str]) -> object | None:
    for name in names:
        if name in row:
            return row[name]
    return None
