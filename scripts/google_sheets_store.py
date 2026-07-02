from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def configured(secrets: Any) -> bool:
    return bool(_secret_get(secrets, "google_sheet_id") and _secret_get(secrets, "gcp_service_account"))


def sync_from_google_sheets(secrets: Any, output_dir: Path) -> None:
    if not configured(secrets):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    client = _client(secrets)
    spreadsheet = client.open_by_key(_secret_get(secrets, "google_sheet_id"))
    _state_from_sheet(spreadsheet, output_dir / "state.json")
    _csv_from_sheet(spreadsheet, "trades", output_dir / "trades.csv")
    _csv_from_sheet(spreadsheet, "daily_snapshots", output_dir / "daily_snapshots.csv")


def sync_to_google_sheets(secrets: Any, output_dir: Path) -> None:
    if not configured(secrets):
        return
    client = _client(secrets)
    spreadsheet = client.open_by_key(_secret_get(secrets, "google_sheet_id"))
    _state_to_sheet(spreadsheet, output_dir / "state.json")
    _csv_to_sheet(spreadsheet, "trades", output_dir / "trades.csv")
    _csv_to_sheet(spreadsheet, "daily_snapshots", output_dir / "daily_snapshots.csv")


def _client(secrets: Any):
    import gspread
    from google.oauth2.service_account import Credentials

    info = dict(_secret_get(secrets, "gcp_service_account"))
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(credentials)


def _state_from_sheet(spreadsheet: Any, path: Path) -> None:
    worksheet = _worksheet(spreadsheet, "state")
    rows = worksheet.get_all_values()
    if len(rows) < 2 or len(rows[1]) < 2 or not rows[1][1]:
        return
    path.write_text(rows[1][1], encoding="utf-8")


def _state_to_sheet(spreadsheet: Any, path: Path) -> None:
    worksheet = _worksheet(spreadsheet, "state")
    worksheet.clear()
    worksheet.update([["key", "json"], ["state", path.read_text(encoding="utf-8") if path.exists() else "{}"]])


def _csv_from_sheet(spreadsheet: Any, sheet_name: str, path: Path) -> None:
    worksheet = _worksheet(spreadsheet, sheet_name)
    rows = worksheet.get_all_values()
    if not rows:
        return
    pd.DataFrame(rows[1:], columns=rows[0]).to_csv(path, index=False, encoding="utf-8-sig")


def _csv_to_sheet(spreadsheet: Any, sheet_name: str, path: Path) -> None:
    worksheet = _worksheet(spreadsheet, sheet_name)
    worksheet.clear()
    if not path.exists():
        worksheet.update([["empty"]])
        return
    frame = pd.read_csv(path).fillna("")
    values = [frame.columns.tolist(), *frame.astype(str).values.tolist()]
    worksheet.update(values if values else [["empty"]])


def _worksheet(spreadsheet: Any, title: str) -> Any:
    try:
        return spreadsheet.worksheet(title)
    except Exception:
        return spreadsheet.add_worksheet(title=title, rows=1000, cols=30)


def _secret_get(secrets: Any, key: str) -> Any:
    try:
        return secrets[key]
    except Exception:
        return None
