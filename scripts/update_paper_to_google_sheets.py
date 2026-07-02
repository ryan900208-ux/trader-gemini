from __future__ import annotations

import json
import os
from pathlib import Path

from google_sheets_store import sync_from_google_sheets, sync_to_google_sheets
from paper_trading_v01 import OUTPUT_DIR, update_paper_trading


def main() -> None:
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    service_account_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not sheet_id or not service_account_json:
        raise SystemExit("GOOGLE_SHEET_ID and GCP_SERVICE_ACCOUNT_JSON are required.")
    secrets = {
        "google_sheet_id": sheet_id,
        "gcp_service_account": json.loads(service_account_json),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sync_from_google_sheets(secrets, OUTPUT_DIR)
    payload = update_paper_trading()
    sync_to_google_sheets(secrets, OUTPUT_DIR)
    print(
        {
            "latest_date": payload["latest_date"],
            "equity": payload["equity"],
            "total_return_pct": payload["total_return_pct"],
            "positions": len(payload["positions"]),
            "pending_orders": len(payload["pending_orders"]),
        }
    )


if __name__ == "__main__":
    main()
