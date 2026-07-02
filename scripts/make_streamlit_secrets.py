from __future__ import annotations

import json
import sys
from pathlib import Path


SHEET_ID = "1hQKOnYSsRdyFI_S56YolbTb4cr57VevPEt8LvhmCjs4"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "work" / "streamlit_secrets.toml"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/make_streamlit_secrets.py path/to/service-account.json")
    source = Path(sys.argv[1])
    output = Path(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_OUTPUT
    info = json.loads(source.read_text(encoding="utf-8"))
    keys = [
        "type",
        "project_id",
        "private_key_id",
        "private_key",
        "client_email",
        "client_id",
        "auth_uri",
        "token_uri",
        "auth_provider_x509_cert_url",
        "client_x509_cert_url",
    ]
    lines = [
        'fixed8_password = "請改成你的網站登入密碼"',
        f'google_sheet_id = "{SHEET_ID}"',
        "",
        "[gcp_service_account]",
    ]
    for key in keys:
        value = str(info.get(key, ""))
        value = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        lines.append(f'{key} = "{value}"')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    print(info.get("client_email", ""))


if __name__ == "__main__":
    main()
