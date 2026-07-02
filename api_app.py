from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("FIXED8_DATA_DIR", str(ROOT / "outputs"))
os.environ.setdefault("FIXED8_PRICE_CACHE_DIR", str(ROOT / "work" / "price_cache"))

from paper_trading_v01 import dashboard_payload, update_paper_trading  # noqa: E402


def _json(data: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(_clean_json(data), status_code=status_code)


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_json(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _authorized(request: Request) -> bool:
    token = os.environ.get("FIXED8_API_TOKEN")
    if not token:
        return True
    return request.headers.get("authorization") == f"Bearer {token}"


def _require_auth(request: Request) -> JSONResponse | None:
    if _authorized(request):
        return None
    return _json({"error": "unauthorized"}, status_code=401)


async def health(request: Request) -> JSONResponse:
    blocked = _require_auth(request)
    if blocked:
        return blocked
    return _json({"status": "ok", "app": "fixed8-api"})


async def dashboard(request: Request) -> JSONResponse:
    blocked = _require_auth(request)
    if blocked:
        return blocked
    return _json(dashboard_payload())


async def signals(request: Request) -> JSONResponse:
    blocked = _require_auth(request)
    if blocked:
        return blocked
    payload = dashboard_payload()
    return _json(
        {
            "latest_date": payload["latest_date"],
            "market_regime": payload["market_regime"],
            "candidate_rows": payload["candidate_rows"],
            "defensive_top": payload["defensive_top"],
            "aggressive_top": payload["aggressive_top"],
            "pending_orders": payload["pending_orders"],
            "exit_signals": payload["exit_signals"],
        }
    )


async def portfolio(request: Request) -> JSONResponse:
    blocked = _require_auth(request)
    if blocked:
        return blocked
    payload = dashboard_payload()
    return _json(
        {
            "latest_date": payload["latest_date"],
            "model_version": payload["model_version"],
            "cash": payload["cash"],
            "equity": payload["equity"],
            "total_return_pct": payload["total_return_pct"],
            "realized_pnl": payload["realized_pnl"],
            "positions": payload["positions"],
        }
    )


async def records(request: Request) -> JSONResponse:
    blocked = _require_auth(request)
    if blocked:
        return blocked
    payload = dashboard_payload()
    return _json(
        {
            "latest_date": payload["latest_date"],
            "trades": payload["trades"],
            "snapshots": payload["snapshots"],
        }
    )


async def update_paper(request: Request) -> JSONResponse:
    blocked = _require_auth(request)
    if blocked:
        return blocked
    return _json(update_paper_trading())


async def update_signals(request: Request) -> JSONResponse:
    blocked = _require_auth(request)
    if blocked:
        return blocked

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{ROOT / 'scripts'}"
    env["FIXED8_DATA_DIR"] = str(ROOT / "outputs")
    env["FIXED8_PRICE_CACHE_DIR"] = str(ROOT / "work" / "price_cache")

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_today_model_signals.py")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    return _json(
        {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[-8000:],
            "stderr": result.stderr[-8000:],
        },
        status_code=200 if result.returncode == 0 else 500,
    )


app = Starlette(
    debug=False,
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/api/dashboard", dashboard, methods=["GET"]),
        Route("/api/signals", signals, methods=["GET"]),
        Route("/api/portfolio", portfolio, methods=["GET"]),
        Route("/api/records", records, methods=["GET"]),
        Route("/api/update-paper", update_paper, methods=["POST"]),
        Route("/api/update-signals", update_signals, methods=["POST"]),
    ],
    middleware=[
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    ],
)
