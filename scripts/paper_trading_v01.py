from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("FIXED8_DATA_DIR", ROOT / "outputs"))
SIGNAL_DIR = Path(os.environ.get("FIXED8_SIGNAL_DIR", DATA_ROOT / "today_model_signals"))
OUTPUT_DIR = DATA_ROOT / "paper_trading_v0_1"
STATE_PATH = OUTPUT_DIR / "state.json"
TRADES_PATH = OUTPUT_DIR / "trades.csv"
SNAPSHOTS_PATH = OUTPUT_DIR / "daily_snapshots.csv"

INITIAL_CASH = 1_000_000.0
MAX_POSITIONS = 5
POSITION_WEIGHT = 0.20
STOP_LOSS = 0.15
COMMISSION_RATE = 0.001425
TAX_RATE = 0.003
SLIPPAGE_RATE = 0.001
MAX_HOLDING_DAYS = 252
MODEL_VERSION = "Rule-Based Strict"


@dataclass
class Position:
    symbol: str
    name: str
    shares: int
    entry_signal_date: str
    entry_date: str
    entry_price: float
    entry_cost: float
    holding_bars: int = 0


def update_paper_trading() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    latest_date = _latest_signal_date()
    features = _read_named_csv(f"latest_features_{latest_date}_named.csv")
    candidates = _read_named_csv(f"final_score_only_{latest_date}_named.csv")
    if state["start_date"] is None:
        state["start_date"] = latest_date
        state["last_signal_date"] = latest_date
        state["pending_orders"] = _make_pending_buys(candidates, latest_date, state)
        _save_state(state)

    state = _execute_pending_orders(state)
    state, exit_signals = _apply_exit_signals(state, features, latest_date)
    state = _create_new_buy_orders(state, candidates, latest_date)
    snapshot = _snapshot(state, features, latest_date, candidates, exit_signals)
    _append_snapshot(snapshot)
    _save_state(state)
    return dashboard_payload()


def dashboard_payload() -> dict:
    latest_date = _latest_signal_date()
    state = _load_state()
    features = _read_named_csv(f"latest_features_{latest_date}_named.csv")
    candidates = _read_named_csv(f"final_score_only_{latest_date}_named.csv")
    aggressive = _read_named_csv(f"aggressive_fund_final_ml_{latest_date}_named.csv")
    summary = _read_named_csv(f"summary_{latest_date}_named.csv")
    holdings = _holdings_table(state, features)
    equity = float(state["cash"] + sum(row["market_value"] for row in holdings))
    total_return = equity / INITIAL_CASH - 1
    return {
        "model_version": MODEL_VERSION,
        "latest_date": latest_date,
        "market_regime": _summary_value(summary, "market_regime"),
        "candidate_rows": _summary_value(summary, "candidate_rows"),
        "cash": state["cash"],
        "equity": equity,
        "total_return_pct": total_return * 100,
        "realized_pnl": _realized_pnl(),
        "positions": holdings,
        "pending_orders": state.get("pending_orders", []),
        "defensive_top": _table_rows(candidates.head(10)),
        "aggressive_top": _table_rows(aggressive.head(10)),
        "exit_signals": _current_exit_signals(state, features, latest_date),
        "trades": _read_trades().tail(30).to_dict("records"),
        "snapshots": _read_snapshots().tail(120).to_dict("records"),
    }


def _load_state() -> dict:
    if STATE_PATH.exists():
        with STATE_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "model_version": MODEL_VERSION,
        "initial_cash": INITIAL_CASH,
        "cash": INITIAL_CASH,
        "start_date": None,
        "last_signal_date": None,
        "positions": [],
        "pending_orders": [],
    }


def _save_state(state: dict) -> None:
    with STATE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


def _latest_signal_date() -> str:
    files = sorted(SIGNAL_DIR.glob("final_score_only_*.csv"))
    if not files:
        raise FileNotFoundError("No final_score_only signal file found. Run scripts/run_today_model_signals.py first.")
    dates = [
        path.stem.replace("final_score_only_", "").replace("_named", "")
        for path in files
    ]
    return sorted(set(dates))[-1]


def _read_named_csv(name: str) -> pd.DataFrame:
    path = SIGNAL_DIR / name
    if not path.exists() and "_named" in name:
        path = SIGNAL_DIR / name.replace("_named", "")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _make_pending_buys(candidates: pd.DataFrame, signal_date: str, state: dict) -> list[dict]:
    held = {position["symbol"] for position in state.get("positions", [])}
    pending = {order["symbol"] for order in state.get("pending_orders", [])}
    slots = MAX_POSITIONS - len(held) - len(pending)
    if slots <= 0 or candidates.empty:
        return []
    orders = []
    for row in candidates[~candidates["symbol"].isin(held | pending)].head(slots).itertuples(index=False):
        orders.append(
            {
                "type": "buy",
                "symbol": row.symbol,
                "name": getattr(row, "name", ""),
                "signal_date": signal_date,
                "status": "pending_next_open",
            }
        )
    return orders


def _execute_pending_orders(state: dict) -> dict:
    remaining = []
    for order in state.get("pending_orders", []):
        if order.get("type") == "sell":
            open_price, trade_date = _next_open(order["symbol"], order["signal_date"])
            if open_price is None:
                remaining.append(order)
                continue
            position = _find_position(state, order["symbol"])
            if position is None:
                continue
            _close_position(state, position, open_price, trade_date, order["signal_date"], order.get("reason", "sell_signal"))
            state["positions"] = [pos for pos in state.get("positions", []) if pos["symbol"] != order["symbol"]]
            continue
        if order.get("type") != "buy":
            remaining.append(order)
            continue
        open_price, trade_date = _next_open(order["symbol"], order["signal_date"])
        if open_price is None:
            remaining.append(order)
            continue
        budget = min(state["cash"], _portfolio_equity(state) * POSITION_WEIGHT)
        buy_price = open_price * (1 + SLIPPAGE_RATE)
        fee_adjusted = buy_price * (1 + COMMISSION_RATE)
        shares = int(budget // fee_adjusted)
        if shares <= 0:
            remaining.append(order)
            continue
        cost = shares * fee_adjusted
        state["cash"] -= cost
        state["positions"].append(
            asdict(
                Position(
                    symbol=order["symbol"],
                    name=order.get("name", ""),
                    shares=shares,
                    entry_signal_date=order["signal_date"],
                    entry_date=trade_date,
                    entry_price=buy_price,
                    entry_cost=cost,
                )
            )
        )
    state["pending_orders"] = remaining
    return state


def _apply_exit_signals(state: dict, features: pd.DataFrame, signal_date: str) -> tuple[dict, list[dict]]:
    remaining = []
    exits = []
    by_symbol = features.set_index("symbol", drop=False) if not features.empty and "symbol" in features else pd.DataFrame()
    for position in state.get("positions", []):
        if position["symbol"] not in by_symbol.index:
            remaining.append(position)
            continue
        row = by_symbol.loc[position["symbol"]]
        position["holding_bars"] = int(position.get("holding_bars", 0)) + 1
        reason = _exit_reason(position, row)
        if not reason:
            remaining.append(position)
            continue
        signal = {"symbol": position["symbol"], "name": position.get("name", ""), "reason": reason, "signal_date": signal_date}
        exits.append(signal)
        if not _has_pending_sell(state, position["symbol"]):
            state.setdefault("pending_orders", []).append(
                {
                    "type": "sell",
                    "symbol": position["symbol"],
                    "name": position.get("name", ""),
                    "signal_date": signal_date,
                    "reason": reason,
                    "status": "pending_next_open",
                }
            )
        remaining.append(position)
    state["positions"] = remaining
    return state, exits


def _create_new_buy_orders(state: dict, candidates: pd.DataFrame, latest_date: str) -> dict:
    if state.get("last_signal_date") == latest_date and state.get("pending_orders"):
        return state
    existing_pending = [order for order in state.get("pending_orders", []) if order.get("signal_date") == latest_date]
    if existing_pending:
        return state
    new_orders = _make_pending_buys(candidates, latest_date, state)
    if new_orders:
        state["pending_orders"] = [*state.get("pending_orders", []), *new_orders]
        state["last_signal_date"] = latest_date
    return state


def _exit_reason(position: dict, row: pd.Series) -> str | None:
    if row.get("market_regime") == "bear":
        return "market_bear"
    close = float(row.get("Close", 0))
    if close <= float(position["entry_price"]) * (1 - STOP_LOSS):
        return "stop_loss"
    if int(position.get("holding_bars", 0)) >= MAX_HOLDING_DAYS:
        return "max_holding_days"
    ma120 = row.get("ma120")
    if pd.notna(ma120) and close < float(ma120):
        return "below_ma120"
    return None


def _close_position(state: dict, position: dict, open_price: float, trade_date: str, signal_date: str, reason: str) -> None:
    proceeds = position["shares"] * open_price * (1 - SLIPPAGE_RATE)
    fee = proceeds * COMMISSION_RATE
    tax = proceeds * TAX_RATE
    net_proceeds = proceeds - fee - tax
    state["cash"] += net_proceeds
    pnl = net_proceeds - float(position["entry_cost"])
    trade = {
        "symbol": position["symbol"],
        "name": position.get("name", ""),
        "entry_signal_date": position["entry_signal_date"],
        "entry_date": position["entry_date"],
        "exit_signal_date": signal_date,
        "exit_date": trade_date,
        "entry_price": position["entry_price"],
        "exit_price": open_price,
        "shares": position["shares"],
        "pnl": pnl,
        "return_pct": open_price / float(position["entry_price"]) - 1,
        "holding_bars": position.get("holding_bars", 0),
        "exit_reason": reason,
    }
    trades = _read_trades()
    trades = pd.concat([trades, pd.DataFrame([trade])], ignore_index=True)
    trades.to_csv(TRADES_PATH, index=False, encoding="utf-8-sig")


def _find_position(state: dict, symbol: str) -> dict | None:
    for position in state.get("positions", []):
        if position["symbol"] == symbol:
            return position
    return None


def _has_pending_sell(state: dict, symbol: str) -> bool:
    return any(order.get("type") == "sell" and order.get("symbol") == symbol for order in state.get("pending_orders", []))


def _portfolio_equity(state: dict) -> float:
    value = float(state["cash"])
    for position in state.get("positions", []):
        close = _last_close(position["symbol"])
        if close is not None:
            value += position["shares"] * close
    return value


def _holdings_table(state: dict, features: pd.DataFrame) -> list[dict]:
    by_symbol = features.set_index("symbol", drop=False) if not features.empty and "symbol" in features else pd.DataFrame()
    rows = []
    for position in state.get("positions", []):
        row = by_symbol.loc[position["symbol"]] if position["symbol"] in by_symbol.index else pd.Series(dtype=object)
        close = float(row.get("Close", _last_close(position["symbol"]) or 0))
        market_value = position["shares"] * close
        unrealized = market_value - float(position["entry_cost"])
        rows.append(
            {
                **position,
                "close": close,
                "market_value": market_value,
                "unrealized_pnl": unrealized,
                "unrealized_return_pct": unrealized / float(position["entry_cost"]) * 100,
                "stop_price": float(position["entry_price"]) * (1 - STOP_LOSS),
                "ma120": float(row.get("ma120")) if pd.notna(row.get("ma120", pd.NA)) else None,
                "exit_reason": _exit_reason(position, row) if not row.empty else None,
            }
        )
    return rows


def _current_exit_signals(state: dict, features: pd.DataFrame, latest_date: str) -> list[dict]:
    rows = []
    for holding in _holdings_table(state, features):
        if holding.get("exit_reason"):
            rows.append(
                {
                    "symbol": holding["symbol"],
                    "name": holding.get("name", ""),
                    "signal_date": latest_date,
                    "reason": holding["exit_reason"],
                }
            )
    return rows


def _snapshot(state: dict, features: pd.DataFrame, latest_date: str, candidates: pd.DataFrame, exits: list[dict]) -> dict:
    holdings = _holdings_table(state, features)
    equity = float(state["cash"] + sum(row["market_value"] for row in holdings))
    return {
        "date": latest_date,
        "cash": state["cash"],
        "market_value": sum(row["market_value"] for row in holdings),
        "equity": equity,
        "total_return_pct": equity / INITIAL_CASH * 100 - 100,
        "positions": len(holdings),
        "pending_orders": len(state.get("pending_orders", [])),
        "candidate_rows": len(candidates),
        "exit_signals": len(exits),
    }


def _append_snapshot(row: dict) -> None:
    snapshots = _read_snapshots()
    snapshots = snapshots[snapshots["date"] != row["date"]] if "date" in snapshots else snapshots
    snapshots = pd.concat([snapshots, pd.DataFrame([row])], ignore_index=True)
    snapshots.to_csv(SNAPSHOTS_PATH, index=False, encoding="utf-8-sig")


def _next_open(symbol: str, signal_date: str) -> tuple[float | None, str | None]:
    frame = _price_frame(symbol)
    if frame.empty:
        return None, None
    signal_ts = pd.Timestamp(signal_date)
    future = frame[frame.index > signal_ts]
    if future.empty:
        return None, None
    row = future.iloc[0]
    return float(row["Open"]), str(future.index[0].date())


def _last_close(symbol: str) -> float | None:
    frame = _price_frame(symbol)
    if frame.empty:
        return None
    return float(frame.iloc[-1]["Close"])


def _price_frame(symbol: str) -> pd.DataFrame:
    path = ROOT / "work" / "price_cache" / f"{symbol.replace('.', '_')}.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame


def _read_trades() -> pd.DataFrame:
    if TRADES_PATH.exists():
        return pd.read_csv(TRADES_PATH)
    return pd.DataFrame(
        columns=[
            "symbol",
            "name",
            "entry_signal_date",
            "entry_date",
            "exit_signal_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "shares",
            "pnl",
            "return_pct",
            "holding_bars",
            "exit_reason",
        ]
    )


def _read_snapshots() -> pd.DataFrame:
    if SNAPSHOTS_PATH.exists():
        return pd.read_csv(SNAPSHOTS_PATH)
    return pd.DataFrame(columns=["date", "cash", "market_value", "equity", "total_return_pct"])


def _realized_pnl() -> float:
    trades = _read_trades()
    return float(trades["pnl"].sum()) if "pnl" in trades and not trades.empty else 0.0


def _summary_value(summary: pd.DataFrame, item: str) -> str:
    if summary.empty or "item" not in summary:
        return ""
    rows = summary[summary["item"] == item]
    return "" if rows.empty else str(rows.iloc[0]["value"])


def _table_rows(frame: pd.DataFrame) -> list[dict]:
    wanted = [
        "symbol",
        "name",
        "Close",
        "fund_ml_85_15",
        "fund_final_ml",
        "final_score",
        "fundamental_score",
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
    cols = [col for col in wanted if col in frame]
    return frame[cols].to_dict("records") if cols else []
