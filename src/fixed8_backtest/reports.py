from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(equity: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame([{"metric": "error", "value": "no equity rows"}])

    curve = equity.set_index("date")["equity"]
    returns = curve.pct_change().dropna()
    total_return = curve.iloc[-1] / curve.iloc[0] - 1 if len(curve) > 1 else 0
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1 / 365.25)
    annual_return = (1 + total_return) ** (1 / years) - 1
    max_drawdown = (curve / curve.cummax() - 1).min()
    sharpe = np.nan
    if not returns.empty and returns.std() != 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)

    win_rate = np.nan
    avg_trade_return = np.nan
    if not trades.empty:
        win_rate = (trades["pnl"] > 0).mean()
        avg_trade_return = trades["return_pct"].mean()

    rows = [
        ("total_return", total_return),
        ("annual_return", annual_return),
        ("max_drawdown", max_drawdown),
        ("sharpe", sharpe),
        ("trades", len(trades)),
        ("win_rate", win_rate),
        ("avg_trade_return", avg_trade_return),
        ("final_equity", curve.iloc[-1]),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])
