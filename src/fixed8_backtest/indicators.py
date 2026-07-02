from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close = df["Close"]
    volume = df["Volume"]

    for window in (5, 20, 60, 120):
        df[f"ret{window}"] = close.pct_change(window)
        df[f"ma{window}"] = close.rolling(window).mean()

    df["ma20_slope"] = df["ma20"].diff(5) / df["ma20"].shift(5)
    df["rsi14"] = rsi(close, 14)
    df["volume_ma20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["volume_ma20"]
    df["ma20_deviation"] = (close / df["ma20"]) - 1
    return df


def market_regime(benchmark: pd.DataFrame, config: dict) -> pd.Series:
    regime_cfg = config["market_regime"]
    df = add_indicators(benchmark)
    bear = (
        (df["Close"] < df[f"ma{regime_cfg['bear_close_below_ma']}"])
        & (df[f"ma{regime_cfg['bear_ma_fast']}"] < df[f"ma{regime_cfg['bear_ma_slow']}"])
    )
    bull = (
        (df["Close"] >= df[f"ma{regime_cfg['bear_close_below_ma']}"])
        & (df[f"ma{regime_cfg['bear_ma_fast']}"] >= df[f"ma{regime_cfg['bear_ma_slow']}"])
    )
    out = pd.Series("neutral", index=df.index, name="market_regime")
    out[bear] = "bear"
    out[bull] = "bull"
    return out
