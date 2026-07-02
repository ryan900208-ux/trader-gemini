from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_universe(path: str | Path, benchmark_symbol: str) -> list[str]:
    frame = pd.read_csv(path)
    symbols = frame["symbol"].dropna().astype(str).str.strip().tolist()
    symbols = [symbol for symbol in symbols if symbol and symbol != benchmark_symbol]
    return sorted(set(symbols))


def download_ohlcv(
    symbols: list[str],
    start: str,
    end: str | None,
    batch_size: int = 80,
    cache_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    if not symbols:
        return {}

    data: dict[str, pd.DataFrame] = {}
    missing = []
    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
        for symbol in symbols:
            cached = _read_cached_ohlcv(cache_path, symbol, start, end)
            if cached is None:
                missing.append(symbol)
            else:
                data[symbol] = cached
    else:
        missing = symbols

    if missing:
        print(f"Downloading {len(missing)} symbols from yfinance; {len(data)} loaded from cache.", flush=True)

    batches = _chunks(missing, max(1, batch_size))
    for batch_number, batch in enumerate(batches, start=1):
        print(f"Downloading batch {batch_number}/{len(batches)} ({len(batch)} symbols)...", flush=True)
        raw = yf.download(
            tickers=batch,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )

        if len(batch) == 1:
            frame = _clean_ohlcv(raw)
            if not frame.empty:
                data[batch[0]] = frame
                if cache_path:
                    _write_cached_ohlcv(cache_path, batch[0], frame)
            continue

        available = set(raw.columns.get_level_values(0))
        for symbol in batch:
            if symbol not in available:
                continue
            frame = _clean_ohlcv(raw[symbol])
            if not frame.empty:
                data[symbol] = frame
                if cache_path:
                    _write_cached_ohlcv(cache_path, symbol, frame)
    return data


def _clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    if "Adj Close" in df.columns:
        ratio = df["Adj Close"] / df["Close"]
        for column in ("Open", "High", "Low", "Close"):
            df[column] = df[column] * ratio
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not set(needed).issubset(df.columns):
        return pd.DataFrame(columns=needed)
    df = df[needed].dropna(subset=["Open", "Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _cache_file(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / f"{symbol.replace('.', '_')}.csv"


def _read_cached_ohlcv(cache_dir: Path, symbol: str, start: str, end: str | None) -> pd.DataFrame | None:
    path = _cache_file(cache_dir, symbol)
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    frame.index = pd.to_datetime(frame.index).tz_localize(None).as_unit("ns")
    start_ts = pd.Timestamp(start).as_unit("ns")
    end_ts = pd.Timestamp(end) if end else None
    if frame.empty:
        return None
    if end_ts is not None and frame.index.max() < end_ts - pd.Timedelta(days=10):
        return None
    return frame[frame.index >= start_ts] if end_ts is None else frame[(frame.index >= start_ts) & (frame.index < end_ts)]


def _write_cached_ohlcv(cache_dir: Path, symbol: str, frame: pd.DataFrame) -> None:
    output = frame.copy()
    output.index.name = "Date"
    output.to_csv(_cache_file(cache_dir, symbol))
