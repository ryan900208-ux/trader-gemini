from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("FIXED8_DATA_DIR", ROOT / "outputs"))
OUTPUT_DIR = DATA_ROOT / "today_model_signals"


def main() -> None:
    universe = pd.read_csv(ROOT / "data" / "universe_twse_all.csv", dtype=str)[["symbol", "name"]]
    name_map = dict(zip(universe["symbol"], universe["name"]))
    files = [path for path in OUTPUT_DIR.glob("*_2026-06-08.csv") if "_named" not in path.name]
    for path in files:
        frame = pd.read_csv(path)
        if "name" in frame.columns:
            frame = frame.drop(columns=["name"])
        if "symbol" in frame.columns:
            location = frame.columns.get_loc("symbol") + 1
            frame.insert(location, "name", frame["symbol"].map(name_map))
        if path.name.startswith("summary_") and "value" in frame.columns:
            frame["value"] = frame["value"].map(lambda value: _add_names_to_symbol_list(value, name_map))
        frame.to_csv(OUTPUT_DIR / f"{path.stem}_named.csv", index=False, encoding="utf-8-sig")
    for path in sorted(OUTPUT_DIR.glob("*named.csv")):
        print(path.name)


def _add_names_to_symbol_list(value: object, name_map: dict[str, str]) -> object:
    if not isinstance(value, str) or ".TW" not in value:
        return value
    parts = []
    for token in value.split(","):
        symbol = token.strip().split()[0]
        name = name_map.get(symbol, "")
        parts.append(f"{symbol} {name}".strip())
    return ",".join(parts)


if __name__ == "__main__":
    main()
