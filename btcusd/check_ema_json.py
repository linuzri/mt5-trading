"""EMA status check — outputs JSON for dashboard. Run from btcusd/ directory."""
import json
import sys
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

auth = json.loads(Path("mt5_auth.json").read_text())
if not mt5.initialize(
    path=r"C:\Program Files\Pepperstone MetaTrader 5\terminal64.exe",
    login=int(auth["login"]),
    password=auth["password"],
    server=auth["server"],
):
    print(json.dumps({"error": f"MT5 init failed: {mt5.last_error()}"}))
    sys.exit(1)

mt5.symbol_select("BTCUSD", True)
rates = mt5.copy_rates_from_pos("BTCUSD", mt5.TIMEFRAME_H1, 0, 60)
if rates is None:
    print(json.dumps({"error": "No candle data"}))
    mt5.shutdown()
    sys.exit(1)

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

last = df.iloc[-1]
prev = df.iloc[-2]
gap = float(last["ema10"] - last["ema50"])
prev_gap = float(prev["ema10"] - prev["ema50"])
gap_pct = (gap / last["close"]) * 100

converging = abs(gap) < abs(prev_gap)
direction = "above" if gap > 0 else "below"

# Last 5 candles
candles = []
for _, r in df.tail(5).iterrows():
    candles.append({
        "time": r["time"].isoformat(),
        "close": round(float(r["close"]), 2),
        "ema10": round(float(r["ema10"]), 2),
        "ema50": round(float(r["ema50"]), 2),
        "gap": round(float(r["ema10"] - r["ema50"]), 2)
    })

result = {
    "timestamp": last["time"].isoformat(),
    "currentPrice": round(float(last["close"]), 2),
    "ema10": round(float(last["ema10"]), 2),
    "ema50": round(float(last["ema50"]), 2),
    "gap": round(gap, 2),
    "gapPercent": round(gap_pct, 3),
    "status": f"EMA10 {direction} EMA50 | {'Converging' if converging else 'Diverging'}",
    "converging": bool(converging),
    "imminent": bool(abs(gap_pct) < 0.3),
    "candles": candles
}

print(json.dumps(result, indent=2))
mt5.shutdown()
