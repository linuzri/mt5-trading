"""Quick EMA status check — run from btcusd/ directory."""
import json
import sys
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

# Load auth
auth = json.loads(Path("mt5_auth.json").read_text())
if not mt5.initialize(
    path=r"C:\Program Files\Pepperstone MetaTrader 5\terminal64.exe",
    login=int(auth["login"]),
    password=auth["password"],
    server=auth["server"],
):
    print(f"MT5 init failed: {mt5.last_error()}")
    sys.exit(1)

mt5.symbol_select("BTCUSD", True)
rates = mt5.copy_rates_from_pos("BTCUSD", mt5.TIMEFRAME_H1, 0, 60)
if rates is None:
    print("No candle data")
    mt5.shutdown()
    sys.exit(1)

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

last = df.iloc[-1]
prev = df.iloc[-2]
gap = last["ema10"] - last["ema50"]
gap_pct = (gap / last["close"]) * 100
prev_gap = prev["ema10"] - prev["ema50"]

print(f"Current candle: {last['time']}")
print(f"Price: {last['close']:.2f}")
print(f"EMA10: {last['ema10']:.2f}")
print(f"EMA50: {last['ema50']:.2f}")
print(f"Gap: {gap:.2f} ({gap_pct:.3f}%)")
print(f"EMA10 {'above' if gap > 0 else 'below'} EMA50")
print(f"Converging: {'yes' if abs(gap) < abs(prev_gap) else 'no'} (prev gap: {prev_gap:.2f})")

# Show last 5 candles
print(f"\nLast 5 H1 candles:")
for _, r in df.tail(5).iterrows():
    print(f"  {r['time']} | close={r['close']:.2f} | EMA10={r['ema10']:.2f} | EMA50={r['ema50']:.2f} | gap={r['ema10']-r['ema50']:.2f}")

mt5.shutdown()
