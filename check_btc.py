import MetaTrader5 as mt5
from datetime import datetime, timezone

mt5.initialize()

# Get M5 candles from 09:00 UTC onwards today
rates = mt5.copy_rates_from(
    'BTCUSD', mt5.TIMEFRAME_M5,
    datetime(2026, 2, 17, 9, 0, tzinfo=timezone.utc), 30
)
if rates is not None:
    print("BTC M5 candles after 09:15 UTC (signal time):")
    print("-" * 55)
    for r in rates:
        t = datetime.fromtimestamp(r[0], tz=timezone.utc)
        marker = " <-- SELL signal here" if t.strftime("%H:%M") == "09:15" else ""
        print(f"{t.strftime('%H:%M')} O:{r[1]:.0f} H:{r[2]:.0f} L:{r[3]:.0f} C:{r[4]:.0f}{marker}")

# ATR calculation
rates_atr = mt5.copy_rates_from(
    'BTCUSD', mt5.TIMEFRAME_M5,
    datetime(2026, 2, 17, 8, 0, tzinfo=timezone.utc), 30
)
if rates_atr is not None:
    ranges = [r[2] - r[3] for r in rates_atr[-14:]]
    atr14 = sum(ranges) / len(ranges)
    print(f"\nATR(14) M5 near signal: ${atr14:.1f}")
    print(f"This means: average M5 candle range is ${atr14:.0f}")
    print(f"For reference: BTC price ~${rates_atr[-1][4]:.0f}")
    print(f"ATR as % of price: {atr14/rates_atr[-1][4]*100:.3f}%")

# Check today's trade count
deals = mt5.history_deals_get(
    datetime(2026, 2, 17, 0, 0, tzinfo=timezone.utc),
    datetime(2026, 2, 17, 23, 59, tzinfo=timezone.utc)
)
btc_deals = [d for d in (deals or []) if 'BTC' in (d.symbol or '')]
print(f"\nBTCUSD deals today: {len(btc_deals)}")
for d in btc_deals:
    t = datetime.fromtimestamp(d.time, tz=timezone.utc)
    print(f"  {t.strftime('%H:%M')} {d.symbol} vol:{d.volume} profit:{d.profit:.2f} comment:{d.comment}")

mt5.shutdown()
