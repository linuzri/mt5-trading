import MetaTrader5 as mt5
from datetime import datetime, timezone
mt5.initialize()

# M5 candles from 09:15 UTC onwards
rates = mt5.copy_rates_from('BTCUSD', mt5.TIMEFRAME_M5, datetime(2026,2,17,9,15,tzinfo=timezone.utc), 20)
if rates is not None:
    print("After 09:15 SELL signal was blocked:")
    for r in rates:
        t = datetime.fromtimestamp(r[0], tz=timezone.utc)
        print(f"  {t.strftime('%H:%M')} C:{r[4]:.0f}")
    
    signal_price = 68354
    latest = rates[-1][4]
    diff = signal_price - latest
    direction = "profit" if diff > 0 else "loss"
    print(f"\nSignal price: ~{signal_price}")
    print(f"Latest price: ~{latest:.0f}")
    print(f"If SELL was taken at 0.05 lot: ${diff * 0.05:.2f} {direction}")

# Current price
tick = mt5.symbol_info_tick('BTCUSD')
if tick:
    print(f"\nCurrent BTC: {tick.bid:.0f}")

mt5.shutdown()
