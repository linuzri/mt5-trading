import MetaTrader5 as mt5
from datetime import datetime, timedelta
mt5.initialize()
info = mt5.account_info()
print(f"Balance: ${info.balance:.2f}")
print(f"Equity: ${info.equity:.2f}")
print(f"Profit: ${info.profit:.2f}")
print(f"Open positions: {mt5.positions_total()}")
positions = mt5.positions_get()
if positions:
    for p in positions:
        side = "BUY" if p.type == 0 else "SELL"
        print(f"  {p.symbol} {side} {p.volume} lots | profit: ${p.profit:.2f} | open: {p.price_open}")

# Recent trades (last 24h)
now = datetime.now()
yesterday = now - timedelta(hours=24)
deals = mt5.history_deals_get(yesterday, now)
if deals:
    print(f"\nTrades last 24h: {len([d for d in deals if d.entry > 0])}")
    total_pnl = sum(d.profit for d in deals if d.entry > 0)
    print(f"24h P/L: ${total_pnl:.2f}")
else:
    print("\nNo trades in last 24h")
mt5.shutdown()
