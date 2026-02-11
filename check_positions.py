import MetaTrader5 as mt5
import json

creds = json.load(open(r'C:\Users\Nazri Hussain\projects\mt5-trading\btcusd\mt5_auth.json'))
mt5.initialize()
mt5.login(creds['login'], password=creds['password'], server=creds['server'])

acc = mt5.account_info()
print(f"Balance: ${acc.balance:.2f}")
print(f"Equity: ${acc.equity:.2f}")
print(f"Floating P/L: ${acc.equity - acc.balance:.2f}")
print(f"Margin Used: ${acc.margin:.2f}")
print()

positions = mt5.positions_get()
if positions:
    print(f"Open positions: {len(positions)}")
    total_swap = 0
    total_profit = 0
    for p in positions:
        total_swap += p.swap
        total_profit += p.profit
        direction = "BUY" if p.type == 0 else "SELL"
        print(f"  {p.symbol} {direction} vol={p.volume} entry={p.price_open:.5f} current={p.price_current:.5f} profit=${p.profit:.2f} swap=${p.swap:.2f}")
    print(f"\nTotal floating profit: ${total_profit:.2f}")
    print(f"Total swap fees: ${total_swap:.2f}")
else:
    print("No open positions")

# Check recent deal history for swap fees
from datetime import datetime, timedelta, timezone
start = datetime(2026, 2, 10, tzinfo=timezone.utc)
end = datetime.now(timezone.utc)
deals = mt5.history_deals_get(start, end)
if deals:
    total_swap_paid = sum(d.swap for d in deals)
    total_commission = sum(d.commission for d in deals)
    print(f"\nLast 2 days swap fees paid: ${total_swap_paid:.2f}")
    print(f"Last 2 days commissions: ${total_commission:.2f}")

mt5.shutdown()
