import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta

mt5.initialize()

# Get all XAUUSD deals for Feb 13
start = datetime(2026, 2, 13, 15, 30, tzinfo=timezone.utc)
end = datetime(2026, 2, 13, 15, 40, tzinfo=timezone.utc)

deals = mt5.history_deals_get(start, end)
if deals:
    for d in deals:
        deal_type = "BUY" if d.type == 0 else "SELL" if d.type == 1 else f"type={d.type}"
        entry_str = "IN" if d.entry == 0 else "OUT" if d.entry == 1 else f"entry={d.entry}"
        dt = datetime.fromtimestamp(d.time, tz=timezone.utc)
        print(f"  {dt} | {deal_type} {entry_str} | Vol: {d.volume} | Price: {d.price:.2f} | Profit: ${d.profit:.2f} | Comment: {d.comment} | Ticket: {d.order}")
else:
    print("No deals found for that period")

# Also check around 21:15
print("\n--- Around 21:15 ---")
start2 = datetime(2026, 2, 13, 21, 10, tzinfo=timezone.utc)
end2 = datetime(2026, 2, 13, 21, 20, tzinfo=timezone.utc)
deals2 = mt5.history_deals_get(start2, end2)
if deals2:
    for d in deals2:
        deal_type = "BUY" if d.type == 0 else "SELL" if d.type == 1 else f"type={d.type}"
        entry_str = "IN" if d.entry == 0 else "OUT" if d.entry == 1 else f"entry={d.entry}"
        dt = datetime.fromtimestamp(d.time, tz=timezone.utc)
        print(f"  {dt} | {deal_type} {entry_str} | Vol: {d.volume} | Price: {d.price:.2f} | Profit: ${d.profit:.2f} | Comment: {d.comment} | Ticket: {d.order}")

mt5.shutdown()
