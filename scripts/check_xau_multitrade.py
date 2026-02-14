import csv
from datetime import datetime, timedelta

trades = []
with open('xauusd/trade_log.csv', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 5:
            continue
        try:
            dt = datetime.fromisoformat(parts[0].replace('+00:00', ''))
            trades.append({'dt': dt, 'dir': parts[1], 'entry': float(parts[2]), 'exit': float(parts[3]), 'profit': float(parts[4])})
        except:
            continue

# Look at the catastrophic trades and nearby trades
print("=== Trades around catastrophic losses ===\n")

bad_times = [
    datetime(2026, 2, 13, 15, 35),
    datetime(2026, 2, 13, 21, 15),
    datetime(2026, 2, 9, 17, 5),
    datetime(2026, 2, 9, 16, 5),
]

for bt in bad_times:
    window = [t for t in trades if abs((t['dt'] - bt).total_seconds()) < 3600]
    print(f"--- Around {bt} ---")
    for t in sorted(window, key=lambda x: x['dt']):
        marker = " <<<" if abs((t['dt'] - bt).total_seconds()) < 120 else ""
        print(f"  {t['dt']} | {t['dir']} | {t['entry']:.2f} -> {t['exit']:.2f} | P/L: ${t['profit']:.2f}{marker}")
    print()

# Check for rapid-fire trades (within 2 min of each other)
print("=== Rapid-fire trades (< 2 min apart) ===")
for i in range(1, len(trades)):
    gap = (trades[i]['dt'] - trades[i-1]['dt']).total_seconds()
    if gap < 120 and gap > 0:
        t1, t2 = trades[i-1], trades[i]
        if abs(t1['profit']) > 20 or abs(t2['profit']) > 20:
            print(f"  {t1['dt']} {t1['dir']} P/L ${t1['profit']:.2f}")
            print(f"  {t2['dt']} {t2['dir']} P/L ${t2['profit']:.2f} (gap: {gap:.0f}s)")
            print(f"  Combined: ${t1['profit'] + t2['profit']:.2f}")
            print()
