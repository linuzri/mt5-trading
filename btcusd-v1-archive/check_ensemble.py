import csv
from collections import defaultdict

rows = list(csv.reader(open('trade_log.csv')))
days = defaultdict(lambda: {'w':0,'l':0,'pnl':0.0})

for r in rows:
    if r[0] >= '2026-02-11':
        pnl = float(r[4])
        d = r[0][:10]
        days[d]['pnl'] += pnl
        if pnl > 0:
            days[d]['w'] += 1
        else:
            days[d]['l'] += 1

total_w = sum(v['w'] for v in days.values())
total_l = sum(v['l'] for v in days.values())
total_t = total_w + total_l
total_pnl = sum(v['pnl'] for v in days.values())

print(f"=== BTCUSD Ensemble Performance (Feb 11-14) ===")
print(f"Total: {total_t} trades, WR {total_w/total_t*100:.1f}%, P/L ${total_pnl:.2f}")
print()
for d in sorted(days):
    v = days[d]
    t = v['w'] + v['l']
    wr = v['w']/t*100
    print(f"  {d}: {t} trades, {wr:.0f}% WR, ${v['pnl']:.2f}")
