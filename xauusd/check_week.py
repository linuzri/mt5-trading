import csv
from collections import defaultdict

rows = list(csv.reader(open('trade_log.csv')))
days = defaultdict(lambda: {'w':0,'l':0,'pnl':0.0,'trades':[]})

for r in rows:
    if r[0] >= '2026-02-07':
        pnl = float(r[4])
        d = r[0][:10]
        days[d]['pnl'] += pnl
        days[d]['trades'].append(pnl)
        if pnl > 0:
            days[d]['w'] += 1
        else:
            days[d]['l'] += 1

total_w = sum(v['w'] for v in days.values())
total_l = sum(v['l'] for v in days.values())
total_t = total_w + total_l
total_pnl = sum(v['pnl'] for v in days.values())
wr = total_w/total_t*100 if total_t else 0

all_wins = [p for v in days.values() for p in v['trades'] if p > 0]
all_losses = [p for v in days.values() for p in v['trades'] if p <= 0]
avg_win = sum(all_wins)/len(all_wins) if all_wins else 0
avg_loss = sum(all_losses)/len(all_losses) if all_losses else 0
biggest_w = max(all_wins) if all_wins else 0
biggest_l = min(all_losses) if all_losses else 0

print("=== XAUUSD Past Week (Feb 7-14) ===")
print(f"Total: {total_t} trades, WR {wr:.1f}%, P/L ${total_pnl:.2f}")
print(f"Wins: {total_w}, Losses: {total_l}")
print(f"Avg win: ${avg_win:.2f}, Avg loss: ${avg_loss:.2f}")
print(f"Biggest win: ${biggest_w:.2f}, Biggest loss: ${biggest_l:.2f}")
if avg_loss != 0:
    print(f"Risk/Reward: {abs(avg_win/avg_loss):.2f}")
print()
green = sum(1 for v in days.values() if v['pnl'] > 0)
red = sum(1 for v in days.values() if v['pnl'] <= 0)
print(f"Green days: {green}, Red days: {red}")
print()
for d in sorted(days):
    v = days[d]
    t = v['w'] + v['l']
    wr2 = v['w']/t*100 if t else 0
    icon = "+" if v['pnl'] > 0 else ""
    status = "GREEN" if v['pnl'] > 0 else "RED"
    print(f"  {status} {d}: {t}t, {wr2:.0f}% WR, {icon}${v['pnl']:.2f}")

# Check if new conservative config (Feb 14) trades exist
feb14 = [r for r in rows if r[0][:10] == '2026-02-14']
print(f"\nFeb 14 trades: {len(feb14)}")
if feb14:
    print("  (New conservative config: max_lot 0.02, $200 daily cap, 5-loss circuit breaker)")
