import csv
from datetime import datetime
from collections import defaultdict

trades = []
with open('xauusd/trade_log.csv', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 5:
            continue
        ts, direction, entry, exit_p, profit_str = parts[0], parts[1], parts[2], parts[3], parts[4]
        try:
            profit = float(profit_str)
            entry = float(entry)
            exit_p = float(exit_p)
            dt = datetime.fromisoformat(ts.replace('+00:00', ''))
            pip_diff = abs(exit_p - entry)
            trades.append({'dt': dt, 'dir': direction, 'entry': entry, 'exit': exit_p, 'profit': profit, 'pips': pip_diff})
        except:
            continue

print(f"=== XAUUSD Deep Dive ({len(trades)} trades) ===\n")

# 1. Biggest losses
print("--- Top 10 Biggest Losses ---")
losses = sorted([t for t in trades if t['profit'] < 0], key=lambda x: x['profit'])
for t in losses[:10]:
    hold = ""
    print(f"  {t['dt']} | {t['dir']} | Entry: {t['entry']:.2f} | Exit: {t['exit']:.2f} | Pips: {t['pips']:.2f} | P/L: ${t['profit']:.2f}")

# 2. Direction breakdown
print("\n--- Direction Breakdown ---")
buys = [t for t in trades if t['dir'] == 'BUY']
sells = [t for t in trades if t['dir'] == 'SELL']
for label, subset in [('BUY', buys), ('SELL', sells)]:
    if not subset:
        continue
    w = sum(1 for t in subset if t['profit'] > 0)
    pnl = sum(t['profit'] for t in subset)
    avg_w = sum(t['profit'] for t in subset if t['profit'] > 0) / max(w, 1)
    l_count = sum(1 for t in subset if t['profit'] < 0)
    avg_l = sum(t['profit'] for t in subset if t['profit'] < 0) / max(l_count, 1)
    print(f"  {label}: {len(subset)} trades | WR: {w/len(subset)*100:.0f}% | P/L: ${pnl:+.2f} | Avg W: ${avg_w:.2f} / Avg L: ${avg_l:.2f}")

# 3. Loss size distribution
print("\n--- Loss Size Distribution ---")
all_losses = [t['profit'] for t in trades if t['profit'] < 0]
if all_losses:
    brackets = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 500)]
    for low, high in brackets:
        count = sum(1 for l in all_losses if low <= abs(l) < high)
        total = sum(l for l in all_losses if low <= abs(l) < high)
        if count > 0:
            print(f"  ${low}-${high}: {count} trades | Total: ${total:.2f}")

# 4. Actual SL distances (in price terms)
print("\n--- Actual Stop Loss Distances (from entry to exit on losses) ---")
loss_pips = [t['pips'] for t in trades if t['profit'] < 0]
if loss_pips:
    print(f"  Min: ${min(loss_pips):.2f} | Max: ${max(loss_pips):.2f} | Avg: ${sum(loss_pips)/len(loss_pips):.2f} | Median: ${sorted(loss_pips)[len(loss_pips)//2]:.2f}")

# 5. Win distances
print("\n--- Actual Take Profit Distances (from entry to exit on wins) ---")
win_pips = [t['pips'] for t in trades if t['profit'] > 0]
if win_pips:
    print(f"  Min: ${min(win_pips):.2f} | Max: ${max(win_pips):.2f} | Avg: ${sum(win_pips)/len(win_pips):.2f} | Median: ${sorted(win_pips)[len(win_pips)//2]:.2f}")

# 6. The -$199 trade specifically
print("\n--- Catastrophic Trades (> $50 loss) ---")
big_losses = [t for t in trades if t['profit'] < -50]
for t in big_losses:
    sl_dist = abs(t['exit'] - t['entry'])
    print(f"  {t['dt']} | {t['dir']} | Entry: {t['entry']:.2f} -> Exit: {t['exit']:.2f}")
    print(f"    Distance: ${sl_dist:.2f} | Loss: ${t['profit']:.2f}")
    print(f"    Config SL is $40 but actual loss was ${abs(t['profit']):.2f} -- SL BLOWN!")

# 7. Hourly performance
print("\n--- Hourly P/L (UTC) ---")
hourly = defaultdict(list)
for t in trades:
    hourly[t['dt'].hour].append(t['profit'])
for h in sorted(hourly.keys()):
    t_list = hourly[h]
    pnl = sum(t_list)
    wr = sum(1 for x in t_list if x > 0) / len(t_list) * 100
    print(f"  {h:02d}:00 | {len(t_list):3d} trades | WR: {wr:.0f}% | P/L: ${pnl:+.2f}")

# 8. Daily trend
print("\n--- Daily P/L (last 10 days) ---")
daily = defaultdict(list)
for t in trades:
    daily[str(t['dt'].date())].append(t['profit'])
for d in sorted(daily.keys())[-10:]:
    t_list = daily[d]
    pnl = sum(t_list)
    w = sum(1 for x in t_list if x > 0)
    print(f"  {d} | {len(t_list):3d}t | WR:{w/len(t_list)*100:3.0f}% | ${pnl:+8.2f}")
